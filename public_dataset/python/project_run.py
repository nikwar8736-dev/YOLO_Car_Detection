import serial
import socket
import threading
import time
import ipaddress
import cv2
import numpy as np
from ultralytics import YOLO

# ---------------- CONFIGURATION ----------------

HOST = "10.179.122.92"
PORT = 81

MODEL_PATH = (
    r"C:\Users\DELL\Documents\YOLO_Car_Detection"
    r"\public_dataset\runs\runs\toy_car_detection"
    r"\weights\best.pt"
)

COM_PORT = "COM7"

CONFIDENCE = 0.15
IMAGE_SIZE = 640
DEVICE = 0

# ---------------- ESP8266 LCD (UDP) ----------------
# MUST be the real IP shown on the ESP8266 LCD / Serial Monitor.
# Leave as None (or a placeholder) to disable the LCD safely.
LCD_IP = "10.179.122.xx"
LCD_PORT = 4210

# ---------------- STREAM TUNING ----------------
SOCKET_TIMEOUT = 5          # seconds without data before reconnecting
RECONNECT_DELAY = 2         # seconds between reconnect attempts

# ---------------- SHARED DATA ----------------

lock = threading.Lock()

latest_frame = None
latest_annotated = None
frame_version = 0

running = True

# ---------------- LOAD MODEL ----------------

print("Loading YOLO model...")

model = YOLO(MODEL_PATH)

print("Model loaded.")
print("Classes:", model.names)

# Warm-up so the first real frame isn't slow
try:
    model.predict(
        np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8),
        imgsz=IMAGE_SIZE,
        device=DEVICE,
        verbose=False
    )
    print("Model warmed up.")
except Exception as e:
    print("Warm-up skipped:", e)

# ---------------- SERIAL CONNECTION ----------------

esp = serial.Serial(COM_PORT, 115200, timeout=0.1)
time.sleep(2)

last_signal = None
candidate_signal = None
candidate_count = 0
last_duration = None

# ---------------- LCD SENDER ----------------

lcd_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
lcd_enabled = False
lcd_error_shown = False

try:
    ipaddress.ip_address(LCD_IP)
    lcd_enabled = True
    print(f"LCD enabled -> {LCD_IP}:{LCD_PORT}")
except Exception:
    print(f"LCD disabled: '{LCD_IP}' is not a valid IP. "
          "Set LCD_IP to the address shown on the ESP8266.")


def send_to_lcd(command):
    """Never raises and never spams the console."""
    global lcd_error_shown

    if not lcd_enabled:
        return

    try:
        lcd_sock.sendto(command.encode("ascii"), (LCD_IP, LCD_PORT))
    except Exception as e:
        if not lcd_error_shown:
            print("LCD send error (shown once):", e)
            lcd_error_shown = True


def update_traffic_light(vehicle_count):

    global last_signal, candidate_signal, candidate_count, last_duration

    if vehicle_count == 0:
        signal = 'G'
        duration = 20
        traffic = "LOW"

    elif vehicle_count == 1:
        signal = 'G'
        duration = 25
        traffic = "LOW"

    elif vehicle_count == 2:
        signal = 'Y'
        duration = 35
        traffic = "MEDIUM"

    elif vehicle_count == 3:
        signal = 'R'
        duration = 40
        traffic = "HIGH"

    else:
        signal = 'R'
        duration = 45
        traffic = "HIGH"

    key = (signal, duration)

    if key == candidate_signal:
        candidate_count += 1
    else:
        candidate_signal = key
        candidate_count = 1

    if candidate_count >= 3 and key != (last_signal, last_duration):

        command = f"{vehicle_count},{traffic},{signal},{duration}\n"

        esp.write(command.encode("ascii"))
        esp.flush()
        send_to_lcd(command)

        print(f"Vehicles: {vehicle_count} | Traffic signal: {signal}")
        print(f"Timer duration: {duration}s\n")

        last_signal = signal
        last_duration = duration

# ---------------- STREAM READER ----------------

def capture_stream():

    global latest_frame, frame_version, running

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    sock.settimeout(SOCKET_TIMEOUT)

    print("Connecting to ESP32-CAM...")

    try:

        sock.connect((HOST, PORT))

        request = (
            "GET /stream HTTP/1.1\r\n"
            f"Host: {HOST}:{PORT}\r\n"
            "Connection: keep-alive\r\n\r\n"
        )

        sock.sendall(request.encode())

        buffer = b""

        def receive_more():
            nonlocal buffer

            data = sock.recv(8192)

            if not data:
                raise ConnectionError("Camera disconnected")

            buffer += data

        def read_until(marker):
            nonlocal buffer

            while marker not in buffer:
                receive_more()

            index = buffer.index(marker)

            result = buffer[:index]
            buffer = buffer[index + len(marker):]

            return result

        def read_exact(size):
            nonlocal buffer

            while len(buffer) < size:
                receive_more()

            result = buffer[:size]
            buffer = buffer[size:]

            return result

        headers = read_until(b"\r\n\r\n")

        print(headers.decode("latin1"))
        print("Camera stream connected.")

        jpeg_buffer = b""

        while running:

            line = read_until(b"\r\n")

            chunk_size = int(
                line.split(b";")[0].strip(), 16
            )

            if chunk_size == 0:
                break

            chunk = read_exact(chunk_size)
            read_exact(2)

            jpeg_buffer += chunk

            while True:

                start = jpeg_buffer.find(b"\xff\xd8")

                if start == -1:
                    jpeg_buffer = (
                        jpeg_buffer[-1:]
                        if jpeg_buffer.endswith(b"\xff")
                        else b""
                    )
                    break

                end = jpeg_buffer.find(
                    b"\xff\xd9", start + 2
                )

                if end == -1:
                    jpeg_buffer = jpeg_buffer[start:]
                    break

                jpg = jpeg_buffer[start:end + 2]
                jpeg_buffer = jpeg_buffer[end + 2:]

                frame = cv2.imdecode(
                    np.frombuffer(jpg, dtype=np.uint8),
                    cv2.IMREAD_COLOR
                )

                if frame is not None:

                    with lock:
                        latest_frame = frame
                        frame_version += 1

    except Exception as e:
        print("Camera error:", e)

    finally:
        sock.close()


def capture_loop():
    """Keeps the camera connected: reconnects after any error/timeout."""

    while running:
        capture_stream()

        if running:
            print(f"Reconnecting in {RECONNECT_DELAY} seconds...")
            time.sleep(RECONNECT_DELAY)

# ---------------- YOLO INFERENCE ----------------

def yolo_worker():

    global latest_annotated, running

    last_processed_version = -1

    while running:

        with lock:

            if latest_frame is None:
                frame = None
                version = -1
            else:
                frame = latest_frame.copy()
                version = frame_version

        if frame is None or version == last_processed_version:
            time.sleep(0.002)
            continue

        last_processed_version = version

        try:
            results = model.predict(
                frame,
                conf=CONFIDENCE,
                imgsz=IMAGE_SIZE,
                device=DEVICE,
                verbose=False
            )
        except Exception as e:
            print("YOLO error:", e)
            time.sleep(0.05)
            continue

        annotated = results[0].plot()

        count = len(results[0].boxes)

        update_traffic_light(count)

        cv2.putText(
            annotated,
            f"Vehicles detected: {count}",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.putText(
            annotated,
            f"Signal: {last_signal or 'WAITING'}",
            (15, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        with lock:
            latest_annotated = annotated


# ---------------- START THREADS ----------------

camera_thread = threading.Thread(
    target=capture_loop,          # auto-reconnect (was capture_stream)
    daemon=True
)

yolo_thread = threading.Thread(
    target=yolo_worker,
    daemon=True
)

camera_thread.start()
yolo_thread.start()

# ---------------- DISPLAY LOOP ----------------

print("Starting live YOLO detection. Press Q to quit.")

try:

    while running:

        with lock:

            if latest_annotated is not None:
                display_frame = latest_annotated.copy()

            elif latest_frame is not None:
                display_frame = latest_frame.copy()

            else:
                display_frame = None

        if display_frame is not None:
            cv2.imshow(
                "ESP32-CAM YOLO Traffic Control",
                display_frame
            )

        if cv2.waitKey(5) & 0xFF == ord("q"):
            break

except KeyboardInterrupt:
    pass

finally:

    running = False

    send_to_lcd("0,OFF,O,0\n")

    if esp.is_open:
        esp.write(b'O')
        esp.close()

    camera_thread.join(timeout=2)
    yolo_thread.join(timeout=2)

    cv2.destroyAllWindows()

    print("Detection stopped.")