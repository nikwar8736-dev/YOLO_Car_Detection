
import socket
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import serial
from ultralytics import YOLO


class TrafficEngine:

    def __init__(
        self,
        host,
        com_port,
        model_path=None,
        port=81,
        baud_rate=115200,
        confidence=0.15,
        image_size=640,
        device=0
    ):
        root = Path(__file__).resolve().parents[1]

        self.host = host
        self.port = port
        self.com_port = com_port
        self.baud_rate = baud_rate

        self.model_path = Path(
            model_path or root / "models" / "best.pt"
        )

        self.confidence = confidence
        self.image_size = image_size
        self.device = device

        self.lock = threading.Lock()
        self.stop_event = threading.Event()

        self.frame = None
        self.count = 0
        self.signal = "WAITING"
        self.fps = 0.0
        self.latency = 0.0

        self.camera_status = "Disconnected"
        self.serial_status = "Disconnected"
        self.model_status = "Not loaded"

        self.history = []
        self.last_signal = None
        self.candidate_signal = None
        self.candidate_count = 0

        self.serial_connection = None
        self.camera_thread = None
        self.yolo_thread = None
        self.model = None

        self.started = False

    def start(self):

        if self.started:
            return

        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"YOLO model not found: {self.model_path}"
            )

        self.model = YOLO(str(self.model_path))
        self.model_status = "Loaded"

        self.stop_event.clear()

        try:
            self.serial_connection = serial.Serial(
                self.com_port,
                self.baud_rate,
                timeout=0.1
            )
            time.sleep(2)
            self.serial_status = "Connected"

        except Exception as e:
            self.serial_status = f"Error: {e}"
            self.serial_connection = None

        self.started = True

        self.camera_thread = threading.Thread(
            target=self.capture_stream,
            daemon=True
        )

        self.yolo_thread = threading.Thread(
            target=self.yolo_worker,
            daemon=True
        )

        self.camera_thread.start()
        self.yolo_thread.start()

    def update_signal(self, vehicle_count):

        if vehicle_count <= 1:
            signal = "G"
        elif vehicle_count == 2:
            signal = "Y"
        else:
            signal = "R"

        if signal == self.candidate_signal:
            self.candidate_count += 1
        else:
            self.candidate_signal = signal
            self.candidate_count = 1

        if self.candidate_count >= 3 and signal != self.last_signal:

            if self.serial_connection and self.serial_connection.is_open:
                try:
                    self.serial_connection.write(signal.encode())
                    self.serial_connection.flush()
                except Exception as e:
                    self.serial_status = f"Error: {e}"
                    self.serial_connection = None

            self.last_signal = signal

            with self.lock:
                self.history.append({
                    "time": time.strftime("%H:%M:%S"),
                    "vehicles": vehicle_count,
                    "signal": signal
                })

                self.history = self.history[-200:]

        with self.lock:
            self.signal = {
                "G": "GREEN",
                "Y": "YELLOW",
                "R": "RED"
            }[self.last_signal or signal]

    def capture_stream(self):

        sock = socket.socket(
            socket.AF_INET,
            socket.SOCK_STREAM
        )

        sock.settimeout(20)

        try:
            sock.connect((self.host, self.port))

            request = (
                "GET /stream HTTP/1.1\r\n"
                f"Host: {self.host}:{self.port}\r\n"
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

            read_until(b"\r\n\r\n")

            jpeg_buffer = b""

            self.camera_status = "Connected"

            while not self.stop_event.is_set():

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
                        with self.lock:
                            self.raw_frame = frame
                            self.frame_version = getattr(
                                self, "frame_version", 0
                            ) + 1

        except Exception as e:
            if not self.stop_event.is_set():
                self.camera_status = f"Error: {e}"

        finally:
            sock.close()

            if not self.stop_event.is_set():
                self.camera_status = "Disconnected"

    def yolo_worker(self):

        last_version = -1
        last_time = time.time()

        while not self.stop_event.is_set():

            with self.lock:
                frame = getattr(self, "raw_frame", None)
                version = getattr(self, "frame_version", 0)

                if frame is not None:
                    frame = frame.copy()

            if frame is None or version == last_version:
                time.sleep(0.005)
                continue

            last_version = version

            start = time.perf_counter()

            try:
                results = self.model.predict(
                    frame,
                    conf=self.confidence,
                    imgsz=self.image_size,
                    device=self.device,
                    verbose=False
                )

                annotated = results[0].plot()
                count = len(results[0].boxes)

                self.update_signal(count)

                cv2.putText(
                    annotated,
                    f"Vehicles: {count}",
                    (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    annotated,
                    f"Signal: {self.signal}",
                    (15, 65),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2
                )

                elapsed = time.perf_counter() - start
                now = time.time()

                with self.lock:
                    self.frame = annotated
                    self.count = count
                    self.latency = elapsed * 1000
                    self.fps = 1 / max(now - last_time, 0.001)

                last_time = now

            except Exception as e:
                self.model_status = f"Error: {e}"
                time.sleep(0.1)

    def snapshot(self):

        with self.lock:
            return {
                "frame": (
                    self.frame.copy()
                    if self.frame is not None
                    else None
                ),
                "count": self.count,
                "signal": self.signal,
                "fps": self.fps,
                "latency": self.latency,
                "camera": self.camera_status,
                "serial": self.serial_status,
                "model": self.model_status,
                "history": list(self.history)
            }

    def stop(self):

        if not self.started:
            return

        self.stop_event.set()

        if self.serial_connection:
            try:
                if self.serial_connection.is_open:
                    self.serial_connection.write(b"O")
                    self.serial_connection.flush()
                    self.serial_connection.close()
            except Exception:
                pass

        for thread in (self.camera_thread, self.yolo_thread):
            if thread and thread.is_alive():
                thread.join(timeout=2)

        self.started = False
        self.camera_status = "Disconnected"
        self.serial_status = "Disconnected"
