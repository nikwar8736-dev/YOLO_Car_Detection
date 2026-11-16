
import socket
import cv2
import numpy as np

HOST = "10.179.122.92"
PORT = 81

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.settimeout(15)

print("Connecting to ESP32-CAM...")

sock.connect((HOST, PORT))

request = (
    "GET /stream HTTP/1.1\r\n"
    f"Host: {HOST}:{PORT}\r\n"
    "Connection: keep-alive\r\n"
    "\r\n"
)

sock.sendall(request.encode())

buffer = b""

def receive_more():
    global buffer
    data = sock.recv(8192)
    if not data:
        raise ConnectionError("ESP32-CAM disconnected")
    buffer += data

def read_until(marker):
    global buffer
    while marker not in buffer:
        receive_more()

    index = buffer.index(marker)
    result = buffer[:index]
    buffer = buffer[index + len(marker):]
    return result

def read_exact(size):
    global buffer
    while len(buffer) < size:
        receive_more()

    result = buffer[:size]
    buffer = buffer[size:]
    return result

# Read HTTP response headers
headers = read_until(b"\r\n\r\n")
print(headers.decode("latin1"))
print("Stream connected. Opening video...")

try:
    while True:

        # Read HTTP chunk size
        line = read_until(b"\r\n")
        chunk_size = int(line.split(b";")[0], 16)

        if chunk_size == 0:
            break

        # Read HTTP chunk payload
        chunk = read_exact(chunk_size)

        # Remove chunk-ending CRLF
        read_exact(2)

        buffer_jpeg = getattr(receive_more, "jpeg_buffer", b"")
        buffer_jpeg += chunk
        receive_more.jpeg_buffer = buffer_jpeg

        while True:
            start = buffer_jpeg.find(b"\xff\xd8")

            if start == -1:
                buffer_jpeg = b""
                break

            end = buffer_jpeg.find(b"\xff\xd9", start)

            if end == -1:
                buffer_jpeg = buffer_jpeg[start:]
                break

            jpg = buffer_jpeg[start:end + 2]
            buffer_jpeg = buffer_jpeg[end + 2:]

            frame = cv2.imdecode(
                np.frombuffer(jpg, dtype=np.uint8),
                cv2.IMREAD_COLOR
            )

            if frame is not None:
                cv2.imshow("ESP32-CAM LIVE", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                raise KeyboardInterrupt

except KeyboardInterrupt:
    print("Video stopped.")

except Exception as e:
    print("Stream error:", e)

finally:
    sock.close()
    cv2.destroyAllWindows()
