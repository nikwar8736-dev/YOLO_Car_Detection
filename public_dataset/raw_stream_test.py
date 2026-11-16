
import socket

HOST = "10.179.122.92"
PORT = 81

print("Connecting to ESP32-CAM...")

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.settimeout(10)

try:
    sock.connect((HOST, PORT))

    request = (
        "GET /stream HTTP/1.1\r\n"
        f"Host: {HOST}:{PORT}\r\n"
        "Connection: keep-alive\r\n"
        "\r\n"
    )

    sock.sendall(request.encode())

    print("Request sent. Waiting for response...")

    data = sock.recv(4096)

    print("\nSERVER RESPONSE:")
    print(data[:1500].decode("latin1", errors="replace"))

    print("\nReceived bytes:", len(data))

except Exception as e:
    print("ERROR:", repr(e))

finally:
    sock.close()

