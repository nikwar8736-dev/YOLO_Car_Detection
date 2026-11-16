
import cv2
import requests
import numpy as np

URL = "http://10.29.155.92:81/stream"

print("Connecting...")

response = requests.get(URL, stream=True, timeout=(10, 30))
print("HTTP Status:", response.status_code)

buffer = b""

for chunk in response.iter_content(chunk_size=4096):

    buffer += chunk

    start = buffer.find(b'\xff\xd8')
    end = buffer.find(b'\xff\xd9')

    if start != -1 and end != -1 and end > start:

        jpg = buffer[start:end + 2]
        buffer = buffer[end + 2:]

        frame = cv2.imdecode(
            np.frombuffer(jpg, dtype=np.uint8),
            cv2.IMREAD_COLOR
        )

        if frame is not None:
            cv2.imshow("ESP32-CAM Live Feed", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

response.close()
cv2.destroyAllWindows()
