import cv2
import urllib.request

URL = "http://10.29.155.92:81"

print("Testing HTTP connection...")

try:
    response = urllib.request.urlopen(URL, timeout=10)
    print("HTTP Status:", response.status)
    print("Content Type:", response.headers.get("Content-Type"))

    data = response.read(1024)
    print("Received bytes:", len(data))

except Exception as e:
    print("HTTP Error:", e)

print("\nTesting OpenCV...")

cap = cv2.VideoCapture(URL, cv2.CAP_FFMPEG)

print("Camera opened:", cap.isOpened())

if cap.isOpened():
    ret, frame = cap.read()

    print("Frame received:", ret)

    if ret:
        print("Resolution:", frame.shape)

        cv2.imshow("ESP32-CAM", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

cap.release()