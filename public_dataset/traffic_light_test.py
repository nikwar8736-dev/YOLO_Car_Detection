import serial
import time

# Change COM5 to your actual ESP32-CAM port
esp = serial.Serial("COM5", 115200, timeout=1)

time.sleep(2)

try:
    while True:
        command = input("Enter R / Y / G / O / Q: ").strip().upper()

        if command == "Q":
            break

        if command in ["R", "Y", "G", "O"]:
            esp.write(command.encode())
            time.sleep(0.1)

            response = esp.readline().decode(
                errors="ignore"
            ).strip()

            print("ESP32:", response)

finally:
    esp.close()
