from ultralytics import YOLO
import torch

if __name__ == "__main__":

    print("GPU:", torch.cuda.get_device_name(0))

    model = YOLO("yolo26n.pt")

    results = model.train(
        data="data.yaml",
        epochs=100,
        imgsz=640,
        batch=8,
        device=0,
        workers=2,
        patience=20,
        cache=False,
        amp=True,
        project="../runs",
        name="toy_car_detection"
    )

    print("Training completed!")
