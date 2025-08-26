from ultralytics import YOLO
m = YOLO("yolov8n.pt")  # auto-downloads tiny model
res = m("https://ultralytics.com/images/bus.jpg", save=True, verbose=False)
print("OK. Saved to:", res[0].save_dir)
