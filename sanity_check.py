import sys
print("Python:", sys.version.split()[0])
try:
    import ultralytics, torch
    print("Ultralytics OK:", ultralytics.__version__)
    print("PyTorch OK:", torch.__version__)
except Exception as e:
    print("Import error:", e)

try:
    from openvino.runtime import Core
    ie = Core()
    print("OpenVINO devices:", ie.available_devices)
except Exception as e:
    print("OpenVINO error:", e)
