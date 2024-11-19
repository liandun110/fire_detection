# 说明
- 算法：YOLOv5

# 使用pytorch模型进行fire detection
```bash
python detect.py --weights model.pt --device=cpu --source=img.png
```

# 将pytorch模型转换成onnx模型
```bash
python export.py --weights model.pt --include onnx --opset 16
```

# 使用onnx模型进行fire detection
```bash
python detect.py --weights model.onnx --device=cpu --source=img.png
```