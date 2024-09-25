from ultralytics import YOLO

model = YOLO('car.pt')  # 将这里换成你模型所在的路径
path = model.export(format="rknn", opset=12)