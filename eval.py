from ultralytics import YOLO

model = YOLO('/home/un/桌面/QY/yolo26/runs/train/weights/best.pt')
model.val(data='/home/un/桌面/QY/yolo26/ultralytics/cfg/datasets/VOC.yaml', save_json=True)