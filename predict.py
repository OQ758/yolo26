from ultralytics import YOLO
# 
model = YOLO('/home/un/桌面/QY/yolo26/runs/train/weights/best.pt')
source="/home/un/桌面/QY/yolo26/ultralytics/assets"
results = model(source,
                save=True, 
                project="/home/un/桌面/QY/yolo26/runs")  # list of Results objects