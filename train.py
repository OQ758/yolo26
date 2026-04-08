from ultralytics import YOLO
import torch
torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
torch.jit._state.disable()
# 视频讲解链接：https://space.bilibili.com/3546660493855256/channel/collectiondetail?sid=3229289
model = YOLO('/home/un/桌面/QY/yolo26/ultralytics/cfg/models/26/yolo26.yaml')

# Train the model
results = model.train(
  data='/home/un/桌面/QY/yolo26/ultralytics/cfg/datasets/VOC.yaml',
  epochs=70, 
  batch=32, # 批量大小，根据GPU内存适当调整（如果256太大，可以尝试64）
  imgsz=640,
  scale=0.5,  # S:0.9; M:0.9; L:0.9; X:0.9 缩放因子，针对模型的规模进行调整
  mosaic=1.0,
  mixup=0.0,  # S:0.05; M:0.15; L:0.15; X:0.2
  copy_paste=0.1,  # S:0.15; M:0.4; L:0.5; X:0.6
  device="0",
  project='/home/un/桌面/QY/yolo26/runs',
)

# # Evaluate model performance on the validation set
# metrics = model.val()

# # Perform object detection on an image
# results = model("path/to/image.jpg")
# results[0].show()
