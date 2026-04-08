import time
from ultralytics import YOLO

# ===================== 你的配置 ======================
IMG_PATH = "ultralytics/assets/bus.jpg"  # 测试图片
MODEL1 = "yolov5n.pt"                    # 你的第一个权重
MODEL2 = "yolo26n.pt"                    # 你的第二个权重
TEST_TIMES = 5                           # 测5次取平均，更准
# =====================================================

print("=" * 60)
print("🚀 YOLOv5n vs YOLO26n 检测速度 & 置信度 对比")
print("=" * 60)

# ---------------------- 测试模型1：YOLOv5n ----------------------
print("\n【1】测试 YOLOv5n")
model1 = YOLO(MODEL1)
model1(IMG_PATH, verbose=False)  # 预热

start = time.time()
for _ in range(TEST_TIMES):
    res1 = model1(IMG_PATH, verbose=False)
end = time.time()
t1 = (end - start) / TEST_TIMES * 1000
c1 = max([box.conf[0].item() for box in res1[0].boxes]) if len(res1[0].boxes) else 0

print(f"✅ 平均速度：{t1:.1f} ms")
print(f"✅ 最高置信度：{c1:.3f}")
res1[0].save(filename="result_yolov5n.jpg")

# ---------------------- 测试模型2：YOLO26n ----------------------
print("\n【2】测试 YOLO26n")
model2 = YOLO(MODEL2)
model2(IMG_PATH, verbose=False)  # 预热

start = time.time()
for _ in range(TEST_TIMES):
    res2 = model2(IMG_PATH, verbose=False)
end = time.time()
t2 = (end - start) / TEST_TIMES * 1000
c2 = max([box.conf[0].item() for box in res2[0].boxes]) if len(res2[0].boxes) else 0

print(f"✅ 平均速度：{t2:.1f} ms")
print(f"✅ 最高置信度：{c2:.3f}")
res2[0].save(filename="result_yolo26n.jpg")

# ---------------------- 最终对比 ----------------------
print("\n" + "=" * 60)
print("📊 最终对比结果")
print("=" * 60)
print(f"YOLOv5n   速度：{t1:>6.1f} ms  | 最高置信度：{c1:.3f}")
print(f"YOLO26n   速度：{t2:>6.1f} ms  | 最高置信度：{c2:.3f}")

faster = "YOLOv5n" if t1 < t2 else "YOLO26n"
better = "YOLOv5n" if c1 > c2 else "YOLO26n"

print(f"\n🏎️  速度更快：{faster}")
print(f"🎯 置信度更高：{better}")
print("=" * 60)

# 自动弹出两张图
res1[0].show()
res2[0].show()