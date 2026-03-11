test.py ทดสอบโมเดลที่ train มา (best.pt) \
train.py ใช้ train YOLOv11 จาก roboflow \
best.pt โมเดลที่ train จาก train.py เสร็จแล้ว \
label รูปจาก roboflow > นำ Datasets จาก roboflow มา train ใน train.py > ได้ best.pt มาทดสอบตรวจจับอุปกรณ์ใน test.py เพื่อดูความถูกต้อง
