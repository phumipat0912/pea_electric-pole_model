from roboflow import Roboflow
from ultralytics import YOLO
import os

def main():
    print("--- 1. กำลังดาวน์โหลด Dataset จาก Roboflow ---")
    
    rf = Roboflow(api_key="Blp44Q5g6Mli0elMU5My")
    project = rf.workspace("ag-tech-ai").project("-ltldn")
    dataset = project.version(1).download("yolov11")
    
    print(f"\nดาวน์โหลดสำเร็จ! ข้อมูลถูกเก็บไว้ที่: {dataset.location}")

    print("\n--- 2. กำลังเริ่มต้นเทรนโมเดล YOLO11 ---")
    
    # โหลดโมเดล YOLO11 ขนาด Nano (yolo11n.pt) 
    # ทันทีที่รัน โค้ดจะไปดาวน์โหลดไฟล์ yolo11n.pt 
    model = YOLO('yolo11n.pt') 

    # ชี้เป้าหมายไปที่ไฟล์ data.yaml ที่ดาวน์โหลดมา
    data_path = os.path.join(dataset.location, "data.yaml")

    # สั่งให้โมเดลเริ่มเรียนรู้จากข้อมูล
    results = model.train(
        data=data_path,
        epochs=100,          # จำนวนรอบในการฝึก (เริ่มที่ 50 ก่อนเพื่อดูแนวโน้ม)
        imgsz=640,          # ขนาดรูปภาพที่ใช้ฝึก (พิกเซล)
        batch=8,            # จำนวนภาพที่ป้อนให้โมเดลเรียนรู้พร้อมกันใน 1 ครั้ง
        project="runs",     # โฟลเดอร์หลักสำหรับเก็บโมเดลที่เทรนเสร็จ
        name="pea_electric-pole_model" # ชื่อโฟลเดอร์ย่อยสำหรับโปรเจ็กต์นี้
    )
    
    print("\n--- 🎉 การเทรนเสร็จสมบูรณ์! ---")
    print("โมเดลที่ฉลาดที่สุดของคุณ (best.pt) ถูกบันทึกไว้ในโฟลเดอร์ runs/pea_yolo11_model/weights/ เรียบร้อยแล้วครับ")

# จำเป็นต้องมีบรรทัดนี้สำหรับการเทรนบน Windows
if __name__ == '__main__':
    main()