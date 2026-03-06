from ultralytics import YOLO

def test_model():
    print("--- 1. กำลังโหลดโมเดลที่เทรนเสร็จแล้ว ---")
    # ⚠️ ตรวจสอบพาธ (Path) ของไฟล์ best.pt ให้ตรงกับในโฟลเดอร์ runs ของคุณ
    # หากคุณใช้โค้ดเดิมที่ผมให้ไป พาธน่าจะเป็นตามด้านล่างนี้ครับ
    model_path = "best.pt" 
    
    try:
        model = YOLO(model_path)
        print("โหลดโมเดลสำเร็จ!")
    except Exception as e:
        print(f"❌ ไม่พบไฟล์โมเดล ตรวจสอบพาธให้ถูกต้องนะครับ: {e}")
        return

    print("\n--- 2. กำลังวิเคราะห์รูปภาพ ---")
    # เปลี่ยนชื่อไฟล์ตรงนี้ให้ตรงกับรูปภาพที่คุณนำมาใส่ในโฟลเดอร์
    image_to_test = "DSCF4192.jpg" 
    
    # สั่งให้โมเดลทำนายผล
    # conf=0.25 หมายถึง ให้แสดงผลเฉพาะสิ่งที่โมเดลมั่นใจ 25% ขึ้นไป (ปรับได้ครับ)
    results = model(image_to_test, conf=0.25) 

    print("\n--- 3. แสดงและบันทึกผลลัพธ์ ---")
    for result in results:
        # เปิดหน้าต่างป๊อปอัปโชว์รูปภาพที่โมเดลวาดกรอบให้แล้ว
        result.show()  
        
        # บันทึกรูปภาพผลลัพธ์ลงในโฟลเดอร์โปรเจ็กต์
        result.save(filename="result_output.jpg") 
        
    print("\n🎉 ทดสอบเสร็จสิ้น! คุณสามารถดูภาพผลลัพธ์ได้ที่ไฟล์ result_output.jpg ครับ")

if __name__ == '__main__':
    test_model()