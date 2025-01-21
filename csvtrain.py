import os
import csv

# ฟังก์ชันสำหรับสร้างไฟล์ CSV
def create_csv(directory, output_csv):
    rows = []
    no = 1

    # เดินสำรวจโฟลเดอร์
    for split in os.listdir(directory):  # Train, Valid
        split_path = os.path.join(directory, split)
        if os.path.isdir(split_path):
            for model_no in os.listdir(split_path):  # model_no
                model_no_path = os.path.join(split_path, model_no)
                if os.path.isdir(model_no_path):
                    for model in os.listdir(model_no_path):  # model
                        model_path = os.path.join(model_no_path, model)
                        if os.path.isdir(model_path):
                            for class_degree in os.listdir(model_path):  # class_degree
                                class_degree_path = os.path.join(model_path, class_degree)
                                if os.path.isdir(class_degree_path):
                                    for class_part in os.listdir(class_degree_path):  # p1, p2, ..., p15
                                        class_part_path = os.path.join(class_degree_path, class_part)
                                        if os.path.isdir(class_part_path):
                                            for img_file in os.listdir(class_part_path):  # Images in p1, p2, ...
                                                img_path = os.path.join(class_part_path, img_file)
                                                if os.path.isfile(img_path):
                                                    # เพิ่มข้อมูลในแถว
                                                    rows.append({
                                                        'no': no,
                                                        'split': split,
                                                        'model_no': model_no,
                                                        'model': model,
                                                        'class_degree': class_degree,
                                                        'class_part': class_part,  # เพิ่ม class_part
                                                        'img_path': img_path
                                                    })
                                                    no += 1

    # เขียนข้อมูลลง CSV
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['no', 'split', 'model_no', 'model', 'class_degree', 'class_part', 'img_path']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

# ตัวอย่างการใช้งาน
directory = "/home/phu/Desktop/Data_test_server"  # โฟลเดอร์หลักที่ต้องการสำรวจ
output_csv = "output.csv"  # ชื่อไฟล์ CSV ที่ต้องการสร้าง
create_csv(directory, output_csv)
print(f"CSV file '{output_csv}' has been created successfully.")
