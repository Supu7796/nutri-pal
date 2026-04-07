import cv2
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# 1. 加载模型（确保 best.pt 在当前文件夹）
model = YOLO('best.pt')

# 2. 营养数据库：集成价格、热量及微量元素
menu_db = {
    'redbull': {
        'name': '红星二锅头', 'price': 15.0, 
        'cal': 220, 'protein': '0g', 'carb': '5.2g', 
        'tips': '酒精热量高，建议适量饮用'
    },
    'oreo': {
        'name': '奥利奥饼干', 'price': 6.0, 
        'cal': 480, 'protein': '4.8g', 'carb': '68g', 
        'tips': '碳水较高，建议搭配高纤维蔬菜'
    },
    'orange': {
        'name': '维他命水', 'price': 4.0, 
        'cal': 45, 'protein': '0.9g', 'carb': '11g', 
        'tips': '维C之王，理想的能量补充'
    }
}

def cv2_img_add_text(img, text, left, top, textColor=(255, 255, 255), textSize=20):
    """支持中文显示的辅助函数"""
    if isinstance(img, np.ndarray):
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("simhei.ttf", textSize, encoding="utf-8")
    except:
        font = ImageFont.load_default()
    draw.text((left, top), text, (textColor[2], textColor[1], textColor[0]), font=font)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

# 开启摄像头
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    # 执行检测
    results = model(frame, stream=True, conf=0.15)
    
    # 核心修复：每一帧开始前初始化所有汇总变量
    total_price = 0.0
    total_cal = 0
    current_items = []

    for r in results:
        for box in r.boxes:
            # 获取识别结果
            cls_id = int(box.cls[0])
            label_eng = model.names[cls_id]
            conf = float(box.conf[0])
            
            # 绘制识别框
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 匹配数据库并累加
            if label_eng in menu_db:
                item = menu_db[label_eng]
                total_price += item['price']
                total_cal += item['cal'] # 累加总热量
                current_items.append(item)
                
                # 框体上方显示名称
                frame = cv2_img_add_text(frame, f"{item['name']}", x1, y1 - 25, (0, 255, 0), 20)

    # 3. 绘制半透明黑色 UI 背景面板
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (350, 200), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

    # 4. 渲染汇总数据与健康建议
    frame = cv2_img_add_text(frame, "🥗 营养健康小助手", 10, 10, (255, 255, 0), 25)
    frame = cv2_img_add_text(frame, f"总计能量: {total_cal} kcal", 10, 50, (0, 255, 0), 22)
    
    if current_items:
        last = current_items[-1]
        # 展示详细营养元素
        info = f"蛋白质: {last['protein']} | 碳水: {last['carb']}"
        frame = cv2_img_add_text(frame, info, 10, 85, (0, 255, 255), 18)
        frame = cv2_img_add_text(frame, f"营养建议: {last['tips']}", 10, 125, (255, 255, 255), 18)
        frame = cv2_img_add_text(frame, f"结算金额: ¥{total_price:.1f}", 10, 160, (50, 50, 255), 22)
    else:
        frame = cv2_img_add_text(frame, "等待扫描中...", 10, 100, (200, 200, 200), 18)

    cv2.imshow("Smart Canteen AI", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()