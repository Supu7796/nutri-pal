import cv2
import pyttsx3
import threading
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# 1. 加载模型与语音引擎
model = YOLO('best.pt')
engine = pyttsx3.init()

# 防止重复说话的变量
last_item = ""
frame_count = 0

menu_db = {
    'redbull': {'name': '红牛饮料', 'price': 15.0, 'cal': 220, 'tips': '提神醒脑'},
    'oreo':    {'name': '奥利奥', 'price': 6.0, 'cal': 480, 'tips': '控制热量'},
    'orange':  {'name': '新鲜甜橙', 'price': 3.5, 'cal': 45, 'tips': '补充维C'}
}

def speak(text):
    """线程化语音播报，不卡顿"""
    def run():
        engine.say(text)
        engine.runAndWait()
    threading.Thread(target=run, daemon=True).start()

def cv2_img_add_text(img, text, left, top, textColor=(255, 255, 255), textSize=25):
    if isinstance(img, np.ndarray):
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("simhei.ttf", textSize, encoding="utf-8")
    except:
        font = ImageFont.load_default()
    draw.text((left, top), text, (textColor[2], textColor[1], textColor[0]), font=font)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    results = model(frame, stream=True, conf=0.3)
    total_price = 0
    names = []

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = model.names[int(box.cls[0])]
            if label in menu_db:
                item = menu_db[label]
                total_price += item['price']
                names.append(item['name'])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                frame = cv2_img_add_text(frame, f"{item['name']}", x1, y1 - 30, (0, 255, 0))

    # --- 语音防抖逻辑 ---
    current_str = ",".join(sorted(set(names)))
    if current_str and current_str != last_item:
        frame_count += 1
        if frame_count > 10: # 稳定10帧再说话
            speak(f"识别成功，总计{total_price}元")
            last_item = current_str
            frame_count = 0
    elif not current_str:
        last_item = ""

    # --- 绘制现代化侧边栏 UI ---
    sidebar = frame.copy()
    cv2.rectangle(sidebar, (0, 0), (300, 200), (40, 40, 40), -1)
    frame = cv2.addWeighted(sidebar, 0.7, frame, 0.3, 0)
    
    frame = cv2_img_add_text(frame, "🍱 AI 智慧结算台", 15, 15, (255, 255, 0), 28)
    frame = cv2_img_add_text(frame, f"总金额: ￥{total_price:.1f}", 15, 65, (0, 255, 0), 35)
    tip = menu_db[model.names[int(r.boxes[0].cls[0])]]['tips'] if names else "请放入菜品..."
    frame = cv2_img_add_text(frame, f"提示: {tip}", 15, 125, (200, 200, 200), 20)

    cv2.imshow("Smart Canteen V2", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()