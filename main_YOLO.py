import os
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

# --- متغیر سراسری برای رسم چندضلعی ---
clicked_points = []

def mouse_callback(event, x, y, flags, param):
    """ثبت نقاط کلیک شده برای تشکیل چندضلعی"""
    global clicked_points
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_points.append([x, y])
        print(f"نقطه ثبت شد -> X: {x}, Y: {y}")


def get_polygon_from_video(video_path):
    """پنجره تعاملی برای رسم چندضلعی روی فریم اول ویدیو"""
    global clicked_points
    clicked_points = []

    cap = cv2.VideoCapture(video_path)
    for i in range(5):
        success, frame = cap.read()
    if not success:
        print("خطا در باز کردن ویدیو!")
        return None

    window_name = "Select ROI Polygon - Click & Press ENTER"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)

    print("\n--- راهنمای ابزار انتخاب ناحیه ---")
    print("1. با کلیک چپ، گوشه‌های محدوده را انتخاب کنید.")
    print("2. دکمه ENTER یا c را بر/ای تایید بزنید.")
    print("3. دکمه r برای ریست و q برای خروج.\n")

    while True:
        display_frame = frame.copy()

        if len(clicked_points) > 0:
            for pt in clicked_points:
                cv2.circle(display_frame, tuple(pt), 6, (0, 0, 255), -1)
            if len(clicked_points) > 1:
                pts_np = np.array(clicked_points, np.int32)
                cv2.polylines(display_frame, [pts_np], isClosed=False, color=(0, 255, 0), thickness=2)

        cv2.imshow(window_name, display_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 13 or key == ord('c'):
            if len(clicked_points) >= 3: break
        elif key == ord('r'):
            clicked_points = []
        elif key == ord('q'):
            clicked_points = []
            break

    cap.release()
    cv2.destroyAllWindows()

    if len(clicked_points) >= 3:
        return np.array(clicked_points, np.int32)
    return None


class VideoCursor:
    """مدیریت ایمن باز و بسته کردن فایل ویدیویی"""

    def __init__(self, file_addr):
        self.addr = file_addr

    def __enter__(self):
        self.cur = cv2.VideoCapture(self.addr)
        return self.cur

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cur.release()
        cv2.destroyAllWindows()

def process_traffic_video(video_path, model_name, polygon_roi):
    """پردازش بهینه‌شده با ماسک تصویر و ردیابی پایدار"""
    print(f"\nدر حال بارگذاری مدل {model_name}...")
    model = YOLO(model_name)

    counts = {'car': 0, 'motorcycle': 0, 'bus': 0, 'truck': 0}
    track_frame_counts = {}
    counted_ids = set()
    MIN_CONSECUTIVE_FRAMES = 3

    with VideoCursor(video_path) as cap:
        success, first_frame = cap.read()
        if not success: return counts

        # ۱. ساخت ماسک سیاه و سفید برای بهینه‌سازی پردازش
        mask = np.zeros(first_frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [polygon_roi], 255)

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # بازگشت به فریم اول

        while cap.isOpened():
            success, frame = cap.read()
            if not success: break

            # ۲. اعمال ماسک: سیاه کردن خارج از چندضلعی قبل از ارسال به YOLO
            masked_frame = cv2.bitwise_and(frame, frame, mask=mask)

            results = model.track(masked_frame, persist=True, tracker='bytetrack.yaml', verbose=False,conf=0.15)
            current_frame_ids_in_roi = set()

            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                ids = results[0].boxes.id.cpu().numpy()
                classes = results[0].boxes.cls.cpu().numpy()

                for box, track_id, cls in zip(boxes, ids, classes):
                    x1, y1, x2, y2 = box
                    track_id = int(track_id)
                    class_name = model.names[int(cls)]

                    # استانداردسازی نام کلاس‌ها
                    if class_name in ['motorbike', 'bicycle']: class_name = 'motorcycle'
                    if class_name not in counts: continue

                    # نقطه مرجع (وسط پایین کادر)
                    center_x, center_y = int((x1 + x2) / 2), int(y2)

                    # ۳. بررسی حضور دقیق در چندضلعی
                    if cv2.pointPolygonTest(polygon_roi, (center_x, center_y), False) >= 0:
                        current_frame_ids_in_roi.add(track_id)

                        if track_id not in counted_ids:
                            track_frame_counts[track_id] = track_frame_counts.get(track_id, 0) + 1
                            if track_frame_counts[track_id] >= MIN_CONSECUTIVE_FRAMES:
                                counts[class_name] += 1
                                counted_ids.add(track_id)

                    # رسم گرافیک (سبز برای شمرده شده، آبی برای در حال بررسی)
                    color = (0, 255, 0) if track_id in counted_ids else (255, 0, 0)
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    cv2.putText(frame, f"{class_name} {track_id}", (int(x1), int(y1)-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # ۴. پاکسازی نویزها
            disappeared_ids = set(track_frame_counts.keys()) - current_frame_ids_in_roi
            for d_id in disappeared_ids:
                if d_id not in counted_ids: track_frame_counts[d_id] = 0

            # رسم کادر چندضلعی و آمار زنده
            cv2.polylines(frame, [polygon_roi], isClosed=True, color=(0, 255, 255), thickness=2)
            cv2.putText(frame, f"Cars: {counts['car']} | Motors: {counts['motorcycle']} | Buses: {counts['bus']}",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow('Integrated Traffic Analysis', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    return counts


if __name__ == '__main__':
    # file_name = '12012 Shoush - 17Shahrivar-2026-09-07_11h30min00s130ms.asf'
    here = os.getcwd()
    os.chdir('video')
    for fn in os.listdir():
        file_name = fn
        os.chdir(here)
        video_path = os.path.join('video' , file_name)

        # اجرای فاز اول: کالیبراسیون و انتخاب چندضلعی
        roi_polygon = get_polygon_from_video(video_path)

        if roi_polygon is not None:
            print("\nمحدوده با موفقیت ثبت شد. در حال شروع پردازش اصلی...")

            # اجرای فاز دوم: ردیابی و شمارش\
            final_stats = process_traffic_video(video_path, 'yolov8n.pt', roi_polygon)

            print("\n=== آمار نهایی ===")
            for mode, count in final_stats.items():
                print(f"{mode.capitalize()}: {count}")

            # فاز سوم: خروجی اکسل
            df = pd.DataFrame(list(final_stats.items()), columns=['Mode', 'Count'])
            df.to_excel(os.path.join('result',f'result{file_name[:3]}.xlsx'), index=False)
            print("\nفایل اکسل با موفقیت ایجاد شد.")
        else:
            print("محدوده‌ای انتخاب نشد. برنامه پایان یافت.")