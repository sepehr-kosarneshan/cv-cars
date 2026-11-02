import os
import cv2
import numpy as np
from ultralytics import YOLO

class VideoCursor:
    """کلاس مدیریت ایمن باز و بسته کردن ویدیو با Context Manager"""

    def __init__(self, file_addr):
        self.addr = file_addr

    def __enter__(self):
        self.cur = cv2.VideoCapture(self.addr)
        return self.cur

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cur.release()
        cv2.destroyAllWindows()


def load_yolo_model(model_name='yolov8n.pt'):
    """بارگذاری مدل هوش مصنوعی (نسخه Nano برای سرعت بالا در پردازش انبوه)"""
    print(f"در حال بارگذاری مدل {model_name}...")
    return YOLO(model_name)


def process_traffic_video(video_path, model, polygon_roi, show_video=False):
    """
    پردازش ویدیو با استفاده از چندضلعیِ ROI، تداوم مسیر و تفکیک وسایل نقلیه (شامل موتور)
    :param video_path: مسیر فایل ویدیو
    :param model: مدل بارگذاری شده YOLO
    :param polygon_roi: آرایه نامپای از نقاط چندضلعیِ لاین خیابان
    :param show_video: نمایش زنده گرافیکی (True/False)
    :return: دیکشنری حاوی آمار شمارش
    """
    car_count = 0
    bus_count = 0
    motorcycle_count = 0

    # دیکشنری برای ثبت تعداد فریم‌های متوالی حضورِ هر ID در محدوده ROI
    track_frame_counts = {}
    # مجموعه IDهای شمرده‌شده جهت جلوگیری از شمارش مجدد
    counted_ids = set()

    # تعداد فریم‌های متوالی مورد نیاز برای تایید اصالت وسیله نقلیه (فیلتر سایه و نویز)
    MIN_CONSECUTIVE_FRAMES = 3

    with VideoCursor(video_path) as cap:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            # ردیابی وسایل نقلیه با ByteTrack
            results = model.track(
                frame,
                persist=True,
                tracker='bytetrack.yaml',
                conf=0.15,
                verbose=False
            )

            current_frame_ids_in_roi = set()

            for r in results:
                boxes = r.boxes
                if boxes.id is not None:
                    xyxy_list = boxes.xyxy.cpu().numpy()
                    ids = boxes.id.cpu().numpy()
                    classes = boxes.cls.cpu().numpy()

                    for box, track_id, cls in zip(xyxy_list, ids, classes):
                        x1, y1, x2, y2 = box
                        track_id = int(track_id)
                        class_name = model.names[int(cls)]

                        # نقطه مرجع: مرکز لبه پایینی کادر (نزدیک لاستیک)
                        center_x = int((x1 + x2) / 2)
                        center_y = int(y2)
                        point = (center_x, center_y)

                        # ۱. بررسی اینکه آیا نقطه مرجع داخل چندضلعی (Polygon ROI) قرار دارد یا خیر
                        inside_roi = cv2.pointPolygonTest(polygon_roi, point, False) >= 0

                        if inside_roi:
                            current_frame_ids_in_roi.add(track_id)

                            # ۲. منطق تداوم مسیر (Track Persistence)
                            if track_id not in counted_ids:
                                track_frame_counts[track_id] = track_frame_counts.get(track_id, 0) + 1

                                # اگر وسیله نقلیه به اندازه کافی (مثلاً ۵ فریم متوالی) در لاین بود، تایید می‌شود
                                if track_frame_counts[track_id] >= MIN_CONSECUTIVE_FRAMES:
                                    if class_name in ['motorcycle', 'motorbike', 'bicycle']:
                                        motorcycle_count += 1
                                        counted_ids.add(track_id)
                                    elif class_name in ['car', 'truck']:
                                        car_count += 1
                                        counted_ids.add(track_id)
                                    elif class_name == 'bus':
                                        bus_count += 1
                                        counted_ids.add(track_id)

                        # رسم کادرها بر اساس نوع وسیله نقلیه در صورت فعال بودن نمایش زنده
                        if show_video:
                            is_motor = class_name in ['motorcycle', 'motorbike', 'bicycle']
                            color = (0, 0, 255) if is_motor else ((0, 255, 0) if inside_roi else (255, 0, 0))
                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                            cv2.putText(
                                frame,
                                f'ID:{track_id} {class_name}',
                                (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (255, 255, 0),
                                2,
                            )

            # ۳. پاکسازی نویزها: اگر ID از محدوده خارج شد، شمارشگر فریم‌های آن صفر می‌شود
            disappeared_ids = set(track_frame_counts.keys()) - current_frame_ids_in_roi
            for d_id in disappeared_ids:
                if d_id not in counted_ids:
                    track_frame_counts[d_id] = 0

            # رسم چندضلعی و آمار زنده روی تصویر در حالت نمایش
            if show_video:
                cv2.polylines(frame, [polygon_roi], isClosed=True, color=(255, 0, 0), thickness=3)
                stats_text = f"Cars: {car_count} | Motors: {motorcycle_count} | Buses: {bus_count}"
                cv2.putText(frame, stats_text, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                cv2.imshow('Traffic Tracking - Polygon ROI', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    return {
        'car_count': car_count,
        'motorcycle_count': motorcycle_count,
        'bus_count': bus_count
    }


def save_results(stats, output_dir='result'):
    """ذخیره نتایج کامل در فایل متنی"""
    os.makedirs(output_dir, exist_ok=True)
    result_file_path = os.path.join(output_dir, 'counts.txt')

    with open(result_file_path, 'w', encoding='utf-8') as f:
        f.write(
            f"car_count : {stats['car_count']}\n"
            f"motorcycle_count : {stats['motorcycle_count']}\n"
            f"bus_count : {stats['bus_count']}"
        )
    print(f"گزارش نهایی در مسیر '{result_file_path}' با موفقیت ذخیره شد.")


# ==================== بخش اجرایی اصلی (Main) ====================
if __name__ == '__main__':
    file_name = '12101 Sadi - Ghaedi-2026-09-07_11h30min00s130ms.asf'
    video_path = os.path.join('video', file_name)

    # کلید نمایش گرافیکی (True برای مشاهده زنده، False برای پردازش سریع پس‌زمینه)
    SHOW_VIDEO_WINDOW = True

    # ۱. بارگذاری مدل سبک YOLOv8n
    yolo_model = load_yolo_model('yolov8s.pt')

    # ۲. تعریف منطقه چندضلعی (Polygon ROI)
    polygon_roi = np.array([
        [1, 185],
        [285, 209],
        [241, 103],
        [158, 97],
    ], np.int32)

    print("شروع پردازش ویدیو با سیستم چندضلعی و تفکیک کامل موتورها و خودروها...")
    final_stats = process_traffic_video(video_path, yolo_model, polygon_roi, show_video=SHOW_VIDEO_WINDOW)

    print("\n--- آمار نهایی استخراج شده ---")
    print(f"خودروها (Cars): {final_stats['car_count']}")
    print(f"موتورها/دوچرخه‌ها (Motors): {final_stats['motorcycle_count']}")
    print(f"اتوبوس‌ها (Buses): {final_stats['bus_count']}")

    save_results(final_stats)