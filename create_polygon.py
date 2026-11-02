import cv2
import numpy as np
import os

# لیست سراسری برای ذخیره نقاط کلیک‌شده
clicked_points = []


def mouse_callback(event, x, y, flags, param):
    """تابع مدیریت کلیک‌های موس روی تصویر"""
    global clicked_points
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_points.append([x, y])
        print(f"نقطه ثبت شد -> X: {x}, Y: {y}")


def get_polygon_from_video(video_path):
    global clicked_points
    clicked_points = []

    cap = cv2.VideoCapture(video_path)
    for i in range(4):
        success, frame = cap.read()

    if not success:
        print("خطا در باز کردن ویدیو یا خواندن فریم اول!")
        return None

    window_name = "Select ROI Polygon - Click & Press ENTER"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)

    print("\n--- راهنمای ابزار کالیبراسیون ---")
    print("1. با [کلیک چپ] روی گوشه‌های لاین خیابان کلیک کنید.")
    print("2. دکمه [ENTER] یا [c] را بزنید تا انتخاب‌ها تایید و ذخیره شوند.")
    print("3. دکمه [r] را بزنید تا تمام نقاط پاک شوند و از نو شروع کنید.")
    print("4. دکمه [q] را بزنید تا خارج شوید.\n")

    while True:
        # یک کپی از فریم برای رسم خطوط و نقاط روی آن
        display_frame = frame.copy()

        # رسم نقاط و چندضلعی به صورت زنده با هر کلیک
        if len(clicked_points) > 0:
            for pt in clicked_points:
                cv2.circle(display_frame, tuple(pt), 6, (0, 0, 255), -1)

            if len(clicked_points) > 1:
                pts_np = np.array(clicked_points, np.int32)
                cv2.polylines(display_frame, [pts_np], isClosed=False, color=(0, 255, 0), thickness=2)

        cv2.imshow(window_name, display_frame)
        key = cv2.waitKey(1) & 0xFF

        # کلید Enter (کد 13) یا c برای تایید نهایی
        if key == 13 or key == ord('c'):
            if len(clicked_points) >= 3:
                print("انتخاب محدوده با موفقیت تایید شد!")
                break
            else:
                print("هشدار: حداقل باید ۳ نقطه برای تشکیل چندضلعی انتخاب کنید!")

        # کلید r برای ریست کردن نقاط
        elif key == ord('r'):
            clicked_points = []
            print("تمام نقاط پاک شدند. دوباره کلیک کنید.")

        # کلید q برای خروج اضطراری
        elif key == ord('q'):
            clicked_points = []
            break

    cap.release()
    cv2.destroyAllWindows()

    if len(clicked_points) >= 3:
        return np.array(clicked_points, np.int32)
    return None


if __name__ == '__main__':
    # مسیر ویدیوی مورد نظر خود را اینجا وارد کنید
    file_name = '12101 Sadi - Ghaedi-2026-09-07_11h30min00s130ms.asf'
    video_path = os.path.join('video', file_name)

    roi_polygon = get_polygon_from_video(video_path)

    if roi_polygon is not None:
        print("\n=== مختصات نهایی آماده برای کپی در کد اصلی ===")
        print("polygon_roi = np.array([")
        for pt in roi_polygon:
            print(f"    [{pt[0]}, {pt[1]}],")
        print("], np.int32)")