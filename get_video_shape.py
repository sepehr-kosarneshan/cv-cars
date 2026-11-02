import cv2
import os

file_name = '6004 Modares - Motahari-2026-09-07_09h30min00s375ms.asf'
video_path = os.path.join('video' , file_name)

cap = cv2.VideoCapture(video_path)
success, frame = cap.read()

if success:
  # گرفتن ابعاد مستقیماً از shape آرایه نامپای فریم اول
  height, width, _ = frame.shape
  print(f"Video Width: {width}, Video Height: {height}")

cap.release()