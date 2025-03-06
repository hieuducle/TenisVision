from ultralytics import YOLO
import cv2

# Load YOLO model
model = YOLO("/home/amin/PycharmProjects/PythonProject/TableTenisVision/runs/detect/train/weights/best.pt")

# Load video
cap = cv2.VideoCapture('data/mel.mp4')

# Check if video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Loop through frames
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform tracking
    frame_process = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_process = cv2.cvtColor(frame_process, cv2.COLOR_GRAY2BGR)
    results = model.track(frame_process, save=True, conf=0.1,tracker='bytetrack.yaml',persist=True)

    # Check if there are any detections
    if results and results[0].boxes is not None and results[0].boxes.id is not None:
        for cls, box,track_id in zip(results[0].boxes.cls,results[0].boxes.xyxy,results[0].boxes.id):

            xmin, ymin, xmax, ymax = map(int, box.tolist())
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(frame,"{}".format(track_id),(xmin,ymin-10),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),1)

    # Display output
    cv2.imshow('Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
# from ultralytics import YOLO
# import cv2
#
# # Load model YOLO
# model = YOLO("/home/amin/PycharmProjects/PythonProject/TableTenisVision/runs/detect/train/weights/best.pt")
#
# # Đọc ảnh và chuyển sang grayscale
# img = cv2.imread('table_tenis.jpg')
# gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# # Chuyển grayscale (1 kênh) sang 3 kênh để YOLO có thể xử lý
# img_rgb = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
#
# # Dự đoán
# results = model.predict(img_rgb, save=True, conf=0.25)
#
# # Hiển thị kết quả (tùy chọn)
# results[0].show()

 # Kiểm tra xem có boxes nào không
