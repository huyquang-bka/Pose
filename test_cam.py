import cv2

url = "rtsp://admin:Atin%402022@0.tcp.ap.ngrok.io:13202/profile2/media.smp"
cap = cv2.VideoCapture(url)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break
