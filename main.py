import time

from pose import *
from yolov5_detect import *
from threading import Thread
from queue import Queue
from LSTM import LSTM_inference

capture_queue = Queue()


def capture(capture_queue):
    video_path = 0
    # video_path = "/home/huyquang/Company/YOLOv4-Cloud-Tutorial/videos/test.mp4"
    # video_path = "rtsp://admin:Atin%402022@192.168.1.233/profile2/media.smp"
    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if capture_queue.qsize() < 1:
            capture_queue.put(frame)
        time.sleep(0.1)


def detect(capture_queue, detection):
    pose = Pose()
    lm_list = []
    time_steps = 10
    label = ""
    while True:
        if capture_queue.qsize() > 0:
            frame = capture_queue.get()
            frame_copy = frame.copy()
            bboxes = detection.detect(frame)
            for bbox in bboxes:
                x1, y1, x2, y2, name = bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, name, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                crop = frame_copy[y1:y2, x1:x2]
                points_list = pose.detect_pose(crop)
                if points_list is None:
                    continue
                lm_list.append(points_list)  
                if len(lm_list) == time_steps:
                    label = LSTM_inference(lm_list)
                    lm_list = []
                # for point in points_dict.values():
                #     x, y = int(point[0] * (x2 - x1)), int(point[1] * (y2 - y1))
                #     point = (x + x1, y + y1)
                #     cv2.circle(frame, point, 3, (0, 255, 0), -1)
            cv2.putText(frame, f"Label {label}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
            frame = cv2.resize(frame, (640, 480))
            cv2.imshow("frame", frame)
            key = cv2.waitKey(1)
            if key == 27:  # ESC
                exit(0)
        else:
            time.sleep(0.001)


if __name__ == "__main__":
    detection = Detection()
    weight_path = r"weights/yolov5s.pt"
    classes = [0]
    conf = 0.3
    imgsz = 640
    device = "cpu"
    detection.setup_model(weight_path, classes, conf, imgsz, device)
    t1 = Thread(target=detect, args=(capture_queue, detection))
    t2 = Thread(target=capture, args=(capture_queue,))
    t1.start()
    t2.start()
