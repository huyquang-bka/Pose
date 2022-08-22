import cv2
import numpy as np
import torch
from utils.augmentations import letterbox
from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.torch_utils import select_device
import time


def relu(x):
    return max(0, int(x))


class Detection():
    def __init__(self):
        self.model_file = 'lp_final.pt'
        # self.data = 'yolov5/data/coco128.yaml'  # dataset.yaml path
        self.imgsz = 640  # inference size (height, width)
        self.conf_thres = 0.5  # confidence threshold
        self.iou_thres = 0.45  # NMS IOU threshold
        self.max_det = 10  # maximum detections per image
        self.device = "0"  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        self.classes = [0, 1]  # filter by class: --class 0, or --class 0 2 3
        self.agnostic_nms = False  # class-agnostic NMS
        self.half = False  # use FP16 half-precision inference
        self.dnn = False  # use OpenCV DNN for ONNX inference
        self.name = None

    def setup_model(self, a_model, classes, conf_thres, img_size, device):
        self.conf_thres = conf_thres
        self.imgsz = img_size
        self.device = device
        self.model_file = a_model
        self.classes = classes
        self.device = select_device(self.device)
        self.model = attempt_load(
            self.model_file, map_location=self.device)  # load FP32 model
        stride = int(self.model.stride.max())  # model stride
        self.imgsz = check_img_size(self.imgsz, s=stride)  # check img_size
        if self.half:
            self.model.half()  # to FP16

        # Get names and colors
        self.names = self.model.module.names if hasattr(
            self.model, 'module') else self.model.names

    def detect(self, img):
        with torch.no_grad():
            detect_list = []
            img_copy = img.copy()
            img = letterbox(img, new_shape=self.imgsz)[0]
            # Convert
            # BGR to RGB, to 3x416x416
            img = img[:, :, ::-1].transpose(2, 0, 1)
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0

            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            # Inference
            pred = self.model(img, augment=False)[0]

            # Apply NMS
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes,
                                       self.agnostic_nms,
                                       max_det=self.max_det)
            # Process detections
            for i, det in enumerate(pred):  # detections per image
                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(
                        img.shape[2:], det[:, :4], img_copy.shape).round()

                    for *xyxy, conf, cls in reversed(det):
                        x1, y1, x2, y2 = list(map(relu, xyxy))
                        detect_list.append([x1, y1, x2, y2, self.names[int(cls)]])
            return detect_list


if __name__ == '__main__':
    detection = Detection()
    weight_path = r"weights/yolov5s.pt"
    classes = [0]
    conf = 0.3
    imgsz = 640
    device = "cpu"
    detection.setup_model(weight_path, classes, conf, imgsz, device)
    cap = cv2.VideoCapture(0)
    t = time.time()
    count = 0
    fps = 0
    while True:
        if time.time() - t > 1:
            fps = count
            t = time.time()
            count = 0
        ret, frame = cap.read()
        if not ret:
            break
        count += 1
        bboxes = detection.detect(frame)
        for bbox in bboxes:
            x1, y1, x2, y2, name = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, name, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2)

        cv2.putText(frame, "FPS: " + str(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        frame = cv2.resize(frame, (640, 480))
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
