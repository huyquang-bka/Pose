import cv2
import mediapipe as mp


class Pose():
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_pose = mp.solutions.pose
        self.pose_detector = self.mp_pose.Pose(static_image_mode=True, model_complexity=2, enable_segmentation=True,
                                               min_detection_confidence=0.5)

    def extract_point(self, results):
        dictionary = {}
        dictionary[0] = (results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.NOSE].x,
                         results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.NOSE].y)
        dictionary[1] = (results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_EYE].x,
                         results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_EYE].y)
        dictionary[2] = (results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_EYE].x,
                         results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_EYE].y)
        dictionary[3] = (results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_EAR].x,
                         results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_EAR].y)
        dictionary[4] = (results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_EAR].x,
                         results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_EAR].y)
        dictionary[5] = (results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                         results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER].y)
        dictionary[6] = (results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                         results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].y)
        dictionary[7] = (results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ELBOW].x,
                         results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ELBOW].y)
        dictionary[8] = (results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ELBOW].x,
                         results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ELBOW].y)
        dictionary[9] = (results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST].x,
                         results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST].y)
        dictionary[10] = (results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST].x,
                          results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST].y)
        dictionary[11] = (results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP].x,
                          results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP].y)
        dictionary[12] = (results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP].x,
                          results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP].y)
        dictionary[13] = (results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_KNEE].x,
                          results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_KNEE].y)
        dictionary[14] = (results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_KNEE].x,
                          results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_KNEE].y)
        dictionary[15] = (results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ANKLE].x,
                          results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ANKLE].y)
        dictionary[16] = (results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ANKLE].x,
                          results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ANKLE].y)

        return dictionary

    def extract_point_list(self, results):
        ls = []
        ls.append(results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.NOSE].x)
        ls.append(results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.NOSE].y)
        ls.append(results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_EYE].x)
        ls.append(results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_EYE].y)
        ls.append(results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_EYE].x)
        ls.append(results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_EYE].y)
        ls.append(results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_EAR].x)
        ls.append(results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_EAR].y)
        ls.append(results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_EAR].x)
        ls.append(results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_EAR].y)
        ls.append(results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER].x)
        ls.append(results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER].y)
        ls.append(results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].x)
        ls.append(results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].y)
        ls.append(results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ELBOW].x)
        ls.append(results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ELBOW].y)
        ls.append(results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ELBOW].x)
        ls.append(results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ELBOW].y)
        ls.append(results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST].x)
        ls.append(results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST].y)
        ls.append(results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST].x)
        ls.append(results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST].y)
        ls.append(results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP].x)
        ls.append(results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP].y)
        ls.append(results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP].x)
        ls.append(results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP].y)
        ls.append(results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_KNEE].x)
        ls.append(results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_KNEE].y)
        ls.append(results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_KNEE].x)
        ls.append(results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_KNEE].y)
        ls.append(results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ANKLE].x)
        ls.append(results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ANKLE].y)
        ls.append(results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ANKLE].x)
        ls.append(results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ANKLE].y)
        return ls

    def detect_pose(self, image):
        results = self.pose_detector.process(
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if results.pose_landmarks is None:
            return None
        ls_points = self.extract_point_list(results)
        return ls_points


if __name__ == '__main__':
    # video_path = "/home/huyquang/Company/YOLOv4-Cloud-Tutorial/videos/test.mp4"
    cap = cv2.VideoCapture(0)
    f = open("head_swing.txt", "w")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        points_dict = detect_pose(frame)
        H, W = frame.shape[:2]
        if points_dict is not None:
            for point in points_dict.values():
                f.write(f"{point[0]},{point[1]},")
                x, y = int(point[0] * W), int(point[1] * H)
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
        f.write("\n")
        frame = cv2.resize(frame, (640, 480))
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) == ord('q'):
            break
