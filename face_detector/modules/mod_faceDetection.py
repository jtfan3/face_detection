import cv2
import mediapipe as mp

class FaceDetection():
    # initialize the face detection class with arguments from https://google.github.io/mediapipe/solutions/face_detection.html
    def __init__(self, model_selection = 0, threshold = 0.5):
        self.model_selection = model_selection
        self.threshold = threshold

        self.mp_draw = mp.solutions.drawing_utils
        self.face_detection = mp.solutions.face_detection.FaceDetection(model_selection = self.model_selection, min_detection_confidence = self.threshold)

    # gets counding boxes using self.face_detection, returns a list of element, elment = (score, bbox_dict)
    def get_bboxs(self, frame):
        mp_detections = self.face_detection.process(frame)
        score_bboxs = []
        if mp_detections.detections:
            for detection in mp_detections.detections:
                score = detection.score[0]
                mp_bbox = detection.location_data.relative_bounding_box
                bbox_dict = {
                    'x_min': mp_bbox.xmin,
                    'y_min': mp_bbox.ymin,
                    'w': mp_bbox.width,
                    'h': mp_bbox.height
                }

                score_bboxs.append([score, bbox_dict])

        return score_bboxs

    # draws the bbox onto the frame
    def draw_bbox(self, score, bbox_dict, frame, col = (255, 0, 255)):
        x_min, y_min, w, h = bbox_dict.values()
        frame_h, frame_w, _ = frame.shape
        bbox = int(x_min * frame_w), int(y_min * frame_h), int(w * frame_w), int(h * frame_h)

        # draw bbox
        cv2.rectangle(frame, bbox, col, 2)
        cv2.putText(frame, str(round(score, 3)), (bbox[0], bbox[1] - 10),
                    cv2.FONT_HERSHEY_COMPLEX, 0.6, col, 1)