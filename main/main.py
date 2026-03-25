import math
from dataclasses import dataclass

import cv2
import mediapipe as mp


@dataclass
class AttentionConfig:
    camera_index: int = 0
    threshold_ratio: float = 0.25
    ear_threshold: float = 0.22
    window_name: str = "Student Attention Tracker"


class StudentAttentionTracker:
    LEFT_EYE_IDX = (33, 160, 158, 133, 153, 144)
    RIGHT_EYE_IDX = (362, 385, 387, 263, 373, 380)

    def __init__(self, config: AttentionConfig) -> None:
        self.config = config

        if not hasattr(mp, "solutions"):
            raise RuntimeError(
                "Your installed mediapipe build does not expose mp.solutions. "
                "Install a compatible version: pip install mediapipe==0.10.14"
            )

        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    @staticmethod
    def _to_pixel(value: float, scale: int) -> int:
        return int(value * scale)

    def _extract_face_geometry(self, face_landmarks, frame_width: int, frame_height: int):
        xs = [self._to_pixel(lm.x, frame_width) for lm in face_landmarks.landmark]
        ys = [self._to_pixel(lm.y, frame_height) for lm in face_landmarks.landmark]

        xmin, xmax = max(min(xs), 0), min(max(xs), frame_width - 1)
        ymin, ymax = max(min(ys), 0), min(max(ys), frame_height - 1)

        face_center_x = (xmin + xmax) // 2
        face_center_y = (ymin + ymax) // 2

        nose_landmark = face_landmarks.landmark[1]
        nose_x = self._to_pixel(nose_landmark.x, frame_width)
        nose_y = self._to_pixel(nose_landmark.y, frame_height)

        face_width = max(xmax - xmin, 1)
        distance = math.dist((nose_x, nose_y), (face_center_x, face_center_y))
        threshold = self.config.threshold_ratio * face_width

        return {
            "box": (xmin, ymin, xmax, ymax),
            "face_center": (face_center_x, face_center_y),
            "nose": (nose_x, nose_y),
            "distance": distance,
            "threshold": threshold,
        }

    @staticmethod
    def _classify(distance: float, threshold: float, avg_ear: float, ear_threshold: float) -> str:
        # D if nose drift is high OR eyes appear closed based on EAR.
        is_distracted = (distance > threshold) or (avg_ear < ear_threshold)
        return "D" if is_distracted else "F"

    def _landmark_to_point(self, face_landmarks, idx: int, frame_width: int, frame_height: int):
        landmark = face_landmarks.landmark[idx]
        x = self._to_pixel(landmark.x, frame_width)
        y = self._to_pixel(landmark.y, frame_height)
        return (x, y)

    @staticmethod
    def _euclidean(p1, p2) -> float:
        return math.dist(p1, p2)

    def _compute_ear(self, face_landmarks, frame_width: int, frame_height: int, eye_indices) -> float:
        p1 = self._landmark_to_point(face_landmarks, eye_indices[0], frame_width, frame_height)
        p2 = self._landmark_to_point(face_landmarks, eye_indices[1], frame_width, frame_height)
        p3 = self._landmark_to_point(face_landmarks, eye_indices[2], frame_width, frame_height)
        p4 = self._landmark_to_point(face_landmarks, eye_indices[3], frame_width, frame_height)
        p5 = self._landmark_to_point(face_landmarks, eye_indices[4], frame_width, frame_height)
        p6 = self._landmark_to_point(face_landmarks, eye_indices[5], frame_width, frame_height)

        vertical_1 = self._euclidean(p2, p6)
        vertical_2 = self._euclidean(p3, p5)
        horizontal = self._euclidean(p1, p4)

        if horizontal == 0:
            return 0.0

        return (vertical_1 + vertical_2) / (2.0 * horizontal)

    def run(self) -> None:
        cap = cv2.VideoCapture(self.config.camera_index)
        if not cap.isOpened():
            print("Could not open camera device. Try camera index 1 or close other camera apps.")
            return

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to capture video")
                    break

                frame_height, frame_width = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = self.face_mesh.process(rgb)

                if result.multi_face_landmarks:
                    for face_landmarks in result.multi_face_landmarks:
                        data = self._extract_face_geometry(face_landmarks, frame_width, frame_height)

                        left_ear = self._compute_ear(
                            face_landmarks, frame_width, frame_height, self.LEFT_EYE_IDX
                        )
                        right_ear = self._compute_ear(
                            face_landmarks, frame_width, frame_height, self.RIGHT_EYE_IDX
                        )
                        avg_ear = (left_ear + right_ear) / 2.0
                        status = self._classify(
                            data["distance"],
                            data["threshold"],
                            avg_ear,
                            self.config.ear_threshold,
                        )
                        status_color = (0, 255, 0) if status == "F" else (0, 0, 255)

                        xmin, ymin, xmax, ymax = data["box"]
                        nose_x, nose_y = data["nose"]
                        center_x, center_y = data["face_center"]

                        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
                        cv2.circle(frame, (nose_x, nose_y), 5, (0, 255, 0), -1)
                        cv2.circle(frame, (center_x, center_y), 4, (255, 255, 0), -1)

                        cv2.line(frame, (nose_x, nose_y), (center_x, center_y), (0, 255, 255), 2)
                        cv2.putText(
                            frame,
                            status,
                            (xmin, max(ymin - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            status_color,
                            2,
                        )

                cv2.imshow(self.config.window_name, frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()


def main() -> None:
    config = AttentionConfig(camera_index=0, threshold_ratio=0.25, ear_threshold=0.22)
    tracker = StudentAttentionTracker(config)
    tracker.run()


if __name__ == "__main__":
    main()
    