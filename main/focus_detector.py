import math
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import mediapipe as mp

from storage import StudentRegistry
from vision_utils import FaceGeometryUtils, build_face_embedding, embedding_distance


@dataclass
class DetectionConfig:
    camera_index: int = 0
    threshold_ratio: float = 0.25
    ear_threshold: float = 0.22
    gaze_offset_threshold: float = 0.18
    recognition_threshold: float = 0.75
    distracted_frames_for_switch: int = 8
    focused_frames_for_recover: int = 4
    window_name: str = "Student Focus Detector"


class StudentFocusDetector:
    LEFT_EYE_IDX = (33, 160, 158, 133, 153, 144)
    RIGHT_EYE_IDX = (362, 385, 387, 263, 373, 380)
    LEFT_EYE_CORNERS = (33, 133)
    RIGHT_EYE_CORNERS = (362, 263)
    LEFT_IRIS_IDX = (474, 475, 476, 477)
    RIGHT_IRIS_IDX = (469, 470, 471, 472)

    def __init__(self, registry: StudentRegistry, config: DetectionConfig) -> None:
        self.registry = registry
        self.config = config
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=5,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.student_state: dict[str, dict[str, float | int | str]] = {}
        self.live_status_path = Path("public") / "live_status.json"
        self.live_status_path.parent.mkdir(parents=True, exist_ok=True)

    def _write_live_status(self, rows: list[dict]) -> None:
        payload = {
            "updated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "students": rows,
        }
        with self.live_status_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2)

    def _extract_face_geometry(self, face_landmarks, frame_width: int, frame_height: int):
        xs = [FaceGeometryUtils.to_pixel(lm.x, frame_width) for lm in face_landmarks.landmark]
        ys = [FaceGeometryUtils.to_pixel(lm.y, frame_height) for lm in face_landmarks.landmark]

        xmin, xmax = max(min(xs), 0), min(max(xs), frame_width - 1)
        ymin, ymax = max(min(ys), 0), min(max(ys), frame_height - 1)

        face_center_x = (xmin + xmax) // 2
        face_center_y = (ymin + ymax) // 2

        nose = face_landmarks.landmark[1]
        nose_x = FaceGeometryUtils.to_pixel(nose.x, frame_width)
        nose_y = FaceGeometryUtils.to_pixel(nose.y, frame_height)

        face_width = max(xmax - xmin, 1)
        distance = math.dist((nose_x, nose_y), (face_center_x, face_center_y))

        return {
            "box": (xmin, ymin, xmax, ymax),
            "distance": distance,
            "threshold": self.config.threshold_ratio * face_width,
        }

    def _compute_ear(self, face_landmarks, frame_width: int, frame_height: int, eye_indices) -> float:
        p1 = FaceGeometryUtils.landmark_to_point(face_landmarks, eye_indices[0], frame_width, frame_height)
        p2 = FaceGeometryUtils.landmark_to_point(face_landmarks, eye_indices[1], frame_width, frame_height)
        p3 = FaceGeometryUtils.landmark_to_point(face_landmarks, eye_indices[2], frame_width, frame_height)
        p4 = FaceGeometryUtils.landmark_to_point(face_landmarks, eye_indices[3], frame_width, frame_height)
        p5 = FaceGeometryUtils.landmark_to_point(face_landmarks, eye_indices[4], frame_width, frame_height)
        p6 = FaceGeometryUtils.landmark_to_point(face_landmarks, eye_indices[5], frame_width, frame_height)

        vertical_1 = FaceGeometryUtils.euclidean(p2, p6)
        vertical_2 = FaceGeometryUtils.euclidean(p3, p5)
        horizontal = FaceGeometryUtils.euclidean(p1, p4)

        if horizontal == 0:
            return 0.0
        return (vertical_1 + vertical_2) / (2.0 * horizontal)

    def _eye_gaze_offset(
        self,
        face_landmarks,
        eye_corners,
        iris_indices,
        frame_width: int,
        frame_height: int,
    ) -> float:
        p1 = FaceGeometryUtils.landmark_to_point(face_landmarks, eye_corners[0], frame_width, frame_height)
        p2 = FaceGeometryUtils.landmark_to_point(face_landmarks, eye_corners[1], frame_width, frame_height)

        iris_center = FaceGeometryUtils.iris_center(face_landmarks, iris_indices, frame_width, frame_height)
        if iris_center is None:
            return 0.0

        d1 = FaceGeometryUtils.euclidean(iris_center, p1)
        d2 = FaceGeometryUtils.euclidean(iris_center, p2)
        eye_span = d1 + d2
        if eye_span == 0:
            return 0.0

        iris_ratio = d1 / eye_span
        return abs(iris_ratio - 0.5)

    def _match_student_name(self, embedding: list[float]) -> tuple[str, float]:
        best_name = "Unknown"
        best_distance = float("inf")

        for student in self.registry.list_students():
            distance = embedding_distance(embedding, student.embedding)
            if distance < best_distance:
                best_distance = distance
                best_name = student.name

        if best_distance > self.config.recognition_threshold:
            return "Unknown", best_distance
        return best_name, best_distance

    def _update_state(self, student_name: str, is_distracted: bool) -> tuple[str, float]:
        if student_name not in self.student_state:
            self.student_state[student_name] = {
                "current_state": "FOCUSED",
                "distracted_streak": 0,
                "focused_streak": 0,
                "focused_frames": 0,
                "tracked_frames": 0,
            }

        row = self.student_state[student_name]

        if is_distracted:
            row["distracted_streak"] = int(row["distracted_streak"]) + 1
            row["focused_streak"] = 0
            if int(row["distracted_streak"]) >= self.config.distracted_frames_for_switch:
                row["current_state"] = "DISTRACTED"
        else:
            row["focused_streak"] = int(row["focused_streak"]) + 1
            row["distracted_streak"] = 0
            if int(row["focused_streak"]) >= self.config.focused_frames_for_recover:
                row["current_state"] = "FOCUSED"

        row["tracked_frames"] = int(row["tracked_frames"]) + 1
        if row["current_state"] == "FOCUSED":
            row["focused_frames"] = int(row["focused_frames"]) + 1

        focus_score = 100.0 * float(row["focused_frames"]) / float(row["tracked_frames"])
        return str(row["current_state"]), focus_score

    def run(self) -> None:
        if not self.registry.list_students():
            print("No registered students found. Please register students first.")
            return

        cap = cv2.VideoCapture(self.config.camera_index)
        if not cap.isOpened():
            print("Could not open camera device.")
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
                frame_rows: list[dict] = []

                if result.multi_face_landmarks:
                    for face_landmarks in result.multi_face_landmarks:
                        geometry = self._extract_face_geometry(face_landmarks, frame_width, frame_height)
                        xmin, ymin, xmax, ymax = geometry["box"]

                        left_ear = self._compute_ear(face_landmarks, frame_width, frame_height, self.LEFT_EYE_IDX)
                        right_ear = self._compute_ear(face_landmarks, frame_width, frame_height, self.RIGHT_EYE_IDX)
                        avg_ear = (left_ear + right_ear) / 2.0

                        left_gaze = self._eye_gaze_offset(
                            face_landmarks,
                            self.LEFT_EYE_CORNERS,
                            self.LEFT_IRIS_IDX,
                            frame_width,
                            frame_height,
                        )
                        right_gaze = self._eye_gaze_offset(
                            face_landmarks,
                            self.RIGHT_EYE_CORNERS,
                            self.RIGHT_IRIS_IDX,
                            frame_width,
                            frame_height,
                        )
                        gaze_offset = (left_gaze + right_gaze) / 2.0

                        head_drift = geometry["distance"] > geometry["threshold"]
                        eyes_closed = avg_ear < self.config.ear_threshold
                        side_gaze = gaze_offset > self.config.gaze_offset_threshold
                        is_distracted = head_drift or eyes_closed or side_gaze

                        embedding = build_face_embedding(face_landmarks)
                        student_name, recognition_distance = self._match_student_name(embedding)

                        if student_name == "Unknown":
                            state_text = "UNKNOWN"
                            focus_score = 0.0
                            color = (0, 165, 255)
                        else:
                            state_text, focus_score = self._update_state(student_name, is_distracted)
                            color = (0, 255, 0) if state_text == "FOCUSED" else (0, 0, 255)

                        signals = []
                        if head_drift:
                            signals.append("head")
                        if eyes_closed:
                            signals.append("eyes")
                        if side_gaze:
                            signals.append("gaze")
                        signal_text = "|".join(signals) if signals else "stable"

                        frame_rows.append(
                            {
                                "name": student_name,
                                "state": state_text,
                                "focus_score": round(focus_score, 2),
                                "ear": round(avg_ear, 3),
                                "gaze_offset": round(gaze_offset, 3),
                                "signals": signals,
                            }
                        )

                        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                        cv2.putText(
                            frame,
                            f"{student_name}: {state_text}",
                            (xmin, max(ymin - 12, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            color,
                            2,
                        )
                        cv2.putText(
                            frame,
                            f"Focus: {focus_score:.1f}%",
                            (xmin, ymax + 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (255, 255, 0),
                            2,
                        )
                        cv2.putText(
                            frame,
                            f"EAR {avg_ear:.2f} | Gaze {gaze_offset:.2f}",
                            (xmin, ymax + 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (220, 220, 220),
                            1,
                        )
                        cv2.putText(
                            frame,
                            f"Signals: {signal_text} | Match {recognition_distance:.2f}",
                            (xmin, ymax + 58),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.45,
                            (200, 255, 255),
                            1,
                        )

                self._write_live_status(frame_rows)

                cv2.imshow(self.config.window_name, frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
