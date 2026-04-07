from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp

from storage import StudentRegistry
from vision_utils import average_embeddings, build_face_embedding


@dataclass
class RegisterConfig:
    camera_index: int = 0
    samples_required: int = 30
    window_name: str = "Student Registration"


class StudentRegistrar:
    def __init__(self, registry: StudentRegistry, config: RegisterConfig) -> None:
        self.registry = registry
        self.config = config
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def register(self, student_id: str, student_name: str) -> None:
        cap = cv2.VideoCapture(self.config.camera_index)
        if not cap.isOpened():
            print("Could not open camera device.")
            return

        embeddings: list[list[float]] = []
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to capture video")
                    break

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = self.face_mesh.process(rgb)

                if result.multi_face_landmarks:
                    landmarks = result.multi_face_landmarks[0]
                    embedding = build_face_embedding(landmarks)
                    if embedding:
                        embeddings.append(embedding)

                    progress = min(len(embeddings), self.config.samples_required)
                    cv2.putText(
                        frame,
                        f"Collecting face samples: {progress}/{self.config.samples_required}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )

                cv2.putText(
                    frame,
                    "Press ESC to cancel",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )

                cv2.imshow(self.config.window_name, frame)

                if len(embeddings) >= self.config.samples_required:
                    break

                if cv2.waitKey(1) & 0xFF == 27:
                    print("Registration cancelled")
                    return
        finally:
            cap.release()
            cv2.destroyAllWindows()

        student_embedding = average_embeddings(embeddings)
        if not student_embedding:
            print("Could not register student because no valid face embeddings were captured")
            return

        snapshot_dir = Path("public") / "students"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        snapshot_path = snapshot_dir / f"{student_id}.jpg"
        cv2.imwrite(str(snapshot_path), frame)

        self.registry.add_student(student_id=student_id, name=student_name, embedding=student_embedding)
        print(f"Student registered: {student_name} ({student_id})")
