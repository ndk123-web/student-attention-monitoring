import math
from typing import Optional


EMBEDDING_IDXS = [1, 33, 61, 133, 152, 199, 263, 291, 362]


class FaceGeometryUtils:
    @staticmethod
    def to_pixel(value: float, scale: int) -> int:
        return int(value * scale)

    @staticmethod
    def euclidean(p1, p2) -> float:
        return math.dist(p1, p2)

    @staticmethod
    def landmark_to_point(face_landmarks, idx: int, frame_width: int, frame_height: int):
        lm = face_landmarks.landmark[idx]
        return (int(lm.x * frame_width), int(lm.y * frame_height))

    @staticmethod
    def iris_center(face_landmarks, iris_indices, frame_width: int, frame_height: int) -> Optional[tuple[int, int]]:
        if len(face_landmarks.landmark) <= max(iris_indices):
            return None

        points = [
            FaceGeometryUtils.landmark_to_point(face_landmarks, idx, frame_width, frame_height)
            for idx in iris_indices
        ]
        avg_x = sum(p[0] for p in points) // len(points)
        avg_y = sum(p[1] for p in points) // len(points)
        return (avg_x, avg_y)


def build_face_embedding(face_landmarks) -> list[float]:
    left_eye = face_landmarks.landmark[33]
    right_eye = face_landmarks.landmark[263]
    nose = face_landmarks.landmark[1]

    scale = math.dist((left_eye.x, left_eye.y), (right_eye.x, right_eye.y))
    if scale == 0:
        return [0.0 for _ in range(len(EMBEDDING_IDXS) * 3)]

    embedding: list[float] = []
    for idx in EMBEDDING_IDXS:
        lm = face_landmarks.landmark[idx]
        embedding.append((lm.x - nose.x) / scale)
        embedding.append((lm.y - nose.y) / scale)
        embedding.append((lm.z - nose.z) / scale)

    return embedding


def average_embeddings(embeddings: list[list[float]]) -> list[float]:
    if not embeddings:
        return []

    length = len(embeddings[0])
    totals = [0.0] * length

    for embedding in embeddings:
        for i, value in enumerate(embedding):
            totals[i] += value

    return [value / len(embeddings) for value in totals]


def embedding_distance(e1: list[float], e2: list[float]) -> float:
    if len(e1) != len(e2):
        return float("inf")

    squared_error = 0.0
    for a, b in zip(e1, e2):
        squared_error += (a - b) ** 2
    return math.sqrt(squared_error)
