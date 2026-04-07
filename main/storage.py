import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class StudentRecord:
    student_id: str
    name: str
    created_at: str
    embedding: list[float]


class StudentRegistry:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.db_path.exists():
            self._write({"students": []})

    def _read(self) -> dict:
        with self.db_path.open("r", encoding="utf-8") as file:
            return json.load(file)

    def _write(self, payload: dict) -> None:
        with self.db_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2)

    def list_students(self) -> list[StudentRecord]:
        payload = self._read()
        return [
            StudentRecord(
                student_id=row["student_id"],
                name=row["name"],
                created_at=row["created_at"],
                embedding=row["embedding"],
            )
            for row in payload.get("students", [])
        ]

    def get_student_by_name(self, name: str) -> Optional[StudentRecord]:
        for student in self.list_students():
            if student.name.lower() == name.lower():
                return student
        return None

    def get_student_by_id(self, student_id: str) -> Optional[StudentRecord]:
        for student in self.list_students():
            if student.student_id == student_id:
                return student
        return None

    def add_student(self, student_id: str, name: str, embedding: list[float]) -> None:
        payload = self._read()
        if any(row["student_id"] == student_id for row in payload.get("students", [])):
            raise ValueError(f"student_id '{student_id}' already exists")

        payload.setdefault("students", []).append(
            {
                "student_id": student_id,
                "name": name,
                "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                "embedding": embedding,
            }
        )
        self._write(payload)

    def delete_student_by_id(self, student_id: str) -> bool:
        payload = self._read()
        before = len(payload.get("students", []))
        payload["students"] = [
            row for row in payload.get("students", []) if row.get("student_id") != student_id
        ]
        after = len(payload.get("students", []))
        if after == before:
            return False
        self._write(payload)
        return True

    def delete_student_by_name(self, name: str) -> bool:
        payload = self._read()
        before = len(payload.get("students", []))
        payload["students"] = [
            row for row in payload.get("students", []) if row.get("name", "").lower() != name.lower()
        ]
        after = len(payload.get("students", []))
        if after == before:
            return False
        self._write(payload)
        return True
