import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from storage import StudentRegistry


app = FastAPI(title="Student Focus API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

registry = StudentRegistry(Path("public") / "students_db.json")
live_status_path = Path("public") / "live_status.json"


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/students")
def students() -> dict:
    rows = []
    for student in registry.list_students():
        rows.append(
            {
                "student_id": student.student_id,
                "name": student.name,
                "created_at": student.created_at,
            }
        )
    return {"students": rows}


@app.get("/api/live-status")
def live_status() -> dict:
    if not live_status_path.exists():
        return {"updated_at": None, "students": []}

    with live_status_path.open("r", encoding="utf-8") as file:
        return json.load(file)


@app.get("/api/overview")
def overview() -> dict:
    total = len(registry.list_students())
    if not live_status_path.exists():
        return {
            "total_registered": total,
            "focused": 0,
            "distracted": 0,
            "unknown": 0,
        }

    with live_status_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    focused = 0
    distracted = 0
    unknown = 0

    for row in payload.get("students", []):
        state = row.get("state", "")
        if state == "FOCUSED":
            focused += 1
        elif state == "DISTRACTED":
            distracted += 1
        else:
            unknown += 1

    return {
        "total_registered": total,
        "focused": focused,
        "distracted": distracted,
        "unknown": unknown,
    }
