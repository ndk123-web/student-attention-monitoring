import argparse
from pathlib import Path

import uvicorn

from focus_detector import DetectionConfig, StudentFocusDetector
from register_student import RegisterConfig, StudentRegistrar
from storage import StudentRegistry


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Student Focus System")
    parser.add_argument(
        "mode",
        choices=["register", "detect", "delete", "list", "api"],
        help="Run registration, detection, delete, list, or api service",
    )
    parser.add_argument("--student-id", default="", help="Used in register/delete mode")
    parser.add_argument("--name", default="", help="Used in register/delete mode")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--host", default="127.0.0.1", help="Used in api mode")
    parser.add_argument("--port", type=int, default=8000, help="Used in api mode")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    registry = StudentRegistry(Path("public") / "students_db.json")

    if args.mode == "register":
        if not args.student_id or not args.name:
            print("register mode requires --student-id and --name")
            return

        registrar = StudentRegistrar(
            registry=registry,
            config=RegisterConfig(camera_index=args.camera_index),
        )
        registrar.register(student_id=args.student_id, student_name=args.name)
        return

    if args.mode == "delete":
        if args.student_id:
            deleted = registry.delete_student_by_id(args.student_id)
            snapshot_path = Path("public") / "students" / f"{args.student_id}.jpg"
            if deleted and snapshot_path.exists():
                snapshot_path.unlink()
            print("Student deleted" if deleted else "Student not found")
            return

        if args.name:
            deleted = registry.delete_student_by_name(args.name)
            print("Student(s) deleted" if deleted else "Student not found")
            return

        print("delete mode requires --student-id or --name")
        return

    if args.mode == "list":
        students = registry.list_students()
        if not students:
            print("No students registered")
            return

        for student in students:
            print(f"{student.student_id} | {student.name} | {student.created_at}")
        return

    if args.mode == "api":
        uvicorn.run("api_server:app", host=args.host, port=args.port, reload=False)
        return

    detector = StudentFocusDetector(
        registry=registry,
        config=DetectionConfig(camera_index=args.camera_index),
    )
    detector.run()


if __name__ == "__main__":
    main()
