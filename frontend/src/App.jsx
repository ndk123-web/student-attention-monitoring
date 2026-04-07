import { useEffect, useMemo, useState } from "react";

function asArray(value) {
  return Array.isArray(value) ? value : [];
}

function StatCard({ title, value, hint, tone = "neutral" }) {
  return (
    <article className={`stat-card stat-${tone}`}>
      <p className="stat-title">{title}</p>
      <p className="stat-value">{value}</p>
      <p className="stat-hint">{hint}</p>
    </article>
  );
}

export default function App() {
  const [students, setStudents] = useState([]);
  const [liveRows, setLiveRows] = useState([]);
  const [updatedAt, setUpdatedAt] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    let active = true;

    const loadStudents = async () => {
      try {
        const response = await fetch("/api/students");
        if (!response.ok) {
          throw new Error("Failed to load students");
        }
        const payload = await response.json();
        if (active) {
          setStudents(asArray(payload?.students));
        }
      } catch (err) {
        if (active) {
          setError(err.message || "Could not fetch students");
        }
      }
    };

    loadStudents();
    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    let active = true;

    const loadLiveStatus = async () => {
      try {
        const response = await fetch("/api/live-status");
        if (!response.ok) {
          throw new Error("Failed to load live status");
        }
        const payload = await response.json();
        if (active) {
          setLiveRows(asArray(payload?.students));
          setUpdatedAt(payload?.updated_at || null);
        }
      } catch (err) {
        if (active) {
          setError(err.message || "Could not fetch live status");
        }
      }
    };

    loadLiveStatus();
    const timer = setInterval(loadLiveStatus, 2000);

    return () => {
      active = false;
      clearInterval(timer);
    };
  }, []);

  const liveByName = useMemo(() => {
    const map = new Map();
    for (const row of asArray(liveRows)) {
      map.set(row.name, row);
    }
    return map;
  }, [liveRows]);

  const focusedCount = asArray(liveRows).filter((row) => row?.state === "FOCUSED").length;
  const distractedCount = asArray(liveRows).filter((row) => row?.state === "DISTRACTED").length;
  const unknownCount = asArray(liveRows).filter((row) => row?.state === "UNKNOWN").length;

  return (
    <main className="page-shell">
      <header className="hero">
        <h1>Student Focus Dashboard</h1>
        <p>Live monitoring of registered students with focus and distraction signals.</p>
        <p className="subtle">Last update: {updatedAt || "No live data yet"}</p>
      </header>

      {error ? <p className="error-text">{error}</p> : null}

      <section className="stats-grid">
        <StatCard title="Registered" value={students.length} hint="Students in registry" />
        <StatCard title="Focused" value={focusedCount} hint="Currently focused" tone="good" />
        <StatCard title="Distracted" value={distractedCount} hint="Needs attention" tone="warn" />
        <StatCard title="Unknown" value={unknownCount} hint="Not recognized" tone="neutral" />
      </section>

      <section className="panel">
        <h2>Student Status</h2>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Student ID</th>
                <th>Name</th>
                <th>State</th>
                <th>Focus Score</th>
                <th>EAR</th>
                <th>Gaze Offset</th>
                <th>Signals</th>
              </tr>
            </thead>
            <tbody>
              {asArray(students).map((student) => {
                const live = liveByName.get(student.name);
                return (
                  <tr key={student.student_id}>
                    <td>{student.student_id}</td>
                    <td>{student.name}</td>
                    <td>
                      <span className={`pill ${live?.state === "FOCUSED" ? "pill-good" : "pill-warn"}`}>
                        {live?.state || "NOT SEEN"}
                      </span>
                    </td>
                    <td>{typeof live?.focus_score === "number" ? `${live.focus_score.toFixed(1)}%` : "-"}</td>
                    <td>{typeof live?.ear === "number" ? live.ear.toFixed(2) : "-"}</td>
                    <td>{typeof live?.gaze_offset === "number" ? live.gaze_offset.toFixed(2) : "-"}</td>
                    <td>{Array.isArray(live?.signals) && live.signals.length ? live.signals.join(", ") : "stable"}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </section>
    </main>
  );
}
