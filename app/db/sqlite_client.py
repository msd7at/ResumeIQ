import sqlite3
import json
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

DB_PATH = os.getenv("SQLITE_DB_PATH", "./resumeiq.db")


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # lets us access columns by name: row["status"]
    return conn


def init_db():
    """Create all tables if they don't exist. Called once at app startup."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.executescript("""
        CREATE TABLE IF NOT EXISTS resume_sessions (
            id              TEXT PRIMARY KEY,
            filename        TEXT NOT NULL,
            user_location   TEXT NOT NULL,
            target_company  TEXT,
            status          TEXT NOT NULL DEFAULT 'uploaded',
            missing_fields  TEXT NOT NULL DEFAULT '[]',
            chunks_count    INTEGER DEFAULT 0,
            created_at      TEXT NOT NULL,
            updated_at      TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS analysis_results (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id      TEXT NOT NULL,
            resume_issues   TEXT NOT NULL DEFAULT '[]',
            skills_found    TEXT NOT NULL DEFAULT '[]',
            questions       TEXT NOT NULL DEFAULT '[]',
            salary_range    TEXT NOT NULL DEFAULT '{}',
            active_companies TEXT NOT NULL DEFAULT '[]',
            created_at      TEXT NOT NULL,
            FOREIGN KEY (session_id) REFERENCES resume_sessions(id)
        );

        CREATE TABLE IF NOT EXISTS chat_history (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id  TEXT NOT NULL,
            role        TEXT NOT NULL,
            message     TEXT NOT NULL,
            timestamp   TEXT NOT NULL,
            FOREIGN KEY (session_id) REFERENCES resume_sessions(id)
        );
    """)

    conn.commit()
    conn.close()


# ── resume_sessions helpers ────────────────────────────────

def create_session(session_id: str, filename: str, user_location: str,
                   target_company: str | None = None) -> None:
    now = datetime.utcnow().isoformat()
    conn = get_connection()
    conn.execute(
        """INSERT INTO resume_sessions
           (id, filename, user_location, target_company, status,
            missing_fields, chunks_count, created_at, updated_at)
           VALUES (?, ?, ?, ?, 'uploaded', '[]', 0, ?, ?)""",
        (session_id, filename, user_location, target_company, now, now)
    )
    conn.commit()
    conn.close()


def update_session_status(session_id: str, status: str,
                          missing_fields: list | None = None,
                          chunks_count: int | None = None) -> None:
    now = datetime.utcnow().isoformat()
    conn = get_connection()

    if missing_fields is not None and chunks_count is not None:
        conn.execute(
            """UPDATE resume_sessions
               SET status=?, missing_fields=?, chunks_count=?, updated_at=?
               WHERE id=?""",
            (status, json.dumps(missing_fields), chunks_count, now, session_id)
        )
    elif missing_fields is not None:
        conn.execute(
            """UPDATE resume_sessions
               SET status=?, missing_fields=?, updated_at=?
               WHERE id=?""",
            (status, json.dumps(missing_fields), now, session_id)
        )
    else:
        conn.execute(
            "UPDATE resume_sessions SET status=?, updated_at=? WHERE id=?",
            (status, now, session_id)
        )

    conn.commit()
    conn.close()


def get_session(session_id: str) -> dict | None:
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM resume_sessions WHERE id=?", (session_id,)
    ).fetchone()
    conn.close()
    if row is None:
        return None
    result = dict(row)
    result["missing_fields"] = json.loads(result["missing_fields"])
    return result


# ── analysis_results helpers ───────────────────────────────

def save_analysis(session_id: str, resume_issues: list, skills_found: list,
                  questions: list, salary_range: dict,
                  active_companies: list) -> None:
    now = datetime.utcnow().isoformat()
    conn = get_connection()
    conn.execute(
        """INSERT INTO analysis_results
           (session_id, resume_issues, skills_found, questions,
            salary_range, active_companies, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (session_id, json.dumps(resume_issues), json.dumps(skills_found),
         json.dumps(questions), json.dumps(salary_range),
         json.dumps(active_companies), now)
    )
    conn.commit()
    conn.close()


def get_analysis(session_id: str) -> dict | None:
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM analysis_results WHERE session_id=? ORDER BY id DESC LIMIT 1",
        (session_id,)
    ).fetchone()
    conn.close()
    if row is None:
        return None
    result = dict(row)
    for field in ["resume_issues", "skills_found", "questions",
                  "salary_range", "active_companies"]:
        result[field] = json.loads(result[field])
    return result


# ── chat_history helpers ───────────────────────────────────

def add_chat_message(session_id: str, role: str, message: str) -> None:
    now = datetime.utcnow().isoformat()
    conn = get_connection()
    conn.execute(
        "INSERT INTO chat_history (session_id, role, message, timestamp) VALUES (?, ?, ?, ?)",
        (session_id, role, message, now)
    )
    conn.commit()
    conn.close()


def get_chat_history(session_id: str) -> list[dict]:
    conn = get_connection()
    rows = conn.execute(
        "SELECT role, message, timestamp FROM chat_history WHERE session_id=? ORDER BY id ASC",
        (session_id,)
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


if __name__ == "__main__":
    init_db()
    print(f"Database initialised at: {Path(DB_PATH).resolve()}")
