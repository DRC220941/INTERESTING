import sqlite3
from typing import List, Dict, Optional
from datetime import datetime

class WorkingMemory:
    """Mémoire de travail (effacée sur réinitialisation)"""
    def __init__(self, db_path: str = "working_memory.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS multiplicateurs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    value REAL NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

    def add_multiple(self, values: List[float]):
        with sqlite3.connect(self.db_path) as conn:
            conn.executemany("INSERT INTO multiplicateurs (value) VALUES (?)", [(v,) for v in values])
            conn.commit()

    def get_all(self) -> List[float]:
        with sqlite3.connect(self.db_path) as conn:
            return [row[0] for row in conn.execute("SELECT value FROM multiplicateurs ORDER BY timestamp").fetchall()]

    def get_last(self, n: int = 1) -> List[float]:
        with sqlite3.connect(self.db_path) as conn:
            return [row[0] for row in conn.execute("SELECT value FROM multiplicateurs ORDER BY timestamp DESC LIMIT ?", (n,)).fetchall()][::-1]

    def reset(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM multiplicateurs")
            conn.commit()

    def count(self) -> int:
        with sqlite3.connect(self.db_path) as conn:
            return conn.execute("SELECT COUNT(*) FROM multiplicateurs").fetchone()[0]

class LearningMemory:
    """Mémoire d'apprentissage (conserve les connaissances)"""
    def __init__(self, db_path: str = "learning_memory.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS model_weights (
                    model_name TEXT PRIMARY KEY,
                    weight REAL NOT NULL DEFAULT 1.0
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS model_performance (
                    model_name TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    error REAL NOT NULL,
                    PRIMARY KEY (model_name, timestamp)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS hypotheses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    description TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('pending', 'validated', 'rejected')),
                    confidence REAL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS scenarios (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pattern TEXT NOT NULL,
                    frequency INTEGER NOT NULL,
                    last_seen DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

    def get_weights(self) -> Dict[str, float]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("SELECT model_name, weight FROM model_weights").fetchall()
            return {row[0]: row[1] for row in rows} if rows else {
                "statistical": 0.4,
                "bayesian": 0.3,
                "timeseries": 0.3,
                "ml": 0.2,
                "anomaly": 0.1
            }

    def update_weight(self, model_name: str, weight: float):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO model_weights (model_name, weight)
                VALUES (?, ?)
            """, (model_name, weight))
            conn.commit()

    def log_performance(self, model_name: str, error: float):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO model_performance (model_name, error)
                VALUES (?, ?)
            """, (model_name, error))
            conn.commit()

    def get_hypotheses(self, status: Optional[str] = None) -> List[Dict]:
        with sqlite3.connect(self.db_path) as conn:
            query = "SELECT id, description, status, confidence, created_at FROM hypotheses"
            params = []
            if status:
                query += " WHERE status = ?"
                params.append(status)
            rows = conn.execute(query, params).fetchall()
            return [{
                "id": row[0],
                "description": row[1],
                "status": row[2],
                "confidence": row[3],
                "created_at": row[4]
            } for row in rows]

    def add_hypothesis(self, description: str, confidence: float = 0.5):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO hypotheses (description, status, confidence)
                VALUES (?, 'pending', ?)
            """, (description, confidence))
            conn.commit()

    def update_hypothesis(self, hypothesis_id: int, status: str, confidence: float):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE hypotheses
                SET status = ?, confidence = ?
                WHERE id = ?
            """, (status, confidence, hypothesis_id))
            conn.commit()
