import sqlite3
from typing import List, Dict
import json

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

    def reset(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM multiplicateurs")
            conn.commit()

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
                    weight REAL NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS model_performance (
                    model_name TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    error REAL NOT NULL,
                    PRIMARY KEY (model_name, timestamp)
                )
            """)
            conn.commit()

    def get_weights(self) -> Dict[str, float]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("SELECT model_name, weight FROM model_weights").fetchall()
            return {row[0]: row[1] for row in rows}

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
