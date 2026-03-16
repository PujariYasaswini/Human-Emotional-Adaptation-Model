import sqlite3
import time
from threading import Lock
from collections import deque

class EmotionHistory:
    def __init__(self, db_path="emotion.db", maxlen=20):
        self.db_path = db_path
        self.lock = Lock()
        self._init_db()

        self._history = deque(maxlen=maxlen)

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
            CREATE TABLE IF NOT EXISTS emotion_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                user_id TEXT,
                session_id TEXT,
                emotion TEXT,
                category TEXT,
                confidence REAL,
                face_score REAL,
                voice_score REAL,
                text_score REAL
            )
            """)

            conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_time
            ON emotion_history(timestamp)
            """)

    def _emotion_category(self, emotion):
        positive = {"Happy", "Excited", "Joy"}
        negative = {"Sad", "Angry", "Nervous", "Disgust"}
        neutral = {"Neutral", "Calm", "Bored"}

        if emotion in positive:
            return "Positive"
        if emotion in negative:
            return "Negative"
        return "Neutral"

    def log(
        self,
        emotion,
        confidence,
        user_id="anonymous",
        session_id="default",
        face_score=None,
        voice_score=None,
        text_score=None
    ):
        # Store in SQLite
        with self.lock, sqlite3.connect(self.db_path) as conn:
            conn.execute("""
            INSERT INTO emotion_history (
                timestamp,
                user_id,
                session_id,
                emotion,
                category,
                confidence,
                face_score,
                voice_score,
                text_score
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                time.time(),
                user_id,
                session_id,
                emotion,
                self._emotion_category(emotion),
                confidence,
                face_score,
                voice_score,
                text_score
            ))

        # Store in memory deque
        self._history.append({
            "timestamp": time.time(),
            "emotion": emotion,
            "confidence": confidence,
            "face_score": face_score,
            "voice_score": voice_score,
            "text_score": text_score,
            "category": self._emotion_category(emotion)
        })

    def add(self, emotion, confidence=1.0):
        """Add an emotion to in-memory history only (optional confidence)."""
        self._history.append({
            "timestamp": time.time(),
            "emotion": emotion,
            "confidence": confidence,
            "category": self._emotion_category(emotion)
        })

    @property
    def history(self):
        """Return the in-memory emotion history."""
        return list(self._history)

    def get_recent(self, user_id="anonymous", seconds=60):
        since = time.time() - seconds
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
            SELECT emotion, confidence, timestamp
            FROM emotion_history
            WHERE user_id = ?
              AND timestamp >= ?
            ORDER BY timestamp DESC
            """, (user_id, since))
            return cursor.fetchall()

    def get_trend(self, session_id="default", window=300):
        since = time.time() - window
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
            SELECT category, COUNT(*)
            FROM emotion_history
            WHERE session_id = ?
              AND timestamp >= ?
            GROUP BY category
            """, (session_id, since))
            return dict(cursor.fetchall())
