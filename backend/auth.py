"""
User Authentication Module

Simple JWT-based authentication for multi-user support.
Users register with username/password, receive a JWT token,
and include it in API requests for user-specific data isolation.
"""
import hashlib
import logging
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import jwt
import bcrypt
from fastapi import HTTPException, Request, WebSocket

from .config import get_settings

logger = logging.getLogger(__name__)

# JWT config
JWT_SECRET = "workout-coach-secret-change-in-production"
JWT_ALGORITHM = "HS256"
JWT_EXPIRY_HOURS = 72


@dataclass
class User:
    id: int
    username: str
    created_at: float

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "username": self.username,
            "created_at": self.created_at,
        }


class AuthManager:
    """Handles user registration, login, and JWT token management."""

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            db_path = str(Path(__file__).parent.parent / "data" / "workout_coach.db")
        self.db_path = db_path
        self._init_table()

    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_table(self):
        conn = self._connect()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                created_at REAL NOT NULL
            )
        """)
        conn.commit()
        conn.close()

    def register(self, username: str, password: str) -> User:
        """Register a new user. Raises ValueError if username taken."""
        if not username or len(username) < 2:
            raise ValueError("Username must be at least 2 characters")
        if not password or len(password) < 4:
            raise ValueError("Password must be at least 4 characters")

        password_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

        conn = self._connect()
        try:
            cursor = conn.execute(
                "INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
                (username.strip(), password_hash, time.time()),
            )
            conn.commit()
            user_id = cursor.lastrowid
            logger.info(f"User registered: {username} (id={user_id})")
            return User(id=user_id, username=username, created_at=time.time())
        except sqlite3.IntegrityError:
            raise ValueError(f"Username '{username}' is already taken")
        finally:
            conn.close()

    def login(self, username: str, password: str) -> str:
        """Authenticate user and return JWT token. Raises ValueError on failure."""
        conn = self._connect()
        row = conn.execute(
            "SELECT * FROM users WHERE username = ?", (username.strip(),)
        ).fetchone()
        conn.close()

        if not row:
            raise ValueError("Invalid username or password")

        if not bcrypt.checkpw(password.encode(), row["password_hash"].encode()):
            raise ValueError("Invalid username or password")

        # Generate JWT
        token = jwt.encode(
            {
                "user_id": row["id"],
                "username": row["username"],
                "exp": time.time() + (JWT_EXPIRY_HOURS * 3600),
            },
            JWT_SECRET,
            algorithm=JWT_ALGORITHM,
        )

        logger.info(f"User logged in: {username}")
        return token

    def verify_token(self, token: str) -> User:
        """Verify a JWT token and return the user. Raises ValueError on failure."""
        try:
            payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])

            if payload.get("exp", 0) < time.time():
                raise ValueError("Token expired")

            return User(
                id=payload["user_id"],
                username=payload["username"],
                created_at=0,
            )
        except jwt.InvalidTokenError as e:
            raise ValueError(f"Invalid token: {e}")

    def get_user_from_request(self, request: Request) -> Optional[User]:
        """Extract user from request Authorization header. Returns None if no auth."""
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return None
        try:
            return self.verify_token(auth_header[7:])
        except ValueError:
            return None

    def get_user_from_websocket(self, websocket: WebSocket) -> Optional[User]:
        """Extract user from WebSocket query parameter."""
        token = websocket.query_params.get("token", "")
        if not token:
            return None
        try:
            return self.verify_token(token)
        except ValueError:
            return None
