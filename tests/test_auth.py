"""
Tests for backend/auth.py

Covers user registration, login, JWT token management, and user isolation.
"""
import tempfile
import pytest
from backend.auth import AuthManager


@pytest.fixture
def auth_mgr():
    """Create auth manager with temporary database."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        mgr = AuthManager(db_path=f.name)
        yield mgr


def test_register_user(auth_mgr):
    """Register a new user successfully."""
    user = auth_mgr.register("testuser", "pass1234")
    assert user.username == "testuser"
    assert user.id > 0


def test_register_duplicate_fails(auth_mgr):
    """Registering the same username twice should fail."""
    auth_mgr.register("testuser", "pass1234")
    with pytest.raises(ValueError, match="already taken"):
        auth_mgr.register("testuser", "otherpass")


def test_register_short_username(auth_mgr):
    """Username under 2 chars should fail."""
    with pytest.raises(ValueError, match="at least 2"):
        auth_mgr.register("a", "pass1234")


def test_register_short_password(auth_mgr):
    """Password under 4 chars should fail."""
    with pytest.raises(ValueError, match="at least 4"):
        auth_mgr.register("testuser", "abc")


def test_login_returns_jwt(auth_mgr):
    """Successful login should return a JWT token string."""
    auth_mgr.register("testuser", "pass1234")
    token = auth_mgr.login("testuser", "pass1234")
    assert isinstance(token, str)
    assert len(token) > 20


def test_login_wrong_password(auth_mgr):
    """Wrong password should fail."""
    auth_mgr.register("testuser", "pass1234")
    with pytest.raises(ValueError, match="Invalid"):
        auth_mgr.login("testuser", "wrongpass")


def test_login_nonexistent_user(auth_mgr):
    """Login with non-existent user should fail."""
    with pytest.raises(ValueError, match="Invalid"):
        auth_mgr.login("nobody", "pass1234")


def test_verify_token(auth_mgr):
    """A valid JWT token should verify and return the user."""
    auth_mgr.register("testuser", "pass1234")
    token = auth_mgr.login("testuser", "pass1234")
    user = auth_mgr.verify_token(token)
    assert user.username == "testuser"


def test_verify_invalid_token(auth_mgr):
    """An invalid token should raise ValueError."""
    with pytest.raises(ValueError, match="Invalid token"):
        auth_mgr.verify_token("not.a.real.token")


def test_password_is_hashed(auth_mgr):
    """Password should be stored as bcrypt hash, not plaintext."""
    import sqlite3
    auth_mgr.register("testuser", "pass1234")
    conn = sqlite3.connect(auth_mgr.db_path)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT password_hash FROM users WHERE username = 'testuser'").fetchone()
    conn.close()
    assert row["password_hash"] != "pass1234"
    assert row["password_hash"].startswith("$2b$")
