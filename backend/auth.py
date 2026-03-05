
# What to change:
#   1. ADVANCED: Configure OAuth2 / campus SSO if required.
#   2. Set SSO_CLIENT_ID, SSO_CLIENT_SECRET, SSO_AUTHORITY_URL in .env.
#
# TODO[USER_ACTION]: CONFIGURE_SSO — replace stub with real OAuth2 integration.

"""
auth.py — Stub authentication module.

Provides a simple role-based access simulation for demo purposes.
In production, replace with campus SSO (OAuth2 / SAML).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# TODO[USER_ACTION]: CONFIGURE_SSO — set these in .env for real OAuth2.
SSO_CLIENT_ID: str = os.getenv("SSO_CLIENT_ID", "")
SSO_CLIENT_SECRET: str = os.getenv("SSO_CLIENT_SECRET", "")
SSO_AUTHORITY_URL: str = os.getenv("SSO_AUTHORITY_URL", "")

# Demo users (role: student | faculty | admin)
DEMO_USERS: Dict[str, Dict[str, Any]] = {
    "student_demo": {"name": "Alice Student", "role": "student", "email": "alice@christuniversity.in"},
    "faculty_demo": {"name": "Dr. Ramesh Kumar", "role": "faculty", "email": "ramesh.k@christuniversity.in"},
    "admin_demo":   {"name": "Admin User", "role": "admin", "email": "admin@christuniversity.in"},
}


def authenticate_demo(username: str) -> Optional[Dict[str, Any]]:
    """Simulate authentication for demo mode.

    Args:
        username: One of the demo user keys.

    Returns:
        User info dict or None if not found.
    """
    user = DEMO_USERS.get(username)
    if user:
        logger.info("Demo auth: %s → %s", username, user["role"])
    else:
        logger.warning("Demo auth failed for: %s", username)
    return user


def check_role(user: Dict[str, Any], required_role: str) -> bool:
    """Check if user has the required role.

    Role hierarchy: admin > faculty > student.
    """
    hierarchy = {"admin": 3, "faculty": 2, "student": 1}
    user_level = hierarchy.get(user.get("role", ""), 0)
    required_level = hierarchy.get(required_role, 0)
    return user_level >= required_level


def get_sso_login_url() -> str:
    """Return the SSO login URL (stub).

    TODO[USER_ACTION]: CONFIGURE_SSO — implement real OAuth2 redirect.
    """
    if SSO_AUTHORITY_URL:
        return f"{SSO_AUTHORITY_URL}/authorize?client_id={SSO_CLIENT_ID}&response_type=code"
    return "#sso-not-configured"
