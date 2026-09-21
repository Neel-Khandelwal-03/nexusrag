"""Password authentication for the chat UI.

Credentials come from settings (``AUTH_USERNAME`` / ``AUTH_PASSWORD``). Comparisons are
constant-time. Behaviour when no password is configured depends on the environment:

* **development/test**: the configured username logs in with any password, so local
  runs need no setup; a warning is logged;
* **staging/production**: every login is refused (fail closed) until a password is set.

Chainlit signs session tokens with ``CHAINLIT_AUTH_SECRET``. Locally a random one is
generated when missing (sessions then reset on restart). Deployed environments must set it.
"""

from __future__ import annotations

import hmac
import secrets
from collections.abc import MutableMapping

from nexusrag.config import Settings

AUTH_SECRET_ENV = "CHAINLIT_AUTH_SECRET"


def _same(a: str, b: str) -> bool:
    return hmac.compare_digest(a.encode("utf-8"), b.encode("utf-8"))


def password_configured(settings: Settings) -> bool:
    return settings.auth_password is not None and bool(settings.auth_password.get_secret_value())


def check_credentials(settings: Settings, username: str, password: str) -> bool:
    """True if ``username``/``password`` may log in (see module docstring for the rules)."""
    user_ok = _same(username.strip(), settings.auth_username)
    if not password_configured(settings):
        return user_ok and not settings.is_deployed
    assert settings.auth_password is not None
    # Compare both even when the username is wrong, so timing doesn't reveal which failed.
    password_ok = _same(password, settings.auth_password.get_secret_value())
    return user_ok and password_ok


def ensure_auth_secret(settings: Settings, environ: MutableMapping[str, str]) -> str | None:
    """Make sure Chainlit has a token-signing secret. Returns a warning to log, if any.

    Raises in staging/production when it's missing: generating one there would silently
    log everyone out on every restart and differ between replicas.
    """
    if environ.get(AUTH_SECRET_ENV):
        return None
    if settings.is_deployed:
        raise RuntimeError(
            f"{AUTH_SECRET_ENV} must be set in {settings.environment} "
            "(generate one with `chainlit create-secret`)."
        )
    environ[AUTH_SECRET_ENV] = secrets.token_urlsafe(48)
    return f"{AUTH_SECRET_ENV} not set; generated a temporary one (sessions reset on restart)."
