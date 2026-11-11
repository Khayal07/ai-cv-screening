"""HTTP hardening: security headers, a request body limit, and a 500 without details.

All three are plain ASGI middleware, so they also cover responses produced by other middleware
and by the static file mount.
"""

import json
import logging
from collections.abc import Awaitable, Callable, MutableMapping
from typing import Any

Scope = MutableMapping[str, Any]
Message = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]

MAX_REQUEST_BYTES = 6 * 1024 * 1024  # a 5 MB PDF plus form overhead
# FastAPI's own docs pages load Swagger UI from a CDN with inline script; they are off in production.
DOCS_PATHS = {"/docs", "/docs/oauth2-redirect", "/redoc"}

log = logging.getLogger("app.errors")


def content_security_policy(frame_ancestors: str) -> str:
    return "; ".join(
        [
            "default-src 'self'",
            "script-src 'self'",
            "style-src 'self'",
            "img-src 'self' data:",
            "connect-src 'self'",
            "font-src 'self'",
            "object-src 'none'",
            "base-uri 'none'",
            "form-action 'self'",
            f"frame-ancestors {frame_ancestors}",
        ]
    )


def security_headers(frame_ancestors: str, production: bool) -> dict[str, str]:
    headers = {
        "Content-Security-Policy": content_security_policy(frame_ancestors),
        "X-Content-Type-Options": "nosniff",
        "Referrer-Policy": "no-referrer",
        "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
        "Cross-Origin-Opener-Policy": "same-origin",
    }
    if production:
        headers["Strict-Transport-Security"] = "max-age=31536000"
    return headers


class SecurityHeadersMiddleware:
    def __init__(self, app: ASGIApp, frame_ancestors: str = "'none'", production: bool = False) -> None:
        self.app = app
        encoded = security_headers(frame_ancestors, production).items()
        self.headers = [(name.lower().encode(), value.encode()) for name, value in encoded]

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        skip_csp = scope.get("path") in DOCS_PATHS

        async def send_with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                present = {name.lower() for name, _ in headers}
                for name, value in self.headers:
                    if name in present or (skip_csp and name == b"content-security-policy"):
                        continue
                    headers.append((name, value))
                message["headers"] = headers
            await send(message)

        await self.app(scope, receive, send_with_headers)


TOO_LARGE = {"detail": "request body is too large"}


class BodySizeLimitMiddleware:
    """413 when Content-Length is too large, or when a streamed body grows past the limit.

    For a streamed body, the 413 is sent as soon as the limit is passed; the app then sees a
    client disconnect, and whatever it answers afterwards is dropped.
    """

    def __init__(self, app: ASGIApp, max_bytes: int = MAX_REQUEST_BYTES) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        if _declared_length(scope) > self.max_bytes:
            await send_json(send, 413, TOO_LARGE)
            return
        received, started, rejected = 0, False, False

        async def limited_receive() -> Message:
            nonlocal received, rejected
            if rejected:
                return {"type": "http.disconnect"}
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    rejected = True
                    if not started:
                        await send_json(send, 413, TOO_LARGE)
                    return {"type": "http.disconnect"}
            return message

        async def guarded_send(message: Message) -> None:
            nonlocal started
            if rejected:
                return
            started = started or message["type"] == "http.response.start"
            await send(message)

        try:
            await self.app(scope, limited_receive, guarded_send)
        except Exception:
            if not rejected:
                raise  # after a 413 the app only fails on the disconnect it was handed


def _declared_length(scope: Scope) -> int:
    for name, value in scope.get("headers", []):
        if name.lower() == b"content-length":
            try:
                return int(value)
            except ValueError:
                return 0
    return 0


class ErrorShieldMiddleware:
    """Turn any unhandled exception into a plain 500. Logs only the exception type, never its text."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        started = False

        async def tracking_send(message: Message) -> None:
            nonlocal started
            started = started or message["type"] == "http.response.start"
            await send(message)

        try:
            await self.app(scope, receive, tracking_send)
        except Exception as exc:  # noqa: BLE001 - the whole point is to catch everything
            log.error("unhandled error: %s", type(exc).__name__)
            if not started:
                await send_json(send, 500, {"detail": "internal error"})


async def send_json(send: Send, status: int, body: dict[str, Any]) -> None:
    payload = json.dumps(body).encode()
    headers = [(b"content-type", b"application/json"), (b"content-length", str(len(payload)).encode())]
    await send({"type": "http.response.start", "status": status, "headers": headers})
    await send({"type": "http.response.body", "body": payload})
