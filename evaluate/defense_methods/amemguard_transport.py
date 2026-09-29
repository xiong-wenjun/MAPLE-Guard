"""A-MemGuard transport-only recovery and metadata journal.

The caller validates model output. Only a failed HTTP/network attempt is retried,
with identical serialized request bytes. No prompts, responses or keys are logged.
"""
from __future__ import annotations

import errno
import http.client
import json
import os
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid


def max_attempts_from_environment():
    value = os.environ.get("AMEMGUARD_TRANSPORT_MAX_ATTEMPTS", "1")
    if value not in ("1", "2", "3"):
        raise ValueError("AMEMGUARD_TRANSPORT_MAX_ATTEMPTS must be 1, 2, or 3")
    return int(value)


def _transient(error):
    if isinstance(error, urllib.error.HTTPError):
        return error.code == 429 or 500 <= error.code <= 599
    if isinstance(error, urllib.error.URLError):
        return _transient(error.reason)
    if isinstance(error, socket.gaierror):
        return error.errno == socket.EAI_AGAIN
    if isinstance(error, (TimeoutError, ConnectionError,
                          http.client.RemoteDisconnected, http.client.IncompleteRead)):
        return True
    return isinstance(error, OSError) and error.errno in {
        errno.ETIMEDOUT, errno.ECONNRESET, errno.ECONNABORTED, errno.ECONNREFUSED,
        errno.EHOSTUNREACH, errno.ENETUNREACH, errno.EPIPE,
    }


def _journal(record):
    path = os.environ.get("MAPLE_CALL_LOG")
    if not path:
        return
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.write(fd, (json.dumps(record, ensure_ascii=False) + "\n").encode())
    finally:
        os.close(fd)


def _response_metadata(record, data):
    if not isinstance(data, dict):
        record["invalid_for_benchmark"] = True
        return
    record["usage"] = data.get("usage")
    choices = data.get("choices", [])
    if not isinstance(choices, list) or any(not isinstance(choice, dict) for choice in choices):
        record["invalid_for_benchmark"] = True
        return
    record["finish_reasons"] = [choice.get("finish_reason") for choice in choices]
    messages = [choice.get("message") for choice in choices]
    record["final_content_present"] = [
        isinstance(message, dict) and isinstance(message.get("content"), str)
        and bool(message["content"].strip()) for message in messages]
    record["reasoning_content_present"] = [
        isinstance(message, dict) and bool(message.get("reasoning_content")) for message in messages]
    record["invalid_for_benchmark"] = (
        "length" in record["finish_reasons"]
        or (record["endpoint"].endswith("/chat/completions")
            and (not choices or not all(record["final_content_present"]))))


def request_json(base, path, body, key, *, timeout, max_attempts):
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = "Bearer " + key
    request = urllib.request.Request(base.rstrip("/") + path,
                                     data=json.dumps(body).encode(), headers=headers)
    parsed = urllib.parse.urlsplit(request.full_url)
    request_id = uuid.uuid4().hex
    for attempt in range(1, max_attempts + 1):
        started = time.monotonic()
        record = {
            "endpoint": parsed.path, "host": parsed.hostname, "model": body.get("model"),
            "max_tokens": body.get("max_tokens"), "temperature": body.get("temperature"),
            "request_bytes": len(request.data), "message_count": len(body.get("messages", [])),
            "timestamp": time.time(), "component": "amemguard_full", "transport": "urllib",
            "request_id": request_id, "attempt": attempt, "max_attempts": max_attempts,
            "timeout_seconds": timeout, "will_retry": False,
        }
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                record["http_status"] = getattr(response, "status", 200)
                data = json.load(response)
            _response_metadata(record, data)
            return data
        except Exception as error:
            record["error_type"] = type(error).__name__
            if isinstance(error, urllib.error.HTTPError):
                record["http_status"] = error.code
                error.close()
            record["retryable_transport_error"] = _transient(error)
            record["will_retry"] = record["retryable_transport_error"] and attempt < max_attempts
            if not record["will_retry"]:
                raise
            record["retry_delay_seconds"] = float(attempt)
        finally:
            record["elapsed_seconds"] = time.monotonic() - started
            _journal(record)
        time.sleep(record["retry_delay_seconds"])
