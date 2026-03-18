"""Shared logging helpers including lightweight sensitive-token redaction."""

from __future__ import annotations

import logging
import re


SENSITIVE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b[\w\.-]+@[\w\.-]+\.[a-z]{2,}\b", re.IGNORECASE),
    re.compile(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    re.compile(r"\b\d{12,19}\b"),
)


def get_logger(name: str) -> logging.Logger:
    """Return a module logger configured with a default stream handler."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


def redact_text(text: str) -> str:
    """Redact common sensitive patterns before writing logs."""
    redacted = text
    for pattern in SENSITIVE_PATTERNS:
        redacted = pattern.sub("[REDACTED]", redacted)
    return redacted