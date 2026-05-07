"""
Domain reputation and existence checks for phishing detection.

This module does not make the final decision but provides
extra signals that complement the ML model. It focuses on:

    * Basic DNS resolution (does the domain exist?)
    * Simple lexical heuristics for suspicious-looking domains
"""

from __future__ import annotations

import math
import socket
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse

import requests


@dataclass
class DomainReputationResult:
    domain: str
    resolves: bool
    resolves_error: Optional[str]
    http_reachable: bool
    http_error: Optional[str]
    length: int
    label_entropy: float
    is_very_long: bool
    is_high_entropy: bool

    @property
    def summary(self) -> str:
        """
        Human‑readable one‑line summary for UI.
        """
        flags = []
        if self.resolves:
            flags.append("resolves")
        else:
            flags.append("does not resolve")

        if self.http_reachable:
            flags.append("http ok")
        else:
            flags.append("http unreachable")

        if self.is_very_long:
            flags.append("very long")
        if self.is_high_entropy:
            flags.append("high‑entropy name")

        return ", ".join(flags)


def _extract_domain(url: str) -> str:
    parsed = urlparse(url.strip())
    host = parsed.netloc or parsed.path
    return host.split(":")[0].lower()


def _shannon_entropy(s: str) -> float:
    """
    Compute Shannon entropy over characters in the string.
    Higher values ≈ more random‑looking names.
    """
    if not s:
        return 0.0
    from collections import Counter

    counts = Counter(s)
    total = float(len(s))
    entropy = 0.0
    for c in counts.values():
        p = c / total
        entropy -= p * math.log2(p)
    return entropy


def check_domain_reputation(url: str) -> DomainReputationResult:
    """
    Perform lightweight checks on the domain part of a URL.
    """
    domain = _extract_domain(url)

    # DNS resolution
    resolves = False
    err: Optional[str] = None
    if domain:
        try:
            socket.gethostbyname(domain)
            resolves = True
        except Exception as exc:  # noqa: BLE001
            resolves = False
            err = str(exc)

    # HTTP reachability (very small timeout to keep UI responsive)
    http_ok = False
    http_err: Optional[str] = None
    try:
        resp = requests.get(url, timeout=3, allow_redirects=True)
        if 200 <= resp.status_code < 400:
            http_ok = True
        else:
            http_ok = False
            http_err = f"HTTP {resp.status_code}"
    except Exception as exc:  # noqa: BLE001
        http_ok = False
        http_err = str(exc)

    # Lexical heuristics
    length = len(domain)
    entropy = _shannon_entropy("".join(ch for ch in domain if ch.isalpha()))

    is_very_long = length > 30
    is_high_entropy = entropy > 3.5 and length > 15

    return DomainReputationResult(
        domain=domain,
        resolves=resolves,
        resolves_error=err,
        http_reachable=http_ok,
        http_error=http_err,
        length=length,
        label_entropy=entropy,
        is_very_long=is_very_long,
        is_high_entropy=is_high_entropy,
    )


__all__ = [
    "DomainReputationResult",
    "check_domain_reputation",
]

