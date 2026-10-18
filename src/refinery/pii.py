"""Regex PII redaction: e-mail addresses, phone numbers, public IPv4 addresses, card numbers.

Card numbers are only redacted when they pass the Luhn checksum, which keeps
ordinary long numbers (order IDs, ISBNs) intact. Private and documentation IP
ranges are left alone, as in FineWeb. A model-based detector (e.g. Presidio
with spaCy NER) would also catch names; that is a deliberate scope limit.
"""
from __future__ import annotations

import ipaddress
import re

EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,24}\b")
PHONE = re.compile(r"(?<![\w+])(?:\+\d{1,3}[\s.-]?)?(?:\(\d{2,4}\)[\s.-]?|\d{2,4}[\s.-])\d{3,4}[\s.-]\d{3,4}(?!\w)")
IPV4 = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
CARD = re.compile(r"\b(?:\d[ -]?){13,19}\b")


def luhn_ok(number: str) -> bool:
    digits = [int(d) for d in number if d.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
    return total % 10 == 0


def _public_ip(text: str) -> bool:
    try:
        ip = ipaddress.ip_address(text)
    except ValueError:
        return False
    return ip.is_global


def redact(text: str) -> tuple[str, dict[str, int]]:
    counts = {"email": 0, "phone": 0, "ip": 0, "card": 0}

    def sub(pattern, tag, key, check=None):
        def repl(m):
            if check and not check(m.group(0)):
                return m.group(0)
            counts[key] += 1
            return tag
        return pattern.sub(repl, text_holder[0])

    text_holder = [text]
    text_holder[0] = sub(EMAIL, "<EMAIL>", "email")
    text_holder[0] = sub(CARD, "<CARD>", "card", luhn_ok)        # before phones: card digits look like phones
    text_holder[0] = sub(IPV4, "<IP>", "ip", _public_ip)
    text_holder[0] = sub(PHONE, "<PHONE>", "phone")
    return text_holder[0], counts
