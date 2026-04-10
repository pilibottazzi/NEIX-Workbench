"""
Shared parsing utilities for all workbench tools.

Consolidates number parsing, string normalization, CSV detection,
and column-finding helpers.
"""
from __future__ import annotations

import csv
import io
import re
import unicodedata
from typing import Any

import pandas as pd


# =========================================================
# STRING HELPERS
# =========================================================
def safe_str(v) -> str:
    """Convert any value to str, treating None/NaN as empty string."""
    if v is None:
        return ""
    try:
        if pd.isna(v):
            return ""
    except Exception:
        pass
    return str(v)


def strip_accents(s: str) -> str:
    """Remove diacritics / combining marks from a string."""
    return "".join(
        ch for ch in unicodedata.normalize("NFD", s)
        if unicodedata.category(ch) != "Mn"
    )


def norm_header(s: str) -> str:
    """Normalize a column header: lowercase, no accents, alphanum only."""
    s = safe_str(s).strip().lower()
    s = strip_accents(s)
    s = re.sub(r"[^a-z0-9]", "", s)
    return s


def norm_cell(x) -> str:
    """Normalize a cell value: lowercase, no accents, collapsed spaces."""
    s = safe_str(x)
    s = strip_accents(s).lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s


def canon(x: Any) -> str:
    """Normalize to uppercase, no accents, collapsed spaces."""
    s = safe_str(x)
    s = strip_accents(s)
    s = re.sub(r"\s+", " ", s).strip().upper()
    return s


# =========================================================
# COLUMN FINDING
# =========================================================
def find_col(headers: list, candidates: list) -> int:
    """Find best-matching column index from a list of candidate names.

    Returns -1 if no match found.
    """
    H = [norm_header(h) for h in headers]

    for cand in candidates:
        c = norm_header(cand)
        if c in H:
            return H.index(c)

    for i, h in enumerate(H):
        for cand in candidates:
            c = norm_header(cand)
            if not c:
                continue
            if h.startswith(c) or c.startswith(h) or (c in h) or (h in c):
                return i

    return -1


# =========================================================
# NUMBER PARSING (Argentine/ES locale)
# =========================================================
def parse_float(v, *, default: float | None = None) -> float | None:
    """Parse a localized number string to float.

    Handles ES locale (1.234,56), US locale (1,234.56), trailing negatives,
    currency symbols, and whitespace. Returns *default* on failure.
    """
    s = safe_str(v).strip()
    if not s or s == "-":
        return default

    neg = False
    if s.endswith("-"):
        neg = True
        s = s[:-1].strip()
    if s.startswith("-"):
        neg = True
        s = s[1:].strip()

    s = re.sub(r"[$ \t]", "", s)

    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif "," in s and "." not in s:
        parts = s.split(",")
        if len(parts[-1]) in (1, 2, 3) and len(parts) == 2:
            s = s.replace(",", ".")
        else:
            s = s.replace(",", "")
    elif "." in s:
        if s.count(".") >= 2:
            s = s.replace(".", "")
        else:
            parts = s.split(".")
            if len(parts[-1]) not in (1, 2, 3):
                s = s.replace(".", "")

    s = re.sub(r"[^\d.]", "", s)
    if not s:
        return default

    try:
        val = float(s)
        return -val if neg else val
    except Exception:
        return default


def parse_int(v) -> int | None:
    """Parse a localized number to integer."""
    s = safe_str(v).strip()
    if not s or s == "-":
        return None
    sign = -1 if s.startswith("-") else 1
    s = re.sub(r"[^\d.,]", "", s)
    s = s.replace(".", "").replace(",", "").strip()
    if not s:
        return None
    try:
        return sign * int(s)
    except Exception:
        return None


def to_num(v) -> float:
    """Parse localized number, returning 0.0 on failure."""
    result = parse_float(v, default=0.0)
    return result if result is not None else 0.0


# =========================================================
# CSV UTILITIES
# =========================================================
def detect_delimiter(raw_bytes: bytes) -> str:
    """Auto-detect CSV delimiter from raw bytes."""
    try:
        sample = raw_bytes[:4096].decode("utf-8", errors="replace")
        dialect = csv.Sniffer().sniff(sample, delimiters=[",", ";"])
        return dialect.delimiter
    except Exception:
        return ";"


def read_csv_auto(uploaded_file) -> pd.DataFrame:
    """Read a CSV file with auto-detected delimiter and latin1 fallback."""
    raw = uploaded_file.getvalue()
    text = raw.decode("latin1", errors="replace")
    lines = text.splitlines()
    first = lines[0] if lines else ""
    sep = ";" if first.count(";") > first.count(",") else ","
    return pd.read_csv(io.StringIO(text), sep=sep, engine="python", dtype=str)
