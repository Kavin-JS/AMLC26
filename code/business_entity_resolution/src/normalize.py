"""
Text normalization for business names and addresses.
"""

import re
import unicodedata


# ── Legal suffix abbreviations ────────────────────────────────────────────────
_LEGAL = {
    r"\bincorporated\b": "inc",
    r"\bcorporation\b": "corp",
    r"\bcorporations\b": "corp",
    r"\blimited liability company\b": "llc",
    r"\blimited liability partnership\b": "llp",
    r"\bprivate limited\b": "pvt ltd",
    r"\bprivate ltd\b": "pvt ltd",
    r"\bprivate\b": "pvt",
    r"\blimited\b": "ltd",
    r"\bcompany\b": "co",
    r"\bbrothers\b": "bros",
    r"\band\b": "&",
    r"\benter(?:prises|prise)\b": "ent",
    r"\btrading\b": "trdg",
    r"\binternational\b": "intl",
    r"\bnational\b": "natl",
    r"\bassociates\b": "assoc",
    r"\bservices\b": "svcs",
    r"\bsolutions\b": "soln",
    r"\bindustries\b": "ind",
    r"\bgroup\b": "grp",
    r"\bmanagement\b": "mgmt",
}

# ── Address abbreviations ─────────────────────────────────────────────────────
_ADDR = {
    r"\bstreet\b": "st",
    r"\broad\b": "rd",
    r"\bavenue\b": "ave",
    r"\bboulevard\b": "blvd",
    r"\bdrive\b": "dr",
    r"\bcourt\b": "ct",
    r"\bcircle\b": "cir",
    r"\blane\b": "ln",
    r"\bplace\b": "pl",
    r"\bsuite\b": "ste",
    r"\bapartment\b": "apt",
    r"\bbuilding\b": "bldg",
    r"\bnorth\b": "n",
    r"\bsouth\b": "s",
    r"\beast\b": "e",
    r"\bwest\b": "w",
    r"\bnortheast\b": "ne",
    r"\bnorthwest\b": "nw",
    r"\bsoutheast\b": "se",
    r"\bsouthwest\b": "sw",
    r"\bhighway\b": "hwy",
    r"\bfreeway\b": "fwy",
    r"\bparkway\b": "pkwy",
    r"\bnagar\b": "ngr",
    r"\bcolony\b": "col",
    r"\bsector\b": "sec",
}

_NAME_RE = [(re.compile(p, re.I), r) for p, r in _LEGAL.items()]
_ADDR_RE = [(re.compile(p, re.I), r) for p, r in _ADDR.items()]

# punctuation to space (keep alphanumerics + &)
_PUNC = re.compile(r"[^\w\s&]")
_WS   = re.compile(r"\s+")


def _to_ascii(text: str) -> str:
    """Try to transliterate unicode → ASCII; keep original chars if it fails."""
    nfkd = unicodedata.normalize("NFKD", text)
    ascii_only = nfkd.encode("ascii", "ignore").decode("ascii")
    # If result is empty (e.g., all Devanagari), fall back
    return ascii_only if ascii_only.strip() else text


def normalize_name(name: str) -> str:
    if not isinstance(name, str) or not name.strip():
        return ""
    t = _to_ascii(name.strip().lower())
    t = _PUNC.sub(" ", t)
    for pat, rep in _NAME_RE:
        t = pat.sub(rep, t)
    t = _WS.sub(" ", t).strip()
    return t


def normalize_address(addr: str) -> str:
    if not isinstance(addr, str) or not addr.strip():
        return ""
    t = _to_ascii(addr.strip().lower())
    t = _PUNC.sub(" ", t)
    for pat, rep in _ADDR_RE:
        t = pat.sub(rep, t)
    t = _WS.sub(" ", t).strip()
    return t


def name_tokens(name: str) -> frozenset:
    """Token set from a normalized name (for overlap features)."""
    return frozenset(normalize_name(name).split())


def addr_tokens(addr: str) -> frozenset:
    """Token set from a normalized address."""
    return frozenset(normalize_address(addr).split())


def numeric_tokens(text: str) -> frozenset:
    """Extract numeric tokens (house numbers, PINs, etc.)."""
    if not isinstance(text, str):
        return frozenset()
    return frozenset(re.findall(r"\d+", text))
