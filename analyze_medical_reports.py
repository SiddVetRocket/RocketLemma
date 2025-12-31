# --- analyze_medical_reports.py ---
"""
Analyze medical reports (Findings/Conclusions) and extract conditions + statuses.

Usage:
    python analyze_medical_reports.py input.csv output.json [limit]

    input.csv should contain columns:
      - ID
      - Findings
      - Conclusion

<<<<<<< HEAD
Outputs a JSON array (one object per report) with:
  - conditions: raw condition hits (per-sentence) with status + reason
  - condition_summary: deduped/aggregated per canonical condition with final status + evidence
  - findings: raw non-condition findings (if enabled) similarly structured
  - finding_summary: deduped/aggregated per canonical finding

This version adds:
  - explicit "uncertain" status (cannot be excluded, suspicious for, differential includes, etc.)
  - safer aggregation: if both present and absent found -> uncertain
  - evidence snippets in the summary for faster manual QA
=======
Behavior:
- Uses MeSH-derived term lists (conditions + findings) via PhraseMatcher.
- If MedSpaCy is installed: TargetMatcher + ConText overrides heuristics.
- Otherwise: sentence-scoped heuristics to assign present/absent/unknown.
- Aggressive filtering of ultra-generic junk "conditions" (disease, abnormality, etc.).
>>>>>>> 3d05d01277cbc18d028bad6d78e2bda522ccc7c5
"""

from __future__ import annotations

import csv
import json
<<<<<<< HEAD
import re
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# -----------------------------
# Config / Heuristics
# -----------------------------
=======
import time
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import pandas as pd
import spacy
from spacy.matcher import PhraseMatcher
>>>>>>> 3d05d01277cbc18d028bad6d78e2bda522ccc7c5

# Sentiment/status labels we use everywhere
PRESENT = "present"
ABSENT = "absent"
UNCERTAINE = "uncertain"  # intentionally spelled unique to avoid shadowing "uncertain" var name
UNKNOWN = "unknown"

<<<<<<< HEAD
# Uncertainty phrases (strong signals)
UNCERTAINTY_PATTERNS = [
    r"\bcannot be excluded\b",
    r"\bcan(?:not|'t)\s+exclude\b",
    r"\bnot excluded\b",
    r"\bmay represent\b",
    r"\bmay reflect\b",
    r"\bmay be\b",
    r"\bpossible\b",
    r"\bpossibly\b",
    r"\bprobable\b",
    r"\bprobably\b",
    r"\blikely\b",
    r"\bsuspicious for\b",
    r"\bconcerning for\b",
    r"\bquestion of\b",
    r"\bconsider\b",
    r"\bconsidering\b",
    r"\bdifferential includes\b",
    r"\bdifferential diagnosis\b",
    r"\bvs\.\b",
    r"\bversus\b",
    r"\br/o\b",
    r"\brule out\b",
    r"\br/o\b",
    r"\bquery\b",
]

# Negation patterns (simple scope; medspacy ConText may override)
NEGATION_PATTERNS = [
    r"\bno\b",
    r"\bwithout\b",
    r"\bnegative for\b",
    r"\bfree of\b",
    r"\bdenies\b",
    r"\babsent\b",
    r"\bnot present\b",
    r"\bno evidence of\b",
    r"\bno sign of\b",
]

# Post-negation / historical mentions that should not imply present
HISTORY_PATTERNS = [
    r"\bhistory of\b",
    r"\bhx of\b",
    r"\bprior\b",
    r"\bprevious\b",
    r"\bstatus post\b",
    r"\bs\/p\b",
]

# If a sentence has BOTH strong present and strong absent, treat as uncertain
CONFLICT_TO_UNCERTAIN = True

# Evidence retention limits in summary
MAX_EVIDENCE_PER_CONDITION = 3

# -----------------------------
# Optional: spaCy / medspacy
# -----------------------------

ENGINE = "spacy_fallback"  # default; may be overridden to medspacy if available

try:
    import spacy  # type: ignore
    _SPACY_OK = True
except Exception:
    spacy = None
    _SPACY_OK = False

try:
    import medspacy  # type: ignore
    from medspacy.context import ConTextComponent  # type: ignore
    _MEDSPACY_OK = True
except Exception:
    medspacy = None
    ConTextComponent = None
    _MEDSPACY_OK = False

# -----------------------------
# Utilities
# -----------------------------
=======

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

MESH_COND_FILE = BASE_DIR / "out" / "conditions_mesh.txt"
MESH_FIND_FILE = BASE_DIR / "out" / "findings_mesh.txt"

USE_SYNONYMS = True
USE_MEDSPACY_IF_AVAILABLE = True

POSSIBLE_COUNTS_AS_PRESENT = False  # keep hedged language as unknown
SPACY_MODEL = "en_core_web_sm"

nlp = spacy.load(SPACY_MODEL)
# Ensure sentence boundaries exist (sm usually has parser, but be defensive)
if "sentencizer" not in nlp.pipe_names and "parser" not in nlp.pipe_names:
    nlp.add_pipe("sentencizer")


# -----------------------------------------------------------------------------
# Expanded generic stoplist
# -----------------------------------------------------------------------------

GENERIC_CANONICAL_STOP = {
    "disease", "diseases",
    "disorder", "disorders",
    "condition", "conditions",
    "abnormality", "abnormalities",
    "pathology", "pathologies",
    "process", "processes",
    "lesion", "lesions",
    "mass", "masses",
    "neoplasm", "neoplasms",
    "tumor", "tumors",
    "infection", "infections",
    "inflammation",
    "injury", "injuries",
    "trauma",
    "change", "changes",
    "appearance",
    "finding", "findings",
    "artifact", "artifacts",
    "technique",
    "effusion",  # optional: comment out if you want to keep
    "edema",     # optional: comment out if you want to keep
    "pain",
    "swelling",
    "enlargement",
}

GENERIC_SPAN_STOP = {
    "disease", "disorder", "condition", "abnormality", "pathology",
    "process", "finding", "findings", "changes", "change", "appearance",
    "technique", "artifact",
}
>>>>>>> 3d05d01277cbc18d028bad6d78e2bda522ccc7c5

GENERIC_REGEX_STOP = [
    r"^airway disease$",
    r"^lower airway disease$",
    r"^upper airway disease$",
    r"^pulmonary pathology$",
    r"^lung disease$",
    r"^cardiac disease$",
    r"^abdominal disease$",
]

<<<<<<< HEAD
def _clean_text(x: Any) -> str:
    if x is None:
        return ""
    s = str(x)
    # Normalize whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _split_sentences(text: str) -> List[str]:
    """
    Lightweight sentence splitter. If spaCy is available, use it.
    Otherwise a conservative regex split.
    """
    text = _clean_text(text)
    if not text:
        return []
    if _SPACY_OK:
        try:
            nlp = _get_spacy_nlp()
            doc = nlp(text)
            sents = [s.text.strip() for s in doc.sents if s.text.strip()]
            return sents if sents else [text]
        except Exception:
            pass
    # Regex fallback
    parts = re.split(r"(?<=[\.\!\?])\s+", text)
    parts = [p.strip() for p in parts if p.strip()]
    return parts if parts else [text]


def _compile_any(patterns: List[str]) -> re.Pattern:
    return re.compile("|".join(f"(?:{p})" for p in patterns), flags=re.I)


_RE_UNC = _compile_any(UNCERTAINTY_PATTERNS)
_RE_NEG = _compile_any(NEGATION_PATTERNS)
_RE_HX = _compile_any(HISTORY_PATTERNS)


def _detect_uncertainty(sentence: str) -> bool:
    return bool(_RE_UNC.search(sentence))


def _detect_negation(sentence: str) -> bool:
    return bool(_RE_NEG.search(sentence))


def _detect_history(sentence: str) -> bool:
    return bool(_RE_HX.search(sentence))


# -----------------------------
# Condition dictionary (placeholder)
# -----------------------------
# In your real project this is populated by MeSH + aliases.
# For this copy/paste version, we keep the interface but allow injection/extension.

@dataclass
class TermEntry:
    canonical: str
    aliases: List[str]


def load_default_terms() -> List[TermEntry]:
    """
    Placeholder defaults. Replace/extend with your MeSH-derived dictionary.
    """
    return [
        TermEntry("pneumonia", ["pneumonia"]),
        TermEntry("pleural effusion", ["pleural effusion", "effusion"]),
        TermEntry("pneumothorax", ["pneumothorax"]),
        TermEntry("mass", ["mass", "lesion", "tumor", "tumour"]),
        TermEntry("lymphoma", ["lymphoma"]),
        TermEntry("lymphadenopathy", ["lymphadenopathy", "adenopathy", "enlarged lymph node", "enlarged lymph nodes"]),
        TermEntry("neuropathy", ["neuropathy", "peripheral neuropathy"]),
    ]


def _build_alias_map(terms: List[TermEntry]) -> Dict[str, str]:
    """
    Map lowercase alias -> canonical.
    """
    m: Dict[str, str] = {}
    for t in terms:
        for a in t.aliases:
            a2 = a.strip().lower()
            if a2:
                m[a2] = t.canonical
    return m


# -----------------------------
# spaCy setup
# -----------------------------
_SPACY_NLP = None


def _get_spacy_nlp():
    global _SPACY_NLP
    if _SPACY_NLP is not None:
        return _SPACY_NLP
    if not _SPACY_OK:
        raise RuntimeError("spaCy not installed.")
    # Use a small English model if present; otherwise blank with sentencizer
    try:
        _SPACY_NLP = spacy.load("en_core_web_sm")
    except Exception:
        _SPACY_NLP = spacy.blank("en")
        if "sentencizer" not in _SPACY_NLP.pipe_names:
            _SPACY_NLP.add_pipe("sentencizer")
    return _SPACY_NLP


# -----------------------------
# Core extraction
# -----------------------------

def _find_terms_in_sentence(sentence: str, alias_map: Dict[str, str]) -> List[Tuple[str, str]]:
    """
    Return list of (canonical, matched_alias) terms found in sentence.
    Simple substring matching with word boundaries.
    """
    s = " " + sentence.lower() + " "
    hits: List[Tuple[str, str]] = []
    for alias, canonical in alias_map.items():
        # Word boundary-ish match: alias surrounded by non-word or spaces
        # We use regex to avoid matching 'mass' inside 'massage'
        pat = re.compile(rf"(?<!\w){re.escape(alias)}(?!\w)", flags=re.I)
        if pat.search(sentence):
            hits.append((canonical, alias))
    return hits


def _classify_sentence_status(sentence: str) -> Tuple[str, str]:
    """
    Heuristic classification: absent vs uncertain vs present vs unknown.
    Returns (status, reason).
    """
    s = _clean_text(sentence)

    hx = _detect_history(s)
    neg = _detect_negation(s)
    unc = _detect_uncertainty(s)

    # History language means do not treat as present
    if hx and not neg and not unc:
        return UNKNOWN, "history_or_prior"

    if neg and unc:
        # e.g., "no evidence to exclude..." etc. treat uncertain
        return (RUNC := "uncertain"), "negation+uncertainty"
    if neg:
        return ABSENT, "negated"
    if unc:
        return (RUNC := "uncertain"), "uncertain_language"

    # If nothing indicates absent/uncertain, default to present when term is mentioned
    return PRESENT, "affirmed"


def _merge_statuses(statuses: List[str]) -> str:
    """
    Aggregate statuses for a canonical condition across multiple hits.

    Rules:
      - if both present and absent -> uncertain (if enabled)
      - else if any present -> present
      - else if any uncertain -> uncertain
      - else if any absent -> absent
      - else unknown
    """
    st = set(statuses)

    if CONFLICT_TO_UNCERTAIN and PRESENT in st and ABSENT in st:
        return (RUNC := "uncertain")

    if PRESENT in st:
        return PRESENT
    if "uncertain" in st:
        return "uncertain"
    if ABSENT in st:
        return ABSENT
    return UNKNOWN


def _summarize_hits(hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Group raw hits by canonical and produce a clean summary:
      - canonical
      - final_status
      - counts by status
      - evidence snippets (up to MAX_EVIDENCE_PER_CONDITION)
    """
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for h in hits:
        grouped.setdefault(h["canonical"], []).append(h)

    summary: List[Dict[str, Any]] = []
    for canonical, items in grouped.items():
        statuses = [it.get("status", UNKNOWN) for it in items]
        final_status = _merge_statuses(statuses)

        counts = {
            PRESENT: sum(1 for s in statuses if s == PRESENT),
            ABSENT: sum(1 for s in statuses if s == ABSENT),
            "uncertain": sum(1 for s in statuses if s == "uncertain"),
            UNKNOWN: sum(1 for s in statuses if s == UNKNOWN),
        }

        # Collect evidence (prefer present/uncertain/absent in that order)
        def score(it: Dict[str, Any]) -> int:
            st = it.get("status", UNKNOWN)
            if st == PRESENT:
                return 3
            if st == "uncertain":
                return 2
            if st == ABSENT:
                return 1
            return 0

        evidence_sorted = sorted(items, key=score, reverse=True)
        evidence = []
        seen_sentences = set()
        for it in evidence_sorted:
            sent = it.get("sentence", "")
            if not sent or sent in seen_sentences:
                continue
            seen_sentences.add(sent)
            evidence.append(
                {
                    "sentence": sent,
                    "status": it.get("status", UNKNOWN),
                    "reason": it.get("reason", ""),
                    "source": it.get("source", ""),
                }
            )
            if len(evidence) >= MAX_EVIDENCE_PER_CONDITION:
                break

        summary.append(
            {
                "canonical": canonical,
                "final_status": final_status,
                "counts": counts,
                "evidence": evidence,
            }
        )

    # Stable ordering: present first, then uncertain, then absent, then unknown
    order = {PRESENT: 0, "uncertain": 1, ABSENT: 2, UNKNOWN: 3}
    summary.sort(key=lambda x: (order.get(x["final_status"], 9), x["canonical"]))
    return summary


def analyze_text_block(
    text: str,
    alias_map: Dict[str, str],
    source: str,
) -> List[Dict[str, Any]]:
    """
    Extract condition hits from a text block (Findings or Conclusion).
    Returns list of raw hits:
      {canonical, matched_alias, status, reason, sentence, source}
    """
    hits: List[Dict[str, Any]] = []
    for sent in _split_sentences(text):
        terms = _find_terms_in_sentence(sent, alias_map)
        if not terms:
            continue
        status, reason = _classify_sentence_status(sent)
        for canonical, matched_alias in terms:
            hits.append(
                {
                    "canonical": canonical,
                    "matched_alias": matched_alias,
                    "status": status,
                    "reason": reason,
                    "sentence": sent,
                    "source": source,
                }
            )
    return hits


# -----------------------------
# CSV IO + Main
# -----------------------------

def read_input_csv(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows

=======
def _norm(s: str) -> str:
    return (s or "").strip().lower()

def is_generic_garbage(canonical: str, span_text: str) -> bool:
    c = _norm(canonical)
    t = _norm(span_text)

    if not c:
        return True

    if c in GENERIC_CANONICAL_STOP:
        # If the span is multiword and looks specific, keep it.
        if len(t.split()) <= 1:
            return True
        if len(c.split()) <= 2 and c in GENERIC_CANONICAL_STOP:
            return True

    if t in GENERIC_SPAN_STOP:
        return True

    for pat in GENERIC_REGEX_STOP:
        if re.match(pat, c):
            return True

    if len(c) <= 4 and c in {"mass", "pain"}:
        return True

    return False


# -----------------------------------------------------------------------------
# Cue lists (sentence-scoped)
# -----------------------------------------------------------------------------

# These should ALWAYS be "unknown" (hedged / differential / not-conclusive language)
UNCERTAIN_CUES = [
    "cannot exclude",
    "cannot rule out",
    "can't exclude",
    "can't rule out",
    "not ruled out",
    "not excluded",
    "cannot be excluded",
    "cannot be ruled out",
    "cannot be entirely ruled out",
    "not conclusive for",
    "not definitive for",
    "not diagnostic for",
    "concerning but not conclusive",
    "concerning but not definitive",
    "cannot confirm",
    "unable to confirm",
    "possible",
    "possibly",
    "may represent",
    "may indicate",
    "could represent",
    "could indicate",
    "question of",
    "suspicious for",
    "concerning for",
    "most concerning for",
    "suspect",
    "suspected",
    "equivocal",
    "indeterminate",
    "vs",
    "versus",
    "differential",
    "rule out",
    "r/o",
    "evaluate for",
    "to evaluate for",
]

PRESENT_CUES = [
    "consistent with",
    "compatible with",
    "indicative of",
    "diagnosis of",
    "diagnostic for",
    "demonstrates",
    "demonstrated",
    "identified",
    "noted",
    "seen",
    "present",
    "there is",
    "there are",
    "shows",
    "showing",
    "reveals",
]

NEGATION_SCOPED_PREFIXES = [
    "no",
    "without",
    "free of",
    "negative for",
    "no evidence of",
    "no sign of",
    "no signs of",
]

# “hard” negation phrases that should NOT be overridden by uncertainty
HARD_NEGATION_PHRASES = [
    "no evidence of",
    "no sign of",
    "no signs of",
    "negative for",
    "without",
    "free of",
]

SEVERITY_CUES = [
    "mild", "moderate", "severe", "marked",
    "progressive", "worsening", "persistent",
    "acute", "chronic",
]

EXAMPLE_CUES = ["e.g.", "eg ", "for example", "such as"]


# -----------------------------------------------------------------------------
# MeSH-based matchers
# -----------------------------------------------------------------------------

def build_matchers(use_synonyms: bool = True):
    if not MESH_COND_FILE.exists() or not MESH_FIND_FILE.exists():
        raise FileNotFoundError(
            f"Missing MeSH term files.\nExpected:\n  {MESH_COND_FILE}\n  {MESH_FIND_FILE}\n"
            f"Run: python mesh_terms_extract.py"
        )

    cond_alias, cond_syns = load_mesh_term_file(str(MESH_COND_FILE))
    find_alias, find_syns = load_mesh_term_file(str(MESH_FIND_FILE))

    cond_matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    find_matcher = PhraseMatcher(nlp.vocab, attr="LOWER")

    # If use_synonyms=True, match aliases (surface forms) and map to canonical via alias dict.
    # If use_synonyms=False, match canonical keys.
    cond_terms = list(cond_alias.keys()) if use_synonyms else list(cond_syns.keys())
    find_terms = list(find_alias.keys()) if use_synonyms else list(find_syns.keys())

    if cond_terms:
        cond_matcher.add("CONDITION", [nlp.make_doc(t) for t in cond_terms])
    if find_terms:
        find_matcher.add("FINDING", [nlp.make_doc(t) for t in find_terms])

    return cond_matcher, find_matcher, cond_alias, find_alias


cond_matcher, find_matcher, cond_alias, find_alias = build_matchers(USE_SYNONYMS)


# -----------------------------------------------------------------------------
# Optional: MedSpaCy integration (TargetMatcher + ConText)
# -----------------------------------------------------------------------------

MEDSPACY_AVAILABLE = False
medspacy_nlp = None

def _silence_medspacy_debug_logs():
    """
    PyRuSH uses loguru and is extremely chatty at DEBUG.
    This disables it without affecting your own prints.
    """
    # loguru-based (PyRuSH)
    try:
        from loguru import logger
        logger.remove()
        # Only show WARNING+ from anything using loguru
        logger.add(sys.stderr, level="WARNING")
    except Exception:
        pass

    # standard logging-based (fallback)
    try:
        import logging
        logging.getLogger().setLevel(logging.WARNING)
        for name in [
            "PyRuSH",
            "PyRuSH.PyRuSHSentencizer",
            "medspacy",
        ]:
            logging.getLogger(name).setLevel(logging.WARNING)
    except Exception:
        pass

def try_setup_medspacy():
    """
    Correct setup for medspacy 1.3.x:
      - Use medspacy.load(enable=[...]) to get pipeline with target_matcher + context
      - Add TargetRules into the existing 'medspacy_target_matcher' pipe
      - ConText will attach negation/uncertainty/etc as span extensions
    """
    global MEDSPACY_AVAILABLE, medspacy_nlp

    if not USE_MEDSPACY_IF_AVAILABLE:
        return

    try:
        _silence_medspacy_debug_logs()

        import medspacy
        from medspacy.target_matcher import TargetRule

        # Only enable what we need; keeps it fast and predictable
        medspacy_nlp = medspacy.load(enable=["medspacy_target_matcher", "medspacy_context"])

        tm = medspacy_nlp.get_pipe("medspacy_target_matcher")

        # Clear rules to avoid duplicates across reruns (dev sessions)
        try:
            tm.rules = []
        except Exception:
            pass

        rules: List[TargetRule] = []

        # Add MeSH alias terms as targets (COND/FIND). We still map alias->canonical ourselves.
        for alias in cond_alias.keys():
            a = (alias or "").strip()
            if a:
                rules.append(TargetRule(literal=a, category="COND"))

        for alias in find_alias.keys():
            a = (alias or "").strip()
            if a:
                rules.append(TargetRule(literal=a, category="FIND"))

        tm.add(rules)

        MEDSPACY_AVAILABLE = True
    except Exception:
        MEDSPACY_AVAILABLE = False
        medspacy_nlp = None

try_setup_medspacy()


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def sentence_for_span(doc, start_char: int, end_char: int) -> str:
    try:
        for sent in doc.sents:
            if sent.start_char <= start_char and end_char <= sent.end_char:
                return sent.text
    except Exception:
        pass
    text = doc.text
    ws = max(0, start_char - 140)
    we = min(len(text), end_char + 140)
    return text[ws:we]

def _term_regex(term: str) -> str:
    term = (term or "").strip()
    if not term:
        return ""
    parts = [re.escape(p) for p in term.split()]
    return r"\b" + r"\s+".join(parts) + r"\b"

def classify_status_sentence_scoped(sentence: str, term_text: str) -> str:
    s = (sentence or "").strip()
    sl = s.lower()
    tl = (term_text or "").strip().lower()
    if not sl or not tl:
        return "unknown"

    term_pat = _term_regex(tl)
    if not term_pat:
        return "unknown"

    # 1) Uncertainty wins (unknown)
    if any(cue in sl for cue in UNCERTAIN_CUES) and re.search(term_pat, sl):
        return "unknown"

    # 2) Scoped negation => absent
    for pref in NEGATION_SCOPED_PREFIXES:
        pat = rf"(?:\b{re.escape(pref)}\b)\s+[^.\n;:]{{0,200}}{term_pat}"
        if re.search(pat, sl):
            if "no change" in sl and re.search(term_pat, sl):
                continue
            return "absent"

    # 3) Present cues => present
    if any(cue in sl for cue in PRESENT_CUES) and re.search(term_pat, sl):
        return "present"

    # 4) Severity near term => present
    sev_pat = rf"\b({'|'.join(SEVERITY_CUES)})\b[^.\n;:]{{0,90}}{term_pat}"
    if re.search(sev_pat, sl):
        return "present"

    # 5) Bullet line implies present unless it looks like examples
    if s.lstrip().startswith(("*", "-", "•")) and re.search(term_pat, sl):
        if any(x in sl for x in EXAMPLE_CUES):
            return "unknown"
        return "present"

    return "unknown"

def _force_unknown_if_uncertain_sentence(sentence: str, current_status: str) -> str:
    """
    Critical fix:
    If ConText incorrectly marks something as NEGATED_EXISTENCE in a sentence like
      - 'cannot be ruled out'
      - 'not conclusive for'
      - 'concerning but not conclusive'
    then we force final status to UNKNOWN.

    We do NOT override true hard negation like 'no evidence of ...' / 'negative for ...'
    """
    s = (sentence or "").strip()
    sl = s.lower()
    if not sl:
        return current_status

    # If there's an explicit hard negation phrase, keep "absent" if we already have absent.
    if current_status == "absent":
        if any(hn in sl for hn in HARD_NEGATION_PHRASES):
            return current_status
        # If the sentence is an example list or differential list, treat as unknown
        if any(cue in sl for cue in UNCERTAIN_CUES):
            return "unknown"

    # If already present, don't downgrade.
    return current_status

def medspacy_status_from_span(span, sentence: str) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Pull ConText status from a target span.
    Then apply sentence-level override to fix mis-tags (not conclusive / cannot rule out).
    """
    mods_out: List[Dict[str, Any]] = []

    # Collect modifiers if available (version-dependent)
    try:
        mods = list(getattr(span._, "modifiers", []))
        for m in mods:
            mods_out.append({
                "term": getattr(m, "literal", None) or getattr(m, "term", None),
                "category": getattr(m, "category", None),
                "direction": getattr(m, "direction", None),
            })
    except Exception:
        pass

    def _get_flag(name: str) -> bool:
        try:
            return bool(getattr(span._, name))
        except Exception:
            return False

    is_neg = _get_flag("is_negated")
    is_unc = _get_flag("is_uncertain")
    is_hx  = _get_flag("is_historical")
    is_fam = _get_flag("is_family")

    if is_neg:
        status = "absent"
    elif is_unc or is_hx or is_fam:
        status = ("present" if POSSIBLE_COUNTS_AS_PRESENT else "unknown")
    else:
        status = "present"

    # 🔥 Fix: ConText sometimes treats "not conclusive for" as negation.
    status = _force_unknown_if_uncertain_sentence(sentence, status)

    return status, mods_out

def summarize_by_canonical(items: List[Dict]) -> List[Dict]:
    by_canon: Dict[str, List[str]] = {}
    for item in items:
        canon = item.get("canonical")
        status = item.get("status", "unknown")
        if not canon:
            continue
        by_canon.setdefault(canon, []).append(status)

    summary = []
    for canon, statuses in by_canon.items():
        if "present" in statuses:
            agg = "present"
        elif statuses and all(s == "absent" for s in statuses):
            agg = "absent"
        else:
            agg = "unknown"
        summary.append({"canonical": canon, "status": agg})
    return summary


# -----------------------------------------------------------------------------
# Extraction (MeSH phrase matcher + sentence-scoped classification, medspacy override)
# -----------------------------------------------------------------------------

def extract_conditions_and_findings(findings_text: str, conclusion_text: str):
    doc_find = nlp(findings_text) if findings_text else None
    doc_conc = nlp(conclusion_text) if conclusion_text else None

    conditions: List[Dict] = []
    findings: List[Dict] = []

    seen = set()

    def add_hit(kind: str, field: str, span_text: str, canonical: str,
                status: str, sentence: str, modifiers=None, source="mesh+spacy"):
        key = (kind, field, _norm(canonical), _norm(sentence))
        if key in seen:
            return
        seen.add(key)

        if is_generic_garbage(canonical, span_text):
            return

        d = {
            "text": span_text,
            "canonical": canonical,
            "status": status,
            "field": field,
            "sentence": sentence,
            "source": source,
        }
        if modifiers is not None:
            d["modifiers"] = modifiers

        (conditions if kind == "condition" else findings).append(d)

    # --- medspacy doc (combined, for ConText) ---
    med_doc = None
    med_index: Dict[Tuple[str, str], List[Any]] = {}

    if MEDSPACY_AVAILABLE and medspacy_nlp is not None:
        combined = " ".join([t for t in [findings_text, conclusion_text] if t]).strip()
        if combined:
            med_doc = medspacy_nlp(combined)

            spans: List[Any] = []

            try:
                spans.extend(list(med_doc.ents))
            except Exception:
                pass

            try:
                for _, group in med_doc.spans.items():
                    spans.extend(list(group))
            except Exception:
                pass

            for sp in spans:
                alias_l = _norm(getattr(sp, "text", ""))
                cat = (getattr(sp, "label_", "") or "").upper().strip()
                if alias_l and cat in ("COND", "FIND"):
                    med_index.setdefault((alias_l, cat), []).append(sp)

    def process_field(doc, field_name: str):
        if doc is None:
            return

        # CONDITIONS
        for _, start, end in cond_matcher(doc):
            span = doc[start:end]
            alias = _norm(span.text)
            canonical = cond_alias.get(alias, alias)
            sent = sentence_for_span(doc, span.start_char, span.end_char)

            status = classify_status_sentence_scoped(sent, span.text)
            modifiers = None
            source = "mesh+spacy"

            if MEDSPACY_AVAILABLE:
                cands = med_index.get((alias, "COND"), [])
                if cands:
                    status, modifiers = medspacy_status_from_span(cands[0], sent)
                    source = "mesh+medspacy"

            # Extra safety: even if status came from medspacy, enforce uncertainty override
            status = _force_unknown_if_uncertain_sentence(sent, status)

            add_hit("condition", field_name, span.text, canonical, status, sent, modifiers, source=source)

        # FINDINGS
        for _, start, end in find_matcher(doc):
            span = doc[start:end]
            alias = _norm(span.text)
            canonical = find_alias.get(alias, alias)
            sent = sentence_for_span(doc, span.start_char, span.end_char)

            status = classify_status_sentence_scoped(sent, span.text)
            modifiers = None
            source = "mesh+spacy"

            if MEDSPACY_AVAILABLE:
                cands = med_index.get((alias, "FIND"), [])
                if cands:
                    status, modifiers = medspacy_status_from_span(cands[0], sent)
                    source = "mesh+medspacy"

            status = _force_unknown_if_uncertain_sentence(sent, status)

            add_hit("finding", field_name, span.text, canonical, status, sent, modifiers, source=source)

    process_field(doc_find, "Findings")
    process_field(doc_conc, "Conclusion")

    return conditions, findings


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
>>>>>>> 3d05d01277cbc18d028bad6d78e2bda522ccc7c5

def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(2)

    input_csv = sys.argv[1]
    output_json = sys.argv[2]
    limit: Optional[int] = None
    if len(sys.argv) >= 4:
        try:
            limit = int(sys.argv[3])
        except Exception:
            limit = None

    terms = load_default_terms()
    alias_map = _build_alias_map(terms)

    rows = read_input_csv(input_csv)
    if limit is not None:
        rows = rows[:limit]

    results: List[Dict[str, Any]] = []
    for r in rows:
        rid = r.get("ID", "") or r.get("Id", "") or r.get("id", "")
        findings = _clean_text(r.get("Findings", ""))
        conclusion = _clean_text(r.get("Conclusion", ""))

<<<<<<< HEAD
        hits_findings = analyze_text_block(findings, alias_map, source="Findings")
        hits_conclusion = analyze_text_block(conclusion, alias_map, source="Conclusion")

        hits_all = hits_findings + hits_conclusion
        summary = _summarize_hits(hits_all)

        results.append(
            {
                "id": rid,
                "findings_text": findings,
                "conclusion_text": conclusion,
                "conditions": hits_all,
                "condition_summary": summary,
            }
        )

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(results)} reports -> {output_json}")
=======
    for _, row in df.iterrows():
        findings_text = str(row.get("Findings", "") or "")
        conclusion_text = str(row.get("Conclusion", "") or "")

        conditions, findings = extract_conditions_and_findings(findings_text, conclusion_text)

        results.append({
            "id": row.get("ID"),
            "conditions": conditions,
            "findings": findings,
            "condition_summary": summarize_by_canonical(conditions),
            "finding_summary": summarize_by_canonical(findings),
            "engine": "medspacy+spacy" if MEDSPACY_AVAILABLE else "spacy_fallback",
            "possible_counts_as_present": POSSIBLE_COUNTS_AS_PRESENT,
        })

    elapsed = time.perf_counter() - t0
    print(f"Processed {len(results)} reports in {elapsed:.2f} seconds")
    print(f"Engine: {'medspacy+spacy' if MEDSPACY_AVAILABLE else 'spacy_fallback'}")

    with output_json.open("w", encoding="utf-8") as out_f:
        json.dump(results, out_f, indent=2)

    print(f"Wrote {output_json}")
>>>>>>> 3d05d01277cbc18d028bad6d78e2bda522ccc7c5


if __name__ == "__main__":
    main()
