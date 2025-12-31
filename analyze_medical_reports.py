# --- analyze_medical_reports.py ---
"""
Analyze medical reports (Findings/Conclusions) and extract conditions + statuses.

Usage:
    python analyze_medical_reports.py input.csv output.json [limit]

    input.csv should contain columns:
      - ID
      - Findings
      - Conclusion

Outputs a JSON array (one object per report) with:
  - conditions: raw condition hits (per-sentence) with status + reason
  - condition_summary: deduped/aggregated per canonical condition with final status + evidence
  - findings: raw non-condition findings (if enabled) similarly structured
  - finding_summary: deduped/aggregated per canonical finding

This version adds:
  - explicit "uncertain" status (cannot be excluded, suspicious for, differential includes, etc.)
  - safer aggregation: if both present and absent found -> uncertain
  - evidence snippets in the summary for faster manual QA
"""

from __future__ import annotations

import csv
import json
import re
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# -----------------------------
# Config / Heuristics
# -----------------------------

# Sentiment/status labels we use everywhere
PRESENT = "present"
ABSENT = "absent"
UNCERTAINE = "uncertain"  # intentionally spelled unique to avoid shadowing "uncertain" var name
UNKNOWN = "unknown"

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


if __name__ == "__main__":
    main()
