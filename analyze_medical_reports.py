# --- analyze_medical_reports.py ---
"""
Analyze medical reports and extract MeSH-based conditions and findings.

Usage:
    python analyze_medical_reports.py input.csv output.json [limit]

    input.csv   : CSV with at least ID, Findings, Conclusion columns
    output.json : JSON array of per-report results
    limit       : optional integer, number of rows to process (e.g. 200)

What it does:
- Loads MeSH-derived term lists (conditions + findings) and matches them in text.
- Prefers MedSpaCy (TargetMatcher + ConTextComponent) when installed.
- Falls back to spaCy PhraseMatcher with a sentence-scoped context classifier:
    * uncertainty checked first (e.g., "not conclusive" -> unknown)
    * negation is phrase-based + token-proximity (no blanket "not " negation)
    * Conclusion defaults to present if mentioned and not negated/uncertain
- Processes Findings and Conclusion separately; Conclusion has precedence in final summary.
- Filters overly-generic concepts like "disease".
- De-dupes repeated hits per (field, canonical, sentence), keeping strongest status.
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import spacy
from spacy.matcher import PhraseMatcher
from spacy.tokens import Span

from mesh_term_loader import load_mesh_term_file


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

MESH_COND_FILE = BASE_DIR / "out" / "conditions_mesh.txt"
MESH_FIND_FILE = BASE_DIR / "out" / "findings_mesh.txt"

USE_SYNONYMS = True

# Prefer MedSpaCy TargetMatcher + ConTextComponent if available
USE_MEDSPACY = True

SPACY_MODEL = "en_core_web_sm"

# If True, "possible/probable/suspected" counts as present; otherwise unknown.
POSSIBLE_COUNTS_AS_PRESENT = False


# -----------------------------------------------------------------------------
# Generic term filtering (big accuracy boost)
# -----------------------------------------------------------------------------

GENERIC_CANONICAL_BLACKLIST = {
    "disease", "disorder", "syndrome", "condition", "finding", "lesion", "abnormality",
    "infection", "inflammation", "cancer", "tumor", "neoplasm", "mass", "pain"
}

def is_generic_canonical(canon: str) -> bool:
    c = (canon or "").strip().lower()
    if not c:
        return True
    return c in GENERIC_CANONICAL_BLACKLIST


# -----------------------------------------------------------------------------
# Context cues (fallback mode uses these)
# -----------------------------------------------------------------------------

# 1) Strong negation phrases (sentence-scoped)
NEGATION_PHRASES = [
    "no evidence of",
    "no sign of",
    "negative for",
    "without evidence of",
    "free of",
    "ruled out",
    "rule out",
    "unlikely",
    "absence of",
    "not identified",
    "not seen",
]

# 2) Uncertainty phrases (checked BEFORE negation so "not conclusive" -> unknown)
# IMPORTANT: "not" is not blanket negation in radiology; it often means uncertainty.
UNCERTAIN_PHRASES = [
    "cannot exclude",
    "cannot rule out",
    "may represent",
    "could represent",
    "possible",
    "possibly",
    "probable",
    "suspected",
    "suspicious for",
    "question of",
    "concern for",
    "concerning for",
    "not conclusive",
    "not definitive",
    "not excluded",
    "not ruled out",
    "equivocal",
    "versus",
    "vs",
]

# 3) Positive cue phrases (fallback; Conclusion also has a default-present rule)
POSITIVE_PHRASES = [
    "consistent with",
    "compatible with",
    "suggestive of",
    "suggests",
    "evidence of",
    "demonstrates",
    "shows",
    "seen",
    "identified",
    "present",
]

# Token-based negation, scoped to a small window before the mention
NEGATION_TOKENS = {"no", "without", "denies"}


# -----------------------------------------------------------------------------
# spaCy setup
# -----------------------------------------------------------------------------

nlp = spacy.load(SPACY_MODEL)
# Ensure sentence boundaries even if parser isn't present
if "parser" not in nlp.pipe_names and "sentencizer" not in nlp.pipe_names:
    nlp.add_pipe("sentencizer")


# -----------------------------------------------------------------------------
# Optional MedSpaCy
# -----------------------------------------------------------------------------

MEDSPACY_AVAILABLE = False
_med_nlp = None
_target_matcher = None
_context = None

try:
    import medspacy
    from medspacy.target_matcher import TargetMatcher
    from medspacy.context import ConTextComponent
    MEDSPACY_AVAILABLE = True
except Exception:
    MEDSPACY_AVAILABLE = False


def _ensure_medspacy_pipeline():
    """Create medspacy pipeline once: TargetMatcher then ConText."""
    global _med_nlp, _target_matcher, _context
    if _med_nlp is not None:
        return

    _med_nlp = medspacy.load(model=SPACY_MODEL)

    _target_matcher = TargetMatcher(_med_nlp)
    _context = ConTextComponent(_med_nlp)

    # Insert target matcher early; context last
    if "target_matcher" not in _med_nlp.pipe_names:
        if "ner" in _med_nlp.pipe_names:
            _med_nlp.add_pipe(_target_matcher, name="target_matcher", before="ner")
        else:
            _med_nlp.add_pipe(_target_matcher, name="target_matcher", first=True)

    if "context" not in _med_nlp.pipe_names:
        _med_nlp.add_pipe(_context, name="context", last=True)


# -----------------------------------------------------------------------------
# Load MeSH terms
# -----------------------------------------------------------------------------

def _invert_alias_map(alias_to_canon: Dict[str, str]) -> Dict[str, List[str]]:
    canon_to_aliases: Dict[str, List[str]] = {}
    for alias, canon in alias_to_canon.items():
        a = (alias or "").strip()
        c = (canon or "").strip()
        if not a or not c:
            continue
        canon_to_aliases.setdefault(c, []).append(a)
    for c in list(canon_to_aliases.keys()):
        canon_to_aliases[c] = sorted(set(canon_to_aliases[c]))
    return canon_to_aliases


def load_mesh_terms():
    if not MESH_COND_FILE.exists() or not MESH_FIND_FILE.exists():
        raise FileNotFoundError(
            f"Missing MeSH term files.\nExpected:\n  {MESH_COND_FILE}\n  {MESH_FIND_FILE}\n"
            f"Run: python mesh_terms_extract.py"
        )

    cond_alias, _cond_syns = load_mesh_term_file(str(MESH_COND_FILE))
    find_alias, _find_syns = load_mesh_term_file(str(MESH_FIND_FILE))

    # Treat alias dicts as alias -> canonical maps for matching
    cond_canon_to_aliases = _invert_alias_map(cond_alias)
    find_canon_to_aliases = _invert_alias_map(find_alias)

    return cond_alias, find_alias, cond_canon_to_aliases, find_canon_to_aliases


cond_alias, find_alias, cond_canon_to_aliases, find_canon_to_aliases = load_mesh_terms()


# -----------------------------------------------------------------------------
# Fallback phrase matchers (when medspacy isn't available)
# -----------------------------------------------------------------------------

def build_phrase_matchers(use_synonyms: bool = True):
    cond_matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    find_matcher = PhraseMatcher(nlp.vocab, attr="LOWER")

    cond_terms = list(cond_alias.keys()) if use_synonyms else list(cond_canon_to_aliases.keys())
    find_terms = list(find_alias.keys()) if use_synonyms else list(find_canon_to_aliases.keys())

    cond_patterns = [nlp.make_doc(t) for t in cond_terms]
    find_patterns = [nlp.make_doc(t) for t in find_terms]

    if cond_patterns:
        cond_matcher.add("CONDITION", cond_patterns)
    if find_patterns:
        find_matcher.add("FINDING", find_patterns)

    return cond_matcher, find_matcher


cond_phrase_matcher, find_phrase_matcher = build_phrase_matchers(USE_SYNONYMS)


# -----------------------------------------------------------------------------
# Dedupe utilities
# -----------------------------------------------------------------------------

def dedupe_hits(hits: List[dict]) -> List[dict]:
    """
    Dedupe by (field, canonical, sentence), keeping strongest status:
      absent > present > unknown
    """
    rank = {"absent": 2, "present": 1, "unknown": 0}
    best: Dict[tuple, dict] = {}

    for h in hits:
        key = (h.get("field"), h.get("canonical"), (h.get("sentence") or "").strip())
        cur = best.get(key)
        if cur is None:
            best[key] = h
            continue
        if rank.get(h.get("status", "unknown"), 0) > rank.get(cur.get("status", "unknown"), 0):
            best[key] = h

    return list(best.values())


# -----------------------------------------------------------------------------
# Summary + precedence
# -----------------------------------------------------------------------------

def summarize_by_canonical(items: List[dict]) -> List[dict]:
    by_canon: Dict[str, List[str]] = {}
    for it in items:
        canon = it.get("canonical")
        st = it.get("status", "unknown")
        if canon:
            by_canon.setdefault(canon, []).append(st)

    out = []
    for canon, statuses in by_canon.items():
        if "present" in statuses:
            agg = "present"
        elif statuses and all(s == "absent" for s in statuses):
            agg = "absent"
        else:
            agg = "unknown"
        out.append({"canonical": canon, "status": agg})
    return out


def merge_summaries_conclusion_over_findings(conc: List[dict], find: List[dict]) -> List[dict]:
    """
    Conclusion dominates Findings:
      - If Conclusion says absent -> absent
      - If Conclusion says present -> present
      - Else fall back to Findings
    """
    f_map = {x["canonical"]: x["status"] for x in find}
    c_map = {x["canonical"]: x["status"] for x in conc}

    canons = sorted(set(f_map) | set(c_map))
    merged = []
    for canon in canons:
        c = c_map.get(canon)
        f = f_map.get(canon)

        if c == "absent":
            st = "absent"
        elif c == "present":
            st = "present"
        elif c == "unknown":
            st = f if f in ("present", "absent") else "unknown"
        else:
            st = f if f else "unknown"

        merged.append({"canonical": canon, "status": st})
    return merged


# -----------------------------------------------------------------------------
# Status classification
# -----------------------------------------------------------------------------

def spacy_status_from_sentence(span: Span, field: str) -> str:
    """
    Improved fallback classifier (Fixes your two cases):
      1) uncertainty FIRST: "not conclusive for X" => unknown (not absent)
      2) negation phrases + token-proximity negation near the hit
      3) positive cues => present
      4) if field is Conclusion and not negated/uncertain => present (major accuracy bump)
      5) default => unknown
    """
    sent = span.sent if hasattr(span, "sent") else span.doc[:]
    s = sent.text.lower()

    # 1) Uncertainty first (so "not conclusive" doesn't get misread as negation)
    for p in UNCERTAIN_PHRASES:
        if p in s:
            return "unknown"

    # 2) Strong negation phrases
    for p in NEGATION_PHRASES:
        if p in s:
            return "absent"

    # 3) Token-proximity negation: check a few tokens before the span in the same sentence
    doc = span.doc
    left_i = max(sent.start, span.start - 5)
    left_ctx = doc[left_i:span.start]
    if any(t.lower_ in NEGATION_TOKENS for t in left_ctx):
        return "absent"

    # 4) Positive cues
    for p in POSITIVE_PHRASES:
        if p in s:
            return "present"

    # 5) KEY RULE: Conclusion mention defaults to present if not negated/uncertain
    if (field or "").strip().lower() == "conclusion":
        return "present"

    return "unknown"


def status_from_modifiers(mods) -> str:
    """
    Convert MedSpaCy ConText modifiers -> present/absent/unknown.
    """
    cats = {getattr(m, "category", "").lower() for m in (mods or [])}

    if any(c in cats for c in ("negation", "negated")):
        return "absent"

    if any(c in cats for c in ("uncertainty", "uncertain", "possible", "probable", "hypothetical")):
        return "present" if POSSIBLE_COUNTS_AS_PRESENT else "unknown"

    if any(c in cats for c in ("historical", "history", "temporality", "experiencer")):
        return "unknown"

    # Conservative default (you can change this later)
    return "unknown"


# -----------------------------------------------------------------------------
# Extraction (MedSpaCy preferred, fallback otherwise)
# -----------------------------------------------------------------------------

def _add_mesh_targets_to_medspacy():
    """
    Adds MeSH-based targets into TargetMatcher once.

    Label encoding:
      COND__<canonical>
      FIND__<canonical>
    """
    if not (USE_MEDSPACY and MEDSPACY_AVAILABLE):
        return

    _ensure_medspacy_pipeline()

    if getattr(_add_mesh_targets_to_medspacy, "_done", False):
        return

    for canon, aliases in cond_canon_to_aliases.items():
        if not canon or not aliases or is_generic_canonical(canon):
            continue
        _target_matcher.add(f"COND__{canon}", aliases)

    for canon, aliases in find_canon_to_aliases.items():
        if not canon or not aliases or is_generic_canonical(canon):
            continue
        _target_matcher.add(f"FIND__{canon}", aliases)

    _add_mesh_targets_to_medspacy._done = True  # type: ignore


def extract_with_medspacy(text: str, field: str) -> Tuple[List[dict], List[dict]]:
    _add_mesh_targets_to_medspacy()

    doc = _med_nlp(text or "")

    conditions: List[dict] = []
    findings: List[dict] = []

    for t in getattr(doc._, "targets", []) or []:
        label = getattr(t, "label_", "") or ""
        mods = getattr(t._, "modifiers", []) or []
        status = status_from_modifiers(mods)

        if label.startswith("COND__"):
            canonical = label[len("COND__"):]
            if is_generic_canonical(canonical):
                continue
            conditions.append({
                "text": t.text,
                "canonical": canonical,
                "start": t.start_char,
                "end": t.end_char,
                "status": status,
                "source": "mesh+medspacy",
                "field": field,
                "sentence": t.sent.text if hasattr(t, "sent") else doc.text,
                "modifiers": [{"term": getattr(m, "term", ""), "category": getattr(m, "category", "")} for m in mods],
            })
        elif label.startswith("FIND__"):
            canonical = label[len("FIND__"):]
            if is_generic_canonical(canonical):
                continue
            findings.append({
                "text": t.text,
                "canonical": canonical,
                "start": t.start_char,
                "end": t.end_char,
                "status": status,
                "source": "mesh+medspacy",
                "field": field,
                "sentence": t.sent.text if hasattr(t, "sent") else doc.text,
                "modifiers": [{"term": getattr(m, "term", ""), "category": getattr(m, "category", "")} for m in mods],
            })

    return dedupe_hits(conditions), dedupe_hits(findings)


def extract_with_spacy_phrasematcher(text: str, field: str) -> Tuple[List[dict], List[dict]]:
    doc = nlp(text or "")
    conditions: List[dict] = []
    findings: List[dict] = []

    seen_cond = set()
    seen_find = set()

    # CONDITIONS
    for _, start, end in cond_phrase_matcher(doc):
        span = doc[start:end]
        alias = span.text.lower()
        canonical = cond_alias.get(alias, alias)

        if is_generic_canonical(canonical):
            continue

        key = (span.start_char, span.end_char, canonical, field)
        if key in seen_cond:
            continue
        seen_cond.add(key)

        status = spacy_status_from_sentence(span, field)

        conditions.append({
            "text": span.text,
            "canonical": canonical,
            "start": span.start_char,
            "end": span.end_char,
            "status": status,
            "source": "mesh+spacy",
            "field": field,
            "sentence": span.sent.text if hasattr(span, "sent") else doc.text,
            "modifiers": [],
        })

    # FINDINGS
    for _, start, end in find_phrase_matcher(doc):
        span = doc[start:end]
        alias = span.text.lower()
        canonical = find_alias.get(alias, alias)

        if is_generic_canonical(canonical):
            continue

        key = (span.start_char, span.end_char, canonical, field)
        if key in seen_find:
            continue
        seen_find.add(key)

        status = spacy_status_from_sentence(span, field)

        findings.append({
            "text": span.text,
            "canonical": canonical,
            "start": span.start_char,
            "end": span.end_char,
            "status": status,
            "source": "mesh+spacy",
            "field": field,
            "sentence": span.sent.text if hasattr(span, "sent") else doc.text,
            "modifiers": [],
        })

    return dedupe_hits(conditions), dedupe_hits(findings)


def extract_from_field(text: str, field: str) -> Tuple[List[dict], List[dict]]:
    if USE_MEDSPACY and MEDSPACY_AVAILABLE:
        _ensure_medspacy_pipeline()
        return extract_with_medspacy(text, field)
    return extract_with_spacy_phrasematcher(text, field)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    if len(sys.argv) < 3:
        print("Usage: python analyze_medical_reports.py input.csv output.json [limit]")
        sys.exit(1)

    input_csv = Path(sys.argv[1])
    output_json = Path(sys.argv[2])
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else None

    if not input_csv.exists():
        print(f"Input CSV not found: {input_csv}")
        sys.exit(1)

    df = pd.read_csv(input_csv)
    if limit is not None:
        df = df.head(limit)

    results = []
    t0 = time.perf_counter()

    for _, row in df.iterrows():
        rid = row.get("ID")
        findings_text = str(row.get("Findings", "") or "")
        conclusion_text = str(row.get("Conclusion", "") or "")

        # Extract separately
        cond_f, find_f = extract_from_field(findings_text, "Findings")
        cond_c, find_c = extract_from_field(conclusion_text, "Conclusion")

        # Combine and dedupe across fields
        all_conditions = dedupe_hits(cond_f + cond_c)
        all_findings = dedupe_hits(find_f + find_c)

        # Summaries per field
        cond_sum_f = summarize_by_canonical(cond_f)
        cond_sum_c = summarize_by_canonical(cond_c)
        find_sum_f = summarize_by_canonical(find_f)
        find_sum_c = summarize_by_canonical(find_c)

        # Merged summaries (Conclusion > Findings)
        cond_summary = merge_summaries_conclusion_over_findings(cond_sum_c, cond_sum_f)
        finding_summary = merge_summaries_conclusion_over_findings(find_sum_c, find_sum_f)

        results.append({
            "id": rid,

            # raw hits (debuggable)
            "conditions": all_conditions,
            "findings": all_findings,

            # per-field summaries (debuggable)
            "condition_summary_findings": cond_sum_f,
            "condition_summary_conclusion": cond_sum_c,
            "finding_summary_findings": find_sum_f,
            "finding_summary_conclusion": find_sum_c,

            # merged (consumable)
            "condition_summary": cond_summary,
            "finding_summary": finding_summary,

            # metadata
            "engine": "medspacy" if (USE_MEDSPACY and MEDSPACY_AVAILABLE) else "spacy_fallback",
            "possible_counts_as_present": POSSIBLE_COUNTS_AS_PRESENT,
        })

    elapsed = time.perf_counter() - t0
    print(f"Processed {len(results)} reports in {elapsed:.2f} seconds")

    with output_json.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Wrote {output_json}")


if __name__ == "__main__":
    main()
