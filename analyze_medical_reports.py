# --- analyze_medical_reports.py ---
"""
Analyze medical reports and extract MeSH-based conditions and findings.

Usage:
    python analyze_medical_reports.py input.csv output.json [limit]

    input.csv   : CSV with at least ID, Findings, Conclusion columns
    output.json : JSON array of per-report results
    limit       : optional integer, number of rows to process (e.g. 200)
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import spacy
from spacy.matcher import PhraseMatcher
from spacy.tokens import Doc, Span

from mesh_term_loader import load_mesh_term_file


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

MESH_COND_FILE = BASE_DIR / "out" / "conditions_mesh.txt"
MESH_FIND_FILE = BASE_DIR / "out" / "findings_mesh.txt"

USE_SYNONYMS = True

nlp = spacy.load("en_core_web_sm")


# -----------------------------------------------------------------------------
# Context cues (dataset-informed)
# -----------------------------------------------------------------------------

# Token-based negation is safer than substring matching.
NEGATION_TOKENS = {"no", "not", "without", "denies", "neither", "nor"}

# Common “negated-by-phrase” patterns seen in your dataset
NEGATION_PHRASES = [
    "no evidence of",
    "no sign of",
    "negative for",
    "without evidence of",
    "absence of",
    "rule out",
    "ruled out",
    "unlikely",
    "not identified",
    "not seen",
    "no focal consolidation",  # common radiology-ish phrasing
]

# Uncertainty / hedging (very common in Conclusion in your dataset)
UNCERTAIN_PHRASES = [
    "may represent",
    "could represent",
    "possible",
    "possibly",
    "probable",
    "suspicious for",
    "cannot rule out",
    "cannot exclude",
    "question of",
    "concern for",
    "likely",
    "versus",
    "vs",
]

# Positive evidence cues (keep conservative)
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

# “Global normal” language: common and important, but not tied to a specific condition span.
# We use this only to bias toward unknown/absent when nothing else is found.
GLOBAL_NORMAL_PHRASES = [
    "unremarkable",
    "within normal limits",
    "no significant abnormality",
    "normal study",
]


# -----------------------------------------------------------------------------
# Build MeSH-based matchers
# -----------------------------------------------------------------------------

def build_matchers(use_synonyms: bool = True):
    if not MESH_COND_FILE.exists() or not MESH_FIND_FILE.exists():
        raise FileNotFoundError(
            f"Missing MeSH term files.\n"
            f"Expected:\n  {MESH_COND_FILE}\n  {MESH_FIND_FILE}\n"
            f"Run: python mesh_terms_extract.py"
        )

    cond_alias, cond_syns = load_mesh_term_file(str(MESH_COND_FILE))
    find_alias, find_syns = load_mesh_term_file(str(MESH_FIND_FILE))

    cond_matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    find_matcher = PhraseMatcher(nlp.vocab, attr="LOWER")

    if use_synonyms:
        cond_terms = list(cond_alias.keys())
        find_terms = list(find_alias.keys())
    else:
        cond_terms = list(cond_syns.keys())
        find_terms = list(find_syns.keys())

    cond_patterns = [nlp.make_doc(t) for t in cond_terms]
    find_patterns = [nlp.make_doc(t) for t in find_terms]

    if cond_patterns:
        cond_matcher.add("CONDITION", cond_patterns)
    if find_patterns:
        find_matcher.add("FINDING", find_patterns)

    return cond_matcher, find_matcher, cond_alias, find_alias


cond_matcher, find_matcher, cond_alias, find_alias = build_matchers(USE_SYNONYMS)


# -----------------------------------------------------------------------------
# Status classification (sentence-scoped, token-aware)
# -----------------------------------------------------------------------------

def _lower(s: str) -> str:
    return (s or "").lower()

def _sentence_for_span(doc: Doc, span: Span) -> Span:
    # If sentencizer is available, use it; otherwise fallback to whole doc
    try:
        return span.sent
    except Exception:
        return doc[:]

def classify_status_span(doc: Doc, span: Span) -> str:
    """
    Conservative classifier:
      - Check within the span's sentence (not a huge character window).
      - Token-aware “no/not/without” handling near the span.
      - Phrase cues for negation/uncertainty/positive.
      - Default is UNKNOWN.
    """
    sent = _sentence_for_span(doc, span)
    sent_text = sent.text
    sent_lower = sent_text.lower()

    # 1) Strong negation phrases in the same sentence
    for p in NEGATION_PHRASES:
        if p in sent_lower:
            # If the sentence contains the matched concept and a negation phrase,
            # treat the hit as absent. This is strict on purpose.
            return "absent"

    # 2) Token proximity negation: “no <X>” “not <X>” “without <X>”
    # Check up to 5 tokens before the span start inside the same sentence.
    # This is the big fix vs substring "no ".
    left_i = max(sent.start, span.start - 5)
    left_ctx = doc[left_i:span.start]
    if any(t.lower_ in NEGATION_TOKENS for t in left_ctx):
        return "absent"

    # 3) Uncertainty cues in same sentence
    for p in UNCERTAIN_PHRASES:
        if p in sent_lower:
            return "unknown"

    # 4) Positive cues in same sentence
    for p in POSITIVE_PHRASES:
        if p in sent_lower:
            return "present"

    # 5) Default
    return "unknown"


# -----------------------------------------------------------------------------
# Summaries / merging with precedence
# -----------------------------------------------------------------------------

def summarize_by_canonical(items: List[dict]) -> List[dict]:
    """
    Aggregation rule:
        - If ANY hit is 'present'  -> overall 'present'
        - ELSE if ALL are 'absent' -> overall 'absent'
        - ELSE                     -> 'unknown'
    """
    by_canon: Dict[str, List[str]] = {}
    for item in items:
        canon = item.get("canonical")
        status = item.get("status", "unknown")
        if canon:
            by_canon.setdefault(canon, []).append(status)

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


def merge_summaries_with_precedence(
    conclusion_summary: List[dict],
    findings_summary: List[dict],
) -> List[dict]:
    """
    Precedence tuned for your dataset:
      - Conclusion dominates Findings
      - If Conclusion says absent -> absent
      - If Conclusion says present -> present
      - Else fall back to Findings
    """
    f_map = {x["canonical"]: x["status"] for x in findings_summary}
    c_map = {x["canonical"]: x["status"] for x in conclusion_summary}

    all_canons = set(f_map) | set(c_map)
    merged = []

    for canon in sorted(all_canons):
        c = c_map.get(canon)
        f = f_map.get(canon)

        if c == "absent":
            merged_status = "absent"
        elif c == "present":
            merged_status = "present"
        elif c == "unknown":
            # Let Findings decide if it is stronger
            merged_status = f if f in ("present", "absent") else "unknown"
        else:
            merged_status = f if f else "unknown"

        merged.append({"canonical": canon, "status": merged_status})

    return merged


# -----------------------------------------------------------------------------
# Extraction
# -----------------------------------------------------------------------------

def extract_from_text(text: str, field: str):
    """
    Run spaCy + matchers on one field (Findings or Conclusion).
    Returns raw hits for conditions & findings.
    """
    doc = nlp(text or "")
    conditions = []
    findings = []

    seen_cond = set()
    seen_find = set()

    # CONDITIONS
    for _, start, end in cond_matcher(doc):
        span = doc[start:end]
        alias = span.text.lower()
        canonical = cond_alias.get(alias, alias)

        key = (span.start_char, span.end_char, canonical, field)
        if key in seen_cond:
            continue
        seen_cond.add(key)

        status = classify_status_span(doc, span)

        conditions.append({
            "text": span.text,
            "canonical": canonical,
            "start": span.start_char,
            "end": span.end_char,
            "status": status,
            "source": "mesh",
            "field": field,
            "sentence": span.sent.text if hasattr(span, "sent") else doc.text,
        })

    # FINDINGS
    for _, start, end in find_matcher(doc):
        span = doc[start:end]
        alias = span.text.lower()
        canonical = find_alias.get(alias, alias)

        key = (span.start_char, span.end_char, canonical, field)
        if key in seen_find:
            continue
        seen_find.add(key)

        status = classify_status_span(doc, span)

        findings.append({
            "text": span.text,
            "canonical": canonical,
            "start": span.start_char,
            "end": span.end_char,
            "status": status,
            "source": "mesh",
            "field": field,
            "sentence": span.sent.text if hasattr(span, "sent") else doc.text,
        })

    return conditions, findings


# -----------------------------------------------------------------------------
# Main: CSV -> JSON
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
        cond_f, find_f = extract_from_text(findings_text, field="Findings")
        cond_c, find_c = extract_from_text(conclusion_text, field="Conclusion")

        # Summaries per field
        cond_sum_f = summarize_by_canonical(cond_f)
        cond_sum_c = summarize_by_canonical(cond_c)
        find_sum_f = summarize_by_canonical(find_f)
        find_sum_c = summarize_by_canonical(find_c)

        # Merged summaries with precedence: Conclusion > Findings
        cond_summary = merge_summaries_with_precedence(cond_sum_c, cond_sum_f)
        finding_summary = merge_summaries_with_precedence(find_sum_c, find_sum_f)

        # Optional: tag “globally normal” if nothing was found
        full_text_lower = (findings_text + "\n" + conclusion_text).lower()
        globally_normal = any(p in full_text_lower for p in GLOBAL_NORMAL_PHRASES)

        result = {
            "id": rid,

            # raw hits
            "conditions": cond_f + cond_c,
            "findings": find_f + find_c,

            # per-field summaries (useful for debugging)
            "condition_summary_findings": cond_sum_f,
            "condition_summary_conclusion": cond_sum_c,
            "finding_summary_findings": find_sum_f,
            "finding_summary_conclusion": find_sum_c,

            # merged (what you usually want)
            "condition_summary": cond_summary,
            "finding_summary": finding_summary,

            "globally_normal_language": globally_normal,
        }

        results.append(result)

    elapsed = time.perf_counter() - t0
    print(f"Processed {len(results)} reports in {elapsed:.2f} seconds")

    with output_json.open("w", encoding="utf-8") as out_f:
        json.dump(results, out_f, indent=2)

    print(f"Wrote {output_json}")


if __name__ == "__main__":
    main()
