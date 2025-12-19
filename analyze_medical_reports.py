# --- analyze_medical_reports.py ---
"""
Analyze medical reports and extract MeSH-based conditions and findings.

Usage:
    python analyze_medical_reports.py input.csv output.json [limit]

    input.csv   : CSV with at least ID, Findings, Conclusion columns
    output.json : JSON array of per-report results
    limit       : optional integer, number of rows to process (e.g. 200)

Behavior:
- Uses MeSH-derived term lists (conditions + findings) via PhraseMatcher.
- If MedSpaCy is installed: TargetMatcher + ConTextComponent overrides heuristics.
- Otherwise: sentence-scoped regex heuristics to assign present/absent/unknown.
- Aggressive filtering of ultra-generic junk "conditions" (disease, abnormality, etc.).
"""

import sys
import json
import time
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd
import spacy
from spacy.matcher import PhraseMatcher

from mesh_term_loader import load_mesh_term_file


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


# -----------------------------------------------------------------------------
# Expanded generic stoplist
# -----------------------------------------------------------------------------
# Goal: remove high-frequency, low-information "conditions" that inflate hits:
#   - disease / disorder / condition / abnormality / pathology / changes / process
# Keep it conservative: we mostly drop these when they are single-token, generic,
# or very short phrases.
GENERIC_CANONICAL_STOP = {
    # ultra-generic medical nouns
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
    # vague body/clinical terms
    "pain",
    "swelling",
    "enlargement",
}

# surface-text stopwords: if a matched span itself is junky
GENERIC_SPAN_STOP = {
    "disease", "disorder", "condition", "abnormality", "pathology",
    "process", "finding", "findings", "changes", "change", "appearance",
    "technique", "artifact",
}

# Additional “junk patterns” often appearing as canonicals or spans.
# These aren’t always wrong, but they are usually too broad to be useful.
GENERIC_REGEX_STOP = [
    r"^airway disease$",           # too broad, often used as a bucket
    r"^lower airway disease$",
    r"^upper airway disease$",
    r"^pulmonary pathology$",
    r"^lung disease$",
    r"^cardiac disease$",
    r"^abdominal disease$",
]

def _norm(s: str) -> str:
    return (s or "").strip().lower()

def is_generic_garbage(canonical: str, span_text: str) -> bool:
    c = _norm(canonical)
    t = _norm(span_text)

    if not c:
        return True

    # 1) Exact canonical stop
    if c in GENERIC_CANONICAL_STOP:
        # If the *span* is multiword and looks specific, keep it.
        # Example: "pulmonary edema" is specific; "edema" is not.
        if len(t.split()) <= 1:
            return True
        # If canonical itself is short and generic phrase, drop
        if len(c.split()) <= 2 and c in GENERIC_CANONICAL_STOP:
            return True

    # 2) Exact span stop (single-word junk)
    if t in GENERIC_SPAN_STOP:
        return True

    # 3) Regex stop for common overly broad phrases
    for pat in GENERIC_REGEX_STOP:
        if re.match(pat, c):
            return True

    # 4) Very short “catch-all” phrases
    if len(c) <= 4 and c in {"mass", "pain"}:
        return True

    return False


# -----------------------------------------------------------------------------
# Cue lists (sentence-scoped)
# -----------------------------------------------------------------------------

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
    # IMPORTANT: “rule out” = uncertainty in clinical writing
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

    cond_terms = list(cond_alias.keys()) if use_synonyms else list(cond_syns.keys())
    find_terms = list(find_alias.keys()) if use_synonyms else list(find_syns.keys())

    if cond_terms:
        cond_matcher.add("CONDITION", [nlp.make_doc(t) for t in cond_terms])
    if find_terms:
        find_matcher.add("FINDING", [nlp.make_doc(t) for t in find_terms])

    return cond_matcher, find_matcher, cond_alias, find_alias


cond_matcher, find_matcher, cond_alias, find_alias = build_matchers(USE_SYNONYMS)


# -----------------------------------------------------------------------------
# Optional: MedSpaCy integration (TargetMatcher + ConTextComponent)
# -----------------------------------------------------------------------------

MEDSPACY_AVAILABLE = False
medspacy_nlp = None
medspacy_target_matcher = None

def try_setup_medspacy():
    global MEDSPACY_AVAILABLE, medspacy_nlp, medspacy_target_matcher
    if not USE_MEDSPACY_IF_AVAILABLE:
        return

    try:
        import medspacy
        from medspacy.target_matcher import TargetMatcher
        from medspacy.context import ConTextComponent

        medspacy_nlp = medspacy.load()
        medspacy_target_matcher = TargetMatcher(medspacy_nlp)

        # Add MeSH alias terms as targets, but we still map alias->canonical ourselves.
        cond_terms = list(cond_alias.keys())
        find_terms = list(find_alias.keys())

        if cond_terms:
            medspacy_target_matcher.add("COND", cond_terms)
        if find_terms:
            medspacy_target_matcher.add("FIND", find_terms)

        medspacy_nlp.add_pipe(medspacy_target_matcher, name="target_matcher", before="ner")

        context = ConTextComponent(medspacy_nlp)
        medspacy_nlp.add_pipe(context, name="context")

        MEDSPACY_AVAILABLE = True
    except Exception:
        MEDSPACY_AVAILABLE = False

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
    # (prevents "not conclusive" from being mistaken as absent)
    if any(cue in sl for cue in UNCERTAIN_CUES) and re.search(term_pat, sl):
        # Treat plain "unlikely X" as unknown rather than absent for safety
        return "unknown"

    # 2) Hit-scoped negation (absent)
    # Covers: "No ... pneumonia", "without ... pneumonia", "negative for pneumonia"
    for pref in NEGATION_SCOPED_PREFIXES:
        pat = rf"(?:\b{re.escape(pref)}\b)\s+[^.\n;:]{{0,200}}{term_pat}"
        if re.search(pat, sl):
            # avoid "no change in pneumonia" edge cases
            if "no change" in sl and re.search(term_pat, sl):
                continue
            return "absent"

    # 3) Strong present cues
    if any(cue in sl for cue in PRESENT_CUES) and re.search(term_pat, sl):
        # guard "no evidence of" already handled above
        return "present"

    # 4) Severity near term -> present (fixes “progressive moderate to severe pneumothorax…”)
    if re.search(rf"\b({'|'.join(SEVERITY_CUES)})\b[^.\n;:]{{0,90}}{term_pat}", sl):
        return "present"

    # 5) Bullet line in conclusion often indicates asserted findings
    if s.lstrip().startswith(("*", "-", "•")) and re.search(term_pat, sl):
        # But if it's an example list, keep unknown
        if any(x in sl for x in EXAMPLE_CUES):
            return "unknown"
        return "present"

    # Default
    return "unknown"

def medspacy_status_from_target(target) -> Tuple[str, List[Dict]]:
    mods_out = []
    try:
        mods = list(getattr(target._, "modifiers", []))
    except Exception:
        mods = []

    cats = []
    for m in mods:
        cat = (getattr(m, "category", "") or "").lower()
        term = (getattr(m, "term", "") or "")
        direction = (getattr(m, "direction", "") or "")
        cats.append(cat)
        mods_out.append({"term": term, "category": cat, "direction": direction})

    # Negation => absent
    if any("neg" in c for c in cats):
        return "absent", mods_out

    # Uncertainty/possible/hypothetical/historical/family => unknown
    if any(any(x in c for x in ("uncertain", "possible", "probable", "hypothetical", "historical", "family", "tempor")) for c in cats):
        return ("present" if POSSIBLE_COUNTS_AS_PRESENT else "unknown"), mods_out

    # No modifiers / no relevant modifiers => present
    return "present", mods_out

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

    def add_hit(kind: str, field: str, span_text: str, canonical: str, status: str, sentence: str, modifiers=None, source="mesh+spacy"):
        # de-dupe at sentence-level per canonical per field
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

    # --- medspacy index (override) ---
    med_doc = None
    med_targets = []
    med_index = {}

    if MEDSPACY_AVAILABLE and medspacy_nlp is not None:
        combined = " ".join([t for t in [findings_text, conclusion_text] if t]).strip()
        if combined:
            med_doc = medspacy_nlp(combined)
            try:
                med_targets = list(getattr(med_doc._, "targets", []))
            except Exception:
                med_targets = []

        for t in med_targets:
            alias_l = _norm(getattr(t, "text", ""))
            lbl = _norm(getattr(t, "label_", "")).upper()
            if alias_l and lbl in ("COND", "FIND"):
                med_index.setdefault((alias_l, lbl), []).append(t)

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

            # Override with medspacy ConText if available
            if MEDSPACY_AVAILABLE:
                cands = med_index.get((alias, "COND"), [])
                if cands:
                    status, modifiers = medspacy_status_from_target(cands[0])
                    source = "mesh+medspacy"

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
                    status, modifiers = medspacy_status_from_target(cands[0])
                    source = "mesh+medspacy"

            add_hit("finding", field_name, span.text, canonical, status, sent, modifiers, source=source)

    process_field(doc_find, "Findings")
    process_field(doc_conc, "Conclusion")

    return conditions, findings


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


if __name__ == "__main__":
    main()
