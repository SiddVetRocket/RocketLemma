# --- analyze_medical_reports.py ---

import sys
import json
import time
import pandas as pd
import spacy
from spacy.matcher import PhraseMatcher

from mesh_term_loader import load_mesh_term_file

###############################################################################
# Settings
###############################################################################

MESH_COND_FILE = "out/conditions_mesh.txt"
MESH_FIND_FILE = "out/findings_mesh.txt"

USE_SYNONYMS = True   # toggle for benchmarks

nlp = spacy.load("en_core_web_sm")


###############################################################################
# Load MeSH Term Files
###############################################################################

def build_matchers(use_synonyms: bool = True):
    cond_alias, cond_syns = load_mesh_term_file(MESH_COND_FILE)
    find_alias, find_syns = load_mesh_term_file(MESH_FIND_FILE)

    cond_matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    find_matcher = PhraseMatcher(nlp.vocab, attr="LOWER")

    if use_synonyms:
        cond_terms = list(cond_alias.keys())
        find_terms = list(find_alias.keys())
    else:
        cond_terms = list(cond_syns.keys())
        find_terms = list(find_syns.keys())

    cond_matcher.add("COND", [nlp.make_doc(t) for t in cond_terms])
    find_matcher.add("FIND", [nlp.make_doc(t) for t in find_terms])

    return cond_matcher, find_matcher, cond_alias, find_alias


cond_matcher, find_matcher, cond_alias, find_alias = build_matchers(USE_SYNONYMS)


###############################################################################
# Extraction
###############################################################################

def extract(text: str):
    doc = nlp(text)

    conditions = []
    findings   = []

    seen_c = set()
    seen_f = set()

    # CONDITIONS
    for _, start, end in cond_matcher(doc):
        span = doc[start:end]
        alias = span.text.lower()
        canonical = cond_alias.get(alias, alias)

        key = (span.start_char, span.end_char, canonical)
        if key in seen_c:
            continue
        seen_c.add(key)

        conditions.append({
            "text": span.text,
            "canonical": canonical,
            "start": span.start_char,
            "end": span.end_char,
            "source": "mesh"
        })

    # FINDINGS
    for _, start, end in find_matcher(doc):
        span = doc[start:end]
        alias = span.text.lower()
        canonical = find_alias.get(alias, alias)

        key = (span.start_char, span.end_char, canonical)
        if key in seen_f:
            continue
        seen_f.add(key)

        findings.append({
            "text": span.text,
            "canonical": canonical,
            "start": span.start_char,
            "end": span.end_char,
            "source": "mesh"
        })

    return conditions, findings


###############################################################################
# Main CSV -> JSON Export
###############################################################################

def main():
    if len(sys.argv) < 3:
        print("Usage: python analyze_medical_reports.py input.csv output.json [limit]")
        sys.exit(1)

    input_csv  = sys.argv[1]
    output_json = sys.argv[2]
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else None

    df = pd.read_csv(input_csv)
    if limit:
        df = df.head(limit)

    results = []
    t0 = time.perf_counter()

    for idx, row in df.iterrows():
        text = " ".join([
            str(row.get("Findings", "")),
            str(row.get("Conclusion", "")),
        ])

        conditions, findings = extract(text)

        results.append({
            "id": row.get("ID"),
            "conditions": conditions,
            "findings": findings,
        })

    elapsed = time.perf_counter() - t0
    print(f"Processed {len(results)} reports in {elapsed:.2f} seconds")

    with open(output_json, "w", encoding="utf-8") as out:
        json.dump(results, out, indent=2)

    print(f"Wrote {output_json}")


if __name__ == "__main__":
    main()
