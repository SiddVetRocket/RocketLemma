# --- mesh_terms_extract.py ---

"""
Downloads MeSH XML and produces two files:

    out/conditions_mesh.txt
    out/findings_mesh.txt

Run:
    python mesh_terms_extract.py
"""

import os
import xml.etree.ElementTree as ET
from urllib.request import urlretrieve

MESH_URL = "https://nlmpubs.nlm.nih.gov/projects/mesh/MESH_FILES/xmlmesh/desc2025.xml"
OUT_DIR = "out"

COND_PATH = os.path.join(OUT_DIR, "conditions_mesh.txt")
FIND_PATH = os.path.join(OUT_DIR, "findings_mesh.txt")

DIAG_PREFIX = "C"          # Diseases
FIND_PREFIX = "C23.888"    # Symptoms and Signs subtree


def ensure_mesh_xml(path="desc2025.xml"):
    if not os.path.exists(path):
        print(f"Downloading MeSH XML from {MESH_URL} ...")
        urlretrieve(MESH_URL, path)
    return path


def parse_tree_numbers(descriptor):
    return [t.text for t in descriptor.findall("./TreeNumberList/TreeNumber")]


def is_diagnosis(tree_numbers):
    return any(tn.startswith(DIAG_PREFIX) for tn in tree_numbers)


def is_finding(tree_numbers):
    return any(tn.startswith(FIND_PREFIX) for tn in tree_numbers)


def get_preferred_term(descriptor):
    return descriptor.findtext("./DescriptorName/String")


def get_all_terms(descriptor):
    terms = []
    for term in descriptor.findall("./ConceptList/Concept/TermList/Term"):
        s = term.findtext("String")
        if s:
            terms.append(s.strip())

    seen = set()
    out = []
    for t in terms:
        k = t.lower()
        if k not in seen:
            out.append(t)
            seen.add(k)
    return out


def main(mesh_xml_path="desc2025.xml"):
    mesh_xml_path = ensure_mesh_xml(mesh_xml_path)
    os.makedirs(OUT_DIR, exist_ok=True)

    tree = ET.parse(mesh_xml_path)
    root = tree.getroot()

    cond_count = 0
    find_count = 0

    with open(COND_PATH, "w", encoding="utf-8") as cond_out, \
         open(FIND_PATH, "w", encoding="utf-8") as find_out:

        for desc in root.findall("./DescriptorRecord"):
            preferred = get_preferred_term(desc)
            if not preferred:
                continue

            tn_list = parse_tree_numbers(desc)
            diagnose = is_diagnosis(tn_list)
            finding = is_finding(tn_list)

            if not (diagnose or finding):
                continue

            all_terms = get_all_terms(desc)
            all_terms = [preferred] + [t for t in all_terms if t.lower() != preferred.lower()]
            line = " | ".join(all_terms) + "\n"

            if diagnose:
                cond_out.write(line)
                cond_count += 1

            if finding:
                find_out.write(line)
                find_count += 1

    print(f"Wrote {cond_count} diagnosis lines → {COND_PATH}")
    print(f"Wrote {find_count} finding lines → {FIND_PATH}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main()
