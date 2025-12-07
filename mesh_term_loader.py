# --- mesh_term_loader.py ---

"""
Loads a MeSH term text file like:
    Preferred | Syn1 | Syn2 | ...

Provides:
    alias_to_canonical  ("viral pneumonia" -> "pneumonia")
    canonical_to_syns  ("pneumonia" -> ["pneumonia", "Viral Pneumonia", ...])
"""

from pathlib import Path
from typing import Dict, List, Tuple


def load_mesh_term_file(path: str) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    alias_to_canonical: Dict[str, str] = {}
    canonical_to_syns: Dict[str, List[str]] = {}

    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"MeSH term file not found: {path}")

    with path_obj.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = [p.strip() for p in line.split("|") if p.strip()]
            if not parts:
                continue

            preferred = parts[0]
            canonical = preferred.lower()
            synonyms = parts

            canonical_to_syns.setdefault(canonical, [])

            for syn in synonyms:
                alias = syn.lower()
                alias_to_canonical[alias] = canonical

                if syn not in canonical_to_syns[canonical]:
                    canonical_to_syns[canonical].append(syn)

    return alias_to_canonical, canonical_to_syns
