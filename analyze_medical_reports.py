"""
Medical Report Analysis using MedSpacy

This script processes medical reports (Findings and Conclusions) from either:
  • a CSV file, or
  • a directory of .txt files

It extracts medical conditions and classifies them as true / false / indeterminate
using medspaCy's context detection.

Author: MedSpacy1 Project (updated to support folder input)
Date: 2025-10-14
"""

import sys
from pathlib import Path
from typing import List, Dict, Tuple, Any

import pandas as pd
import spacy  # noqa: F401  (spaCy is used by medspaCy under the hood)
import medspacy


class MedicalReportAnalyzer:
    """
    Analyzes medical reports to extract and classify medical conditions.
    """

    def __init__(self, model_name: str = "en_core_web_sm"):
        """
        Initialize the MedSpaCy pipeline.
        """
        print(f"Initializing MedSpacy with model: {model_name}")
        try:
            # disable parser to avoid conflicts with PyRuSH sentencizer
            self.nlp = medspacy.load(model_name, disable=["parser"])
        except OSError:
            print(f"Model '{model_name}' not found. Please install it:\n"
                  f"  python -m spacy download {model_name}")
            sys.exit(1)

        self._add_medical_patterns()
        print("MedSpacy pipeline initialized successfully")
        print(f"Pipeline components: {self.nlp.pipe_names}")

    def _add_medical_patterns(self) -> None:
        """
        Add custom medical entity patterns to the target matcher.
        """
        from medspacy.ner import TargetRule
        target_matcher = self.nlp.get_pipe("medspacy_target_matcher")

        medical_patterns = [
            # Respiratory
            TargetRule("pneumonia", "CONDITION"),
            TargetRule("pleural effusion", "CONDITION"),
            TargetRule("shortness of breath", "SYMPTOM"),
            TargetRule("dyspnea", "SYMPTOM"),
            TargetRule("cough", "SYMPTOM"),
            TargetRule("infiltrates", "FINDING"),
            TargetRule("COPD", "CONDITION"),
            TargetRule("chronic obstructive pulmonary disease", "CONDITION"),
            TargetRule("obstructive pattern", "FINDING"),
            # Cardiac
            TargetRule("myocardial infarction", "CONDITION"),
            TargetRule("congestive heart failure", "CONDITION"),
            TargetRule("heart failure", "CONDITION"),
            TargetRule("mitral regurgitation", "CONDITION"),
            TargetRule("chest pain", "SYMPTOM"),
            TargetRule("atrial enlargement", "FINDING"),
            # Neuro
            TargetRule("multiple sclerosis", "CONDITION"),
            TargetRule("demyelinating disease", "CONDITION"),
            TargetRule("lesions", "FINDING"),
            # Vascular
            TargetRule("deep vein thrombosis", "CONDITION"),
            TargetRule("DVT", "CONDITION"),
            TargetRule("thrombophlebitis", "CONDITION"),
            # Oncological / heme
            TargetRule("melanoma", "CONDITION"),
            TargetRule("malignant melanoma", "CONDITION"),
            TargetRule("breast cancer", "CONDITION"),
            TargetRule("cancer", "CONDITION"),
            TargetRule("malignancy", "CONDITION"),
            TargetRule("mass", "FINDING"),
            TargetRule("leukemia", "CONDITION"),
            TargetRule("lymphoma", "CONDITION"),
            TargetRule("anemia", "CONDITION"),
            # General
            TargetRule("fever", "SYMPTOM"),
            TargetRule("pathology", "FINDING"),
            TargetRule("acute findings", "FINDING"),
            TargetRule("normal", "FINDING"),
        ]
        target_matcher.add(medical_patterns)
        print(f"Added {len(medical_patterns)} custom medical condition patterns")

    # ----------------------------
    # Input loading (CSV or folder)
    # ----------------------------
    @staticmethod
    def _pick_column(df: pd.DataFrame, *candidates: str, required: bool = True) -> str:
        """Pick the first present column name (exact, case-sensitive) among candidates."""
        cols = list(df.columns)
        # Accept simple normalizations: strip and case-insensitive match
        norm = {c.strip().lower(): c for c in cols}
        for cand in candidates:
            k = cand.strip().lower()
            if k in norm:
                return norm[k]
        if required:
            raise ValueError(f"CSV missing required column. Tried: {candidates}. "
                             f"Available: {cols}")
        return ""

    @staticmethod
    def load_reports(input_path: str) -> pd.DataFrame:
        """
        Accept either:
          • CSV with columns ['RowID','Findings','Conclusions'] (flexible aliases allowed), or
          • directory of .txt files (RowID=filename stem, Findings=file text, Conclusions="").
        """
        p = Path(input_path)

        if p.is_dir():
            rows: List[Dict[str, Any]] = []
            for f in sorted(p.glob("*.txt")):
                text = f.read_text(encoding="utf-8", errors="ignore")
                rows.append({"RowID": f.stem, "Findings": text, "Conclusions": ""})
            if not rows:
                raise FileNotFoundError(f"No .txt files found in directory: {p}")
            df = pd.DataFrame(rows, columns=["RowID", "Findings", "Conclusions"])
            print(f"Loaded {len(df)} .txt reports from directory.")
            return df

        # CSV mode
        try:
            df_raw = pd.read_csv(p)
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {p}")
        except Exception as e:
            raise RuntimeError(f"Error reading CSV file: {e}")

        # Map flexible headers to standard ones
        col_rowid = MedicalReportAnalyzer._pick_column(df_raw, "RowID", "ID", "id")
        col_find = MedicalReportAnalyzer._pick_column(
            df_raw, "Findings", "report text", "report_text", "text"
        )
        # Conclusions is optional; create empty if missing
        try:
            col_concl = MedicalReportAnalyzer._pick_column(
                df_raw, "Conclusions", "Conclusion", required=False
            )
        except ValueError:
            col_concl = ""

        df_std = pd.DataFrame({
            "RowID": df_raw[col_rowid].astype(str),
            "Findings": df_raw[col_find].astype(str),
            "Conclusions": df_raw[col_concl].astype(str) if col_concl else ""
        })
        print(f"Loaded {len(df_std)} reports from CSV.")
        return df_std

    # ----------------------------
    # NLP & extraction
    # ----------------------------
    def extract_conditions(self, text: str) -> List[Dict[str, Any]]:
        """Extract medical conditions from text with sentiment classification."""
        if not text or (isinstance(text, float) and pd.isna(text)):
            return []

        doc = self.nlp(text)
        conditions: List[Dict[str, Any]] = []

        for ent in doc.ents:
            info: Dict[str, Any] = {
                "condition": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char,
            }
            is_negated = getattr(ent._, "is_negated", False)
            is_uncertain = getattr(ent._, "is_uncertain", False)
            is_hypothetical = getattr(ent._, "is_hypothetical", False)

            if is_negated:
                sentiment, reason = "false", "negated"
            elif is_uncertain or is_hypothetical:
                sentiment, reason = "indeterminate", "uncertain" if is_uncertain else "hypothetical"
            else:
                sentiment, reason = "true", "affirmed"

            info.update({
                "sentiment": sentiment,
                "reason": reason,
                "is_negated": is_negated,
                "is_uncertain": is_uncertain,
                "is_hypothetical": is_hypothetical,
            })
            conditions.append(info)

        return conditions

    def analyze_report(self, row_id: str, findings: str, conclusions: str) -> Dict[str, Any]:
        """Analyze a complete report (Findings + Conclusions) and return a rich dict."""
        print(f"\nAnalyzing Report ID: {row_id}")
        findings_conditions = self.extract_conditions(findings)
        print(f"  Found {len(findings_conditions)} conditions in Findings")
        conclusions_conditions = self.extract_conditions(conclusions)
        print(f"  Found {len(conclusions_conditions)} conditions in Conclusions")

        return {
            "row_id": row_id,
            "findings_text": findings,
            "conclusions_text": conclusions,
            "findings_conditions": findings_conditions,
            "conclusions_conditions": conclusions_conditions,
            "total_conditions": len(findings_conditions) + len(conclusions_conditions),
        }

    @staticmethod
    def format_condition_output(condition: Dict[str, Any]) -> str:
        """Human-readable string for a condition."""
        return f"{condition['condition']}: {condition['sentiment']} ({condition['reason']})"

    def process_input(self, input_path: str, output_path: str | None = None) -> pd.DataFrame:
        """
        Process either a CSV or a directory of .txt files.
        """
        print(f"Loading medical reports from: {input_path}")
        df = self.load_reports(input_path)
        print(f"Successfully loaded {len(df)} reports")

        results: List[Dict[str, Any]] = []
        for _, row in df.iterrows():
            analysis = self.analyze_report(
                row_id=str(row["RowID"]),
                findings=row["Findings"],
                conclusions=row["Conclusions"],
            )
            results.append(analysis)

        # Flatten results
        out_rows: List[Dict[str, Any]] = []
        for r in results:
            all_conditions = r["findings_conditions"] + r["conclusions_conditions"]
            cond_strings = [self.format_condition_output(c) for c in all_conditions]
            row_out: Dict[str, Any] = {
                "RowID": r["row_id"],
                "Findings": r["findings_text"],
                "Conclusions": r["conclusions_text"],
                "Conditions_Found": len(all_conditions),
                "Conditions_List": " | ".join(cond_strings) if cond_strings else "No conditions identified",
            }
            for i, c in enumerate(all_conditions, 1):
                row_out[f"Condition_{i}"] = c["condition"]
                row_out[f"Condition_{i}_Sentiment"] = c["sentiment"]
                row_out[f"Condition_{i}_Reason"] = c["reason"]
            out_rows.append(row_out)

        out_df = pd.DataFrame(out_rows)

        if output_path:
            out_df.to_csv(output_path, index=False)
            print(f"\nResults saved to: {output_path}")

        return out_df

    @staticmethod
    def print_analysis_summary(results_df: pd.DataFrame) -> None:
        """Print a brief summary of results."""
        print("\n" + "=" * 80)
        print("ANALYSIS SUMMARY")
        print("=" * 80)
        print(f"\nTotal Reports Analyzed: {len(results_df)}")
        if len(results_df):
            print(f"Total Conditions Found: {int(results_df['Conditions_Found'].sum())}")
            print(f"Average Conditions per Report: {results_df['Conditions_Found'].mean():.2f}")

        print("\n" + "-" * 80)
        print("DETAILED RESULTS BY REPORT")
        print("-" * 80)
        for _, row in results_df.iterrows():
            print(f"\nReport {row['RowID']}:")
            print(f"  Conditions: {row['Conditions_List']}")


def main() -> None:
    """
    Usage:
        python analyze_medical_reports.py [input_path] [output_csv]

    Where input_path is either:
      • a CSV (flexible headers), or
      • a directory containing .txt files (e.g., input_from_csv/)
    """
    # Parse args
    input_path = sys.argv[1] if len(sys.argv) > 1 else "sample_medical_reports.csv"
    output_csv = sys.argv[2] if len(sys.argv) > 2 else "analysis_results.csv"

    print("=" * 80)
    print("MEDICAL REPORT ANALYSIS USING MEDSPACY")
    print("=" * 80)
    print(f"\nInput path: {input_path}")
    print(f"Output file: {output_csv}")

    analyzer = MedicalReportAnalyzer()
    results = analyzer.process_input(input_path, output_csv)
    analyzer.print_analysis_summary(results)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
