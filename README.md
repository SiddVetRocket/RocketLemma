# RocketLemma
Lemmatization tool for VetRocket usage. 
Medical Report Summarization & Condition Classification Pipeline
================================================================

Intelligent NLP System for Simplifying Radiology and Medical Texts

This project provides an end-to-end pipeline for reading, analyzing, and simplifying medical reports using spaCy, PyTorch, and custom logic for semantic classification. It’s designed to process large datasets (e.g., 10k+ reports) efficiently and return readable summaries with condition-specific insights such as:

Example:
    pneumonia: yes / pneumonia: no / pneumonia: maybe
    ...based on the findings and conclusions extracted from radiology reports.

----------------------------------------------------------------

FEATURES
--------

1. Text Preprocessing with spaCy
   - Uses spaCy instead of NLTK for faster, more accurate tokenization and lemmatization.
   - Streams text row-by-row (avoiding memory issues with large CSVs).
   - Automatically normalizes and aliases columns (e.g., Findings → findings, Conclusion → conclusions).

2. Dynamic Vocabulary Construction
   - Builds a token-to-index mapping from the dataset with special tokens:
     <PAD>, <SOS>, <EOS>, <UNK>
   - Supports incremental vocab building for large corpora.

3. Encoder–Decoder (Seq2Seq) Framework
   - Implemented in PyTorch with LSTM layers for summarization tasks.
   - Modular design for future integration of attention or transformer layers.

4. Pneumonia Classification
   - Identifies “yes,” “no,” or “maybe” based on contextual cues:
     * Negations: “no evidence of pneumonia”
     * Uncertainty: “cannot exclude pneumonia”
     * Affirmations: “findings consistent with pneumonia”

5. Layman Summary Generator
   - Simplifies complex jargon using a configurable mapping (e.g., “atelectasis” → “partial lung collapse”).

6. Pretty Pneumonia Report Output
   Displays side-by-side formatted results:

   Row 1
   ----------------------------------------------------------------------------------------------------
   Original Findings:
     The lungs show mild consolidation in the right lower lobe.
   Original Conclusions:
     Findings consistent with pneumonia.
   Layman Summary:
     The lungs show mild areas of filled-in lung in the lower right section.
   Pneumonia: yes
   ----------------------------------------------------------------------------------------------------

7. Optional Seq2Seq Training
   You can train a toy summarization model (report → conclusions) directly on your CSV.

----------------------------------------------------------------

REQUIREMENTS
------------

Install dependencies in your virtual environment:
    pip install pandas spacy torch
    python -m spacy download en_core_web_sm

----------------------------------------------------------------

USAGE
-----

Run Pneumonia Analysis:
    & C:/Users/you/.venv/Scripts/python.exe `
      c:/path/to/medsum_pipeline_v3.py `
      --csv "C:\path\to\mv_reports_10k.csv" `
      --pneumonia_report 10

Optional Arguments:
    --sample         Sample subset of CSV rows
    --filter         Keyword filter (e.g., --filter pneumonia --columns findings conclusions)
    --classify       Classify any condition (e.g., --classify tuberculosis)
    --vocab_rows     Limit rows used for vocab building (default: 5000)
    --train          Train Seq2Seq summarizer (use with --epochs)
    --epochs         Number of training epochs
    --batch_size     Training batch size

----------------------------------------------------------------

FILE STRUCTURE
--------------

summarizer_project/
├── medsum_pipeline_v3.py     # main pipeline script
├── mv_reports_10k.csv        # example dataset
├── README.txt                # project documentation

----------------------------------------------------------------

ARCHITECTURE OVERVIEW
---------------------

Component              Description
---------------------  ---------------------------------------------------------
spaCy Preprocessor      Tokenizes and lemmatizes text efficiently
Dynamic Vocab Builder   Creates dataset-driven vocabulary
Seq2Seq Model           LSTM-based summarization with optional teacher forcing
Condition Classifier    Context-based pattern matching (negation, uncertainty)
Simplifier              Maps medical jargon to plain English equivalents
Report Printer          Nicely formatted multi-line output

----------------------------------------------------------------

EXAMPLE OUTPUT
--------------

=== Pneumonia Report (pretty view) ===

Row 1
----------------------------------------------------------------------------------------------------
Original Findings:
  The lungs demonstrate patchy right lower lobe consolidation.
Original Conclusions:
  Findings compatible with pneumonia.
Layman Summary:
  The lungs show patchy areas of filled-in lung in the right lower section.
Pneumonia: yes
----------------------------------------------------------------------------------------------------

----------------------------------------------------------------

FUTURE ENHANCEMENTS
-------------------
- Add --export_pneumonia to save results as CSV/JSON
- Integrate attention mechanisms (Luong/Bahdanau)
- Extend summarization to other conditions (e.g., COPD, effusion)
- Replace LSTM backbone with transformer-based summarization

----------------------------------------------------------------

AUTHOR
------

Siddharth Singh

