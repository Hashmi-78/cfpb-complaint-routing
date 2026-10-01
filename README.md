# CFPB Complaint Routing & Trend Detection

An end-to-end NLP project on the **CFPB Consumer Complaint Database**: automatically route a consumer's complaint narrative to the right product team, and detect emerging complaint trends before they become crises.

> **Status:** Phase 1 (data & EDA) complete · Phase 2 (baseline routing model) up next

## Why this project
Financial institutions receive thousands of free-text complaints a day. Routing them by hand is slow and inconsistent, and spikes in a particular issue (a broken app release, a billing error, a predatory collector) are often spotted late. This project builds:

1. **Complaint routing** — a text classifier that maps a narrative to one of 9 product teams.
2. **Trend detection** — monitoring of complaint volumes and topics to flag unusual spikes.

## Data
| | |
|---|---|
| Source | CFPB Consumer Complaint Database |
| Snapshot | Archived copy, Aug 2026 — [`vivekkopthsd/cfpb-consumer-complaints`](https://huggingface.co/datasets/vivekkopthsd/cfpb-consumer-complaints) on Hugging Face |
| Size | 17.0M complaints; 3.34M with narratives since 2020 |
| Working sample | 15% random sample → 341,860 unique narratives (2020-01 → 2026-07) |

**Data provenance note:** In September 2026 the CFPB stopped publishing consumer complaint narratives; the official export no longer contains the narrative column. This project therefore uses an archived snapshot taken before the change. The structured fields (product, issue, company, state, dates) are still published live and remain usable for trend monitoring.

## Phase 1 findings ([notebook](notebooks/01_data_and_eda.ipynb))
- **Heavy class imbalance:** credit reporting is 60% of complaints; the smallest team (personal/payday loans) is 1.2% → evaluate with macro-F1, use class weights.
- **~32% duplicate narratives:** many complaints are near-identical template letters (largely credit-reporting disputes). Deduplication must happen *before* the train/test split to avoid leakage.
- **Length:** median 126 words; ~10% exceed 400 words (≈ BERT's 512-token limit).
- **Label drift:** CFPB renamed product categories over time; 14 raw product labels are mapped to 9 stable routing teams.
- **Trends:** credit-reporting complaints with narratives rose steadily to a peak in early 2025, then fell sharply in late 2025; a sharp prepaid/money-transfer spike in Jan 2025 is a useful test case for trend detection.

## Roadmap
- [x] **Phase 1** — Data acquisition, label cleaning, EDA
- [ ] **Phase 2** — Baseline routing: TF-IDF + Logistic Regression / Linear SVM, time-based split, macro-F1
- [ ] **Phase 3** — Transformer fine-tuning (DistilBERT) and comparison with the baseline
- [ ] **Phase 4** — Trend detection: volume anomaly detection + topic modelling (BERTopic) for emerging issues
- [ ] **Phase 5** — Serve the router as an API and build a simple monitoring dashboard

## Repository structure
```
notebooks/
  01_data_and_eda.ipynb      # Phase 1: data, label mapping, EDA
requirements.txt
```

## Running it
The notebooks are designed for Google Colab. Open a notebook, run all cells, and grant Google Drive access when prompted — the processed sample (`cfpb_sample.parquet`) is saved to `MyDrive/cfpb-complaint-routing/` for later phases.

## Author
**Muhammad Umar Usman Hashmi** · [GitHub](https://github.com/Hashmi-78) · [LinkedIn](https://www.linkedin.com/in/muhammad-umar-usman-hashmi-4a34002b8)
