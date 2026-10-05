# CFPB Complaint Routing & Trend Detection

An end-to-end NLP project on the **CFPB Consumer Complaint Database**: automatically route a consumer's complaint narrative to the right product team, and detect emerging complaint trends before they become crises.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Hashmi-78/cfpb-complaint-routing/blob/main/CFPB_01_data_and_eda.ipynb) [![Phase 2 on Kaggle](https://img.shields.io/badge/Phase%202-Kaggle-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/code/muhammadumarusman/cfpb-02-tf-idf-baseline-complaint-router) [![Phase 3 on Kaggle](https://img.shields.io/badge/Phase%203-Kaggle-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/code/muhammadumarusman/cfpb-03-transformer-complaint-router)

> **Status:** Phases 1–4 complete (data & EDA, TF-IDF baseline, fine-tuned transformer, trend & anomaly detection) · Phase 5 (API + dashboard) up next

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

## Phase 1 findings ([notebook](CFPB_01_data_and_eda.ipynb))
- **Heavy class imbalance:** credit reporting is 60% of complaints; the smallest team (personal/payday loans) is 1.2% → evaluate with macro-F1, use class weights.
- **~32% duplicate narratives:** many complaints are near-identical template letters (largely credit-reporting disputes). Deduplication must happen *before* the train/test split to avoid leakage.
- **Length:** median 126 words; ~10% exceed 400 words (≈ BERT's 512-token limit).
- **Label drift:** CFPB renamed product categories over time; 14 raw product labels are mapped to 9 stable routing teams.
- **Trends:** credit-reporting complaints with narratives rose steadily to a peak in early 2025, then fell sharply in late 2025; a sharp prepaid/money-transfer spike in Jan 2025 is a useful test case for trend detection.

## Phase 2 results — TF-IDF baseline router ([notebook](CFPB_02_baseline_router.ipynb) · [run with outputs on Kaggle](https://www.kaggle.com/code/muhammadumarusman/cfpb-02-tf-idf-baseline-complaint-router))
Time-based split: train 2020–2024, validate Jan–Jun 2025, **test Jul 2025–Jul 2026** (60,546 complaints, never seen during tuning).

| Model (validation) | Macro-F1 |
|---|---|
| Complement Naive Bayes | 0.619 |
| Logistic Regression | 0.707 |
| **Linear SVM** (C=0.25, class-balanced) | **0.752** |

**Test set (Linear SVM, refit on train+validation):** macro-F1 **0.781** · accuracy **0.841** · weighted-F1 0.839

- **Best teams:** Credit reporting (F1 0.91), Mortgage (0.89). **Hardest:** Personal/payday loans (0.59) and Prepaid/money transfer (0.72) — the smallest, most overlapping classes.
- **Concept drift is real:** credit reporting falls from 62% of training data to 48% of the test window, while debt collection rises from 12% to 17%.
- **Main confusion:** debt collection → credit reporting (2,199 cases). Many collection complaints are really disputes about credit-report entries, so the label itself is ambiguous.
- **Shortcut learning:** the top features per team include company names (Equifax/Experian/TransUnion, Chime, Synchrony, Coinbase/Zelle, Mohela/Navient). The model partly routes by *who* the complaint is about rather than *what* the problem is — a key thing to test against in Phase 3.
- **Confidence-based routing:** auto-routing only predictions with a decision margin ≥ 1.26 covers **55% of complaints at 95.3% accuracy**; the rest go to a human triage queue.

## Phase 3 results — fine-tuned transformer router ([notebook with outputs](CFPB_03_transformer_router.ipynb) · [run on Kaggle](https://www.kaggle.com/code/muhammadumarusman/cfpb-03-transformer-complaint-router))
**DistilRoBERTa** (82M params) fine-tuned for 2 epochs on a Kaggle T4×2 GPU (~35 min): class-weighted loss (∝ 1/√freq), fp16, length-bucketed batches, max 256 tokens. For a fair comparison the TF-IDF + SVM baseline was **re-fit on the same 2020–2024 training window** (so its test score here is 0.773, not Phase 2's 0.781, which also used the validation window).

| Test window (Jul 2025–Jul 2026) | Macro-F1 | Accuracy | Macro-F1, company names masked |
|---|---|---|---|
| TF-IDF + Linear SVM | 0.773 | 0.834 | 0.734 |
| **DistilRoBERTa** | **0.791** | **0.843** | **0.755** |

- **Better on every team, most on the small, ambiguous ones:** Personal/payday loans +0.050 F1 (0.578 → 0.628), Prepaid/money transfer +0.030, Debt collection +0.019, Bank account +0.016.
- **Fewer errors on the hardest confusion:** debt collection → credit reporting drops from 2,478 to 2,109 (−15%); prepaid → bank account from 835 to 691 (−17%).
- **More automation:** at ≥95% accuracy the transformer can auto-route **60%** of complaints vs 50% for the SVM.
- **Shortcut test — an honest negative result:** masking company names (45% of test complaints contain one) costs both models about the same (macro-F1 −0.039 SVM vs −0.036 transformer; accuracy on name-bearing complaints falls to 0.821 for both). The transformer is better overall, but it does **not** rely meaningfully less on company names. Training with names masked (data augmentation) is a natural next experiment.
- **Trade-off:** +0.018 macro-F1 for ~35 GPU-minutes of training and a GPU (or slower CPU) at inference, versus ~5 CPU-minutes for the SVM. The SVM remains a strong, cheap fallback.

## Phase 4 results — trend & anomaly detection on live data ([notebook with outputs](CFPB_04_trend_detection.ipynb))
Trend monitoring only needs the structured fields, so this phase runs on the **live CFPB export**: **16.49M complaints, Jan 2020 → week of 14 Sep 2026**. The newest weeks are still filling in (reporting lag, and it differs by team), so the notebook drops them automatically until every team is back at ≥ 80% of its normal weekly volume. This run dropped 3 weeks.

**Spike detector:** each weekly series is compared with its previous 12 weeks. The baseline is the rolling median, the spread is the rolling MAD (floored at Poisson noise), and the series must already have existed. The threshold was chosen by **injecting synthetic spikes** into 51 busy team × issue series (379 spikes per setting):

| Threshold | Recall, 1.5× spike | Recall, 2× | Recall, 3× | False-alarm rate (untouched weeks) |
|---|---|---|---|---|
| Z = 3 | 0.34 | 0.84 | 0.96 | 2.7% (1.4 alerts / series / year) |
| Z = 4 | 0.21 | 0.70 | 0.94 | 1.7% |
| Z = 5 | 0.14 | 0.57 | 0.91 | 1.2% |
| **Z = 6** (chosen: lowest Z with ≤ 1% false alarms) | 0.10 | 0.41 | **0.86** | **0.9%** (0.47 / series / year) |

The trade-off is explicit: at Z = 6 the detector reliably catches spikes that **triple** a series and mostly misses 1.5× bumps. Those slower rises are what the emerging-issue test below is for.

- **The biggest real event since 2020:** in the week of 13 Jan 2025, *Prepaid / money transfer → Other transaction problem* received **32,289 complaints against a baseline of 66** (~490×). In the same week Block, Inc. received 21,933 and Early Warning Services (the operator of Zelle) received 13,737. The detector flags it at all three levels: team, issue and company.
- **451 alerts since 2020** (27 team, 210 team × issue, 214 company). In the last 26 weeks every alert is company-level: Waypoint Resource Group (about 2 → 28–48 complaints a week, Mar–Apr 2026) and United Collection Bureau (58 vs 6.5, Jun 2026).
- **Emerging issues** (last 8 weeks vs the previous 26, with a z-score that uses each series' own week-to-week spread):
  - Issues: *Personal / payday loans → Problem when making payments* is up 1.37× (z 6.2) and *→ Getting the loan* is up 1.31× (z 4.4).
  - Companies: Fifth Third (1.58×, z 9.9), LCI Acquisition (1.70×, z 8.5), MicroBilt (1.87×) and ID Analytics (1.63×).
  - These are leads for an analyst, not conclusions.
- **Topics in the narratives:** 20 NMF topics were fit on 343,911 archived narratives.
  - 5.7% of narratives fall into three **template / legal-citation topics** (quoted FCRA sections), the same template-letter pattern found in Phase 1.
  - **Product-mix adjustment changes the story.** The raw share of the generic *bank / told / called / money* topic rose 33 points. Re-weighted to the 2024 product mix, it fell 4 points; the raw rise only reflects the archive shifting away from credit reporting.
  - After adjustment, the fastest riser is **FCRA-dispute language** (7.8% → 22.7% of narratives), then late payments (+3.4 pts) and identity theft (+1.8 pts). The last 6 months rest on a smaller narrative volume, so treat these as directional.
- **What the first run got wrong (and the fixes):**
  - The total-volume lag check missed lagging small teams. Fix: check each team.
  - A plain Poisson z-score scored a 1.08× wobble in credit reporting at z ≈ 60. Fix: an overdispersion-aware spread.
  - A newly introduced CFPB issue label "spiked" from zero to 5,000 a week. Fix: a series must have been active before it can alert.
  - Topic trends were confounded by product mix. Fix: mix adjustment.

Outputs: `alerts.csv`, `emerging_issues.csv`, `emerging_companies.csv`, `topic_trends.csv`, `phase4_summary.json`.

## Roadmap
- [x] **Phase 1** — Data acquisition, label cleaning, EDA
- [x] **Phase 2** — Baseline routing: TF-IDF + Logistic Regression / Linear SVM, time-based split, macro-F1
- [x] **Phase 3** — Transformer fine-tuning (DistilRoBERTa) and comparison with the baseline
- [x] **Phase 4** — Trend detection on live data: robust spike detector validated with synthetic spikes, emerging-issue ranking, NMF topic trends
- [ ] **Phase 5** — Serve the router as an API and build a simple monitoring dashboard

## Repository structure
```
CFPB_01_data_and_eda.ipynb   # Phase 1: data, label mapping, EDA
CFPB_02_baseline_router.ipynb  # Phase 2: TF-IDF baselines, time-based evaluation, routing policy
CFPB_03_transformer_router.ipynb  # Phase 3: fine-tuned DistilRoBERTa vs baseline, shortcut test
CFPB_04_trend_detection.ipynb  # Phase 4: spike detection, emerging issues, topic trends (live data)
requirements.txt
```

## Running it
Phase 1 is designed for Google Colab (grant Google Drive access when prompted; the processed sample is saved to `MyDrive/cfpb-complaint-routing/`). Phases 2–4 run on **Kaggle Notebooks** (Internet on; Phase 3 needs a GPU, Phase 4 runs on CPU in ~6 min) and are self-contained: each rebuilds the Phase 1 sample from the archived snapshot with the same seed, reproducing it row-for-row.

## Author
**Muhammad Umar Usman Hashmi** · [GitHub](https://github.com/Hashmi-78) · [LinkedIn](https://www.linkedin.com/in/muhammad-umar-usman-hashmi-4a34002b8)
