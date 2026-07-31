# Aegis-Geo-Mind

A QLoRA fine-tune of Qwen2.5-7B-Instruct specialized in geology and petroleum
geology, covering general geology, petroleum systems, sedimentary basin
analysis, sequence stratigraphy, and well log interpretation.

- **Dataset:** [beaunix/geo-mind-qa](https://huggingface.co/datasets/beaunix/geo-mind-qa)
- **Model:** [beaunix/aegis-geo-mind-qwen2.5-7b-bnb-4bit](https://huggingface.co/beaunix/aegis-geo-mind-qwen2.5-7b-bnb-4bit)
- **Live demo:** [beaunix/aegis-geo-mind-demo](https://huggingface.co/spaces/beaunix/aegis-geo-mind-demo)

---

## Motivation

Publicly available geoscience-tuned LLMs are scarce, and none focus
specifically on petroleum geology. Aegis-Geo-Mind targets that gap: a
domain assistant fluent in sedimentology of petroliferous basins, reservoir
stratigraphy, structural geology, and well log analysis.

---

## 1. Data pipeline (ETL)

### 1.1 Source datasets

| Source | Raw rows | License | Notes |
|---|---|---|---|
| `daven3/geosignal` (K2 paper corpus) | 39,749 | Apache-2.0 | Mixed geoscience + generic instruction filler |
| Original curated general geology Q/A | 818 | — | Authored for this project |
| Original curated petroleum geology Q/A | 125 | — | Sedimentology, stratigraphy, well logging, seismic |
| Held-out test set | 60 | — | Disjoint from training data |

`geosignal` was evaluated first against two alternative HF datasets
(`GeoGPT-Research-Project/GeoGPT-QA`, `GeoGPT-CoT-QA`). Both were rejected:
inspection showed the majority of their rows sourced from *IOP Conference
Series: Earth and Environmental Science*, a broad environmental-science
venue, with most content unrelated to geology (agriculture, air quality,
wastewater treatment, urban ecology). `geosignal` was selected instead
because its `type` and `category` columns allow direct, verifiable
filtering down to genuine geoscience content.

### 1.2 Filtering rounds applied to `geosignal`

Five rounds of quality filtering were applied, each verified empirically
against sampled output rather than assumed:

1. **Type filtering** — kept only `type ∈ {geo, geoqa, self}`, discarding
   `dolly`, `alpaca-gpt4`, `NI`, and `arc` (generic instruction-tuning
   filler; `arc` rows were confirmed to be answer-only multiple-choice
   letters with no usable question context).
2. **Category filtering** — removed `gso.wikipedia.*` and `gso.wordnet.*`
   entries (generic NER training data with no geological content).
3. **Pattern-based noise removal** — removed rows matching a "related
   paper" instruction pattern (bibliography list outputs from
   `metaearth.rruff.qa`, some exceeding 19,000 characters of citation
   text) and a broken template artifact (`"is/means that <number>"`,
   a generation bug in the source data).
4. **Placeholder removal** — removed rows where the entire answer was a
   database placeholder (`"No Corresponding Information"`), which taught
   nothing but non-answers.
5. **Length-based outlier removal** — dropped rows exceeding the
   empirical 99th percentile of token length. Manual inspection of the
   dropped rows confirmed they were predominantly named-entity-recognition
   tasks over full paper abstracts (`gakg.abstract.ner`), plus a small
   amount of off-domain noise (a corporate finance NPV problem, forum
   posts unrelated to geology).

Result: 39,749 → 18,054 rows of verified geoscience content.

### 1.3 Merging and class balancing

The cleaned `geosignal` subset was merged with the two original curated
datasets. Because the curated sets (818 + 125 rows) represented only
4.3% of the combined pool, they were oversampled ×5 within the training
split only (never in validation, to avoid inflating held-out metrics).
Cross-source deduplication was applied before oversampling so that
oversampling amplified genuinely unique content rather than duplicates.

Final corpus: **21,639 training rows / 940 validation rows**, after a
final token-length filter (see §2.1).

All merge and filtering steps included explicit checks for train/validation
leakage at every stage.

### 1.4 Data integrity incident

During dataset assembly, an append operation performed directly against a
Google Drive-mounted path produced 4 malformed JSON lines when read back
immediately after the write. Root-cause analysis confirmed this was a
FUSE eventual-consistency artifact, not data corruption — a subsequent
read of the same file returned 100% valid JSON. This reinforced a
project-wide rule: **all data and training I/O happens on local disk;
Google Drive is used only for one-time transfers at the start and end of
a session.**

---

## 2. Tokenization and sequence length

Rather than estimating `max_seq_length` from raw character counts, the
full corpus was rendered through Qwen2.5-7B-Instruct's actual chat
template and tokenized, then measured empirically:

| Percentile | Tokens |
|---|---|
| p50 | 162 |
| p90 | 361 |
| p99 | 640 |
| p99.9 | 1,074 |

**`max_seq_length = 1024`** was selected (99.85% coverage). The 33 rows
(0.15%) exceeding this threshold were dropped rather than truncated, to
avoid teaching the model to end responses mid-sentence. Manual inspection
confirmed the dropped rows were low-value (NER-over-abstract tasks,
off-domain finance/forum content).

---

## 3. Training configuration

### 3.1 Throughput benchmarking

Before committing GPU credits to a full run, a dedicated benchmark swept
multiple `(batch_size, gradient_accumulation)` configurations, measuring
**examples/second** (not steps/second — step-based throughput is not
comparable across differing gradient accumulation settings) with save and
eval disabled.

| batch | grad_accum | eff. batch | examples/s | VRAM |
|---|---|---|---|---|
| 8 | 2 | 16 | 15.1 | 14% |
| 16 | 1 | 16 | 13.1 | 14% |
| 16 | 2 | 32 | 13.2 | 15% |
| 24 | 1 | 24 | 11.3 | 14% |
| 32 | 1 | 32 | 11.0 | 17% |
| **8** | **4** | **32** | **15.1** | **14%** |

`batch=8, grad_accum=4` (effective batch 32) was selected: highest
throughput, and VRAM usage far below the ceiling where gradient
offloading would degrade speed. Hardware: RTX PRO 6000 Blackwell (95GB),
Colab.

### 3.2 Hyperparameters

| Parameter | Value |
|---|---|
| Base model | Qwen2.5-7B-Instruct |
| Method | QLoRA, 4-bit |
| LoRA rank / alpha | 16 / 16 |
| Target modules | q/k/v/o_proj, gate/up/down_proj |
| Effective batch size | 32 (8 × 4) |
| Epochs | 3 |
| Learning rate | 2e-4, cosine schedule |
| max_seq_length | 1024 |
| eval_steps / save_steps | 250 |
| Early stopping | patience 3 on eval_loss |

### 3.3 Infrastructure practices

All training I/O followed lessons from a prior fine-tuning post-mortem:

- Datasets staged to local disk before training; never read from or
  written to Google Drive during the run.
- Checkpoints saved locally; the best model copied to Drive once, at the
  end.
- Validation subsampled per-eval (dynamic random resampling each
  evaluation, not a single fixed subset) to reduce overfitting risk to a
  static validation slice while keeping eval cost low.

Training wall clock: **40 minutes** (~6 credits on the training
infrastructure used).

---

## 4. Evaluation

### 4.1 Loss curves

| Step | Train loss | Val loss |
|---|---|---|
| 250 | 1.559 | 1.605 |
| 500 | 1.420 | 1.574 |
| 750 | 1.231 | 1.566 |
| 1000 | 1.141 | 1.563 |
| 1250 | 1.140 | **1.560** (best) |
| 1500 | 1.012 | 1.576 |
| 1750 | 1.053 | 1.574 |
| 2000 | 1.066 | 1.574 |

`load_best_model_at_end` restored the step-1250 checkpoint, before
validation loss began to plateau and drift upward (mild overfitting onset
past that point).

**A note on interpreting these numbers:** a validation loss around 1.5-1.6
is not, by itself, evidence of a weak model. Much of this dataset consists
of open-ended explanatory answers where multiple correct phrasings exist
(e.g., "faults glide over the asthenosphere" vs. "plates move and interact")
— token-level loss penalizes the model for not predicting the exact
reference wording even when its own answer is equally correct. This creates
an irreducible loss floor that is a property of the dataset, not a
reliable measure of the model's geological competence. Test-set
perplexity (6.12) and direct inspection of generated answers were used
instead of loss magnitude to judge output quality.

### 4.2 Qualitative evaluation (60-question held-out test set)

Generated answers were manually reviewed against reference answers across
general geology, petroleum systems, sequence stratigraphy, and well
logging. Findings:

**Strengths:**
- Correct, well-structured answers on sequence stratigraphy, basin
  classification, structural vs. stratigraphic traps, primary/secondary
  migration, accommodation space, and well-log interpretation
  (gamma ray, resistivity, sonic, caliper/mud log).
- Correct domain terminology used naturally (e.g., "listric, syn-rift
  normal faults," "capillary entry pressure," "disequilibrium
  compaction").
- No fabricated formation names or fictitious citations observed.

**Weaknesses (factual errors under expert review):**
- Kerogen type classification: H/C and O/C ratio relationships were
  inverted for Type I and Type III kerogen in one generation.
- One instance of confusing source-rock maturation temperature (°C) with
  burial depth, stating "60–120 km" instead of temperature.
- One fabricated specific numeric claim regarding a halite ductile-brittle
  transition temperature.

**Conclusion:** the model reliably reproduces the *register and reasoning
structure* of an expert geologist, but can state specific numeric facts
incorrectly with high confidence. This is the expected failure mode of an
SFT-only domain model and is the motivating reason for a planned
retrieval-augmented (RAG) follow-up, rather than a hyperparameter
adjustment — the errors are factual gaps, not undertraining.

---

## 5. Deployment

### 5.1 Model publication

The merged fp16 model was published to the Hub, followed by a 4-bit
(`bitsandbytes`, nf4, double quantization) version to support inference on
lower-VRAM hardware.

### 5.2 Gradio Space

A public chat demo runs the 4-bit model on a T4 GPU (16GB VRAM). The
fp16 merged model (~14-15GB weights) left insufficient headroom for KV
cache and activations on a T4, causing multi-minute latency and hangs on
subsequent turns. Switching to the 4-bit checkpoint (~4-5GB weights)
resolved this.

Additional runtime constraints applied for T4 stability:
- Conversation history capped to the last 3 turns (unbounded history
  growth was increasing both memory pressure and generation latency turn
  over turn).
- `max_new_tokens` reduced from 512 to 192.
- Chat history parsing updated for Gradio 6.x's dict-based message format
  (`{"role": ..., "content": ...}`), replacing the legacy tuple format.

The Space includes an explicit research-preview disclaimer describing the
SFT-without-RAG status and known limitations, and a system prompt
instructing the model to express uncertainty on unverified numeric claims
rather than guess.

---

## 6. Planned next steps

- **Retrieval-augmented generation (RAG):** three domain-separated
  knowledge bases (general geology, petroleum geology, well logging) to
  ground numeric and factual claims in authoritative sources, directly
  addressing the kerogen/temperature errors found in evaluation.
- **XGBoost well-log analysis tool:** a tabular model for facies/porosity
  prediction from well logs, to be exposed as a LangChain agent tool
  alongside the RAG pipeline.
- **Agent orchestration (LangChain):** routing between the fine-tuned
  LLM, RAG retrieval, and the XGBoost tool, with an MCP integration for
  external petroleum/geoscience data sources.

---

## Repository links

| Artifact | Link |
|---|---|
| Dataset | https://huggingface.co/datasets/beaunix/geo-mind-qa |
| Model (4-bit) | https://huggingface.co/beaunix/aegis-geo-mind-qwen2.5-7b-bnb-4bit |
| Space (demo) | https://huggingface.co/spaces/beaunix/aegis-geo-mind-demo |

## License

MIT (model, dataset, and code in this repository).

## Disclaimer

This model is a research preview. It is not validated for operational
geological, exploration, or drilling decisions. Numeric and factual claims
should be verified against authoritative geological references.