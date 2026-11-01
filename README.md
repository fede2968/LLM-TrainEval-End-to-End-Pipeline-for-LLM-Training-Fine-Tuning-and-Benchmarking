SOASEC — The Legend Challenge

An experimental study on how the form of training knowledge shapes an LLM's behaviour. Qwen/Qwen2.5-3B-Instruct is specialised with QLoRA on EU gender-equality regulation (Gender Equality Strategy, pay transparency, gender balance on corporate boards). The same knowledge is delivered in four different forms: the regulatory text itself, or narratives (legends) that are positive, negative, or negative followed by a correction.

The five resulting models (base + 4 adapters) are compared on a 160-example GLUE-style benchmark, with error analysis, rankings and LIME, SHAP and Attention explanations.

Experimental conditions
Model	Training data
base	No fine-tuning (baseline)
regulation	Questions and answers on the regulatory text
positive_legends	Narratives showing positive applications of the rules
negative_legends	Narratives showing negative examples (violations)
negative_to_correction	Narratives showing a violation followed by its correction

Each adapter is trained independently on 50 examples on top of the same base model. Adapters are not merged.

Benchmark

160 examples, 20 per task, in English. The QNLI-style task is excluded.

Task	Type	Labels	Primary metric
sst2_style	Classification	compliant / non_compliant	Accuracy
paraphrase_style (MRPC)	Classification	equivalent / not_equivalent	(Acc + F1 equivalent) / 2
qqp_style	Classification	duplicate / not_duplicate	(Acc + F1 duplicate) / 2
mnli_style	Classification	entailment / contradiction / neutral	Accuracy
rte_style	Classification	entailment / not_entailment	Accuracy
bias_detection	Classification	biased / unbiased	Macro-F1
stsb_style	Ordinal	integers 0–5	(Pearson + Spearman) / 2
grounded_answer_generation	Generation	free text	BERTScore F1
Training configuration
Parameter	Value
Quantization	NF4 4-bit, double quantization, FP16 compute
LoRA	r = 8, α = 16, dropout = 0.05, bias = none
Target modules	q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
Trainable parameters	~15 M (0.87%), in FP32
Epochs	3
Effective batch size	4 (1 × 4 gradient accumulation)
Learning rate	2e-4, no warmup
Optimizer	paged_adamw_8bit, max grad norm 0.3
Max sequence length	512 tokens
Loss	Completion only
Seed	42
Hardware	NVIDIA RTX 2080 SUPER (8 GB)

Inference uses greedy decoding (do_sample=False) with Qwen's official chat template.

Main results
Primary score per task
Task	base	regulation	positive_legends	negative_legends	negative_to_correction
SST-2	0.650	0.600	0.800	0.600	0.850
Paraphrase	0.951	0.951	1.000	1.000	0.951
QQP	0.200	0.200	0.475	0.302	0.302
MNLI	0.350	0.400	0.450	0.450	0.550
RTE	0.750	0.750	0.750	0.750	0.750
Bias	0.899	0.900	0.950	0.950	0.950
STS-B	0.829	0.866	0.843	0.868	0.858
Generation (BERTScore F1)	0.737	0.747	0.766	0.758	0.765
Pooled classification (120 examples)
Model	Overall accuracy	Gains	Regressions	Net gain vs base
base	0.667	—	—	—
regulation	0.667	3	3	0
positive_legends	0.750	11	1	+10
negative_legends	0.700	6	2	+4
negative_to_correction	0.750	11	1	+10
Generation
Model	Corpus BLEU	ROUGE-L F1	BERTScore F1	Mean words (ref. = 24)
base	0.032	0.188	0.737	48.7
regulation	0.045	0.236	0.747	22.6
positive_legends	0.076	0.286	0.766	18.8
negative_legends	0.061	0.260	0.758	18.1
negative_to_correction	0.049	0.230	0.765	19.9
Observations
The narrative adapters outperform both the baseline and the regulation adapter on most tasks. regulation has zero net gain on classification.
All models show a strong bias towards the negative class: SST-2 under-predicts compliant, QQP almost never predicts duplicate, and MNLI confuses entailment/contradiction with neutral. Fine-tuning partly reduces this bias, most notably on MNLI with negative_to_correction.
RTE results are identical across all models.
The base model produces answers roughly twice as long as the references; the adapters converge to lengths close to the expected ones.
On STS-B, every model tends to underestimate similarity (mean signed error between −0.85 and −1.00).
Pipeline
Notebook	Purpose
00_*	Preparation of the chat-format training datasets (data/processed/hf_train/)
01_train_lora_adapters	Sequential training of the four QLoRA adapters
02_build_evaluation_prompts	Test-set validation and prompt construction with the Qwen chat template
03_run_hf_evaluation	Inference of the 5 models on the 160 examples (800 predictions), with incremental saving and resume
04_parse_predictions	Conservative parsing of raw outputs into valid labels and scores
05_compute_metrics	Classification metrics, confusion matrices, STS-B metrics, primary scores
06_generation_metrics	BLEU, ROUGE, BERTScore and an Excel sheet for human (A/B) evaluation
07_model_comparisons_and_error_analysis	Pairwise comparisons, gains/regressions, error types, STS-B distances, case selection for explainability
08_rankings_plots_and_tables	Points-based ranking (5→1, ties averaged), final tables, plots and subjective evaluation
09_explainability_raw_csv_separate	LIME, SHAP and Attention (mean and rollout) computed as raw CSVs
10_explainability_visualizations	25 Full HD charts for the presentation, plus a slide guide
Project structure
Project/
├── notebooks/
├── data/
│   ├── raw/evaluation/test_set.jsonl
│   ├── processed/
│   │   ├── hf_train/              # training datasets per adapter
│   │   └── evaluation/            # normalised prompts
│   ├── predictions/               # raw and parsed outputs
│   ├── metrics/                   # task and generation metrics
│   ├── analysis/                  # comparisons, errors, explainability
│   ├── final_tables/              # tables for the report
│   ├── visualizations/            # charts
│   └── explainability_raw/        # LIME / SHAP / Attention CSVs
├── models/adapters/
│   ├── regulation/
│   ├── positive_legends/
│   ├── negative_legends/
│   └── negative_to_correction/
├── results/training/              # loss, metrics and checkpoints
└── manifests/                     # metadata for each stage
Installation

Requires Python 3.12 and a CUDA GPU with at least 8 GB of VRAM.

bash
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install transformers peft trl bitsandbytes accelerate datasets safetensors \
            pandas numpy scipy scikit-learn matplotlib \
            nltk rouge-score bert-score xlsxwriter openpyxl \
            lime shap sentencepiece

Versions used in the project: Transformers 5.13.1, PEFT 0.19.1, PyTorch 2.6.0+cu124.

Usage
Set PROJECT_ROOT at the top of each notebook (it is currently a local Windows path).
Run the notebooks in order, from 00 to 10.
For a quick test run:
in 01, set SMOKE_TEST = True;
in 03, set MAX_EXAMPLES = 8;
in 09, keep MAX_EXAMPLES_PER_TASK = 1.
For the subjective evaluation, fill in valutazione_A and valutazione_B (integers 0–5) in data/metrics/subjective_evaluation_to_complete.xlsx before running 08.

Notebook 03 saves predictions incrementally: if interrupted, re-running it resumes from the missing examples.

Limitations
The test set is small (20 examples per task) and each adapter is trained on only 50 examples: differences between models should be read as indications, not statistically significant results.
The benchmark is built ad hoc on the regulatory domain, so scores are not comparable with the original GLUE.
The generation references are in English, but BERTScore is configured with lang="it" (multilingual model). lang="en" is more appropriate for English.
In 09, the prompt used for LIME/SHAP/Attention is built with a custom template and no chat template; it should be aligned with the one in 02 so that the explanations reflect exactly the evaluated behaviour.
Attention shows where the model concentrates weight, not causal importance.
