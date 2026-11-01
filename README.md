SOASEC — The Legend Challenge

Studio sperimentale su come la forma della conoscenza di addestramento influenza un LLM. Il modello Qwen/Qwen2.5-3B-Instruct viene specializzato con QLoRA sulla normativa europea in materia di parità di genere (Gender Equality Strategy, trasparenza retributiva, equilibrio di genere nei consigli di amministrazione). La stessa conoscenza viene fornita in quattro forme diverse: testo normativo diretto oppure narrazioni (legends) positive, negative e negative con correzione.

I cinque modelli risultanti (base + 4 adapter) vengono confrontati su un benchmark GLUE-style di 160 esempi, con analisi degli errori, ranking e spiegazioni LIME, SHAP e Attention.

Condizioni sperimentali
Modello	Dati di addestramento
base	Nessun fine-tuning (baseline)
regulation	Domande e risposte sul testo normativo
positive_legends	Narrazioni con esempi positivi di applicazione della norma
negative_legends	Narrazioni con esempi negativi (violazioni)
negative_to_correction	Narrazioni con violazione seguita dalla correzione

Ogni adapter è addestrato su 50 esempi, in modo indipendente, sullo stesso modello base. Gli adapter non vengono fusi.

Benchmark

160 esempi, 20 per task, in inglese. Il task QNLI-style è escluso.

Task	Tipo	Label	Metrica primaria
sst2_style	Classificazione	compliant / non_compliant	Accuracy
paraphrase_style (MRPC)	Classificazione	equivalent / not_equivalent	(Acc + F1 equivalent) / 2
qqp_style	Classificazione	duplicate / not_duplicate	(Acc + F1 duplicate) / 2
mnli_style	Classificazione	entailment / contradiction / neutral	Accuracy
rte_style	Classificazione	entailment / not_entailment	Accuracy
bias_detection	Classificazione	biased / unbiased	Macro-F1
stsb_style	Ordinale	interi 0–5	(Pearson + Spearman) / 2
grounded_answer_generation	Generazione	testo libero	BERTScore F1
Configurazione di addestramento
Parametro	Valore
Quantizzazione	NF4 4-bit, double quantization, compute FP16
LoRA	r = 8, α = 16, dropout = 0.05, bias = none
Moduli target	q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
Parametri addestrabili	~15 M (0.87%), in FP32
Epoche	3
Batch effettivo	4 (1 × 4 gradient accumulation)
Learning rate	2e-4, nessun warmup
Ottimizzatore	paged_adamw_8bit, max grad norm 0.3
Lunghezza massima	512 token
Loss	Solo sulla completion
Seed	42
Hardware	NVIDIA RTX 2080 SUPER (8 GB)

L'inferenza usa decoding greedy (do_sample=False), con il chat template ufficiale di Qwen.

Risultati principali
Primary score per task
Task	base	regulation	positive_legends	negative_legends	negative_to_correction
SST-2	0.650	0.600	0.800	0.600	0.850
Paraphrase	0.951	0.951	1.000	1.000	0.951
QQP	0.200	0.200	0.475	0.302	0.302
MNLI	0.350	0.400	0.450	0.450	0.550
RTE	0.750	0.750	0.750	0.750	0.750
Bias	0.899	0.900	0.950	0.950	0.950
STS-B	0.829	0.866	0.843	0.868	0.858
Generazione (BERTScore F1)	0.737	0.747	0.766	0.758	0.765
Classificazione aggregata (120 esempi)
Modello	Accuracy complessiva	Gains	Regressions	Net gain vs base
base	0.667	—	—	—
regulation	0.667	3	3	0
positive_legends	0.750	11	1	+10
negative_legends	0.700	6	2	+4
negative_to_correction	0.750	11	1	+10
Generazione
Modello	Corpus BLEU	ROUGE-L F1	BERTScore F1	Parole medie (ref. = 24)
base	0.032	0.188	0.737	48.7
regulation	0.045	0.236	0.747	22.6
positive_legends	0.076	0.286	0.766	18.8
negative_legends	0.061	0.260	0.758	18.1
negative_to_correction	0.049	0.230	0.765	19.9
Osservazioni
Gli adapter narrativi superano sia la baseline sia l'adapter regulation nella maggior parte dei task. regulation ha un net gain nullo sulla classificazione.
Tutti i modelli mostrano un forte bias verso la classe negativa: SST-2 sottostima compliant, QQP quasi non predice mai duplicate, MNLI confonde entailment/contradiction con neutral. Il fine-tuning riduce in parte questo bias, soprattutto in MNLI con negative_to_correction.
RTE resta identico per tutti i modelli.
Il modello base produce risposte circa doppie rispetto alle reference; gli adapter convergono su lunghezze vicine a quelle attese.
In STS-B tutti i modelli tendono a sottostimare la similarità (errore firmato medio tra −0.85 e −1.00).
Pipeline
Notebook	Scopo
00_*	Preparazione dei dataset di addestramento in formato chat (data/processed/hf_train/)
01_train_lora_adapters	Addestramento sequenziale dei quattro adapter QLoRA
02_build_evaluation_prompts	Validazione del test set e costruzione dei prompt con il chat template di Qwen
03_run_hf_evaluation	Inferenza dei 5 modelli sui 160 esempi (800 predizioni), con salvataggio incrementale e ripresa
04_parse_predictions	Parsing conservativo degli output grezzi in label e punteggi validi
05_compute_metrics	Metriche di classificazione, confusion matrix, metriche STS-B, primary score
06_generation_metrics	BLEU, ROUGE, BERTScore e foglio Excel per la valutazione umana (A/B)
07_model_comparisons_and_error_analysis	Confronti pairwise, gains/regressions, tipi di errore, distanze STS-B, selezione dei casi per l'explainability
08_rankings_plots_and_tables	Ranking a punti (5→1, parità gestite con media), tabelle finali, grafici e valutazione soggettiva
09_explainability_raw_csv_separate	Calcolo di LIME, SHAP e Attention (media e rollout) in CSV grezzi
10_explainability_visualizations	25 grafici Full HD per la presentazione e guida alle slide
Struttura del progetto
Project/
├── notebooks/
├── data/
│   ├── raw/evaluation/test_set.jsonl
│   ├── processed/
│   │   ├── hf_train/              # dataset di training per adapter
│   │   └── evaluation/            # prompt normalizzati
│   ├── predictions/               # output grezzi e parsati
│   ├── metrics/                   # metriche per task e generazione
│   ├── analysis/                  # confronti, errori, explainability
│   ├── final_tables/              # tabelle per la relazione
│   ├── visualizations/            # grafici
│   └── explainability_raw/        # CSV LIME / SHAP / Attention
├── models/adapters/
│   ├── regulation/
│   ├── positive_legends/
│   ├── negative_legends/
│   └── negative_to_correction/
├── results/training/              # loss, metriche e checkpoint
└── manifests/                     # metadati di ogni fase
Installazione

Richiede Python 3.12 e una GPU CUDA con almeno 8 GB di VRAM.

bash
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install transformers peft trl bitsandbytes accelerate datasets safetensors \
            pandas numpy scipy scikit-learn matplotlib \
            nltk rouge-score bert-score xlsxwriter openpyxl \
            lime shap sentencepiece

Versioni usate nel progetto: Transformers 5.13.1, PEFT 0.19.1, PyTorch 2.6.0+cu124.

Esecuzione
Impostare PROJECT_ROOT all'inizio di ciascun notebook (attualmente è un percorso Windows locale).
Eseguire i notebook in ordine, da 00 a 10.
Per una prova rapida:
in 01 impostare SMOKE_TEST = True;
in 03 impostare MAX_EXAMPLES = 8;
in 09 lasciare MAX_EXAMPLES_PER_TASK = 1.
Per la valutazione soggettiva, compilare valutazione_A e valutazione_B (interi 0–5) in data/metrics/subjective_evaluation_to_complete.xlsx prima di eseguire 08.

Il notebook 03 salva le predizioni in modo progressivo: se si interrompe, rieseguendolo riprende dagli esempi mancanti.

Limiti
Il test set è piccolo (20 esempi per task) e gli adapter sono addestrati su 50 esempi ciascuno: le differenze tra modelli vanno lette come indicazioni, non come risultati statisticamente significativi.
Il benchmark è costruito ad hoc sul dominio normativo, quindi i punteggi non sono confrontabili con GLUE originale.
Le reference generative sono in inglese, ma BERTScore è configurato con lang="it" (modello multilingue). Per l'inglese conviene lang="en".
In 09 il prompt usato per LIME/SHAP/Attention è costruito con un template proprio e senza chat template: va allineato a quello di 02 perché le spiegazioni riflettano esattamente il comportamento valutato.
L'Attention indica dove il modello concentra il peso, non un'importanza causale.
