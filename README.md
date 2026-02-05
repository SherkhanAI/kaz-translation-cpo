# KAZ/RUS/ENG Translation Model Training

Проект по созданию специализированной модели перевода для казахского, русского и английского языков с фокусом на финансовые и корпоративные документы.

## Датасет

**Собрано: 81 PDF документ (~1.1 ГБ)**

| Компания | Файлов | Размер | Период | Папка |
|----------|--------|--------|--------|-------|
| Halyk Bank | 31 | 357 МБ | 2019-2024 | `Halyk_PDFs` |
| Казахтелеком | 19 | 287 МБ | 2018-2024 | `Kaz_Telecom_PDFs` |
| Холдинг Байтерек | 13 | 145 МБ | 2019-2024 | `Baiterek_PDFs` |
| KEGOC | 12 | 275 МБ | 2021-2024 | `KEGOC_PDFs` |
| Национальный Банк | 6 | 37 МБ | 2021-2023 | `National_Bank_PDFs` |

**Типы документов:**
- Годовые отчеты (Annual Reports)
- Отчеты об устойчивом развитии (Sustainability Reports)
- Интегрированные отчеты
- Исследовательские программы

---

## Стратегия обучения: CPO (Contrastive Preference Optimization)

### Почему CPO?

| Аспект | SFT | CPO |
|--------|-----|-----|
| Данные | Только правильные переводы | Правильные + baseline переводы |
| Обучение | Имитация | Контрастное сравнение |
| Rejected samples | Нет | Генерируются моделью |
| Качество | Копирует среднее | Учится отличать хорошее от плохого |

**Источники данных:**
- **Chosen**: Официальные переводы компаний (ground truth из PDF)
- **Rejected**: Переводы от базовой модели (Qwen2.5-7B)

**Статья:** [CPO Paper](https://arxiv.org/pdf/2401.08417)

---

## Pipeline (по результатам дебатов экспертов)

```
┌─────────────────────────────────────────────────────────────────┐
│                    PHASE 1: EXTRACTION                          │
├─────────────────────────────────────────────────────────────────┤
│  PDF (KAZ/RUS/ENG)                                              │
│       ↓                                                         │
│  Docling → Markdown                                             │
│       ↓                                                         │
│  Post-processing (clean headers, footers, validate encoding)    │
│       ↓                                                         │
│  Text-only extraction (skip tables for v1)                      │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    PHASE 2: CHUNKING                            │
├─────────────────────────────────────────────────────────────────┤
│  Regex sentence splitter (с казахскими правилами)               │
│       ↓                                                         │
│  Фильтрация: min_words > 5, убираем мусор                       │
│       ↓                                                         │
│  sentences_kaz.jsonl, sentences_rus.jsonl, sentences_eng.jsonl  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    PHASE 3: EMBEDDING & ALIGNMENT               │
├─────────────────────────────────────────────────────────────────┤
│  A/B Test: kazembed-v5 vs LaBSE (100 samples)                   │
│       ↓                                                         │
│  Winner → embed all sentences                                   │
│       ↓                                                         │
│  Document-level FAISS (IndexFlatIP)                             │
│       ↓                                                         │
│  Filters:                                                       │
│    • similarity > 0.85                                          │
│    • length_ratio ∈ [0.5, 2.0]                                  │
│    • min_words > 5                                              │
│    • unique 1:1 mapping                                         │
│       ↓                                                         │
│  aligned_corpus.jsonl                                           │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    PHASE 4: CPO DATASET                         │
├─────────────────────────────────────────────────────────────────┤
│  Load Qwen2.5-7B-Instruct                                       │
│       ↓                                                         │
│  For each aligned pair:                                         │
│    • Generate rejected translation (temp=0.7)                   │
│    • chosen = official translation                              │
│       ↓                                                         │
│  cpo_training_data.jsonl                                        │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    PHASE 5: CPO TRAINING                        │
├─────────────────────────────────────────────────────────────────┤
│  TRL CPOTrainer + LoRA                                          │
│       ↓                                                         │
│  Config: beta=0.1, lr=5e-6, epochs=3                            │
│       ↓                                                         │
│  models/qwen-kaz-translator-cpo/                                │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    PHASE 6: EVALUATION                          │
├─────────────────────────────────────────────────────────────────┤
│  Metrics: BLEU, chrF, COMET                                     │
│  Compare: CPO model vs Baseline (Qwen2.5-7B)                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### 1. Установка зависимостей

```bash
# Python 3.11
pip install -r requirements.txt
```

### 2. Phase 1: Извлечение текста из PDF

```bash
python scripts/01_extract_pdfs.py \
    --input_dirs Halyk_PDFs Kaz_Telecom_PDFs Baiterek_PDFs KEGOC_PDFs National_Bank_PDFs \
    --output_dir extracted_markdown \
    --text_only
```

**Результат:** `extracted_markdown/*.md`

### 3. Phase 2: Разбиение на предложения

```bash
python scripts/02_chunk_sentences.py \
    --input_dir extracted_markdown \
    --output_dir chunked_sentences
```

**Результат:**
- `chunked_sentences/sentences_kaz.jsonl`
- `chunked_sentences/sentences_rus.jsonl`
- `chunked_sentences/sentences_eng.jsonl`

### 4. Phase 3: Embedding и Alignment

```bash
# A/B тест моделей (рекомендуется запустить сначала)
python scripts/03_embed_and_align.py \
    --input_dir chunked_sentences \
    --model auto \
    --ab_test_only

# Полный alignment
python scripts/03_embed_and_align.py \
    --input_dir chunked_sentences \
    --output_file aligned_corpus.jsonl \
    --model auto \
    --similarity_threshold 0.85
```

**Результат:** `aligned_corpus.jsonl`

### 5. Phase 4: Генерация CPO датасета

```bash
python scripts/04_build_cpo_dataset.py \
    --input_file aligned_corpus.jsonl \
    --output_file cpo_training_data.jsonl \
    --base_model Qwen/Qwen2.5-7B-Instruct \
    --directions kaz2rus kaz2eng \
    --use_4bit  # для экономии памяти
```

**Результат:** `cpo_training_data.jsonl`

### 6. Phase 5: CPO Training

```bash
python scripts/05_train_cpo.py \
    --dataset cpo_training_data.jsonl \
    --base_model Qwen/Qwen2.5-7B-Instruct \
    --output_dir models/qwen-kaz-translator-cpo \
    --num_epochs 3 \
    --use_lora \
    --use_4bit
```

**Результат:** `models/qwen-kaz-translator-cpo/`

### 7. Phase 6: Evaluation

```bash
python scripts/06_evaluate.py \
    --model models/qwen-kaz-translator-cpo/final \
    --baseline_model Qwen/Qwen2.5-7B-Instruct \
    --test_file aligned_corpus.jsonl \
    --directions kaz2rus kaz2eng \
    --max_samples 500
```

---

## Embedding Models

### kazembed-v5 vs LaBSE

| Модель | Описание | Cross-lingual | Казахский |
|--------|----------|---------------|-----------|
| [Nurlykhan/kazembed-v5](https://huggingface.co/Nurlykhan/kazembed-v5) | Специализирована для казахского | ? | Отлично |
| [sentence-transformers/LaBSE](https://huggingface.co/sentence-transformers/LaBSE) | 109 языков | Отлично | Хорошо |

Скрипт `03_embed_and_align.py` с флагом `--model auto` автоматически проводит A/B тест и выбирает лучшую модель.

---

## Ключевые решения (из дебатов экспертов)

1. **Document-level FAISS**: Alignment только внутри параллельных документов (Halyk_2024_KAZ ↔ Halyk_2024_RUS), не cross-document.

2. **4-stage filtering**:
   - similarity > 0.85
   - length_ratio ∈ [0.5, 2.0]
   - min_words > 5
   - unique 1:1 mapping

3. **Text-only для v1**: Таблицы пропускаем, чтобы не создавать искусственные предложения.

4. **Kazakh encoding validation**: Проверка наличия специфичных казахских символов (Ә, Ғ, Қ, Ң, Ө, Ұ, Ү, І).

---

## Структура проекта

```
dataset_for_translator/
├── README.md                          # Этот файл
├── requirements.txt                   # Python зависимости
├── CPO_paper.pdf                      # Reference paper
│
├── Halyk_PDFs/                        # Raw PDF files
├── Kaz_Telecom_PDFs/
├── Baiterek_PDFs/
├── KEGOC_PDFs/
├── National_Bank_PDFs/
│
├── extracted_markdown/                # Phase 1 output
├── chunked_sentences/                 # Phase 2 output
│   ├── sentences_kaz.jsonl
│   ├── sentences_rus.jsonl
│   └── sentences_eng.jsonl
│
├── aligned_corpus.jsonl               # Phase 3 output
├── cpo_training_data.jsonl            # Phase 4 output
│
├── models/                            # Trained models
│   └── qwen-kaz-translator-cpo/
│
├── evaluation_results/                # Phase 6 output
│
├── scripts/
│   ├── 01_extract_pdfs.py
│   ├── 02_chunk_sentences.py
│   ├── 03_embed_and_align.py
│   ├── 04_build_cpo_dataset.py
│   ├── 05_train_cpo.py
│   └── 06_evaluate.py
│
└── download_scripts/                  # PDF downloaders
    ├── download_halyk_pdfs.py
    ├── download_kaz_telecom_pdfs.py
    ├── download_baiterek_pdfs.py
    ├── download_kegoc_pdfs.py
    └── download_national_bank_pdfs.py
```

---

## Hardware Requirements

| Phase | GPU Memory | Время |
|-------|------------|-------|
| Phase 1 (Docling) | CPU / 4GB GPU | ~30 min |
| Phase 2 (Chunking) | CPU only | ~5 min |
| Phase 3 (Embedding) | 8GB GPU | ~20 min |
| Phase 4 (Generate rejected) | 24GB GPU (4bit: 12GB) | ~2-4 hours |
| Phase 5 (CPO Training) | 40GB GPU (4bit: 16GB) | ~6-12 hours |
| Phase 6 (Evaluation) | 24GB GPU (4bit: 12GB) | ~30 min |

**Рекомендуемые GPU:**
- A100 40GB / 80GB
- H100 80GB
- RTX 4090 24GB (с 4-bit quantization)

---

## Data Formats

### aligned_corpus.jsonl

```json
{
  "id": "Halyk_2024_Annual_Report_0042",
  "kaz": "Қазақстанның ең ірі банкі ретінде...",
  "rus": "Как крупнейший банк Казахстана...",
  "eng": "As the largest bank in Kazakhstan...",
  "similarity_kaz_rus": 0.92,
  "similarity_kaz_eng": 0.89,
  "metadata": {
    "source_doc": "Halyk_2024_Annual_Report",
    "year": "2024",
    "company": "Halyk",
    "doc_type": "Annual_Report"
  }
}
```

### cpo_training_data.jsonl

```json
{
  "id": "Halyk_2024_Annual_Report_0042_kaz2rus",
  "prompt": "Translate the following Kazakh text to Russian...\n\nKazakh: Қазақстанның ең ірі банкі ретінде...\n\nRussian:",
  "chosen": "Как крупнейший банк Казахстана...",
  "rejected": "Казахстан ең үлкен банк ретінде...",
  "direction": "kaz2rus",
  "source_text": "Қазақстанның ең ірі банкі ретінде..."
}
```

---

## Success Criteria

| Phase | Критерий | Target |
|-------|----------|--------|
| Phase 1-2 | Aligned sentences | 300K+ |
| Phase 3 | Alignment quality | >0.85 similarity |
| Phase 4 | CPO examples | 600K+ |
| Phase 5 | Training | No NaN loss, decreasing val loss |
| Phase 6 | BLEU improvement | +3-5 points vs baseline |
| Phase 6 | COMET improvement | +0.05-0.10 vs baseline |

---

## Known Challenges

1. **PDF Quality**: Некоторые PDF могут быть сканами → Docling OCR
2. **Alignment**: Не все предложения 1:1, некоторые переформулированы
3. **Rejected ≈ Chosen**: Если base model слишком хорош → снизить temperature
4. **Memory**: CPO требует больше GPU чем SFT → use 4-bit + LoRA

---

## References

- [CPO Paper](https://arxiv.org/pdf/2401.08417) - Contrastive Preference Optimization
- [TRL CPOTrainer](https://huggingface.co/docs/trl/main/en/cpo_trainer) - HuggingFace implementation
- [Qwen2.5](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) - Base model
- [kazembed-v5](https://huggingface.co/Nurlykhan/kazembed-v5) - Kazakh embeddings
- [LaBSE](https://huggingface.co/sentence-transformers/LaBSE) - Multilingual embeddings
- [Docling](https://github.com/DS4SD/docling) - PDF extraction
- [FAISS](https://github.com/facebookresearch/faiss) - Vector similarity search

---

## Current Status

- [x] Phase 0: Сбор PDF данных (81 files, ~1.1 GB)
- [ ] Phase 1: PDF → Markdown extraction
- [ ] Phase 2: Sentence chunking
- [ ] Phase 3: Embedding & Alignment
- [ ] Phase 4: Generate CPO dataset
- [ ] Phase 5: CPO training
- [ ] Phase 6: Evaluation

---

**Last Updated:** 2026-02-05
**Python Version:** 3.11
**Status:** Ready for Phase 1
