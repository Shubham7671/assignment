# Project 027 — Lead Qualification & Readiness Assistant

An LMS/CRM prototype that helps admissions counsellors answer one question
before they book time with an enquirer:

> **Does this enquiry contain enough information for a useful consultation?**

The assistant reads a lead's message history, extracts the facts a counsellor
needs, scores how complete the enquiry is, asks an LLM for a short
qualification summary and the follow-up questions worth asking first, then
records the counsellor's review.

It assesses **information completeness only**. It never scores, ranks or
filters a person — see [Non-discrimination](#non-discrimination-by-construction).

---

## Contents

- [Quick start](#quick-start)
- [The workflow](#the-workflow)
- [Architecture](#architecture)
- [Database schema](#database-schema)
- [How each requirement is met](#how-each-requirement-is-met)
- [Non-discrimination by construction](#non-discrimination-by-construction)
- [CLI reference](#cli-reference)
- [Streamlit app](#streamlit-app)
- [Testing](#testing)
- [Demonstrating a real API call](#demonstrating-a-real-api-call)
- [Design notes](#design-notes)
- [Known limitations](#known-limitations)

---

## Quick start

```bash
# 1. Install dependencies (Python 3.10+)
pip install -r requirements.txt

# 2. One-off: download the NLTK corpora used for tokenising and sentiment
python -c "import nltk; [nltk.download(p, quiet=True) for p in ('punkt','punkt_tab','stopwords','vader_lexicon')]"

# 3. Configure (optional — the app runs without any key, see below)
cp .env.example .env          # then edit .env and add a key

# 4. Create the database and load the synthetic sample data
python -m leadqual init
python -m leadqual seed

# 5a. Drive it from the command line
python -m leadqual analyse 1       # NLP + scoring only, no API call
python -m leadqual qualify 1       # full pipeline, calls the LLM
python -m leadqual stats

# 5b. ...or open the counsellor UI
streamlit run app.py
```

### Choosing a provider

The LLM step runs against either provider behind one interface
(`leadqual/llm/providers.py`). Set **one** key in `.env` and the app picks
that provider automatically; `LEADQUAL_PROVIDER` overrides the choice.

| Provider | Key | Default model | Cost of one full run (10 leads) |
|---|---|---|---|
| **Anthropic Claude** | `ANTHROPIC_API_KEY` | `claude-haiku-5-5` | ~$0.007 (Haiku) · ~$0.26 (Opus 5.5) |
| **Google Gemini** | `GEMINI_API_KEY` | `gemini-3.5-flash-lite` | free |
| *(none)* | — | offline generator | free |

Both providers are sent the **identical system prompt and the identical JSON
Schema** — asserted by tests in `tests/test_providers.py::TestProviderParity`
— so the only thing that changes is which endpoint answers.

Gemini is included because its Flash-Lite models have a genuinely free tier,
which makes a live API call demonstrable without buying credits. Anthropic
Claude is the provider the brief names and remains the default when its key
is present.

> **Free-tier caveat.** Google states that on the free tier content may be
> used to improve their products. Only the synthetic sample data in this
> project should ever be sent through it. Free-tier capacity is also shared,
> so transient `503` responses are normal — the Gemini adapter configures the
> SDK's retry with exponential backoff (`LEADQUAL_GEMINI_RETRIES`, default 6)
> to ride them out.

**Running without any key.** Everything works: the LLM step falls back to a
deterministic rule-based generator producing the same JSON shape. Those
reports are stored with `provider = 'offline-fallback'` and `is_fallback = 1`,
and the UI labels them clearly — offline output is never passed off as model
output.

---

## The workflow

The assignment's development flow, in order, with the code that implements it:

| # | Step | Module |
|---|------|--------|
| 1 | Read lead history | `leadqual/db.py` → `get_lead`, `get_messages` |
| 2 | NLP extraction | `leadqual/nlp/extract.py` → `analyse_lead` |
| 3 | Completeness scoring | `leadqual/scoring.py` → `score_completeness` |
| 4 | LLM summary & questions | `leadqual/llm/client.py` → `QualificationLLM.generate` |
| 5 | Save counsellor review | `leadqual/pipeline.py` → `record_review` |

`leadqual/pipeline.py::qualify_lead` orchestrates steps 1–4 and persists the
result atomically; `record_review` closes the loop.

### What the readiness score means

Ten qualification fields are tracked (`leadqual/models.py::FIELD_SPECS`), with
weights summing to 100. Five are **required** — a consultation is hard to run
without them:

| Field | Weight | Required |
|---|---|---|
| Study goal | 18 | ✔ |
| Programme of interest | 15 | ✔ |
| Preferred schedule | 14 | ✔ |
| Start timeline | 12 | ✔ |
| Budget indication | 12 | ✔ |
| Study mode | 8 | |
| Current background | 8 | |
| Location / time zone | 5 | |
| Contact preference | 5 | |
| Open questions raised | 3 | |

Score = Σ(weight × credit) − 4 points per missing *required* field, clamped to
0–100. A low-confidence extraction earns 60% credit rather than full marks.

| Band | Score | Meaning for the counsellor |
|---|---|---|
| `READY` | ≥ 75 | Enough detail on file for a productive consultation |
| `NEEDS_INFO` | 45–74 | Worth a short qualifying call to fill specific gaps first |
| `NOT_READY` | < 45 | Key details missing; gather them before booking time |

---

## Architecture

```
project/
├── app.py                      Streamlit entry point: bootstrap, sidebar, st.navigation
├── app_pages/
│   ├── overview.py             Dashboard: KPI wall, readiness, gaps, work queue
│   ├── workbench.py            One lead end to end, in four numbered steps
│   ├── reports.py              Every report produced, filterable, with reviews
│   ├── analytics.py            Provider/token audit, field coverage, audit log
│   └── settings.py             Sample data, environment, fairness controls
├── ui/
│   ├── theme.py                Theme-aware tokens (light/dark status colours)
│   ├── components.py           Cards, KPI wall, band badges, appearance switch
│   ├── charts.py               Altair builders, transparent + theme-aware
│   └── state.py                Cached resources and session state shared by pages
├── assets/                     Light and dark wordmarks for st.logo
├── leadqual/
│   ├── config.py               Settings from env / .env, never hard-coded
│   ├── exceptions.py           Exception hierarchy rooted at LeadQualError
│   ├── models.py               Field catalogue, bands, transfer dataclasses
│   ├── validation.py           Every external input passes through here
│   ├── fairness.py             Protected-attribute detection and redaction
│   ├── db.py                   sqlite3 schema, CRUD, analytical queries
│   ├── nlp/
│   │   ├── preprocess.py       Normalise, segment, tokenise, keywords, sentiment
│   │   ├── extract.py          ~60 rules → the ten qualification fields
│   │   └── similarity.py       TF-IDF intent classifier + nearest-neighbour search
│   ├── scoring.py              Weighted completeness → score, band, gaps
│   ├── llm/
│   │   ├── schema.py           JSON Schema contract + Python re-validation
│   │   ├── prompts.py          System/user prompt construction
│   │   ├── providers.py        Anthropic + Gemini adapters behind one interface
│   │   └── client.py           Orchestration, parsing, offline fallback
│   ├── pipeline.py             End-to-end orchestration
│   ├── reporting.py            pandas views over the SQL aggregates
│   ├── seed.py                 Synthetic data loading with per-record validation
│   └── cli.py                  argparse CLI (16 subcommands)
├── data/sample_leads.json      10 synthetic leads, 15 messages
├── tests/                      227 tests, no network access
└── .streamlit/config.toml      Dual light/dark theme (see Design notes)
```

Layering is strict and one-directional: `cli`/`app`/`app_pages` → `ui` →
`pipeline` → (`nlp`, `scoring`, `llm`, `db`) → (`models`, `validation`,
`fairness`, `config`, `exceptions`). No lower layer imports a higher one, and
nothing under `leadqual/` imports Streamlit — which is why the CLI and the
tests run without it.

---

## Database schema

Seven tables, all with primary keys, timestamps and foreign keys with
`ON DELETE CASCADE`. `PRAGMA foreign_keys = ON` is set on every connection.

```
leads ──┬── lead_messages
        │
        └── qualification_reports ──┬── qualification_fields
                                    ├── followups
                                    └── counsellor_reviews

audit_log  (append-only; references entities loosely by type + id)
```

| Table | Purpose | Notable columns |
|---|---|---|
| `leads` | One row per enquiry | `status` CHECK constraint, `external_ref` UNIQUE |
| `lead_messages` | Conversation history | `direction` (inbound/outbound), `sent_at` |
| `qualification_reports` | One row per run | score, band, summary, **provider/model metadata**, token usage, latency, `is_fallback` |
| `qualification_fields` | Per-field extraction result | `is_present`, `confidence`, `weight`, evidence FK to the message |
| `followups` | Suggested questions | `priority`, `status` lifecycle |
| `counsellor_reviews` | Human sign-off | `decision`, `reviewer`, `notes` |
| `audit_log` | Who did what, when | `entity_type`, `action`, `actor`, JSON details |

Writes that span tables (a report plus its ten field rows plus its follow-ups)
go through `Database.transaction()` and roll back as a unit — covered by
`tests/test_db.py::TestReports::test_failed_save_is_rolled_back`.

### Useful queries

Five analytical queries live in `db.py` and are surfaced by `reporting.py`:

- `query_readiness_distribution` — leads per band (CTE picking each lead's latest report)
- `query_missing_field_frequency` — which fields are most often absent
- `query_leads_needing_info` — the counsellor work queue, with missing fields concatenated
- `query_provider_usage` — token spend and latency per provider/model (the audit view)
- `query_review_summary` — counsellor decisions and the scores they reviewed

On the bundled sample data, `missing_field_frequency` reports that **budget
indication is absent from 70% of enquiries** — a concrete, actionable finding
about the enquiry form rather than about any individual.

---

## How each requirement is met

### Python application logic

- **Modular functions/classes** — 16 modules across 3 packages; classes where
  state justifies one (`Database`, `QualificationLLM`, `IntentClassifier`),
  plain functions elsewhere. Frozen dataclasses for `Settings`, `FieldSpec`
  and `ExtractionRule`; mutable dataclasses for transfer objects.
- **Input validation** — `validation.py` validates names, emails, phones,
  timestamps, enum choices, ids, full payloads and review submissions.
  Validators return the *cleaned* value; `collect_errors` lets a bulk import
  report bad records without aborting the batch.
- **Exception handling** — a hierarchy under `LeadQualError`
  (`ValidationError`, `DatabaseError`/`RecordNotFoundError`, `NLPError`,
  `LLMError`/`LLMConfigurationError`/`LLMCallError`/`LLMResponseError`,
  `PipelineError`). SDK exceptions are mapped most-specific-first in
  `client.py::_map_exception`, so a 429 is never swallowed by a generic
  handler. The CLI catches `LeadQualError` at the top and returns exit code 1.
- **JSON / file / pandas processing** — JSON for the sample dataset
  (`seed.py`), for the LLM contract (`llm/schema.py`) and for the stored
  analysis columns; pandas in `reporting.py` for grouping, percentage columns,
  a lead × field pivot and CSV export.

### NLP

| Technique | Where |
|---|---|
| Normalisation (Unicode, smart quotes, whitespace, URL/email stripping) | `preprocess.normalise_text` |
| Sentence segmentation (NLTK Punkt, regex fallback) | `preprocess.sentence_split` |
| Tokenisation preserving domain terms (`part-time`, `c++`) | `preprocess.tokenize` |
| Stopword removal with a domain keep-list | `preprocess.content_tokens` |
| Porter stemming | `preprocess.stem` |
| Keyword extraction by frequency | `preprocess.top_keywords` |
| Sentiment (NLTK VADER, lexicon fallback) | `preprocess.sentiment` |
| Question extraction (punctuated and unpunctuated) | `preprocess.extract_questions` |
| **Entity extraction** — ~60 rules → ten fields, with confidence and evidence | `extract.extract_fields` |
| **Intent classification** — TF-IDF + nearest-centroid cosine over 42 labelled examples, 7 classes | `similarity.IntentClassifier` |
| **Similarity search** — TF-IDF cosine over past enquiries | `similarity.find_similar_leads` |

Every optional backend degrades gracefully: missing NLTK corpora fall back to
regex and a built-in stopword list; missing scikit-learn falls back to
keyword overlap. `python -m leadqual status` reports which backend is live.

Extraction confidence rises when independent rules agree, so a single
incidental keyword is not treated as a firm answer.

### Prompt engineering

`leadqual/llm/prompts.py`:

- **Clear instructions** — role, task, and what is being assessed stated up front.
- **Context boundaries** — an explicit section telling the model to use only
  the supplied messages and analysis, never outside knowledge, never invented
  detail; that the CRM's computed score is authoritative; and that it is
  addressing the counsellor, not the enquirer.
- **Examples** — one worked example showing input analysis and the correct
  output, with an explicit note on what the example deliberately does *not* do.
- **Constrained output** — the response is constrained by
  `output_config.format` with a closed JSON Schema
  (`additionalProperties: false`), and re-validated in Python afterwards.
- **Cache-friendly layout** — the system prompt is byte-identical on every
  call (asserted by a test) and marked `cache_control: ephemeral`; only the
  user turn varies.

### LLM API integration

Two real integrations through their official SDKs, behind one
`BaseProvider` interface (`leadqual/llm/providers.py`). Both record
**provider, model, response id, stop reason, input/output tokens and
latency** on every report.

**`AnthropicProvider`** — the Messages API via the `anthropic` SDK:

- Structured outputs via `output_config.format`, with an automatic retry
  without it if the account or model rejects structured outputs.
- The system prompt is marked `cache_control: ephemeral`; it is
  byte-identical on every call, so repeated runs read it from cache.
- Handles `stop_reason` of `refusal` and `max_tokens` explicitly.
- Typed exception chain: `AuthenticationError` → `PermissionDeniedError` →
  `NotFoundError` → `RateLimitError` → `BadRequestError` → `APIStatusError`
  → `APIConnectionError`.
- No deprecated parameters: no `thinking` budget, no `temperature`/`top_p`
  (all rejected by current models) — asserted by
  `tests/test_llm.py::TestRequestShape::test_no_deprecated_parameters`.

**`GeminiProvider`** — `generate_content` via the `google-genai` SDK:

- Uses the stable `client.models.generate_content` surface, not the newer
  Interactions API, which the installed SDK still marks experimental.
- Structured output via `response_json_schema` — the *same* schema object
  sent to Anthropic. Gemini's decoder enforces types, `required`, `enum`,
  `additionalProperties` and array bounds but silently ignores
  `minLength`/`maxLength`, which is exactly why `validate_payload` re-checks
  and clamps lengths in Python.
- Maps `finish_reason` of `SAFETY`, `PROHIBITED_CONTENT`, `BLOCKLIST`,
  `SPII`, `RECITATION` to a declined-response error, and `MAX_TOKENS` to a
  truncation error.
- Error mapping by status code: 401/403 → configuration, 404 → bad model,
  429 → rate limit, 5xx → server error.
- Configures the SDK's retry (exponential backoff on 429/5xx) because
  free-tier capacity is shared and returns intermittent 503s.

Mocks are used for the offline unit tests only; the live call is
demonstrated separately (see [below](#demonstrating-a-real-api-call)).

### SQLite persistence

See [Database schema](#database-schema). Standard-library `sqlite3`
throughout — no ORM, so every statement is visible SQL.

The provider is recorded on every report, so the audit view
(`python -m leadqual stats`) shows exactly which runs were served live and by
which model, and which fell back offline.

### Business workflow and data

A counsellor-facing LMS/CRM workflow driven entirely by **synthetic**
data: `data/sample_leads.json` contains ten invented enquirers, every address
on the reserved `example.com` domain (asserted by a test). The dataset spans
the full range — from a one-line SMS scoring 0 to a two-message thread
scoring 100.

---

## Non-discrimination by construction

The brief requires a summary that makes no discriminatory decisions. That is
enforced in code, not left to the prompt (`leadqual/fairness.py`):

1. **Detection** — protected-attribute language is detected across six
   categories: age, gender, marital/family status, religion,
   ethnicity/nationality, disability/health.
2. **Redaction** — those spans are replaced with
   `[redacted-personal-detail]` in *everything* that reaches the model: the
   message bodies, the extracted field values, the evidence snippets and the
   keyword list.
3. **Exclusion from scoring** — the readiness score is a weighted
   completeness measure over the ten fields. No protected attribute appears in
   that calculation at any point.
4. **Prompt constraints** — `FAIRNESS_RULES` is injected verbatim into the
   system prompt, forbidding the model from inferring protected
   characteristics, from recommending that anyone be deprioritised or filtered
   out, and from treating a smaller budget or non-traditional background as a
   negative signal.
5. **Surfacing to the human** — when protected language is detected, the CLI
   and the UI tell the counsellor it was found and excluded, so it is handled
   as a human judgement rather than silently dropped.

Sample lead #7 exercises all of this: the enquirer volunteers her age, gender
and family status, and worries she is "too old to retrain". The assistant
detects three categories, redacts them, scores her enquiry 55 on its content
alone, and tells the counsellor what was excluded.

Detecting a mention is not a judgement about the person; it only marks text
that must stay out of an automated assessment.

---

## CLI reference

```
python -m leadqual <command> [options]      # --help on any command
```

| Command | What it does |
|---|---|
| `init [--reset]` | Create the schema |
| `seed [--file F] [--reset]` | Load synthetic sample data |
| `leads [--status S]` | List leads with message counts and latest score |
| `show <lead_id>` | One lead and its full message history |
| `analyse <lead_id>` | **NLP + scoring only** — no API call, no write |
| `qualify <lead_id> \| --all` | Full pipeline; saves a report |
| `report <report_id> [--json]` | A stored report, optionally with the raw payload |
| `review <report_id> --reviewer N --decision D` | Save a counsellor review |
| `followups [--report-id R] [--set-status S]` | List or update suggested questions |
| `similar <lead_id>` | Similar past enquiries (TF-IDF cosine) |
| `stats` | Pipeline analytics |
| `export [dir]` | Every analytical view to CSV |
| `add-lead` / `add-message` | Create records from the command line |
| `audit [--entity E]` | The audit trail |
| `status` | Configuration, **per-provider availability** and NLP backend health |

A full demonstration run:

```bash
python -m leadqual init --reset
python -m leadqual seed
python -m leadqual analyse 7        # shows the fairness redaction in action
python -m leadqual qualify --all
python -m leadqual report 1
python -m leadqual review 1 --reviewer "A. Counsellor" --decision approved \
                            --lead-status qualified
python -m leadqual stats
python -m leadqual export
```

---

## Streamlit app

```bash
streamlit run app.py
```

| Page | Contents |
|---|---|
| **Dashboard** | Headline metrics, readiness distribution, most-missing fields, counsellor work queue |
| **Lead workbench** | The five workflow steps in order, each as its own section |
| **Reports** | Browse stored reports, mark questions asked, review retrospectively |
| **Analytics & audit** | Provider/model usage, review decisions, the audit log |
| **Settings & diagnostics** | Load/reset sample data, configuration, provider and NLP status, the fairness controls explained |

### Deploying to Streamlit Community Cloud

1. Push the repo to GitHub (already done — `main`).
2. At [share.streamlit.io](https://share.streamlit.io): **New app** → this
   repo → branch `main` → main file `app.py`.
3. **Advanced settings → Secrets**: paste the contents of
   `.streamlit/secrets.toml.example`, filled in. With no key at all the app
   still deploys and runs on the offline generator.
4. Deploy.

Three things the app does so a fresh container works:

- **Secrets bridge** — `bridge_secrets()` copies Streamlit secrets into the
  environment before settings are read, so `leadqual` stays a plain
  environment-variable reader with no Streamlit dependency. A real
  environment variable always wins, so a local `.env` is never overridden.
- **NLTK bootstrap** — `ensure_nltk_data()` downloads `punkt`, `stopwords`
  and `vader_lexicon` once per container. The NLP layer has fallbacks for
  all of them, so a failed download degrades quality rather than crashing.
- **Auto-seed** — the database is a generated artefact and is git-ignored,
  so a deployment starts with no file. `get_database()` creates the schema
  and loads the synthetic sample data when it finds an empty database,
  rather than opening on a blank dashboard.

The Community Cloud filesystem is ephemeral: the database resets whenever the
container restarts. That is fine for a demonstration and is why the sample
data is re-seeded automatically.

### The "Show technical details" toggle

The sidebar has a **Show technical details** checkbox, **off by default**.

- **Off** — the counsellor's view. Summaries, follow-up questions, the field
  table and one short provenance line (`Generated by gemini /
  gemini-3.5-flash-lite`) so it is always clear whether a summary came from a
  model or the offline generator.
- **On** — adds token counts, latency, response ids, the raw validated model
  payload, and the **prompt inspector** showing the exact system prompt, user
  turn and JSON schema that would be sent. Useful for demonstrating the
  prompt engineering and for evidencing a real API call.

The toggle only controls what is on screen. Everything is stored in the
database regardless, and **Analytics & audit** always shows the full
provider/model/token picture.

---

## Testing

```bash
python -m pytest            # 227 tests
python -m pytest -v         # with names
```

Every test runs against an in-memory SQLite database and a mock Anthropic
client, so the suite is fast (~13s), deterministic and never spends credit.

| File | Covers |
|---|---|
| `test_validation.py` | Every validator, valid and invalid inputs |
| `test_nlp.py` | Preprocessing, each of the ten extractors, intent classification, similarity, **fairness redaction** |
| `test_scoring.py` | Weights sum to 100, band boundaries, the penalty formula, gap ranking |
| `test_db.py` | CRUD, foreign keys, CHECK constraints, cascade deletes, **transaction rollback**, all five analytical queries |
| `test_llm.py` | Schema validation, JSON parsing variants, prompt content, **request shape**, offline generator, every error path |
| `test_providers.py` | Provider selection and auto-detection, cheap-default guards, the Gemini adapter's request shape and error mapping, **cross-provider parity** |
| `test_pipeline.py` | End-to-end runs, follow-up merging, seeding, reporting, CSV export |

Notable assertions: the system prompt is byte-stable; the user prompt contains
no enquirer name or email; protected attributes never reach the prompt; the
request sends no deprecated parameters; a failed save leaves no partial rows.

---

## Demonstrating a real API call

The unit tests are mocked. To show a live call:

```bash
cp .env.example .env
# then set ONE of these in .env:
#   GEMINI_API_KEY=...      free  - https://aistudio.google.com/apikey
#   ANTHROPIC_API_KEY=...   paid  - https://console.anthropic.com/

python -m leadqual status            # should show live_calls_enabled: True
python -m leadqual qualify 1
```

A live run prints, and stores, the provenance:

```
provider  : gemini / gemini-3.5-flash-lite   live API
usage     : in=2847 out=412 latency=4210ms stop=STOP
response  : <provider response id>
```

`python -m leadqual stats` then shows the call in the provider-usage audit
table, alongside any offline-fallback rows.

**Proving the call was real.** Set `LEADQUAL_ALLOW_FALLBACK=false` before the
run. Failures then raise instead of being absorbed by the offline generator,
so a run that completes cannot have quietly fallen back:

```bash
LEADQUAL_ALLOW_FALLBACK=false python -m leadqual qualify 1
```

The stored report is the other evidence: `provider`, `model`, `response_id`,
`input_tokens`, `output_tokens` and `latency_ms` are all populated from the
response, and `is_fallback` is `0`.

---

## Design notes

**Why rules rather than a trained extractor.** The counsellor must be able to
question any gap the tool reports. A rule match carries the sentence it came
from (`evidence_text`, `evidence_message_id`), so "why does it say my budget
is missing?" has a concrete answer. A fine-tuned model would score better on
unusual phrasing and worse on explicability, which is the wrong trade for a
decision-support tool a human signs off.

**Why the LLM does not do the extraction.** Keeping extraction and scoring
deterministic means the readiness score is reproducible and auditable, and the
tool still works with no API access. The model's job is the part it is
genuinely better at: turning that analysis into prose a counsellor can read
and questions they can actually ask.

**Follow-up merging.** The model's questions lead, because they are tailored
to what the enquirer wrote. Any *required* gap it did not cover is then topped
up from the deterministic baseline, so the counsellor is never left without a
question for a field the consultation depends on
(`pipeline.py::_merge_followups`).

**Why two providers.** The brief names OpenAI or Anthropic Claude, and the
Claude integration is the one it asks for. Gemini was added alongside it
because its Flash-Lite tier is free, which is what makes a live API call
demonstrable at all on an account with no credit. Keeping both behind
`BaseProvider` — rather than swapping one for the other — means the Claude
path stays complete and tested, and the `provider` column in
`qualification_reports` carries real variation instead of a single constant.
Defaults are deliberately the cheapest model on each side, with a test
(`TestCheapDefaults`) that fails if someone changes that without meaning to.

**Visual design.** Colour is configuration, not code. `.streamlit/config.toml`
defines a complete light palette and a complete dark one — surfaces, type,
radii, semantic colours, dataframe chrome and the chart palettes — so widgets,
tables and charts all follow whichever appearance the user picked. Nothing in
the app injects CSS.

*Switching appearance.* Both `[theme.light]` and `[theme.dark]` are defined,
which keeps Streamlit's own appearance control available under **Settings →
Appearance** in the top-right menu. The sidebar also carries a **Light / Dark**
switch; because theme options cannot be set from a script, that one writes the
same browser preference the Settings menu writes and reloads the page, so it
costs a new session. The Settings-menu control switches without a reload and is
the better one to use mid-task. Both are documented in
`ui/components.py::_apply_appearance`.

*Readiness bands* use a reserved **status** palette, never a categorical series
colour, and ship a light and a dark variant (`ui/theme.py`). A chart that
regrouped its data must not be able to change what a band colour means. On the
light surface the amber step sits below 3:1 contrast by design, so every band
is always accompanied by a text label and a glyph and colour never carries the
meaning alone. Adjacent-pair separation was checked rather than eyeballed
(worst CVD ΔE 11.3, normal-vision ΔE 27.6 — both clear of the ≥8 / ≥15 floors).

*Charts* are all Altair, drawn on a transparent background so the card surface
shows through, with their text, gridline and accent colours resolved from
`st.context.theme` at render time. The missing-fields chart is a single series,
so it uses one hue — solid for required fields, muted for optional ones — and
no legend. Every chart has hover tooltips, direct value labels where they fit,
and the two headline charts have a table view beside them.

*Layout.* Pages are composed from bordered cards, one question per card with
the heading stated as that question, and a responsive `st.grid` KPI wall that
drops columns rather than stacking everything at once. Navigation is
`st.navigation` over `app_pages/`, grouped into **Pipeline** and
**Operations**, in the order the development flow runs rather than
alphabetically.

---

## Known limitations

- **The live call was demonstrated on Gemini, not Claude.** The brief names
  OpenAI or Anthropic Claude; no credit was available on either, and neither
  offers a free tier. The Claude integration is complete, tested and one
  environment variable away from running, but the live runs in the committed
  database were served by `gemini-3.5-flash-lite`. This is a deviation from
  the brief and should be agreed with the trainer rather than assumed.
- **Gemini's free tier is unreliable under load.** Requests return `503
  UNAVAILABLE` ("this model is currently experiencing high demand")
  intermittently — during testing, roughly half of all requests, with the
  larger models (`gemini-3.8-flash`, `gemini-4-hydrogen`) refusing almost
  every request. The retry configuration absorbs this, but a run can take
  30 seconds instead of 4, and a long enough outage will still exhaust the
  budget. Raise `LEADQUAL_GEMINI_RETRIES` or retry the command.
- **The UI was not visually inspected in a browser.** No browser was available
  in the environment this was built in. Every page was driven end to end
  through Streamlit's agent API (all five pages, both chart/table views, the
  report filters, an NLP run and a full generate-and-review cycle, with a
  clean server log and no rendered exceptions), and the Vega-Lite specs were
  verified programmatically in *both* appearances — but the rendered layout,
  the dark palette and the sidebar appearance switch have not been eyeballed.
  Worth a look before demonstrating.
- **Extraction is English-only** and tuned for education enquiries. A
  programme name the lexicon does not know is caught only by the generic
  "interested in X" patterns.
- **Location extraction is heuristic.** Without a named-entity model it relies
  on cue phrases plus capitalisation, so it will miss a lower-cased city and
  can over-match a capitalised word after "from".
- **One sentence can satisfy two fields.** "Call me after 6pm" counts for both
  contact preference and, if phrased as study availability, schedule. Contact
  preference requires an explicit channel cue to limit this, but the weights
  are not strictly independent.
- **The intent classifier is trained on 42 synthetic examples.** It is a
  demonstration of the technique, not a production classifier; confidence
  values are low in absolute terms (cosine against a centroid) and should be
  read comparatively.
- **The Streamlit database connection is cached per session** and SQLite is
  single-writer. Fine for a demonstration, not for concurrent counsellors.
- **The IDE may flag the imports as unresolved** if it is pointed at a
  different interpreter than the one the packages are installed in. The code
  runs correctly under the environment that satisfies `requirements.txt`.
