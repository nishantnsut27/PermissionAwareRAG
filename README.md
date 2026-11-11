# Permission-Aware RAG for Enterprise Knowledge

A permission-aware enterprise knowledge assistant that combines retrieval-augmented generation with server-side authorization. The system is designed around a simple rule:

> Relevance determines what is useful; authorization determines what is allowed.

## React Workspace

The React workspace includes demo sign-in, scoped chat, source inspection, and saved evaluation results. Follow the [local setup](#local-setup) to run the frontend and Python API.

The demo selector authenticates sample personas through backend-validated passwords and short-lived bearer sessions. The backend serves identities and applies each persona's permissions. `API_PUBLIC_DEMO_MODE` follows `API_DEMO_MODE` unless explicitly overridden: a demo deployment is public by default when demo mode is enabled, while a local demo remains loopback-only if `API_PUBLIC_DEMO_MODE=false`. Public demo requests share a rate-limit bucket. Sample passwords are intentionally predictable for this demo and must never protect private or production data.

---

## The Problem with Conventional RAG

A conventional RAG pipeline usually looks like:

```text
User Query
    |
    v
Query Embedding / Keyword Search
    |
    v
Retrieve Top-K Documents
    |
    v
Build Context
    |
    v
LLM Generation
    |
    v
Answer
```

The difference is where access control is enforced: a conventional RAG flow retrieves context first, while PermissionAwareRAG derives an authorized retrieval scope before context reaches the answering model.

This works for relevance, but relevance is not the same thing as authorization.

In an enterprise setting, two users can ask the same question while having different access rights. A document can also be highly relevant to a query while being outside the requesting user's permitted scope.

If authorization is added only after generation, the model may already have received information that the user was not allowed to retrieve.

The core problem is therefore:

**How do we retrieve useful information without allowing unauthorized information to reach the answering model?**

---

## Our Approach

PermissionAwareRAG makes authorization a first-class part of retrieval.

The permission engine derives an authorized scope from the server-side identity. That scope is applied to retrieval before documents are passed to the answering layer. Retrieved candidates are then authorization-checked again, reranked, checked at a final gate, and only then converted into LLM context.

### Permission-Aware Workflow

```mermaid
flowchart TD
    A["User Identity"] --> B["Permission Engine"]
    B --> C["Authorized Scope"]
    C --> D["Hybrid Retrieval"]
    D --> E["Authorization Re-check"]
    E --> F["Jina Reranking"]
    F --> G["Final Authorization Gate"]
    G --> H["Context Construction"]
    H --> I["LLM Generation"]
    I --> J["Deterministic Sources"]
    J --> K["Answer"]

    Q["User Query + Bounded Conversation"] --> D
```

The important security boundary is between retrieval and generation:

```text
Identity
   |
   v
Permission Engine
   |
   v
Authorized Retrieval Scope
   |
   v
Qdrant Hybrid Search
   |
   v
Post-Retrieval Authorization Check
   |
   v
Reranking
   |
   v
Final Authorization Gate
   |
   v
Authorized Context
   |
   v
LLM
```

### Workflow Reference

![Permission-aware RAG workflow](images/Permission_aware_rag_workflow.png)

---

## Authorization Model

The authorization layer combines role-based and attribute-based rules.

Access decisions consider:

- Role
- Department
- Clearance level
- Seller scope
- Document type
- Document classification
- Organization-wide document rules

Clearance controls the maximum classification a user can access. Seller scope controls seller-specific material. The role matrix controls which document types a role can read. Organization-wide restricted material can additionally require department alignment.

Authorization is implemented centrally in the `PermissionEngine`.

### Permission Matrix

| User | Department | Role | Clearance | Seller Scope |
|---|---|---|---|---|
| Aditya Verma | Business | Account Manager | CONFIDENTIAL | S001, S003 |
| Rahul Sharma | Support | Support Engineer | CONFIDENTIAL | S001, S002 |
| Vikram Singh | Engineering | Software Engineer | CONFIDENTIAL | S001, S002 |
| Neha Gupta | Operations | System Engineer | RESTRICTED | S002, S004 |
| Admin | IT | Platform Administrator | RESTRICTED | All sellers |

The UI displays the effective seller scope returned by the server. A caller can narrow its requested scope, but cannot use the request to expand its authorization.

![Permission matrix](images/permission_matrix.png)

---

## Retrieval Pipeline

### 1. Document Ingestion

The ingestion pipeline recursively discovers PDF and TXT documents, extracts content, derives metadata, validates metadata, cleans the text, and creates token-aware chunks.

Current dataset:

- 63 documents
- 52 PDF files
- 11 TXT files
- 220 generated chunks in the published visibility-projected revision
- 0 failed documents
- 0 validation errors

Chunking uses:

- `RecursiveCharacterTextSplitter`
- `cl100k_base` tokenizer
- 800-token target size
- 120-token overlap
- Page-level provenance for PDF content

Metadata is preserved with every chunk so authorization decisions can be made after retrieval.

### 2. Indexing

Each chunk is indexed in Qdrant with both dense and sparse representations.

**Dense retrieval**

- Jina `jina-embeddings-v4`
- 2048-dimensional vectors
- Cosine similarity

**Sparse retrieval**

- Local BM25 encoder
- `k1 = 1.2`
- `b = 0.75`

**Hybrid retrieval**

- Dense top-k and sparse top-k candidates
- Reciprocal Rank Fusion
- Final retrieval pool before authorization-aware answering

The published index contains 220 Qdrant points. The 2026-10-31 read-only verification recorded corpus `corpus-584718318df8d652ff66-5ed2d2e1` and collection `enterprise_knowledge__r_9d180aecc5e11fd7ac7e_db6b6fe9`; these identify the verified artifacts, rather than a new publication in this change.

### 3. Permission-Aware Retrieval

The permission engine converts the identity into a Qdrant filter containing:

- Allowed classifications
- Allowed document types
- An explicit seller-scoped branch
- An explicit organization-wide branch, with department matching for restricted organization documents

The Qdrant prefilter represents these alternatives as boolean branches. Independent permission checks still run after retrieval and again before context construction. Malformed or unknown authorization metadata is denied, and request-supplied filters never grant access.

The system deliberately fails closed. An invalid identity, invalid scope, or unusable authorization filter does not silently turn into an unrestricted search.

### 4. Authorization Re-check

Candidate chunks are passed through the central permission engine again.

This provides defense in depth:

```text
Qdrant pre-filter
      |
      v
Candidate chunks
      |
      v
PermissionEngine.can_access()
      |
      v
Authorized chunks only
```

The LLM boundary is protected by an explicit invariant: an unauthorized chunk must never be included in the context sent for generation.

### 5. Reranking

Authorized candidates are reranked using Jina's multilingual reranker.

Reranking changes relevance ordering; it does not grant access.

A final authorization gate runs after reranking so that ranking can never reintroduce unauthorized content.

### 6. Context Construction and Citations

Personal-portfolio questions ("my sellers", "my accounts", "my portfolio", and
multilingual equivalents/paraphrases) always use the current authenticated
principal's server-derived seller scope and per-seller evidence retrieval,
regardless of `MULTI_SELLER_RETRIEVAL`. Request identity attributes and conversation
claims cannot supply or expand this inventory. Follow-ups resolve only their
topic from bounded prior user questions and recompute scope for the current
principal. Assistant history is omitted from portfolio generation.

Direct personal-language cues are resolved locally; other wording uses a strict
multilingual intent classification through the existing model adapter. The
classifier returns only `portfolio` or `other`, never identity attributes or
sellers. Invalid, truncated, refused or unavailable classification fails before
retrieval. This extra stage shares the API request deadline, adds a bounded model
request for questions without direct personal cues, and has not been evaluated
with a live provider.

Portfolio evidence requires seller-scoped provenance. Shared organization-wide
chunks are conservatively omitted from portfolio context; whole seller passages
mentioning a known account outside the effective scope are also omitted without
rewriting corpus text, ACLs or citation offsets. Seller ID/name aliases are
server-owned in `answering/src/answer_scope.py`, including Unicode and spacing
normalization. Generated headings, tables, text and source inventory are checked
for excluded seller references. Accounts without included evidence get an
application-generated unavailable note, including when all evidence is absent.
Portfolio streams hold model text until this validation completes.

Explicit organization-wide questions follow existing document permissions and
policy. The prior ORG-001 commercial-fact restriction remains unchanged. Whether
other shared seller facts should become portfolio evidence requires reviewed,
trusted seller attribution at a passage/fact boundary; shared readability alone
does not establish that attribution. This change publishes no corpus or index.
The conservative omission can reduce portfolio evidence coverage. The alias
guard does not establish semantic grounding or recognize every possible
translated/invented alias; multilingual classifier accuracy and answer quality
require a comparable evaluation before any improvement is claimed.

For explicit multi-account questions without personal-portfolio intent, opt in
with `MULTI_SELLER_RETRIEVAL=seller_balanced`
(restart the API after changing configuration). The default `global` retains the
existing hybrid baseline. Routing uses only the current question and the
server-resolved principal's scope, intersected with any requested seller filter;
request identity fields, UI state, and conversation claims cannot add sellers.
Single-account and ambiguous questions retain the global path.

The balanced path searches seller-only partitions under the existing permission
policy, sharing one query embedding and resolved immutable index revision. It
divides the existing dense, sparse, and answering candidate limits across sellers,
shares one aggregate retrieval deadline, reranks each seller separately under one
reranker deadline, and reserves
one surviving passage per seller before filling remaining `ANSWER_TOP_K` slots.
Independent post-retrieval and final authorization checks remain in place.
Context packing divides `MAX_CONTEXT_TOKENS` and `MAX_CONTEXT_CHARS`, counting
seller headers and separators. Small budgets or thresholds can still omit evidence;
no limits are increased to force coverage. Organization-wide passages remain in the
global baseline and do not consume the balanced seller slots.

When only some sellers have included evidence, generation is instructed to answer
the supported parts and the application explicitly marks other in-scope sellers
as unavailable in the included sources. This is a completed answer with limited
coverage; the existing `partial` outcome still means generation truncation.
An empty context retains the insufficient-information response (personal portfolio
responses also name each in-scope account lacking included evidence). Evidence presence
does not prove that every requested fact is supported. Offline mocked regressions
run with `python -B scripts/run_offline_suite.py security_tests.seller_balanced_retrieval`
and `security_tests.portfolio_answer_scope`; the wrapper rejects external network
requests while permitting the loopback HTTP regressions. No comparable
provider-backed quality or latency benchmark has been run for this path. Compare
both configurations on the same frozen corpus/index, identities, questions,
budgets, and provider settings before drawing quality conclusions. Per-seller
context coverage and end-to-end latency, groundedness, abstention, and leakage
remain benchmark work.

Only authorized chunks that fit the context budget are included in the final context.

Sources are generated deterministically from those same authorized chunks. The application adds the source list after generation, rather than allowing the model to invent page numbers or citations.

Sparse retrieval uses BM25 with positive Robertson/Sparck Jones IDF, `k1=1.2`, `b=0.75`, and linear query-term frequency. Document vectors carry IDF-weighted BM25 term weights; query vectors carry term frequency only, so the dot product applies IDF once. Tokenization normalizes Unicode hyphens and underscores to ASCII hyphens. Retrieval loads the vocabulary from the resolved immutable revision, validates it against that revision's manifest, and filters indexed points by its sparse fingerprint. Legacy compatibility artifacts are not used for active retrieval.

Context packing uses the configured `MAX_CONTEXT_TOKENS` and counts passage headers and separators. The chosen hosted model has no local tokenizer mapping available here, so the current offline counter uses UTF-8 bytes as a conservative upper bound on tokens; `MAX_CONTEXT_CHARS` remains an additional cap. The model receives stable evidence IDs, but the UI lists available included evidence without claiming a validated claim-to-source relationship. Reranker thresholds remain provisional; no held-out quality or latency benchmark has been run.

New chunks preserve chunk-relative page spans. Citation pages are derived after overlap trimming and excerpting from the exact included ranges. Legacy multi-page excerpts without page offsets omit page numbers because precise attribution cannot be verified; publishing page spans for the existing corpus requires a new immutable build.

---

## Conversation Safety

Conversation history is treated as untrusted input.

It can help resolve references in follow-up questions, but it cannot:

- Change the user's identity
- Change clearance
- Expand seller scope
- Grant a role
- Override authorization decisions

The retrieval filter is always derived from the server-side identity.

This is important because a previous assistant response or a statement inside a conversation is not an authorization credential.

---

## Grounded Answer Generation

The answering layer receives only the authorized context.

The system prompt requires the model to:

- Use only supplied context
- Avoid inventing enterprise facts
- State when the context is insufficient
- Preserve identifiers and figures from the source
- Avoid generating its own source list
- Treat question, conversation, and retrieved documents as untrusted data

The current configured generation model is `openai/gpt-oss-120b` through the Groq-compatible API.

The answering service also supports streaming responses through the `/query/stream` endpoint.

---

## Application Architecture

```mermaid
flowchart LR
    U["React UI"] --> API["Answering HTTP API"]

    API --> ID["Identity Resolution"]
    ID --> PE["Permission Engine"]

    PE --> RET["Permission-Aware Retrieval"]
    RET --> Q["Qdrant"]

    RET --> AUTH["Authorization Re-check"]
    AUTH --> RR["Jina Reranker"]
    RR --> GATE["Final Authorization Gate"]

    GATE --> CTX["Context Builder"]
    CTX --> LLM["LLM"]

    LLM --> SRC["Verified Source Builder"]
    SRC --> API
    API --> U

    DATA["PDF / TXT Dataset"] --> ING["Ingestion"]
    ING --> CH["Chunks + Metadata"]
    CH --> IDX["Indexing"]
    IDX --> Q
```

---

## Real Application UI

The React workspace exposes the permission-aware workflow:

1. Select a prototype identity.
2. View the identity's effective seller scope.
3. Ask a question about authorized enterprise material.
4. Retrieve and rerank authorized documents.
5. Generate a grounded answer.
6. Inspect the cited sources.
7. Review saved evaluation results.


---

## Evaluation

The repository contains a reviewed v1.2 golden evaluation dataset with 90 cases covering:

- Authorized factual questions
- Multi-document retrieval
- Cross-seller queries
- Follow-up questions
- Unauthorized access attempts
- Mixed-access cases
- Policy and governance questions
- Ambiguous questions
- Prompt-injection attempts

The dashboard defaults to [the latest results by case](evaluation/runs/latest-cases-20261111.json), saved on 11 November 2026. It combines the latest recorded outcome for each of 23 targeted cases across three provider-backed runs. Earlier passing cases keep their original run context; this is not a new 90-case baseline. Individual reports remain selectable.

| Metric | Current 23-case snapshot |
|---|---:|
| Passed | 14 |
| Failed | 9 |
| Errors / pending | 0 / 0 |
| Recall@5 | 88.04% |
| Precision@5 | 27.83% |
| MRR | 80.07% |
| Groundedness | 100.00% |
| Answer relevancy | 98.26% |
| Answer correctness | 67.39% |
| Security pass rate | 100.00% |
| Answer leakage rate | 0.00% |

All 23 cases have measured quality scores. The latest execution covered 13 cases once each: 4 PASS, 9 FAIL, 0 ERROR. Failures were preserved with their causes and were not repeatedly retried. See the [before/after comparison and failure explanations](evaluation/runs/remaining-and-failed-20261111-summary.json).

Current PASS/FAIL measures answer quality and security. Retrieval remains a separate diagnostic. Thresholds are groundedness 70%, correctness 60%, and relevancy 60%; retrieval Recall@5 has a separate 50% threshold. The historical 90-case v1.1 baseline (70 PASS / 20 FAIL) used combined retrieval/answer/security status and earlier expectations, so its pass rate is not directly comparable.

The saved snapshot preserves source-report hashes and per-case provenance. It can be rebuilt for a new publication with `scripts/publish_evaluation_snapshot.py`; it reads reports only and cannot be resumed as a live evaluation. Publish a new snapshot with a new name and the original report IDs:

```powershell
.\.venv\Scripts\python.exe -B scripts/publish_evaluation_snapshot.py --name latest-cases-20261111 --source runs/failed-cases-live-20261106-targeted.json --source runs/execution-fixes-20261106-targeted.json --source runs/failed-cases-rerun-20261111-targeted.json
```

The existing snapshot is preserved; choose a new `--name` for a later publication.

The evaluation dashboard reads saved results and does not automatically start a quota-consuming evaluation run from the web UI.

---

## Environment

Copy `.env.example` to `.env` and provide the required service credentials.

Core variables include:

```env
JINA_API_KEY_1=
JINA_API_KEY_2=
# Legacy JINA_API_KEY is also supported when no numbered keys are configured.
QDRANT_URL=
QDRANT_API_KEY=

GROQ_API_KEY_1=
GROQ_API_KEY_2=
GROQ_API_KEY_3=
GROQ_API_KEY_4=
GROQ_API_KEY_5=
```

The repository also supports configuration for embedding, reranking, retrieval, generation, API, conversation limits, and evaluation settings. See `.env.example` for the complete configuration surface.

Never commit real API keys.

---

## Local Setup

### 1. Create the environment

```bash
py -3.14 -m venv .venv
```

Activate the environment and install dependencies:

```bash
pip install -r requirements.lock
```

### 2. Configure environment variables

```bash
copy .env.example .env
```

Fill in the required credentials.

### 3. Run ingestion

From the repository root:

```bash
python ingestion/run_ingestion.py
```

This generates the processed document catalog, chunks, and ingestion reports.

### 4. Build the Qdrant index

```bash
python indexing/pipeline.py
```

This embeds the active corpus revision and builds a new immutable Qdrant collection. The job validates exact point IDs and count, ACL metadata against the reviewed permission fixture, embedding and sparse configuration, and a local manifest before atomically changing the `<QDRANT_COLLECTION_NAME>_active` alias. Failed builds leave the prior alias intact. The existing published revision was verified on 2026-10-31; no new index was published for the subsequent issue fixes.

### 5. Start the answering API

```bash
python answering/serve.py
```

The local API listens on:

```text
http://127.0.0.1:8000
```

### 6. Start the React UI

Install Node.js 24 or newer, then run in another terminal:

```bash
cd frontend
npm ci
npm run dev
```

The UI opens at `http://127.0.0.1:8501`. Its components, styles, and HTTP transport live in `frontend/src/`. The public demo directory is cached for five minutes per API URL. Restored tab sessions are verified by the backend before chat is enabled. **Check service** runs diagnostics on demand. Saved evaluations load when opened; the latest report refreshes every 30 seconds while visible, and historical selections stay fixed.

For an API at another address, copy `frontend/.env.example` to `frontend/.env.local` and set `VITE_API_BASE_URL`. Configure `API_ALLOWED_ORIGINS` on the backend with the frontend's exact origin.

For deployment, use `frontend` as the project root, `npm run build` as the build command, and `dist` as the output directory. `frontend/vercel.json` provides the SPA route fallback. Set `VITE_API_BASE_URL` to the deployed API's HTTPS URL before building.

### Windows quick start

For a one-command local run on Windows, use the included `run.bat`:

```bat
run.bat
```

Install the Python dependencies and run `npm ci` in `frontend` first; `.venv` must exist and `.env` must be configured. The script starts a fresh API/UI pair on loopback, choosing available ports near 8000 and 8501 and leaving existing services running. It waits for both services, sets the UI's `VITE_API_BASE_URL` and the backend's allowed origin to the selected ports, and opens the application in your browser.

Both services run in the background. Their URLs, process IDs, and log paths are saved in `.cache/launcher.json`; the console also prints the application URL. Use `run.bat --no-browser` to start without opening a tab, or `run.bat --api-port 8141 --ui-port 8511` to choose different preferred ports. Startup failures print the relevant log path and clean up only processes started by that invocation. Startup checks do not run model queries or rebuild the index.

---

## API Surface

The answering service exposes:

| Method | Endpoint | Purpose |
|---|---|---|
| GET, HEAD | `/ping` | Tiny uptime probe: HTTP 200, plain-text `ok` (2 bytes for GET); no database/model calls |
| GET | `/health` | Service health check |
| GET | `/live` | Process liveness only |
| GET | `/ready` | Local configuration, Qdrant alias, and active manifest readiness (no model provider calls) |
| GET | `/users` | Authenticated identity directory |
| POST | `/query` | Generate a complete answer |
| POST | `/query/stream` | Stream an answer through SSE |
| GET | `/evaluation` | Minimal saved report summary and run listing |
| GET | `/evaluation/matrix` | Detailed report; authenticated operator/admin and feature flag required |
| GET | `/demo/users` | Demo identity directory (`API_DEMO_MODE=true`; public by default unless overridden) |
| POST | `/demo/login` | Validate a demo password and issue a one-hour bearer session |
| POST | `/demo/query`, `/demo/query/stream` | Demo persona query; requires the session from `/demo/login` |

Evaluation execution is intentionally kept outside the HTTP request path and is run explicitly from the evaluation CLI.

For the Render uptime cron job, deploy the backend change, then use
`https://permissionawarerag-t80d.onrender.com/ping` with method **GET** every
**14 minutes** and expect HTTP **200** with body **ok**. No authorization header
or request body is required. This is a liveness probe, not an index-readiness
check. A sleeping Render service can serve its own wake-up page before Python
is running; this endpoint cannot shrink that platform-generated response or
guarantee uninterrupted uptime. If that occurs, wake the service once, verify
`/ping` returns `ok`, then retest the cron job. The static React frontend is hosted separately from the Python API.

Query routes require a backend-issued demo session; request-body `user_id` values cannot select or change the persona. For local use, set `API_DEMO_MODE=true` and keep `API_HOST=127.0.0.1`; set `API_PUBLIC_DEMO_MODE=false` to explicitly keep demo routes local. For a deployed synthetic-data demo, set `API_DEMO_MODE=true` and `API_HOST=0.0.0.0` on the API service, then set the frontend's `VITE_API_BASE_URL` to the API's HTTPS public URL and allow the frontend origin through `API_ALLOWED_ORIGINS`. Public demo mode follows the demo setting by default. Passwords follow `firstname@login` (for example, `aditya@login`) and the backend returns a one-hour session. This demo login is only for synthetic data. Public demo requests share a global per-process budget. `EVALUATION_REPORTS_ENABLED=true` only enables the detailed route after operator authorization.

The answering service also needs the immutable revision bundle matching the Qdrant active alias. The deployed bundle is stored under `indexing/revisions/` and includes its `manifest.json` and `sparse_vocab.json`; keep both files in the deployment checkout whenever the alias points to that collection. They are runtime inputs for revision validation and sparse retrieval, not disposable build output. If a new Qdrant revision is published, deploy its matching manifest and vocabulary with the service. Without them the API fails closed with `Active Qdrant alias has no valid local manifest`.

Authorization evaluation expectations are stored in the independently reviewed `access_control/fixtures/expected_access_matrix.json`, rather than generated from the production permission engine. Reports track candidate, post-filter, context, citation, and answer-level exposure separately. Each run records and fingerprints the resolved collection, manifest, revision vocabulary, and emergency revocations, pins that revision for retrieval, and stops if the active state changes. Resume requires an identical input/configuration fingerprint; attempts retain their own configuration and timestamps, and changed parent answers invalidate dependent follow-up results. Detailed traces, including denied candidate IDs, are restricted to authenticated operators.

Buffered and streaming generation use explicit `answer`, `partial`, `refusal`, and `error` outcomes. A stream is complete only after valid provider finish metadata and the terminal marker; output-limit truncation is reported as partial. The UI renders the canonical terminal payload and does not treat provisional deltas as the final answer.

**E035 policy decision:** Seller-identifying commercial and financial facts in ORG-001 are seller-restricted, consistent with ORG-001 page 4 and the user's decision. Ingestion now projects the page-5 seller-commercial section out of shared ORG-001 chunks; seller-scoped account records retain the facts. The published 220-point revision implements this projection. ACL/prefilter decisions are represented in the reviewed access fixture and covered by offline security checks. E035's golden refusal remains unchanged. No fresh model-generated E035 benchmark is claimed.

---

## Technology Stack

| Layer | Technology |
|---|---|
| UI | React 19, TypeScript, Vite |
| API | Python HTTP server |
| Ingestion | Python, pypdf |
| Chunking | LangChain text splitters, tiktoken |
| Dense embeddings | Jina Embeddings v4 |
| Sparse retrieval | Local BM25 |
| Vector database | Qdrant |
| Hybrid fusion | Reciprocal Rank Fusion |
| Reranking | Jina Reranker |
| LLM | `openai/gpt-oss-120b` via Groq-compatible API |
| Evaluation | Python-based golden dataset and judge pipeline |

---

## Phase 4 Reliability Controls

Ingestion writes a content-fingerprinted immutable corpus revision and advances a local `ingestion/processed/active.json` pointer only when extraction, metadata, source stability, and integrity checks pass. Compatibility JSONL files are still written for existing tools; index builds read the active immutable revision when one exists. Prior revisions are retained. Unknown seller scope, conflicting seller/classification labels, unknown document types, unreadable encrypted PDFs, extraction failures, and configured file/page/text limit violations fail or quarantine the document. Positional header/footer removal requires repeated edge placement; repeated body headings are retained. Chunk records carry page numbers, absolute text offsets, detected section spans, and metadata schema version.

Each Qdrant build uses a unique immutable collection. It records corpus/document/chunk fingerprints, model and vector dimensions, sparse vocabulary fingerprint, schema version, and exact point IDs. The Qdrant active alias is switched only after validation. Deletion, shortening, and ACL restriction therefore require building a new exact point set; old revisions remain stored and are not rollback eligible after the corpus or ACL fingerprint changes. A rollback is allowed only to a complete revision with the same corpus/ACL fingerprint. Urgent document revocation uses the trusted local `indexing/revisions/revoked_documents.json` denylist, which is read on each retrieval and merged into both dense and sparse filters; malformed revocation data fails retrieval closed. Retained inactive revisions need an operator-managed retention and physical-erasure process before sensitive data can be considered erased.

The index uses the alias/manifest lifecycle. The 2026-10-31 review verified `enterprise_knowledge_active` resolving to `enterprise_knowledge__r_9d180aecc5e11fd7ac7e_db6b6fe9`, all 220 point IDs and payload/ACL hashes, and all 315 expected user/document filter decisions. `/ready` checks configuration and the alias/manifest without model-provider calls. `/health` and `/live` indicate liveness. The deployed sample application uses backend-validated demo sessions and synthetic data.

The API starts its request deadline before parsing headers, applies absolute header/body budgets and bounded JSON/schema validation, limits concurrent handlers, and charges request/token windows by verified provider subject. Socket shutdown interrupts trickling input and active generation provider I/O at the deadline. Generation checks the deadline on stream frames and before accepting completion; nested stages cannot restart the budget. Safe limits are configurable with `API_*` in `.env.example`. Dense and sparse retrieval may run concurrently against the same pinned revision and server-derived authorization filters. Embedding retries are bounded and limited to network, timeout, rate-limit, and server failures; permanent HTTP errors are not retried. Reranker failure falls back to authorized fusion order and is recorded as degraded mode.

Operator-only evaluation traces include a stable request ID, retrieval/authorization, rerank, context and generation timing, selected evidence references, policy version, index revision, provider finish reason, and available token usage. Denied point IDs remain restricted to authenticated operators. The local authorization audit log contains identities, IDs, decisions and reasons but not query text or passage contents; it rotates at 10 MiB and retains one prior file. On sink I/O failure, the error is logged and authorization still follows the already-computed decision. This local file is not a tamper-proof or centralized audit service.

### Offline regression suite

Supported local runtime: Python 3.14.6 (`.python-version`). Backend dependencies are pinned in `requirements.lock`. Run `scripts/test_offline.ps1` on Windows for the backend authorization, answer quality, reliability, evaluation, API contract, and liveness suites. CI runs the same modules through `scripts/run_offline_suite.py`, which rejects external network requests and permits isolated loopback HTTP tests. In `frontend`, run `npm test`, `npm run build`, and `npm run test:e2e` for the React unit tests, production checks, and desktop/mobile browser coverage. Install the browser once with `npx playwright install chromium`.

Runtime audit logs, generated ingestion/indexing reports, local revision pointers, and temporary inspection artifacts are ignored by Git. The checked-in corpus, permission fixtures, evaluation reports, and published index manifest/vocabulary remain inputs to the backend and its regression suite.

---

## Security Design Principles

### Authorization is not a prompt

The model is never asked whether a user should have access to a document. The permission engine makes that decision before context construction.

### Relevance is not authorization

A semantically relevant document can still be inaccessible. Retrieval therefore operates inside an authorization-derived scope.

### Reranking does not grant access

Reranking can change ordering, but every selected chunk is checked again.

### Conversation is not identity

Previous messages cannot change the server-side identity or permissions.

### Fail closed

Invalid identities and invalid authorization scopes are rejected instead of being converted into unrestricted retrieval.

### Citations come from authorized evidence

The application constructs the final source list from the chunks that were actually authorized and included in context.

---
