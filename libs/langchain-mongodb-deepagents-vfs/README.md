# langchain-mongodb-deepagents-vfs

**MongoDB Atlas-backed virtual filesystem search adapter for LangChain DeepAgents.**

`langchain-mongodb-deepagents-vfs` implements DeepAgents' `BackendProtocol`, routing `grep`, `glob`, and `ls` through MongoDB Atlas (vector search + full-text search + hybrid `$rankFusion`) while forwarding all other file operations (`read`, `write`, `edit`, `upload_files`, `download_files`) directly to S3.

## Install

```bash
pip install langchain-mongodb-deepagents-vfs
```

### Embedding providers

The adapter supports two embedding providers via optional extras:

```bash
# AWS Bedrock (default — uses boto3 credential chain, no extra API key needed)
pip install "langchain-mongodb-deepagents-vfs[bedrock]"

# OpenAI
pip install "langchain-mongodb-deepagents-vfs[openai]"
```

## Quickstart

```python
from langchain_mongodb_deepagents_vfs import MongoFilesystemBackend

# Non-blocking: index provisioning, initial sync, and the watcher
# all start in a background thread. grep/glob/ls block until ready.
backend = MongoFilesystemBackend(
    s3_bucket_name="my-docs-bucket",
    mongodb_connection_string="mongodb+srv://user:pass@cluster.mongodb.net/",
)

# Every method returns the result type declared by
# `deepagents.backends.protocol` — GrepResult, GlobResult, LsResult, …

# Hybrid search (full-text + vector via $rankFusion)
result = backend.grep("authentication flow", path="docs/")
for match in result.matches or []:
    # Plain GrepMatch dicts, most relevant first (rank is the list order)
    print(match["path"], match["line"], match["text"][:80])

# Glob by path pattern → FileInfo dicts. Standard glob semantics: the pattern
# is relative to `path`, and "*" does not cross "/" — use "**/*.pdf" to recurse.
pdfs = backend.glob("*.pdf", path="reports/")
print([m["path"] for m in pdfs.matches or []])

# List directory → FileInfo dicts (directories end in "/" and set is_dir)
ls = backend.ls("docs/")
for entry in ls.entries or []:
    print(entry["path"], "DIR" if entry["is_dir"] else "FILE")

# Pass-through file operations
backend.write("docs/new.txt", "hello world")
content = backend.read("docs/new.txt").file_data["content"]

# Conditional in-place edit (ETag-verified read-modify-write)
backend.edit("docs/new.txt", old_string="hello", new_string="goodbye")

# Graceful shutdown (or use as a context manager)
backend.stop()
```

### Context manager

```python
with MongoFilesystemBackend(
    s3_bucket_name="my-docs-bucket",
    mongodb_connection_string="mongodb+srv://...",
) as backend:
    result = backend.grep("authentication flow")
# watcher is stopped automatically on exit
```

## Configuration

| Parameter | Required | Default | Description |
|---|---|---|---|
| `s3_bucket_name` | Yes | — | S3 bucket name |
| `mongodb_connection_string` | Yes | — | Atlas connection string |
| `embedding_model` | No | `BedrockEmbeddings(titan-embed-text-v2:0)` | Any LangChain `Embeddings` instance |
| `embedding_dimensions` | No | `1024` | Vector dimensions; must match the model |
| `llm` | No | — | Reserved for future agent LLM integration |
| `watcher` | No | `"polling"` | `"polling"` or `"sqs"` |
| `sqs_queue_url` | If `watcher="sqs"` | — | Full SQS queue URL |
| `aws_region` | No | `AWS_DEFAULT_REGION` env, then the boto3 chain | AWS region for S3, SQS **and Bedrock embeddings** — all three resolve it the same way, so they cannot end up split across regions |
| `s3_prefix` | No | `""` | Only sync/watch objects under this S3 prefix |
| `debug` | No | `False` | Re-raise exceptions instead of returning error DTOs (local dev) |

### Embedding provider selection

The provider is resolved at construction time from the `EMBEDDING_PROVIDER` environment variable (default `bedrock`). Override the model with `EMBEDDING_MODEL`.

| `EMBEDDING_PROVIDER` | Default model | Required credential |
|---|---|---|
| `bedrock` (default) | `amazon.titan-embed-text-v2:0` | boto3 credential chain (IAM role, `~/.aws/credentials`, etc.) |
| `openai` | `text-embedding-3-small` | `OPENAI_API_KEY`, or `AZURE_OPENAI_ENDPOINT` for Azure OpenAI |

`openai` serves both OpenAI and Azure OpenAI — there is no separate `azure`
provider, matching the convention in the sibling packages of this monorepo.
Azure is selected when `AZURE_OPENAI_ENDPOINT` is set and `OPENAI_API_KEY` is
not; if both are present, `OPENAI_API_KEY` wins.

With `bedrock`, the region comes from `aws_region`, then `AWS_DEFAULT_REGION`,
then the boto3 chain. There is no hardcoded fallback: if no region resolves you
get an explicit `NoRegionError` rather than a silent default that may not have
the model enabled.

Pass a `LangChain Embeddings` instance directly to `embedding_model` to bypass provider lookup entirely.

## Why not `MongoDBStore` + `StoreBackend`?

DeepAgents already ships [`StoreBackend`](https://github.com/langchain-ai/deepagents), which implements the same `BackendProtocol` on top of LangGraph's `BaseStore`. Since `langgraph-store-mongodb`'s `MongoDBStore` *is* a `BaseStore`, this works today and needs no new package:

```python
from deepagents.backends import StoreBackend
from langgraph.store.mongodb import MongoDBStore

backend = StoreBackend(store=MongoDBStore.from_conn_string(...))
```

The two are close cousins — LangGraph's Store is already filesystem-shaped (hierarchical namespaces, keys, prefix search), and `StoreBackend` maps a file path to a single Store key, much as this package maps it to `source_path`. So the difference is not the data model; it is **where search runs** and **how large a file can be**.

Note that `MongoDBStore` itself is not the limiting factor on search: `MongoDBStore.search(query=...)` performs Atlas Vector Search, with optional reranking. The gap is that `StoreBackend` never calls it that way — its `grep` loads items and matches literally in Python, so the Store's semantic search goes unused. The rows below therefore describe the pairing as `StoreBackend` wires it up, not the ceiling of what `MongoDBStore` can do.

| | `MongoDBStore` + `StoreBackend`                                                                          | This package |
|---|----------------------------------------------------------------------------------------------------------|---|
| `grep` / `glob` execution | `StoreBackend` fetches every item in the namespace, then filters in Python                               | runs as an aggregation inside MongoDB; only matches come back |
| `grep` matching | literal substring — `StoreBackend` does not pass `query` to `MongoDBStore.search()`, so no vector search | hybrid `$rankFusion` — full-text **and** vector, so natural-language queries work |
| Unit of storage | one Store item per file, read and written whole                                                          | file chunked into 512-token documents |
| Large files | whole file must fit in memory on every `grep`                                                            | only matching chunks are returned; `line_start` gives real line numbers |
| Where bytes live | in MongoDB, as the Store value                                                                           | in S3; MongoDB holds only chunks and embeddings |
| Source of truth | MongoDB                                                                                                  | the S3 bucket — objects written by other tools are picked up automatically |

The chunking is the part that cannot be retrofitted onto `StoreBackend`: it depends on one Store item per file path, and splitting a file into many items would break that identity. Chunking is also what makes semantic `grep` useful, since embedding a whole document into one vector loses the passage-level detail an agent is searching for.

**Use `MongoDBStore` + `StoreBackend`** for an agent scratchpad — modest numbers of small, agent-authored files, where exact-match `grep` is fine and you want writes and reads in one system.

**Use this package** when the corpus is large, pre-existing, and lives in S3 (documents, PDFs, spreadsheets), and the agent needs to find things by meaning rather than by exact string.

### A note on search freshness

Neither option gives read-your-writes for search. Atlas Search and Vector Search are eventually consistent by design: `mongod` replicates to the `mongot` process, which indexes asynchronously, so any newly written document becomes searchable a short time after the write acknowledges. Point reads are unaffected — `read` here goes straight to S3, and a `MongoDBStore.get()` (or a `search()` with only a `filter` and no `query`) is a normal MongoDB query.

This package adds two further hops on top of that: a `write` lands in S3, and the object is only chunked and embedded once the watcher notices it — up to the poll interval (10s default) with `PollingWatcher`, or near real-time with `SQSWatcher`. Budget for `write` → `grep` lag on the order of the watcher interval plus Atlas indexing time, and don't rely on a file being greppable immediately after writing it.

## Design Decisions

### Non-blocking constructor

`MongoFilesystemBackend.__init__` returns immediately. Index provisioning, initial sync, and the background watcher all start in a single daemon thread. Search methods (`grep`, `glob`, `ls`) block internally until the first sync completes, so the first call may take longer than subsequent ones.

### Supported file formats

The chunker extracts text from the following formats before embedding:

| Extension | Parser |
|---|---|
| `.txt`, `.md`, `.rst`, `.csv` | UTF-8 decode |
| `.pdf` | pypdf (layout mode + plain fallback) |
| `.docx` | python-docx |
| `.xlsx` | openpyxl (one page per sheet) |
| `.xls` | xlrd (one page per sheet) |
| `.pptx` | python-pptx (one page per slide, includes speaker notes) |
| `.ppt` | olefile OLE stream scan (best-effort) |

Format is detected by file extension; magic bytes are used as a fallback for extensionless objects.

### Chunking strategy

- **512 tokens / chunk, 64-token overlap** (tiktoken `cl100k_base`)
- Each chunk stores `source_path`, `chunk_index`, `page_number`, `char_start`, `char_end`, `line_start`
- `line_start` is what makes `grep` return DeepAgents-compatible `GrepMatch.line`

### Embedding model

- **AWS Bedrock `amazon.titan-embed-text-v2:0` @ 1024 dimensions** — default; uses the boto3 credential chain, no extra API key needed
- **OpenAI `text-embedding-3-small` @ 1024 dimensions** — alternative; 10× cheaper than `ada-002`, strong MTEB benchmarks
- 1024 dims balances semantic fidelity with storage cost at 100k-document scale

### Search

`grep` uses `$rankFusion` (equal 0.5/0.5 weights) combining:
- **Atlas Full-Text Search** (`lucene.standard` on `content`)
- **Atlas Vector Search** (cosine similarity on `embedding`)

If the embedding API is unavailable at query time, `grep` falls back to full-text only. On non-Atlas MongoDB (mongomock, community server), `grep` falls back to a regex query.

`glob` has a single implementation on every cluster. It matches with [`wcmatch`](https://facelessuser.github.io/wcmatch/) using the same flags as DeepAgents' own backends, so it follows the standard glob semantics the `BackendProtocol` documents: `*` stays within one path segment, `**` recurses, and `?`, `[abc]`, `{a,b}` all work. Atlas Search's `wildcard` operator cannot express `**`, `[abc]` or `{a,b}`, so matching identically everywhere means matching in Python; only one document per distinct key is fetched, making the cost O(files) rather than O(chunks).

### Watcher

- **PollingWatcher** (default): ETag diff on a 10-second interval, zero AWS infra required. Backs off gracefully on S3 list failures.
- **SQSWatcher** (production): S3 event notifications via SQS long-polling (20s), near real-time. Requires S3 → SQS event notifications configured in AWS.

### Object size limit

Every read is capped at **64 MiB** (`MAX_READ_BYTES` in
`langchain_mongodb_deepagents_vfs.backends.base`). The cap is enforced inside
`S3Backend.read()`, so it applies uniformly to `read`, `edit`,
`download_files`, the initial sync and the watchers — not just the public
`read` API. Ingest paths run unattended across a thread pool, so an unbounded
read there is a memory-exhaustion vector rather than merely a slow query.

`write` and `upload_files` refuse content over the same limit: accepting a
larger write would store an object this backend could then only ever fail to
read.

An oversized object is skipped rather than fatal — the rest of the corpus
still indexes. During the initial sync it is counted in
`SyncReport.failed`; in the watchers it is logged and skipped, with no
counter, so the log is the only record.

Chunking applies its own bounds on top of the read cap: OOXML archives are
checked for entry count, uncompressed size and expansion ratio before
parsing, and PDF extraction is bounded by page count and total extracted
characters.

### ETag-based idempotency

Initial sync and every watcher ingest pass are idempotent: objects whose ETag hasn't changed since the last run are skipped. Restarting the backend after a partial failure resumes cheaply without re-embedding unchanged files.

## Error Handling

Every public method returns a DTO. If an error occurs, the `error` field contains a stable `ErrorCode` and a human-readable message — no raw stack traces ever surface to the caller.

```python
result = backend.grep("query")
if result.error:
    print(result.error)  # "[E5001] The grep search operation failed. Detail: ..."
```

Enable `debug=True` during local development to re-raise the original exception with a full traceback instead.

Every error carries a stable `[EXXXX]` code from the `ErrorCode` enum in `langchain_mongodb_deepagents_vfs.errors`.

### Checking initialization outcome

Index provisioning, initial sync and watcher startup all run in a background
thread, so their failures cannot be raised from the constructor. They are
recorded instead.

A search call returning guarantees the index and sync stages have finished —
the readiness gate is released between the sync and the watcher start, so
those two entries are always present by then. A watcher-start failure is
recorded a moment later and can be missed by a check made immediately after
the first search.

```python
backend.grep("anything")          # blocks until initialization finishes

if backend.init_errors:
    print(backend.init_errors)    # e.g. ["3 of 50 objects failed to index"]

report = backend.initial_sync_report   # None if the sync raised outright
if report and report.failed:
    print(f"{report.failed} of {report.seen} objects are not searchable")
```

Worth checking in any long-running deployment: a partially failed sync or a
watcher that never started both leave a collection that looks healthy and is
quietly incomplete, which is otherwise easy to mistake for a search bug.

## Requirements

- Python 3.11+ (matches `deepagents`)
- MongoDB Atlas M0+ (Vector Search and Full-Text Search require M10+ for production; Atlas Local works for dev)
- AWS S3 bucket + appropriate IAM permissions (`s3:GetObject`, `s3:PutObject`, `s3:ListBucket`, `s3:DeleteObject`)
- AWS credentials accessible via the boto3 credential chain (required for S3 and for Bedrock embeddings)
- `OPENAI_API_KEY` env var — only if using `EMBEDDING_PROVIDER=openai`

### IAM permissions

Minimum policy for the S3 backend:

```json
{
  "Effect": "Allow",
  "Action": ["s3:GetObject", "s3:PutObject", "s3:DeleteObject", "s3:ListBucket"],
  "Resource": ["arn:aws:s3:::my-docs-bucket", "arn:aws:s3:::my-docs-bucket/*"]
}
```

When using `watcher="sqs"`, also grant `sqs:ReceiveMessage` and `sqs:DeleteMessage` on the queue.

When using Bedrock embeddings, also grant `bedrock:InvokeModel` for `amazon.titan-embed-text-v2:0`.

## Testing

```bash
just install              # uv sync --frozen
just unit_tests           # fast, no external services
just integration_tests    # moto + mongomock (no real credentials needed)
just e2e_tests            # full stack, moto + mongomock; real_e2e tests skip unless creds are set
```

The `real_e2e` and `watcher_e2e` marked tests run against real Atlas + S3 + an
embedding provider and are skipped automatically when the required environment
variables (`MONGODB_URI`, AWS credentials, etc.) are absent.

## Contributing

See the repository-level [CONTRIBUTING.md](https://github.com/langchain-ai/langchain-mongodb/blob/main/CONTRIBUTING.md).
