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

# Glob by filename pattern → FileInfo dicts
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
| `aws_region` | No | `AWS_DEFAULT_REGION` env | AWS region for S3 and SQS clients |
| `s3_prefix` | No | `""` | Only sync/watch objects under this S3 prefix |
| `debug` | No | `False` | Re-raise exceptions instead of returning error DTOs (local dev) |

### Embedding provider selection

The provider is resolved at construction time from the `EMBEDDING_PROVIDER` environment variable (default `bedrock`). Override the model with `EMBEDDING_MODEL`.

| `EMBEDDING_PROVIDER` | Default model | Required credential |
|---|---|---|
| `bedrock` (default) | `amazon.titan-embed-text-v2:0` | boto3 credential chain (IAM role, `~/.aws/credentials`, etc.) |
| `openai` | `text-embedding-3-small` | `OPENAI_API_KEY` env var |

Pass a `LangChain Embeddings` instance directly to `embedding_model` to bypass provider lookup entirely.

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

If the embedding API is unavailable at query time, `grep` falls back to full-text only. On non-Atlas MongoDB (mongomock, community server), both `grep` and `glob` fall back to regex queries.

### Watcher

- **PollingWatcher** (default): ETag diff on a 10-second interval, zero AWS infra required. Backs off gracefully on S3 list failures.
- **SQSWatcher** (production): S3 event notifications via SQS long-polling (20s), near real-time. Requires S3 → SQS event notifications configured in AWS.

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
