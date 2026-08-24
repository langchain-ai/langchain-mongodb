# Trusted Publishing Setup (first release of a new library)

One-time setup required **before the first release of any new library** in this
monorepo. Existing libraries (`langchain-mongodb`, `langgraph-checkpoint-mongodb`,
`langgraph-store-mongodb`) are already configured — this is only for a package
that has never been published.

Companion to [`RELEASE.md`](RELEASE.md), which covers the steady-state release
process and assumes this is already done.

## Why it's needed

The release workflow publishes via [PyPI Trusted Publishing][tp] (OIDC) rather
than API tokens — there is no token to configure, and `twine` is not used
manually. But trusted publishing cannot attach a publisher to a project that
does not exist yet, so a brand-new package needs a **"pending publisher"**
registered by hand first. Without it, the release run fails at the publish
step, after a green build.

[tp]: https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/

## Before you start

- You need an account with permission to add publishers on PyPI.
- For a package owned by an organisation, this may need whoever owns the
  sibling packages, who is not necessarily the person doing the release.

## Register the pending publisher

Go to <https://pypi.org/manage/account/publishing/>
(Account settings → **Publishing** in the left sidebar).

Scroll to **"Add a new pending publisher"** and select the **GitHub** tab.

| Field | Value |
|---|---|
| PyPI Project Name | *the distribution name from `pyproject.toml`*, e.g. `langchain-mongodb-deepagents-vfs` |
| Owner | `langchain-ai` |
| Repository name | `langchain-mongodb` |
| Workflow name | `_release.yml` |
| Environment name | **leave empty** |

Click **Add**. That is the only registration required — TestPyPI is no longer
part of the release process.

## The two easy-to-get-wrong bits

### The workflow name is the file containing the publish job

PyPI validates the `job_workflow_ref` OIDC claim: the workflow file containing
**the job that requests the token**, not necessarily the workflow you dispatch.
Here they are the same file (`_release.yml`), but if the publish job is ever
moved into a reusable workflow, this field must name *that* file instead.

Verified against [warehouse source][wh]: the claim is checked at
`warehouse/oidc/models/github.py:156`, compared against
`.github/workflows/{workflow_filename}` at line 242.

[wh]: https://github.com/pypi/warehouse/blob/main/warehouse/oidc/models/github.py

### Environment name must be blank

PyPI's UI describes an environment as "optional but strongly recommended", but
neither publish job in this repo declares an `environment:` key. If you fill
this field in, the OIDC claim will not match and the upload is rejected.

If a future change adds `environment:` to that job, the publisher config must
be updated to match.

## Gotchas

- **A pending publisher does not reserve the name.** It only becomes a real
  publisher on the first successful upload. Until then, someone else could
  claim the name — so don't register it months ahead of releasing.
- **Metadata is validated locally before upload.** `_release.yml` runs
  `validate-pyproject` and `twine check --strict` in the build job, so a
  malformed artifact fails there rather than at the PyPI upload step.

## Verifying

Registration does **not** create the project, so the package will still 404
until the first successful publish. Before releasing, confirm the name is
still unclaimed. For example:

```bash
PKG=langchain-mongodb-deepagents-vfs
curl -s -o /dev/null -w '%{http_code}\n' https://pypi.org/pypi/$PKG/json
# 404 = name unclaimed, pending publisher will convert on first upload
```

After the release it should return 200 and the pending publisher will have
converted automatically — no further configuration needed for subsequent
releases.

## Note on signing

Both publish steps currently set `attestations: false`, commented as a
temporary workaround. **No Sigstore / PEP 740 attestations are produced** on
either index. If signed releases are wanted, that flag is what to change — it
is independent of the trusted-publishing setup above.
