# Benchmark evidence storage policy

This directory keeps reviewable summaries and provenance, not an unbounded raw
artifact archive. The repository enforces both limits against the Git index:

- no tracked file under this directory may exceed 1 MiB (1,048,576 bytes);
- all tracked files under this directory may total at most 16 MiB
  (16,777,216 bytes).

Do not add path-specific exceptions to either limit. Large local measurements
may remain untracked while they are being reviewed. Prefer not to publish an
oversized raw artifact when the repository runner, exact command, clean source
and binary identities, input hashes, and compact result are sufficient to
reproduce and audit the claim. In that case, omit the raw bytes and keep the
README, compact CSV/JSON result, unavailable coordinates, correctness
qualification, and limitations in Git.

External archival is optional and should be used only when the exact raw bytes
are themselves necessary evidence or the result cannot be reproduced. When it
is used, retain an `EXTERNAL_ARTIFACTS.tsv` file in the issue bundle with at
least these fields:

```text
sha256<TAB>bytes<TAB>url<TAB>producing_revision<TAB>retrieval_command
```

Do not mark otherwise qualified evidence unverified merely because a
reproducible oversized raw artifact was intentionally omitted. If a claim
specifically depends on externally archived bytes, however, those bytes must be
durably retrievable and hash-verifiable for that acceptance row to pass.

## Legacy oversized artifacts

Issue #348 removed pre-existing over-limit JSON files from the current tree.
Their exact bytes remain addressable at the last main revision that tracked
them; `legacy-large-artifacts.tsv` records the path, size, digest, and revision.
For example:

```bash
git show <source_commit>:<path> > /tmp/artifact.json
sha256sum /tmp/artifact.json
```

This cleanup reduces current checkout payload and prevents further growth. It
does not rewrite published Git history; a history-purging migration requires
separate authorization because it disrupts existing clones and open branches.
