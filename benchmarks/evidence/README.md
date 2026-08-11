# Benchmark evidence storage policy

This directory keeps reviewable summaries and provenance, not an unbounded raw
artifact archive. The repository enforces both limits against the Git index:

- no tracked file under this directory may exceed 1 MiB (1,048,576 bytes);
- all tracked files under this directory may total at most 16 MiB
  (16,777,216 bytes).

Do not add path-specific exceptions to either limit. Large local measurements
may remain untracked while they are being reviewed. Before using an oversized
raw artifact as final evidence, place the exact bytes in durable external
storage and retain an `EXTERNAL_ARTIFACTS.tsv` file in the issue bundle with at
least these fields:

```text
sha256<TAB>bytes<TAB>url<TAB>producing_revision<TAB>retrieval_command
```

The compact README, CSV/JSON summaries, unavailable coordinates, correctness
qualification, and external metadata stay in Git. If the external artifact is
not durably retrievable and hash-verifiable, its acceptance row is
`UNVERIFIED`, not `PASS`.

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
