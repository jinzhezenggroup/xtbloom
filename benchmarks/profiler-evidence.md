# Profiler evidence policy

Raw profiler captures are prohibited from the repository. They can embed the
target process environment, credentials, API keys, filesystem paths, cgroup
state, and session metadata.

Do not commit:

- `*.nsys-rep`, `*.ncu-rep`, or `*.qdstrm`;
- `*.sqlite`, `*.sqlite.dbb`, or `*.csv.db`; or
- `*.prof`.

`.gitignore` and the `forbid-raw-profiler-captures` pre-commit hook enforce
this rule, including force-added files.

Only sanitized derived summaries may be archived under
`benchmarks/evidence/`:

- `nsys stats` CSV reports;
- `ncu --csv` console output; and
- reviewed text or JSON summaries.

Every evidence README must record the profiler version, exact extraction
command, measured target, and how the derived data supports the stated claim.
Never use `ncu --export` as a sanitization step; it writes a native
`.ncu-rep` capture.
