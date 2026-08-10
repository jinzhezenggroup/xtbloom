# Skills for AI agents

User-facing skills live in [`skills/`](../../skills). Contributor workflows
remain separate under [`.agents/skills/`](../../.agents/skills).

## Choose a skill

| Skill | Use it for |
| --- | --- |
| `xtbloom-install-and-diagnose` | Install from a source checkout, select CPU or CUDA deliberately, inspect native-library discovery, and prove which backend actually executes. |
| `xtbloom-run-python-inference` | Write single-system or native ragged-batch Python calculations with correct units, spin, warm-start, and per-system failure handling. |
| `xtbloom-integrate-ase-dpdata` | Attach xTBloom to ASE or dpdata for energies, forces, dataset labeling, or adapter-level relaxation. |
| `xtbloom-use-zero-copy-ml` | Connect NumPy, CuPy, JAX, or PyTorch arrays through Array API and DLPack, including caller-owned outputs and the positions-only PyTorch gradient. |
| `xtbloom-integrate-c-api` | Build an installed C or C++ consumer with versioned descriptors, caller-owned buffers, ragged batches, CUDA memory tags, and result diagnostics. |
| `xtbloom-couple-qmmm` | Add explicit point charges or a caller-supplied `b + A q` charge-response operator and account for the force terms xTBloom intentionally excludes. |

## Load a skill

Ask the agent to load the matching `SKILL.md`, for example:

```text
Read skills/xtbloom-run-python-inference/SKILL.md and use that skill to write
a CUDA ragged-batch calculation for these XYZ structures while preserving
per-system failures.
```

For a client that installs skills from GitHub, use the repository and skill
path directly:

```text
Repository: jinzhezenggroup/xtbloom
Path: skills/xtbloom-run-python-inference
```

Each skill directory is self-contained. Clients may ignore
`agents/openai.yaml` when they use only the portable `SKILL.md` convention.
