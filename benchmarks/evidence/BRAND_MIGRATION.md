# Pre-release brand migration

Issue #281 renamed the project to xTBloom before its first public release. The
archived benchmark bundles were migrated mechanically so their engine labels,
library names, commands, and artifact paths use the released name consistently.

This migration did not rerun or reinterpret any measurement. Numeric samples,
hardware and toolchain facts, convergence results, profiler observations, and
scientific correctness fields remain the observations recorded by each bundle.
Bundle checksum manifests were refreshed because the textual labels and paths
changed. The exact pre-migration bytes remain available from Git commit
`a5887616a6ea0f0448d2707cc3a6588d8cf65176`.
