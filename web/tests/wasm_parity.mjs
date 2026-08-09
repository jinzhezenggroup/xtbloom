import assert from "node:assert/strict";

import { runWebCases, validateWebCases } from "./wasm_smoke.mjs";

if (process.argv.length !== 4) {
  throw new Error("usage: node wasm_parity.mjs <wasm32-site> <wasm64-site>");
}

const wasm32 = await runWebCases(process.argv[2]);
const wasm64 = await runWebCases(process.argv[3]);
validateWebCases(wasm32);
validateWebCases(wasm64);

/* Pointer width must not affect model results or adapter failure semantics.
 * Exact equality is expected because both variants use the same compiler,
 * binary64 operations, parameters, and single-threaded execution order. */
assert.deepEqual(wasm32, wasm64);
console.log(
  "wasm32/wasm64 parity ok, energy", wasm32.withForces.energy_Eh,
  "optimized", wasm32.optimized.energy_final_Eh,
);
