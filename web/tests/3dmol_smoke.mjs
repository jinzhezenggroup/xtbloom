import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";

import { THREE_DMOL_SOURCES } from "../bootstrap.js";

const EXPECTED_SHA256 =
  "f7cc78921ae72e7623e89cdd111434f58c2efddd2ffda1cd212644b406fb8016";
const EXPECTED_BYTES = 537792;

function sha256(bytes) {
  return createHash("sha256").update(bytes).digest("hex");
}

async function fetchPinned(url) {
  const response = await fetch(url, { signal: AbortSignal.timeout(60000) });
  assert.equal(response.ok, true, `${url}: HTTP ${response.status}`);
  return new Uint8Array(await response.arrayBuffer());
}

const localBytes = new Uint8Array(await readFile(new URL(
  "../node_modules/3dmol/build/3Dmol-min.js",
  import.meta.url,
)));
assert.equal(localBytes.byteLength, EXPECTED_BYTES);
assert.equal(sha256(localBytes), EXPECTED_SHA256);

for (const source of THREE_DMOL_SOURCES.filter(({ id }) => id !== "local")) {
  const remoteBytes = await fetchPinned(source.url);
  assert.equal(remoteBytes.byteLength, EXPECTED_BYTES, source.id);
  assert.equal(sha256(remoteBytes), EXPECTED_SHA256, source.id);
  assert.deepEqual(remoteBytes, localBytes, source.id);
}

console.log("3Dmol pinned multi-source smoke passed");
