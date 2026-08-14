# Browser demo

The [xTBloom browser demo](https://xtbloom.jinzhezeng.group) runs GFN1-xTB or
GFN2-xTB entirely on the client. GFN2-xTB is selected by default. Molecular
coordinates and results remain in the browser; the site does not upload a
calculation to a server.

## Try a molecule

1. Choose **GFN2-xTB** (the default) or **GFN1-xTB**.
2. Choose a preset such as **C₆₀ fullerene**, paste XYZ coordinates in
   angstrom, or enter a SMILES such as `CCO`.
3. For SMILES, select **Generate 3D** to add explicit hydrogens and create a
   pre-relaxed conformer.
4. Select **Compute energy** for a single point with optional analytic forces.
5. Select **Optimize geometry** to run the demo's L-BFGS adapter and inspect
   its energy trajectory.

On phones, SCC and optimizer controls are grouped under **Advanced settings**
without changing their values. Completed calculations and failures bring the
result panel into view; reduced-motion preferences disable animated scrolling.

SMILES conformer generation is bounded to two minutes. Flexible drug-sized
molecules can take substantially longer than small examples such as ethanol,
especially on phones. If that limit is reached, the page terminates the
uninterruptible OpenChemLib task and restores a clean generator automatically;
retry after the generator reports that it is ready. Editing the SMILES or
selecting **Reset** also cancels the old task, so its coordinates cannot replace
newer input.

Each optimization step after the first is seeded from the previous step's
converged electronic state (native SCC warm start), reusing electronic state
across successive geometries. A new optimization or a standalone single-point
calculation always starts fresh and never inherits another run's electronic
state.

The 3D view is an input-validation preview, independent of the calculation
path. Valid XYZ coordinates render automatically about 400 ms after you stop
typing, and a generated SMILES structure renders immediately. Malformed input
(invalid element symbols, missing or non-numeric coordinates, more than 512
atoms) is flagged inline beside the coordinates box while the last valid
structure stays on screen, and **Compute energy** / **Optimize geometry**
remain disabled until a valid structure is present. **Reset** clears the
SMILES box, coordinates, settings, preview, results, and errors, then restores
the water template.

The URL
[`?smiles=CCO`](https://xtbloom.jinzhezeng.group/?smiles=CCO) waits for the
SMILES and xTBloom workers, generates ethanol, runs the demo optimizer, and
writes the final coordinates back to the editor. Encode a literal `+` in a
charged SMILES as `%2B`.

The engine loader reports completed files and received bytes across the full
Worker, JavaScript, WebAssembly, and data resource set. A dependency-free
bootstrap first verifies the versioned app/helper module graph, while the page
itself retries a transient failure loading that bootstrap. A small version
manifest is revalidated on refresh; unchanged versioned resources stay in the
browser cache, while a changed manifest pulls the new generation. The manifest
also provides exact decoded sizes for byte progress. Transient startup network
failures are retried automatically as one coherent resource generation; the
manual retry keeps the current molecule and settings instead of reloading the
page.

The molecular viewer separately probes a small prefix of the pinned 3Dmol.js
bundle from JSDMirror, jsDelivr, and the site origin, then downloads the fastest
verified source with ranked fallback. The optional SMILES worker reuses the
measured JSDMirror/jsDelivr order for its pinned OpenChemLib module and resource
pair. Close probe results prefer JSDMirror for recognized mainland-China time
zones and jsDelivr elsewhere. Provider failures do not prevent the core WASM
engine from loading; 3Dmol retains a site-local fallback, while ordinary XYZ
calculations never require OpenChemLib.

## Results

The page reports energy, atomic charges, optional analytic forces, convergence
state, elapsed browser time, and an interactive molecular view. Geometry
optimization additionally exposes the energy trajectory and optimized
coordinates.

The C60 preset remains the deployed engine's GFN2-xTB regression. It exercises
a 60-atom, 240-orbital neutral singlet and is checked against native
public-C-ABI energy, charge, force, SCC-status, and iteration references in
both wasm32 and wasm64 CI builds. A charged H3 structure independently checks
GFN1-xTB energy and forces against the hash-pinned tblite golden, including a
GFN2-to-GFN1-to-GFN2 sequence through one Web context.

Browser timing is for interactivity only. It depends on the device, browser,
download cache, and single-threaded WebAssembly execution through the
Eigen-backed LP64 LAPACKE/CBLAS compatibility provider. It is not part of
xTBloom's published native benchmark evidence.

## Scope

- The deployed engine is the single-threaded CPU backend compiled to wasm32.
- The method selector exposes GFN1-xTB and GFN2-xTB; GFN2-xTB is the default.
  This does not add GFN1 CUDA support.
- Its Web-only LP64 LAPACKE/CBLAS side module uses pinned Eigen 5.0.1 while
  preserving the same loader symbols and public xTBloom C ABI as before.
- Eigen is downloaded only when building the Web demo, with a fixed archive
  SHA-256; native builds, installs, and Python wheels do not acquire or carry it.
- The demo supports modern browsers with WebAssembly, module Workers, and
  WebGL for molecular visualization.
- SMILES-to-3D and L-BFGS optimization are browser-adapter features, not
  stable C ABI capabilities.
- The demo is for exploration, not a production scientific environment.
- Solvation, molecular dynamics, Hessians, and lattice/PBC inputs remain
  unsupported here.

Implementation, build, dependency, and parity details are documented in
[`web/README.md`](../../web/README.md).
