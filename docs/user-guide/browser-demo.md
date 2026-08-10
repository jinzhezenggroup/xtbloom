# Browser demo

The [xTBloom browser demo](https://xtbloom.jinzhezeng.group) runs GFN2-xTB
entirely on the client. Molecular coordinates and results remain in the
browser; the site does not upload a calculation to a server.

## Try a molecule

1. Choose a preset, paste XYZ coordinates in angstrom, or enter a SMILES such
   as `CCO`.
2. For SMILES, select **Generate 3D** to add explicit hydrogens and create a
   pre-relaxed conformer.
3. Select **Compute energy** for a single point with optional analytic forces.
4. Select **Optimize geometry** to run the demo's L-BFGS adapter and inspect
   its energy trajectory.

The URL
[`?smiles=CCO`](https://xtbloom.jinzhezeng.group/?smiles=CCO) waits for the
SMILES and xTBloom workers, generates ethanol, runs the demo optimizer, and
writes the final coordinates back to the editor. Encode a literal `+` in a
charged SMILES as `%2B`.

## Results

The page reports energy, atomic charges, optional analytic forces, convergence
state, elapsed browser time, and an interactive molecular view. Geometry
optimization additionally exposes the energy trajectory and optimized
coordinates.

Browser timing is for interactivity only. It depends on the device, browser,
download cache, and demo numerical layer and is not part of xTBloom's published
native benchmark evidence.

## Scope

- The deployed engine is the single-threaded CPU backend compiled to wasm32.
- The demo supports modern browsers with WebAssembly, module Workers, and
  WebGL for molecular visualization.
- SMILES-to-3D and L-BFGS optimization are browser-adapter features, not
  stable C ABI capabilities.
- The demo is for exploration, not a production scientific environment.
- GFN1-xTB, solvation, molecular dynamics, Hessians, and lattice/PBC inputs
  remain unsupported.

Implementation, build, dependency, and parity details are documented in
[`web/README.md`](../../web/README.md).
