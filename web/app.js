/* gpuxtb web demo front end (bilingual zh/en).
 * Loads the wasm32 module through the Emscripten factory and wires the two
 * adapter entry points (single-point compute, L-BFGS geometry optimize) to
 * the input/output panels. All user-facing strings come from the I18N
 * dictionary below; errors arrive from wasm as stable ASCII codes. */

import {
  angstromToBohr,
  initializeWorker,
  postToReadyWorker,
  withTimeout,
} from "./app_helpers.js";

const EH2EV = 27.211386245988;
const EH2KCAL = 627.509474063;
const EHB2EVA = EH2EV / 0.529177210903;
const K2EH = 3.166811563e-6;

const ELEMENT_SYMBOLS = [
  "", "H","He","Li","Be","B","C","N","O","F","Ne","Na","Mg","Al","Si","P","S","Cl","Ar",
  "K","Ca","Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn","Ga","Ge","As","Se","Br","Kr",
  "Rb","Sr","Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd","In","Sn","Sb","Te","I","Xe",
  "Cs","Ba","La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu",
  "Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg","Tl","Pb","Bi","Po","At","Rn","Fr","Ra","Ac",
  "Th","Pa","U","Np","Pu","Am","Cm","Bk","Cf","Es","Fm","Md","No","Lr"
];

const PRESETS = {
  water: { xyz: "O  0.00000000  0.00000000  0.00000000\nH  0.00000000  0.00000000  0.95720000\nH  0.00000000  0.75718000 -0.58552000", charge: 0, unpaired: 0 },
  ketene: { xyz: "C  0.00000000  0.00000000  0.54294240\nH -0.94138558  0.00000000  1.07531674\nH  0.94138558  0.00000000  1.07531674\nC  0.00000000  0.00000000 -0.76523014\nO  0.00000000  0.00000000 -1.92834574", charge: 0, unpaired: 0 },
  oh: { xyz: "O  0.00000000  0.00000000  0.00000000\nH  0.00000000  0.00000000  0.97051100", charge: 0, unpaired: 1 },
  benzene: { xyz: "C 1.39000000 0.00000000 0.00000000\nC 0.69500000 1.20377531 0.00000000\nC -0.69500000 1.20377531 0.00000000\nC -1.39000000 0.00000000 0.00000000\nC -0.69500000 -1.20377531 0.00000000\nC 0.69500000 -1.20377531 0.00000000\nH 2.48000000 0.00000000 0.00000000\nH 1.24000000 2.14774300 0.00000000\nH -1.24000000 2.14774300 0.00000000\nH -2.48000000 0.00000000 0.00000000\nH -1.24000000 -2.14774300 0.00000000\nH 1.24000000 -2.14774300 0.00000000", charge: 0, unpaired: 0 },
  ethanol: { xyz: "C -0.05410000  0.37050000  0.00000000\nC  1.27860000 -0.37750000  0.00000000\nO  2.38310000  0.51100000 -0.01750000\nH -0.04270000  1.45560000  0.00000000\nH -0.54830000  0.00000000 -0.89740000\nH -0.54830000  0.00000000  0.89740000\nH  1.36610000 -1.01150000  0.88490000\nH  1.36610000 -1.01150000 -0.88490000\nH  3.19200000  0.03930000 -0.03490000", charge: 0, unpaired: 0 },
};

const I18N = {
  zh: {
    tagline: "浏览器内运行的原生 GFN2-xTB 单点能与几何优化 · C++17 WASM32 · CPU 后端",
    engine_loading: "引擎加载中…",
    panel_input: "输入",
    presets_label: "模板分子",
    preset_ketene: "乙烯酮",
    preset_ethanol: "乙醇",
    preset_benzene: "苯",
    mol_title: "分子可视化",
    mol_hint: "实时显示当前坐标；计算、优化、应用优化坐标后自动更新。",
    mol_unavailable: "当前浏览器不支持 WebGL 分子可视化。",
    opt_running: "优化中… {{n}}/{{max}} 步 · E = {{e}} Eh",
    opt_done: "完成 ✓",
    xyz_label: "坐标（XYZ，单位：埃 Å）",
    xyz_placeholder: "每行：元素符号 x y z（埃，Å）",
    charge_label: "分子电荷 q / e",
    unpaired_label: "未配对电子数",
    etemp_label: "电子温度 K",
    etemp_tip: "电子温度即 k_B·T 能量标度；0 = 精确 0 K（默认）",
    maxiter_label: "最大 SCC 迭代",
    etol_label: "能量收敛 (Eh)",
    qtol_label: "电荷收敛 (e)",
    forces_label: "同时计算解析力（forces）",
    run: "计算能量",
    reset: "重置",
    opt_title: "几何优化",
    opt_hint: "L-BFGS · 解析力",
    opt_maxiter_label: "最大迭代",
    opt_gradtol_label: "收敛力 (Eh/bohr)",
    opt_maxmove_label: "单步位移上限 (Å)",
    opt_run: "几何优化",
    panel_output: "结果",
    stat_atoms: "原子",
    stat_iter: "SCC 迭代",
    stat_conv: "收敛",
    stat_ms: "耗时 ms",
    th_atom: "原子", th_q: "电荷 q",
    th_fx: "Fx (Eh/bohr)", th_fy: "Fy (Eh/bohr)", th_fz: "Fz (Eh/bohr)", th_fmag: "|F| (eV/Å)",
    copy_json: "复制 JSON",
    opt_apply: "把优化坐标填回输入框",
    copy_done: "已复制",
    roadmap: "路线图",
    roadmap_smiles_title: "SMILES → 结构",
    tag_pending: "暂不实现",
    roadmap_smiles_desc: "输入 SMILES 自动生成三维坐标，然后直接计算。",
    btn_pending: "即将推出",
    roadmap_opt_title: "几何优化",
    tag_done: "已支持",
    roadmap_opt_desc: "内置 L-BFGS 优化器，使用解析力收敛到稳定结构。在左侧“优化”区配置后点击“几何优化”。",
    opt_go: "去优化",
    footer: "由 gpuxtb 驱动 —— 同一套 C ABI 的 C++17 原生库编译为不依赖 Memory64 的 wasm32，面向支持 WebAssembly 与模块 Worker 的现代 iOS Safari、Safari、Chrome 和 Firefox。BLAS/LAPACK 层为演示用最小实现，经 numpy 与原生 gpuxtb 逐位验证。仅供演示，非科学计算生产环境。",
    overlay_loading: "正在加载 WASM 引擎…",
    overlay_compute: "正在计算单点能…",
    overlay_opt: "正在几何优化（逐梯度迭代，可能需要几秒）…",
    stat_converged: "是",
    stat_not_conv: "否",
    opt_converged_line: "优化收敛 · 最大力 {{f}} Eh/bohr · ΔE = {{de}} Eh",
    opt_not_conv_line: "优化未达梯度阈值 · 最大力 {{f}} Eh/bohr · 已迭代 {{n}} 步",
    no_xyz: "请先输入坐标。",
    opt_apply_done: "已把优化后的坐标写入输入框。",
    copy_fail: "复制失败（浏览器权限）",
    smiles_msg: "SMILES → 结构暂未实现，敬请期待。",
    engine_ok: "引擎就绪",
    engine_fail: "引擎加载失败",
    load_fail: "无法加载 wasm32 引擎。请使用支持 WebAssembly 和模块化 Web Worker 的现代浏览器。\n详情：",
    load_timeout: "加载超时——网络可能较慢，请重试。若持续失败请检查是否能正常访问本页面资源。",
    load_retry: "重试",
    load_downloading: "正在下载 WASM 引擎：{{pct}}%",
    traj_title: "能量迭代轨迹（Eh）",
    engine_call_fail: "引擎调用失败：",
    err_xyz_parse: "无法解析坐标：每行请提供「元素符号 x y z」，单位 Å",
    err_xyz_too_many: "原子数超过 512 上限",
    err_ctx: "GPU 上下文创建失败：{{e}}",
    err_alloc: "内存分配失败",
    err_init: "内部初始化失败",
    err_compute: "计算失败：{{e}}",
    err_scc_initial: "SCC / 特征求解失败（初始结构）",
    err_nan_initial: "初始能量为非有限值",
    err_nan_step: "优化中出现非有限能量",
    err_linesearch: "线搜索失败：无法找到能量下降步长",
    err_step_sp: "线搜索中单点计算失败",
    err_initial_calc: "初始结构计算失败：{{e}}",
    err_opt: "几何优化失败：{{e}}",
    err_unknown: "未知错误",
  },
  en: {
    tagline: "Native GFN2-xTB single-point + geometry optimization in your browser · C++17 WASM32 · CPU backend",
    engine_loading: "engine loading…",
    panel_input: "Input",
    presets_label: "Preset molecules",
    preset_ketene: "Ketene",
    preset_ethanol: "Ethanol",
    preset_benzene: "Benzene",
    mol_title: "Molecule",
    mol_hint: "Live view of the current coordinates; refreshes after compute, optimize, or applying optimized coordinates.",
    mol_unavailable: "WebGL molecular visualization is not available in this browser.",
    opt_running: "Optimizing… step {{n}}/{{max}} · E = {{e}} Eh",
    opt_done: "done ✓",
    xyz_label: "Coordinates (XYZ, angstrom)",
    xyz_placeholder: "One atom per line: Symbol x y z (Å)",
    charge_label: "Molecular charge q / e",
    unpaired_label: "Unpaired electrons",
    etemp_label: "Electronic temperature K",
    etemp_tip: "Electronic temperature is the k_B·T energy scale; 0 = exact 0 K (default)",
    maxiter_label: "Max SCC iterations",
    etol_label: "Energy tolerance (Eh)",
    qtol_label: "Charge tolerance (e)",
    forces_label: "Also compute analytical forces",
    run: "Compute energy",
    reset: "Reset",
    opt_title: "Geometry optimization",
    opt_hint: "L-BFGS · analytic forces",
    opt_maxiter_label: "Max iterations",
    opt_gradtol_label: "Force tolerance (Eh/bohr)",
    opt_maxmove_label: "Max step displacement (Å)",
    opt_run: "Optimize geometry",
    panel_output: "Results",
    stat_atoms: "atoms",
    stat_iter: "SCC iter.",
    stat_conv: "conv.",
    stat_ms: "time ms",
    th_atom: "Atom", th_q: "charge q",
    th_fx: "Fx (Eh/bohr)", th_fy: "Fy (Eh/bohr)", th_fz: "Fz (Eh/bohr)", th_fmag: "|F| (eV/Å)",
    copy_json: "Copy JSON",
    opt_apply: "Load optimized coords back to input",
    copy_done: "copied",
    roadmap: "Roadmap",
    roadmap_smiles_title: "SMILES → structure",
    tag_pending: "not implemented",
    roadmap_smiles_desc: "Generate 3D coordinates from a SMILES string, then compute.",
    btn_pending: "Coming soon",
    roadmap_opt_title: "Geometry optimization",
    tag_done: "supported",
    roadmap_opt_desc: "Built-in L-BFGS optimizer using analytic forces. Configure it in the left panel, then click “Optimize geometry”.",
    opt_go: "Try it",
    footer: "Powered by gpuxtb — the same native C ABI library compiled to wasm32 without requiring Memory64, targeting modern iOS Safari, Safari, Chrome, and Firefox with WebAssembly and module Worker support. The BLAS/LAPACK layer is a minimal demo implementation, validated bit-for-bit against numpy and native gpuxtb. Demo only, not a production scientific environment.",
    overlay_loading: "Loading the WASM engine…",
    overlay_compute: "Computing single point…",
    overlay_opt: "Optimizing geometry (gradient steps, may take a few seconds)…",
    stat_converged: "yes",
    stat_not_conv: "no",
    opt_converged_line: "Optimization converged · max force {{f}} Eh/bohr · ΔE = {{de}} Eh",
    opt_not_conv_line: "Gradient tolerance not reached · max force {{f}} Eh/bohr · {{n}} steps",
    no_xyz: "Please enter coordinates first.",
    opt_apply_done: "Optimized coordinates loaded back into the input box.",
    copy_fail: "Copy failed (browser permissions)",
    smiles_msg: "SMILES → structure is not implemented yet.",
    engine_ok: "engine ready",
    engine_fail: "engine failed to load",
    load_fail: "Could not load the wasm32 engine. Use a modern browser with WebAssembly and module Worker support.\nDetails: ",
    load_timeout: "Load timed out — network may be slow. Please retry. If it keeps failing, check that the page and its assets can be reached.",
    load_retry: "Retry",
    load_downloading: "Downloading WASM engine: {{pct}}%",
    traj_title: "Energy trajectory (Eh)",
    engine_call_fail: "Engine call failed: ",
    err_xyz_parse: "Cannot parse coordinates: each line must be “Symbol x y z” in A",
    err_xyz_too_many: "More than 512 atoms",
    err_ctx: "Context creation failed: {{e}}",
    err_alloc: "Out of memory",
    err_init: "Internal initialization failed",
    err_compute: "Compute failed: {{e}}",
    err_scc_initial: "SCC / eigensolver failed (initial structure)",
    err_nan_initial: "Non-finite initial energy",
    err_nan_step: "Non-finite energy during optimization",
    err_linesearch: "Line search failed: no acceptable downhill step",
    err_step_sp: "Single-point failure during line search",
    err_initial_calc: "Initial structure compute failed: {{e}}",
    err_opt: "Geometry optimization failed: {{e}}",
    err_unknown: "Unknown error",
  },
};

/* ------------------------------------------------------------------ */

const $ = (id) => document.getElementById(id);

let lang = "zh";
try {
  lang = localStorage.getItem("gpuxtb-lang") ||
    (navigator.language && navigator.language.toLowerCase().startsWith("zh") ? "zh" : "en");
} catch { /* storage unavailable */ }

const t = (key, vars) => {
  let s = (I18N[lang] && I18N[lang][key]) || I18N.zh[key] || key;
  if (vars) {
    for (const [k, v] of Object.entries(vars)) {
      s = s.split("{{" + k + "}}").join(String(v));
    }
  }
  return s;
};

function tf(key, vars) { return t(key, vars); }

function applyI18n() {
  document.documentElement.lang = lang === "zh" ? "zh-CN" : "en";
  document.title = lang === "zh" ? "gpuxtb · GFN2-xTB 在线计算（WASM）" : "gpuxtb · GFN2-xTB in-browser (WASM)";
  document.querySelectorAll("[data-i18n]").forEach((el) => {
    if (el.id === "engine-badge") return; /* state-managed separately */
    el.textContent = t(el.dataset.i18n);
  });
  document.querySelectorAll("[data-i18n-placeholder]").forEach((el) => {
    el.placeholder = t(el.dataset.i18nPlaceholder);
  });
  $("lang-toggle").textContent = lang === "zh" ? "EN" : "中文";
  $("etemp-tip").title = t("etemp_tip");
  refreshBadge();
  updateXyzHint();
}

$("lang-toggle").addEventListener("click", () => {
  lang = lang === "zh" ? "en" : "zh";
  try { localStorage.setItem("gpuxtb-lang", lang); } catch { /* ignore */ }
  applyI18n();
  // re-render dynamic labels on the results panel
  if (window.__lastMode === "compute" && window.__lastResult) renderCompute(window.__lastResult);
  if (window.__lastMode === "optimize" && window.__lastResult) renderOptimize(window.__lastResult);
});

/* ------------------------------------------------------------------ */
/* Engine lifecycle (runs inside a Web Worker so sync wasm calls never  */
/* block the UI thread).                                                */

let engineState = "loading"; /* loading | ready | error */
let worker = null;
let engineBusy = false;
let msgSeq = 0;
const pending = new Map();

function syncEngineControls() {
  const enabled = engineState === "ready" && worker !== null && !engineBusy;
  $("run").disabled = !enabled;
  $("opt-run").disabled = !enabled;
}

function refreshBadge() {
  const b = $("engine-badge");
  b.classList.toggle("ok", engineState === "ready");
  b.classList.toggle("err", engineState === "error");
  b.textContent =
    engineState === "ready" ? t("engine_ok")
    : engineState === "error" ? t("engine_fail")
    : t("engine_loading");
  syncEngineControls();
}

function rejectPendingCalls(error) {
  for (const entry of pending.values()) entry.reject(error);
  pending.clear();
}

function handleWorkerMessage(m) {
  if (m.type === "error") {
    const error = new Error(String(m.error || "worker error"));
    if (worker) worker.terminate();
    worker = null;
    engineState = "error";
    rejectPendingCalls(error);
    refreshBadge();
    hideOverlay();
    setError((t("load_fail") + error.message).trim());
    $("retry").hidden = false;
  } else if (m.type === "step") {
    handleStepMessage(m);
  } else if (m.type === "result") {
    const entry = pending.get(m.id);
    if (!entry) return;
    pending.delete(m.id);
    if (m.ok) entry.resolve(m); else entry.reject(new Error(m.error || "worker error"));
  }
}

async function initWorker(wasmBinary) {
  worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
  const ready = await initializeWorker(worker, wasmBinary, handleWorkerMessage);
  engineState = "ready";
  refreshBadge();
  $("ver-badge").textContent = "v" + ready.version;
  $("xyz").value = PRESETS.water.xyz;
  updateXyzHint();
  initMoleculeViewer();
  updateMoleculeViewer(PRESETS.water.xyz);
}

function callWorker(cmd, args, onStep) {
  return new Promise((resolve, reject) => {
    const id = ++msgSeq;
    pending.set(id, onStep ? { resolve, reject, onStep } : { resolve, reject });
    try {
      postToReadyWorker(worker, engineState, { type: "call", id, cmd, args });
    } catch (error) {
      pending.delete(id);
      reject(error);
    }
  });
}

function handleStepMessage(m) {
  const entry = pending.get(m.id);
  if (entry && entry.onStep) {
    entry.onStep(m);
  }
}

function getElementSymbols(xyz) {
  const symbols = [];
  for (const line of xyz.split(/\r?\n/)) {
    const t = line.trim();
    if (!t) continue;
    const tok = t.split(/\s+/)[0];
    const n = Number(tok);
    symbols.push(Number.isInteger(n) && n >= 1 && n <= 103 ? (ELEMENT_SYMBOLS[n] || "?") : tok);
  }
  return symbols;
}

const overlayText = $("overlay-text");
const overlay = $("overlay");
function showOverlay(key) { overlayText.textContent = t(key); overlay.hidden = false; }
function hideOverlay() { overlay.hidden = true; }
$("retry").addEventListener("click", () => { window.location.reload(); });

function fmt(x, digits = 6) {
  if (x === null || x === undefined || Number.isNaN(x)) return "NaN";
  return Number(x).toFixed(digits);
}

function countAtoms(xyz) {
  let n = 0;
  for (const line of xyz.split(/\r?\n/)) {
    if (line.trim()) n++;
  }
  return n;
}

let __loadingPct = 0;
function updateLoader(pct) {
  __loadingPct = pct;
  $("load-bar-wrap").hidden = false;
  $("load-bar-fill").style.width = Math.min(100, Math.max(0, pct)) + "%";
  $("load-bar-text").textContent = Math.round(pct) + "%";
  $("overlay-text").textContent = tf("load_downloading", { pct: Math.round(pct) });
}
async function fetchProgress(url, options = {}) {
  const resp = await fetch(url, options);
  if (!resp.ok) throw new Error(`HTTP ${resp.status} ${resp.statusText}`);
  const total = Number(resp.headers.get("content-length")) || 0;
  const reader = resp.body.getReader();
  const chunks = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    got += value.length;
    const pct = total ? (got / total) * 100 : 0;
    if (pct >= 10) updateLoader(pct);
  }
  const n = chunks.reduce((a, c) => a + c.length, 0);
  const out = new Uint8Array(n);
  let o = 0;
  for (const c of chunks) { out.set(c, o); o += c.length; }
  return out;
}

function updateXyzHint() {
  $("xyz-hint").textContent =
    lang === "zh"
      ? `已识别的原子数：${countAtoms($("xyz").value)}`
      : `Recognized atoms: ${countAtoms($("xyz").value)}`;
}

function setError(msg) {
  const el = $("error");
  if (msg) {
    el.textContent = msg;
    el.hidden = false;
  } else {
    el.hidden = true;
  }
}

function errorText(d) {
  if (d && d.error_code) {
    return t(d.error_code, { e: d.error ? d.error : "" });
  }
  return (d && d.error) || t("err_unknown");
}

function energyParts(eh) {
  return { eh, ev: eh * EH2EV, kcal: eh * EH2KCAL };
}

function renderCompute(d) {
  setError(null);
  const e = energyParts(d.energy_Eh);
  $("energy").textContent = fmt(e.eh, 10) + " Eh";
  $("energy-ev").textContent = fmt(e.ev, 6) + " eV";
  $("energy-kcal").textContent = fmt(e.kcal, 4) + " kcal/mol";
  $("stat-atoms").textContent = d.charges.length;
  $("stat-iter").textContent = d.scc_iterations;
  $("stat-conv").textContent = d.scc_converged ? t("stat_converged") : t("stat_not_conv");

  const tbody = $("atom-table").querySelector("tbody");
  tbody.innerHTML = "";
  d.charges.forEach((c, i) => {
    const f = d.forces ? d.forces[i] : null;
    const fr = f ? Math.sqrt(f.fx_eh_bohr ** 2 + f.fy_eh_bohr ** 2 + f.fz_eh_bohr ** 2) * EHB2EVA : null;
    const tr = document.createElement("tr");
    tr.innerHTML =
      `<td>${i + 1}</td><td>${c.element}</td><td class="sym">${ELEMENT_SYMBOLS[c.element] || "?"}</td>` +
      `<td class="num">${fmt(c.q, 5)}</td>` +
      (f
        ? `<td class="num">${fmt(f.fx_eh_bohr, 6)}</td><td class="num">${fmt(f.fy_eh_bohr, 6)}</td><td class="num">${fmt(f.fz_eh_bohr, 6)}</td><td class="num">${fmt(fr, 4)}</td>`
        : `<td class="num">–</td><td class="num">–</td><td class="num">–</td><td class="num">–</td>`);
    tbody.appendChild(tr);
  });
  $("table-wrap").hidden = false;
  $("output-tools").hidden = false;
  window.__lastResult = d;
  window.__lastMode = "compute";
}

function renderOptimize(d) {
  setError(null);
  const e0 = energyParts(d.energy_initial_Eh);
  const e1 = energyParts(d.energy_final_Eh);
  $("energy").textContent = fmt(e1.eh, 10) + " Eh";
  $("energy-ev").textContent = `${fmt(e1.ev, 6)} eV  ·  ${lang === "zh" ? `初值 ${fmt(e0.ev, 6)} eV` : `start ${fmt(e0.ev, 6)} eV`}`;
  $("energy-kcal").textContent = fmt(e1.kcal, 4) + " kcal/mol";
  $("stat-atoms").textContent = d.charges.length;
  $("stat-iter").textContent = d.iterations;
  $("stat-conv").textContent = d.converged ? t("stat_converged") + " ✓" : t("stat_not_conv");

  const note = d.converged
    ? tf("opt_converged_line", { f: fmt(d.force_max_final_Eh_bohr, 6), de: (d.energy_final_Eh - d.energy_initial_Eh).toFixed(6) })
    : tf("opt_not_conv_line", { f: fmt(d.force_max_final_Eh_bohr, 6), n: d.iterations });

  const tbody = $("atom-table").querySelector("tbody");
  let html = `<tr><td colspan="8" style="text-align:center;color:var(--muted)">${note}</td></tr>`;
  d.charges.forEach((c, i) => {
    const f = d.forces && d.forces[i] ? d.forces[i] : null;
    const fr = f ? Math.sqrt(f.fx_eh_bohr ** 2 + f.fy_eh_bohr ** 2 + f.fz_eh_bohr ** 2) * EHB2EVA : null;
    html +=
      `<tr><td>${i + 1}</td><td>${c.element}</td><td class="sym">${ELEMENT_SYMBOLS[c.element] || "?"}</td>` +
      `<td class="num">${fmt(c.q, 5)}</td>` +
      (f
        ? `<td class="num">${fmt(f.fx_eh_bohr, 6)}</td><td class="num">${fmt(f.fy_eh_bohr, 6)}</td><td class="num">${fmt(f.fz_eh_bohr, 6)}</td><td class="num">${fmt(fr, 4)}</td>`
        : `<td class="num">–</td><td class="num">–</td><td class="num">–</td><td class="num">–</td>`);
  });
  tbody.innerHTML = html;
  $("table-wrap").hidden = false;
  $("output-tools").hidden = false;
  $("opt-apply").hidden = false;

  renderTrajectory(d);
  window.__lastResult = d;
  window.__lastMode = "optimize";
}

function renderTrajectory(d) {
  const traj = $("traj");
  if (!d.trajectory || d.trajectory.length < 2) { traj.hidden = true; return; }
  traj.hidden = false;
  $("traj-title").textContent = t("traj_title");
  const es = d.trajectory;
  let lo = es[0], hi = es[0];
  for (const e of es) { if (e < lo) lo = e; if (e > hi) hi = e; }
  const pad = (hi - lo) * 0.08 || 1;
  lo -= pad; hi += pad;
  const W = 600, H = 150, L = 8, R = 8, T = 10, B = 20;
  const x = (i) => L + (i / (es.length - 1)) * (W - L - R);
  const y = (e) => T + (1 - (e - lo) / (hi - lo)) * (H - T - B);
  const pts = es.map((e, i) => `${x(i).toFixed(1)},${y(e).toFixed(1)}`).join(" ");
  const last = es.length - 1;
  let svg =
    `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none">` +
    `<line x1="${L}" y1="${y(es[0])}" x2="${x(last)}" y2="${y(es[0])}" stroke="var(--border)" stroke-dasharray="3,4"/>` +
    `<polyline points="${pts}" fill="none" stroke="var(--accent)" stroke-width="2"/>` +
    es.map((e, i) =>
      `<circle cx="${x(i)}" cy="${y(e)}" r="2.6" fill="${i === last ? "var(--good)" : "var(--accent2)"}"/>`
    ).join("") +
    `<text x="${x(0)}" y="${H - 4}" font-size="9" fill="var(--muted)">${fmt(es[0], 6)}</text>` +
    `<text x="${x(last) - 70}" y="${H - 4}" font-size="9" fill="var(--good)">${fmt(es[last], 6)}</text>` +
    `</svg>`;
  $("traj-svg").innerHTML = svg;
}

/* ---- 3Dmol molecular viewer ---- */
let molViewer = null;
let molUnavailable = false;

function xyzTo3Dmol(xyz) {
  const lines = xyz.split(/\r?\n/).filter((l) => l.trim());
  return lines.length + "\n" + "gpuxtb\n" + lines.join("\n") + "\n";
}

function initMoleculeViewer() {
  if (typeof window.$3Dmol === "undefined") { molUnavailable = true; }
  if (molUnavailable) return;
  try {
    molViewer = window.$3Dmol.createViewer($("mol"), {
      backgroundColor: "#0c1420",
      antialias: true,
    });
  } catch (e) {
    molUnavailable = true;
  }
  if (molUnavailable) {
    $("mol").innerHTML = `<div class="mol-placeholder">${t("mol_unavailable")}</div>`;
  }
}

function updateMoleculeViewer(xyz) {
  if (molUnavailable || !molViewer || !xyz || !xyz.trim()) return;
  try {
    molViewer.removeAllModels();
    molViewer.addModel(xyzTo3Dmol(xyz), "xyz");
    molViewer.setStyle({}, {
      sphere: { scale: 0.25, colorscheme: "Jmol" },
      stick: { radius: 0.12, colorscheme: "Jmol" },
    });
    molViewer.zoomTo();
    molViewer.render();
    $("mol-hint").textContent = t("mol_hint");
  } catch (e) { /* ignore per-frame viewer errors */ }
}

/* ---- optimization replay (scrubber over recorded step frames) ---- */
let optFrames = [];
let optSymbols = [];
let replayTimer = null;
let replayPlaying = false;

function renderOptFrame(frame) {
  if (!frame) return;
  $("replay-label").textContent = `#${frame.iter} · ${fmt(frame.energy, 6)} Eh`;
  if (molViewer && !molUnavailable && frame.symbols && frame.symbols.length === frame.natoms) {
    const lines = [];
    for (let i = 0; i < frame.natoms; i++) {
      lines.push(
        `${frame.symbols[i]} ${frame.coords[i * 3].toFixed(6)} ${frame.coords[i * 3 + 1].toFixed(6)} ${frame.coords[i * 3 + 2].toFixed(6)}`,
      );
    }
    updateMoleculeViewer(lines.join("\n"));
  }
}

function stopReplay() {
  if (replayTimer) { clearInterval(replayTimer); replayTimer = null; }
  replayPlaying = false;
  $("replay-play").textContent = "▶";
}

function playReplay() {
  if (optFrames.length < 2) return;
  if (replayPlaying) { stopReplay(); return; }
  replayPlaying = true;
  $("replay-play").textContent = "⏸";
  replayTimer = setInterval(() => {
    const maxV = parseInt($("replay-slider").max, 10);
    let v = parseInt($("replay-slider").value, 10) + 1;
    if (v > maxV) { stopReplay(); return; }
    $("replay-slider").value = String(v);
    renderOptFrame(optFrames[v]);
  }, 150);
}

function showReplay() {
  stopReplay();
  if (optFrames.length < 2) { $("replay").hidden = true; return; }
  const last = optFrames.length - 1;
  $("replay-slider").max = String(last);
  $("replay-slider").value = String(last);
  renderOptFrame(optFrames[last]);
  $("replay").hidden = false;
}

$("replay-play").addEventListener("click", playReplay);
$("replay-slider").addEventListener("input", () => {
  stopReplay();
  const v = parseInt($("replay-slider").value, 10);
  renderOptFrame(optFrames[v]);
});

/* ---- preset wiring ---- */
Object.entries(PRESETS).forEach(([key, p]) => {
  const btn = document.querySelector(`[data-preset="${key}"]`);
  btn.addEventListener("click", () => {
    document.querySelectorAll(".chip").forEach((c) => c.classList.remove("active"));
    btn.classList.add("active");
    $("xyz").value = p.xyz;
    $("charge").value = p.charge;
    $("unpaired").value = p.unpaired;
    updateXyzHint();
    updateMoleculeViewer(p.xyz);
    setError(null);
  });
});

$("xyz").addEventListener("input", updateXyzHint);

function collectOptions() {
  return {
    charge: parseFloat($("charge").value) || 0,
    unpaired: parseInt($("unpaired").value, 10) || 0,
    etempK: parseFloat($("etemp").value) || 0,
    etol: parseFloat($("etol").value) || 1e-8,
    qtol: parseFloat($("qtol").value) || 1e-5,
    maxiter: parseInt($("maxiter").value, 10) || 250,
    forces: $("forces").checked,
  };
}

async function withPending(fn) {
  engineBusy = true;
  syncEngineControls();
  try {
    await fn();
  } finally {
    engineBusy = false;
    syncEngineControls();
  }
}

async function runCompute() {
  const xyz = $("xyz").value;
  if (!xyz.trim()) { setError(t("no_xyz")); return; }
  const o = collectOptions();
  showOverlay("overlay_compute");
  try {
    const t0 = performance.now();
    const m = await callWorker("compute",
      [xyz, o.charge, o.unpaired, o.etempK * K2EH, o.etol, o.qtol, o.maxiter, o.forces ? 1 : 0]);
    const dt = performance.now() - t0;
    const d = JSON.parse(m.raw);
    if (!d.ok) { setError(errorText(d)); return; }
    renderCompute(d);
    updateMoleculeViewer(xyz);
    $("stat-ms").textContent = fmt(dt, 1);
  } catch (e) {
    setError(t("engine_call_fail") + e.message);
  } finally {
    hideOverlay();
  }
}

async function runOptimize() {
  const xyz = $("xyz").value;
  if (!xyz.trim()) { setError(t("no_xyz")); return; }
  const o = collectOptions();
  const optMax = parseInt($("opt-maxiter").value, 10) || 200;
  const gradTol = parseFloat($("opt-gradtol").value) || 4.5e-4;
  const maxMoveAngstrom = parseFloat($("opt-maxmove").value) || 0.4;
  /* No blocking overlay: the engine runs in the worker, so the page stays
   * responsive and the 3Dmol viewer animates each accepted step. */
  const symbols = getElementSymbols(xyz);
  optFrames = [];
  optSymbols = symbols;
  stopReplay();
  $("replay").hidden = true;
  const statusShownAt = performance.now();
  const MIN_STATUS_MS = 400;
  $("mol-status").hidden = false;
  $("mol-status").textContent = tf("opt_running", { n: 0, max: optMax, e: "…" });
  try {
    const t0 = performance.now();
    const m = await callWorker("optimize",
      [xyz, o.charge, o.unpaired, o.etempK * K2EH, o.etol, o.qtol, o.maxiter, optMax, gradTol, angstromToBohr(maxMoveAngstrom)],
      (step) => {
        $("mol-status").textContent = tf("opt_running", { n: step.iter, max: optMax, e: fmt(step.energy, 6) });
        const frame = { iter: step.iter, natoms: step.natoms, coords: step.coords, energy: step.energy, fmax: step.fmax, symbols };
        optFrames.push(frame);
        const idx = optFrames.length - 1;
        $("replay-slider").max = String(idx);
        $("replay-slider").value = String(idx);
        renderOptFrame(frame);
      });
    const dt = performance.now() - t0;
    const d = JSON.parse(m.raw);
    if (!d.ok) { setError(errorText(d)); return; }
    renderOptimize(d);
    updateMoleculeViewer(d.geometry);
    showReplay();
    $("stat-ms").textContent = fmt(dt, 1);
    $("mol-status").textContent = t("opt_done");
  } catch (e) {
    setError(t("engine_call_fail") + e.message);
  } finally {
    $("mol-hint").textContent = t("mol_hint");
    const shown = performance.now() - statusShownAt;
    if (shown < MIN_STATUS_MS) {
      await new Promise((r) => setTimeout(r, MIN_STATUS_MS - shown));
    }
    $("mol-status").hidden = true;
  }
}

$("run").addEventListener("click", () => withPending(runCompute));
$("opt-run").addEventListener("click", () => withPending(runOptimize));
$("opt-apply").addEventListener("click", () => {
  const d = window.__lastResult;
  if (d && d.geometry) {
    $("xyz").value = d.geometry;
    updateXyzHint();
    updateMoleculeViewer(d.geometry);
    setError(t("opt_apply_done"));
    $("opt-apply").hidden = true;
  }
});
$("reset").addEventListener("click", () => {
  document.querySelectorAll(".chip").forEach((c) => c.classList.remove("active"));
  $("xyz").value = PRESETS.water.xyz;
  $("charge").value = 0;
  $("unpaired").value = 0;
  $("etemp").value = 0;
  $("maxiter").value = 250;
  $("etol").value = "1e-8";
  $("qtol").value = "1e-5";
  $("forces").checked = true;
  updateXyzHint();
  updateMoleculeViewer(PRESETS.water.xyz);
  stopReplay();
  optFrames = [];
  $("replay").hidden = true;
  setError(null);
  $("energy").textContent = "—";
  $("energy-ev").textContent = "—";
  $("energy-kcal").textContent = "—";
  $("table-wrap").hidden = true;
  $("output-tools").hidden = true;
  $("opt-apply").hidden = true;
  window.__lastResult = null;
  window.__lastMode = null;
});

$("copy-json").addEventListener("click", async () => {
  if (!window.__lastResult) return;
  try {
    await navigator.clipboard.writeText(JSON.stringify(window.__lastResult, null, 2));
    $("copy-done").hidden = false;
    setTimeout(() => { $("copy-done").hidden = true; }, 1200);
  } catch {
    setError(t("copy_fail"));
  }
});

$("smiles-alert").addEventListener("click", () => {
  setError(t("smiles_msg"));
});

/* ---- bootstrap ---- */
(async () => {
  applyI18n();
  engineState = "loading";
  refreshBadge();
  const LOAD_TIMEOUT_MS = 60000;
  const abortController = new AbortController();
  try {
    showOverlay("overlay_loading");
    updateLoader(0);
    const initialize = (async () => {
      // Download the main wasm on the UI thread with a real progress bar, then
      // wait for .data loading, module instantiation, and side-module readiness
      // inside the worker before considering the engine loaded.
      const wasmUrl = new URL("gpuxtb_web.wasm", import.meta.url).href;
      const wasmBinary = await fetchProgress(wasmUrl, { signal: abortController.signal });
      updateLoader(100);
      $("overlay-text").textContent = "…";
      $("load-bar-wrap").hidden = true;
      await initWorker(wasmBinary);
    })();
    await withTimeout(initialize, LOAD_TIMEOUT_MS, () => {
      abortController.abort();
      if (worker) worker.terminate();
    });
    hideOverlay();
  } catch (e) {
    abortController.abort();
    if (worker) {
      worker.terminate();
      worker = null;
    }
    engineState = "error";
    rejectPendingCalls(e instanceof Error ? e : new Error(String(e)));
    refreshBadge();
    hideOverlay();
    setError(String(e && e.message).includes("TIME_OUT") ? t("load_timeout") : t("load_fail") + String(e && e.message));
    $("retry").hidden = false;
  }
})();
