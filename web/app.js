/* xTBloom web demo front end (bilingual zh/en).
 * Loads the wasm32 module through the Emscripten factory and wires the two
 * adapter entry points (single-point compute, L-BFGS geometry optimize) to
 * the input/output panels. All user-facing strings come from the I18N
 * dictionary below; errors arrive from wasm as stable ASCII codes. */

const appModuleUrl = new URL(import.meta.url);
const appContentVersion = appModuleUrl.searchParams.get("xtbloom_version");
const appBootstrapToken = appModuleUrl.searchParams.get("xtbloom_bootstrap");
const appHelpersUrl = new URL("./app_helpers.js", import.meta.url);
if (appContentVersion) appHelpersUrl.searchParams.set("xtbloom_version", appContentVersion);
if (appBootstrapToken) appHelpersUrl.searchParams.set("xtbloom_bootstrap", appBootstrapToken);

function currentAppImportOrAbort() {
  if (
    appBootstrapToken &&
    globalThis.__XTBLOOM_APP_BOOT_TOKEN !== appBootstrapToken
  ) {
    throw new DOMException("Application import superseded", "AbortError");
  }
}

currentAppImportOrAbort();
const {
  angstromToBohr,
  canStartUrlSmiles,
  clampProgressPercent,
  fetchResourceBatch,
  initializeWorker,
  isRetryableLoadError,
  postToReadyWorker,
  readSmilesQuery,
  runWithRetries,
  validateEngineManifest,
  withTimeout,
} = await import(appHelpersUrl.href);
currentAppImportOrAbort();

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
    tagline: "无需安装或上传，在浏览器中运行 GFN2-xTB 单点与几何优化",
    engine_loading: "引擎加载中…",
    panel_input: "输入",
    presets_label: "模板分子",
    preset_ketene: "乙烯酮",
    preset_ethanol: "乙醇",
    preset_benzene: "苯",
    smiles_label: "SMILES → 三维结构",
    smiles_placeholder: "例如：CCO 或 c1ccccc1",
    smiles_download_button: "正在下载…",
    smiles_download_status: "正在后台下载 SMILES 三维结构生成资源；其他功能不受影响。",
    smiles_ready_button: "生成三维结构",
    smiles_ready_status: "结构生成器已就绪。将补全显式氢并执行 MMFF94 预优化。",
    smiles_retry_button: "重试下载",
    smiles_generate_button: "正在生成…",
    smiles_generate_status: "正在生成显式氢三维构象并进行 MMFF94 预优化…",
    smiles_generated: "已生成 {{n}} 个原子的三维结构；形式电荷 {{q}}。",
    smiles_load_failed: "SMILES 结构生成资源加载失败：{{e}}",
    smiles_url_optimizing: "已从地址读取 SMILES，正在自动执行 xTBloom 几何优化…",
    smiles_url_done: "地址中的 SMILES 已生成并优化；最终坐标已写回输入框。",
    smiles_url_failed: "地址中的 SMILES 自动生成/优化失败：{{e}}",
    smiles_go: "去生成",
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
    stat_opt_steps: "优化步数",
    stat_conv: "收敛",
    stat_ms: "耗时 ms",
    th_atom: "原子", th_q: "电荷 q",
    th_fx: "Fx (Eh/bohr)", th_fy: "Fy (Eh/bohr)", th_fz: "Fz (Eh/bohr)", th_fmag: "|F| (eV/Å)",
    copy_json: "复制 JSON",
    opt_apply: "把优化坐标填回输入框",
    copy_done: "已复制",
    roadmap: "浏览器中可以尝试",
    roadmap_smiles_title: "SMILES → 结构",
    roadmap_smiles_desc: "输入 SMILES，生成显式氢三维构象并进行 MMFF94 预优化。",
    roadmap_opt_title: "几何优化",
    tag_done: "已支持",
    roadmap_opt_desc: "内置 L-BFGS 优化器，使用解析力收敛到稳定结构。在左侧“优化”区配置后点击“几何优化”。",
    opt_go: "去优化",
    footer: "由 xTBloom 驱动 —— 同一套 C ABI 的 C++17 原生库编译为不依赖 Memory64 的 wasm32；可选的 SMILES 三维结构由固定版本的 OpenChemLib 在浏览器中生成并以 MMFF94 预优化。BLAS/LAPACK 层为演示用最小实现。仅供演示，非科学计算生产环境。",
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
    engine_ok: "引擎就绪",
    engine_fail: "引擎加载失败",
    load_fail: "无法加载 wasm32 引擎。请使用支持 WebAssembly 和模块化 Web Worker 的现代浏览器。\n详情：",
    load_timeout: "多次加载超时——网络可能较慢，请重试。若持续失败请检查是否能正常访问本页面资源。",
    load_retry: "重试",
    load_checking: "正在检查引擎资源版本…",
    load_downloading: "正在读取引擎资源（缓存或网络，第 {{attempt}}/{{max}} 次）…",
    load_progress_known: "{{done}}/{{total}} 个文件 · {{loaded}} / {{size}} · {{pct}}%",
    load_progress_unknown: "{{done}}/{{total}} 个文件 · 已接收 {{loaded}}",
    load_initializing: "{{done}}/{{total}} 个文件已就绪，正在初始化引擎…",
    load_retrying: "网络暂时不可用，{{wait}} 秒后自动重试（{{attempt}}/{{max}}）…",
    traj_title: "能量迭代轨迹（Eh）",
    engine_call_fail: "引擎调用失败：",
    err_xyz_parse: "无法解析坐标：每行请提供「元素符号 x y z」，单位 Å",
    err_xyz_too_many: "原子数超过 512 上限",
    err_ctx: "计算上下文创建失败：{{e}}",
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
    smiles_err_empty: "请输入 SMILES。",
    smiles_err_too_long: "SMILES 过长（最多 2048 个字符）。",
    smiles_err_parse: "无法解析 SMILES。",
    smiles_err_conformer: "无法生成无碰撞的三维构象。",
    smiles_err_fragments: "暂不支持用点号连接的盐或多片段 SMILES；其片段间相对位置没有定义。",
    smiles_err_atoms: "SMILES 没有生成有效原子。",
    smiles_err_too_many: "补氢后原子数超过 512 上限。",
    smiles_err_element: "SMILES 含有 GFN2-xTB 不支持的元素。",
    smiles_err_radical: "自动流程不猜测自旋：自由基请改用 XYZ，并手动填写未配对电子数。",
    smiles_err_mmff: "MMFF94 预优化失败。",
    smiles_err_coords: "结构生成器返回了非有限坐标。",
    smiles_err_library: "SMILES 结构生成器尚未就绪。",
    smiles_err_timeout: "SMILES 三维结构生成超时，请缩短分子或重试。",
    smiles_err_unknown: "SMILES 三维结构生成失败。",
    err_unknown: "未知错误",
  },
  en: {
    tagline: "Run GFN2-xTB in your browser—no install or uploads",
    engine_loading: "engine loading…",
    panel_input: "Input",
    presets_label: "Preset molecules",
    preset_ketene: "Ketene",
    preset_ethanol: "Ethanol",
    preset_benzene: "Benzene",
    smiles_label: "SMILES → 3D structure",
    smiles_placeholder: "e.g. CCO or c1ccccc1",
    smiles_download_button: "Downloading…",
    smiles_download_status: "Downloading the SMILES 3D generator in the background; other features remain available.",
    smiles_ready_button: "Generate 3D",
    smiles_ready_status: "Structure generator ready. Explicit hydrogens and an MMFF94 pre-relaxation will be applied.",
    smiles_retry_button: "Retry download",
    smiles_generate_button: "Generating…",
    smiles_generate_status: "Generating an explicit-hydrogen 3D conformer and running MMFF94 pre-relaxation…",
    smiles_generated: "Generated a {{n}}-atom 3D structure with formal charge {{q}}.",
    smiles_load_failed: "Could not load the SMILES structure generator: {{e}}",
    smiles_url_optimizing: "SMILES read from the URL; running automatic xTBloom geometry optimization…",
    smiles_url_done: "The URL SMILES was generated and optimized; final coordinates were written to the input.",
    smiles_url_failed: "Automatic URL SMILES generation/optimization failed: {{e}}",
    smiles_go: "Generate",
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
    stat_opt_steps: "optimization steps",
    stat_conv: "conv.",
    stat_ms: "time ms",
    th_atom: "Atom", th_q: "charge q",
    th_fx: "Fx (Eh/bohr)", th_fy: "Fy (Eh/bohr)", th_fz: "Fz (Eh/bohr)", th_fmag: "|F| (eV/Å)",
    copy_json: "Copy JSON",
    opt_apply: "Load optimized coords back to input",
    copy_done: "copied",
    roadmap: "Try in the browser",
    roadmap_smiles_title: "SMILES → structure",
    roadmap_smiles_desc: "Generate an explicit-hydrogen 3D conformer from SMILES and pre-relax it with MMFF94.",
    roadmap_opt_title: "Geometry optimization",
    tag_done: "supported",
    roadmap_opt_desc: "Built-in L-BFGS optimizer using analytic forces. Configure it in the left panel, then click “Optimize geometry”.",
    opt_go: "Try it",
    footer: "Powered by xTBloom — the same native C ABI library compiled to wasm32 without requiring Memory64. Optional SMILES 3D structures are generated in-browser by a pinned OpenChemLib release and pre-relaxed with MMFF94. The BLAS/LAPACK layer is a minimal demo implementation. Demo only, not a production scientific environment.",
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
    engine_ok: "engine ready",
    engine_fail: "engine failed to load",
    load_fail: "Could not load the wasm32 engine. Use a modern browser with WebAssembly and module Worker support.\nDetails: ",
    load_timeout: "Repeated loading attempts timed out. Please retry; if it persists, check that the page assets are reachable.",
    load_retry: "Retry",
    load_checking: "Checking the engine resource version…",
    load_downloading: "Loading engine resources from cache or network (attempt {{attempt}}/{{max}})…",
    load_progress_known: "{{done}}/{{total}} files · {{loaded}} / {{size}} · {{pct}}%",
    load_progress_unknown: "{{done}}/{{total}} files · {{loaded}} received",
    load_initializing: "{{done}}/{{total}} files ready; initializing the engine…",
    load_retrying: "Network loading failed temporarily; retrying in {{wait}} s ({{attempt}}/{{max}})…",
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
    smiles_err_empty: "Enter a SMILES string.",
    smiles_err_too_long: "The SMILES is too long (maximum 2048 characters).",
    smiles_err_parse: "Could not parse the SMILES string.",
    smiles_err_conformer: "Could not generate a collision-free 3D conformer.",
    smiles_err_fragments: "Dot-disconnected salts or multi-fragment SMILES are not supported because their relative placement is undefined.",
    smiles_err_atoms: "The SMILES produced no valid atoms.",
    smiles_err_too_many: "The explicit-hydrogen structure exceeds the 512-atom limit.",
    smiles_err_element: "The SMILES contains an element unsupported by GFN2-xTB.",
    smiles_err_radical: "Spin is not guessed: use XYZ for radicals and enter the unpaired-electron count explicitly.",
    smiles_err_mmff: "MMFF94 pre-optimization failed.",
    smiles_err_coords: "The structure generator returned non-finite coordinates.",
    smiles_err_library: "The SMILES structure generator is not ready.",
    smiles_err_timeout: "SMILES 3D generation timed out; use a smaller molecule or retry.",
    smiles_err_unknown: "SMILES 3D generation failed.",
    err_unknown: "Unknown error",
  },
};

/* ------------------------------------------------------------------ */

const $ = (id) => document.getElementById(id);

let lang = "zh";
try {
  lang = localStorage.getItem("xtbloom-lang") ||
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
  document.title = lang === "zh" ? "xTBloom · 浏览器内运行 GFN2-xTB" : "xTBloom · Run GFN2-xTB in your browser";
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
  syncSmilesControls();
  updateXyzHint();
}

$("lang-toggle").addEventListener("click", () => {
  lang = lang === "zh" ? "en" : "zh";
  try { localStorage.setItem("xtbloom-lang", lang); } catch { /* ignore */ }
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
let engineLoadGeneration = 0;
let engineLoadController = null;

/* The optional OpenChemLib worker has an independent lifecycle: its CDN
 * download or conformer search must never gate ordinary XYZ/xTBloom controls. */
let smilesWorker = null;
let smilesResourceState = "loading"; /* loading | ready | error */
let smilesBusy = false;
let smilesMsgSeq = 0;
const smilesPending = new Map();
let smilesLoadTimer = null;
let smilesStatusKey = "smiles_download_status";
let smilesStatusVars = null;
let smilesStatusTone = "";
let urlSmiles = null;
let urlSmilesStarted = false;

function setSmilesStatus(key, vars = null, tone = "") {
  smilesStatusKey = key;
  smilesStatusVars = vars;
  smilesStatusTone = tone;
  syncSmilesControls();
}

function syncSmilesControls() {
  const input = $("smiles");
  const button = $("smiles-generate");
  const spinner = $("smiles-spinner");
  const label = $("smiles-button-text");
  const status = $("smiles-status");
  if (!input || !button || !spinner || !label || !status) return;

  const loading = smilesResourceState === "loading";
  const failed = smilesResourceState === "error";
  const generating = smilesBusy;
  button.disabled = loading || generating || engineBusy ||
    (!failed && input.value.trim() === "");
  spinner.hidden = !(loading || generating);
  label.textContent = generating
    ? t("smiles_generate_button")
    : loading
      ? t("smiles_download_button")
      : failed
        ? t("smiles_retry_button")
        : t("smiles_ready_button");
  status.textContent = t(smilesStatusKey, smilesStatusVars || undefined);
  status.classList.toggle("ok", smilesStatusTone === "ok");
  status.classList.toggle("err", smilesStatusTone === "err");
}

function rejectSmilesPending(error) {
  for (const entry of smilesPending.values()) entry.reject(error);
  smilesPending.clear();
}

function failSmilesWorker(error) {
  if (smilesLoadTimer !== null) {
    clearTimeout(smilesLoadTimer);
    smilesLoadTimer = null;
  }
  if (smilesWorker) smilesWorker.terminate();
  smilesWorker = null;
  smilesResourceState = "error";
  smilesBusy = false;
  rejectSmilesPending(error);
  setSmilesStatus("smiles_load_failed", { e: error.message }, "err");
}

function handleSmilesWorkerMessage(message) {
  if (message.type === "ready") {
    if (smilesLoadTimer !== null) {
      clearTimeout(smilesLoadTimer);
      smilesLoadTimer = null;
    }
    smilesResourceState = "ready";
    setSmilesStatus("smiles_ready_status", null, "ok");
    void maybeRunUrlSmiles();
    return;
  }
  if (message.type === "load-error") {
    failSmilesWorker(new Error(String(message.error || "OpenChemLib load failed")));
    return;
  }
  if (message.type !== "result") return;
  const entry = smilesPending.get(message.id);
  if (!entry) return;
  smilesPending.delete(message.id);
  if (message.ok) {
    entry.resolve(message.result);
  } else {
    const error = new Error(String(message.error || "SMILES generation failed"));
    error.code = message.errorCode || "smiles_err_unknown";
    entry.reject(error);
  }
}

function startSmilesWorker() {
  if (smilesLoadTimer !== null) clearTimeout(smilesLoadTimer);
  if (smilesWorker) smilesWorker.terminate();
  rejectSmilesPending(new Error("SMILES worker restarted"));
  smilesResourceState = "loading";
  smilesBusy = false;
  setSmilesStatus("smiles_download_status");
  try {
    smilesWorker = new Worker(new URL("./smiles_worker.js", import.meta.url), { type: "module" });
  } catch (error) {
    failSmilesWorker(error instanceof Error ? error : new Error(String(error)));
    return;
  }
  smilesWorker.onmessage = (event) => handleSmilesWorkerMessage(event.data);
  smilesWorker.onerror = (event) => {
    failSmilesWorker(new Error((event && event.message) || "SMILES worker error"));
  };
  smilesLoadTimer = setTimeout(() => {
    failSmilesWorker(new Error("OpenChemLib resource download timed out"));
  }, 60000);
}

function callSmilesWorker(smiles) {
  return new Promise((resolve, reject) => {
    if (
      smilesResourceState !== "ready" || !smilesWorker ||
      typeof smilesWorker.postMessage !== "function"
    ) {
      const error = new Error("OpenChemLib is not ready");
      error.code = "smiles_err_library";
      reject(error);
      return;
    }
    const id = ++smilesMsgSeq;
    smilesPending.set(id, { resolve, reject });
    try {
      smilesWorker.postMessage({ type: "generate", id, smiles });
    } catch (error) {
      smilesPending.delete(id);
      reject(error);
    }
  });
}

async function requestSmilesGeometry(smiles) {
  const GENERATION_TIMEOUT_MS = 30000;
  try {
    return await withTimeout(callSmilesWorker(smiles), GENERATION_TIMEOUT_MS, () => {
      const error = new Error("SMILES generation timed out");
      error.code = "smiles_err_timeout";
      failSmilesWorker(error);
    });
  } catch (error) {
    if (error instanceof Error && error.message === "TIME_OUT") {
      error.code = "smiles_err_timeout";
    }
    throw error;
  }
}

function smilesErrorText(error) {
  if (error && error.code) {
    return t(error.code, { e: error.message || "" });
  }
  return error && error.message ? error.message : t("smiles_err_unknown");
}

function syncEngineControls() {
  const enabled = engineState === "ready" && worker !== null &&
    !engineBusy && !smilesBusy;
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

function engineMessageError(message, fallback = "worker error") {
  const detail = message && typeof message.error === "object"
    ? message.error
    : { message: message?.error };
  const error = new Error(String(detail?.message || fallback));
  if (detail?.name) error.name = String(detail.name);
  if (detail?.phase) error.phase = String(detail.phase);
  return error;
}

function handleWorkerMessage(sourceWorker, generation, m) {
  // A failed attempt may deliver a late ready/error/result after its successor
  // started. Only the currently published Worker may affect UI or promises.
  if (generation !== engineLoadGeneration || sourceWorker !== worker) return;
  if (m.type === "error") {
    const error = engineMessageError(m);
    sourceWorker.terminate();
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

async function createEngineWorker(
  generation,
  { workerUrl, moduleUrl, helpersUrl, wasmBinary, dataBinary },
  onCreate = () => {},
) {
  const candidate = new Worker(workerUrl, { type: "module" });
  onCreate(candidate);
  try {
    const ready = await initializeWorker(candidate, {
      wasmBinary,
      dataBinary,
      moduleUrl,
      helpersUrl,
    }, (message) => handleWorkerMessage(candidate, generation, message));
    return { candidate, ready };
  } catch (error) {
    candidate.terminate();
    throw error;
  }
}

function publishReadyEngine(candidate, ready) {
  if (worker && worker !== candidate) worker.terminate();
  worker = candidate;
  engineState = "ready";
  refreshBadge();
  $("ver-badge").textContent = "v" + ready.version;
  /* Preserve coordinates entered or generated while the WASM worker was
   * loading; only supply the water example when the editor is still empty. */
  if (!$("xyz").value.trim()) $("xyz").value = PRESETS.water.xyz;
  updateXyzHint();
  initMoleculeViewer();
  updateMoleculeViewer($("xyz").value);
  void maybeRunUrlSmiles();
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
$("retry").addEventListener("click", () => { void startEngineLoad({ forceReload: true }); });

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
const ENGINE_LOAD_MAX_ATTEMPTS = 3;
const ENGINE_ATTEMPT_TIMEOUT_MS = 60000;
const ENGINE_ASSET_IDS = ["worker", "helpers", "module", "wasm", "data"];

function formatByteCount(value) {
  const bytes = Math.max(0, Number(value) || 0);
  const units = ["B", "KB", "MB", "GB"];
  let scaled = bytes;
  let unit = 0;
  while (scaled >= 1000 && unit < units.length - 1) {
    scaled /= 1000;
    unit++;
  }
  const digits = unit === 0 || scaled >= 100 ? 0 : scaled >= 10 ? 1 : 2;
  return `${scaled.toFixed(digits)} ${units[unit]}`;
}

async function loadEngineManifest(attempt, forceReload, signal) {
  $("overlay-text").textContent = t("load_checking");
  if (attempt === 1 && !forceReload && globalThis.__XTBLOOM_BOOTSTRAP_MANIFEST) {
    const bootstrapped = globalThis.__XTBLOOM_BOOTSTRAP_MANIFEST;
    delete globalThis.__XTBLOOM_BOOTSTRAP_MANIFEST;
    return validateManifestForLoadedApp(bootstrapped);
  }
  const url = new URL("engine-manifest.json", import.meta.url);
  const bustCache = forceReload || attempt > 1;
  let response;
  try {
    response = await fetch(url, {
      signal,
      cache: bustCache ? "reload" : "no-cache",
    });
  } catch (cause) {
    const error = new Error("Network error while checking the engine manifest", { cause });
    error.retryable = true;
    throw error;
  }
  if (!response.ok) {
    const error = new Error(`HTTP ${response.status} while checking the engine manifest`);
    error.status = response.status;
    throw error;
  }
  let rawManifest;
  try {
    rawManifest = await response.json();
  } catch (cause) {
    const error = new Error("Invalid engine manifest response", { cause });
    error.retryable = true;
    throw error;
  }
  return validateManifestForLoadedApp(rawManifest);
}

function validateManifestForLoadedApp(rawManifest) {
  const manifest = validateEngineManifest(rawManifest, ["app", ...ENGINE_ASSET_IDS]);
  // A deployment may finish between retry attempts. Never combine an already
  // evaluated app/helper graph with a newer Worker/WASM/data generation.
  if (appContentVersion && manifest.version !== appContentVersion) {
    const error = new TypeError("An engine update was detected; refresh the page to load it coherently");
    error.retryable = false;
    throw error;
  }
  return manifest;
}

function engineAssets(manifest) {
  const engineIds = new Set(ENGINE_ASSET_IDS);
  return manifest.assets.filter((asset) => engineIds.has(asset.id)).map((asset) => {
    const url = new URL(asset.path, import.meta.url);
    url.searchParams.set("xtbloom_version", manifest.version);
    return { ...asset, url: url.href };
  });
}

function updateLoader(progress, attempt, reset = false) {
  const bounded = clampProgressPercent(progress.barPercent);
  __loadingPct = reset ? bounded : Math.max(__loadingPct, bounded);
  $("load-bar-wrap").classList.remove("indeterminate");
  $("load-bar-wrap").hidden = false;
  $("load-bar-fill").style.width = __loadingPct + "%";
  const vars = {
    done: progress.completedFiles,
    total: progress.totalFiles,
    loaded: formatByteCount(progress.loadedBytes),
    size: progress.totalBytes === null ? "" : formatByteCount(progress.totalBytes),
    pct: progress.percent === null ? "" : Math.round(progress.percent),
  };
  $("load-bar-text").textContent = tf(
    progress.totalBytes === null ? "load_progress_unknown" : "load_progress_known",
    vars,
  );
  $("overlay-text").textContent = tf("load_downloading", {
    attempt,
    max: ENGINE_LOAD_MAX_ATTEMPTS,
  });
}

function setLoaderInitializing(progress) {
  $("load-bar-fill").style.width = "100%";
  $("load-bar-text").textContent = tf(
    progress.totalBytes === null ? "load_progress_unknown" : "load_progress_known",
    {
      done: progress.completedFiles,
      total: progress.totalFiles,
      loaded: formatByteCount(progress.loadedBytes),
      size: progress.totalBytes === null ? "" : formatByteCount(progress.totalBytes),
      pct: 100,
    },
  );
  $("overlay-text").textContent = tf("load_initializing", {
    done: progress.completedFiles,
    total: progress.totalFiles,
  });
}

function setLoaderRetrying({ nextAttempt, maxAttempts, waitMs }) {
  $("overlay-text").textContent = tf("load_retrying", {
    wait: Math.max(1, Math.ceil(waitMs / 1000)),
    attempt: nextAttempt,
    max: maxAttempts,
  });
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
  $("stat-iter-label").textContent = t("stat_iter");
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
  /* The optimizer reports L-BFGS steps here, not the final single point's SCC
   * iteration count. Keep the visible label coupled to the result mode. */
  $("stat-iter-label").textContent = t("stat_opt_steps");
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
  return lines.length + "\n" + "xTBloom\n" + lines.join("\n") + "\n";
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

function applyGeneratedGeometry(result) {
  document.querySelectorAll(".chip").forEach((chip) => chip.classList.remove("active"));
  $("xyz").value = result.xyz;
  $("charge").value = String(result.formalCharge);
  /* Radical SMILES are rejected by the helper, so zero is the only supported
   * automatic spin state. Users retain the explicit XYZ route for radicals. */
  $("unpaired").value = "0";
  updateXyzHint();
  updateMoleculeViewer(result.xyz);
  setError(null);
}

async function generateSmilesGeometry() {
  const smiles = $("smiles").value.trim();
  if (!smiles) {
    const error = new Error("SMILES is empty");
    error.code = "smiles_err_empty";
    throw error;
  }
  smilesBusy = true;
  setSmilesStatus("smiles_generate_status");
  syncEngineControls();
  try {
    const result = await requestSmilesGeometry(smiles);
    applyGeneratedGeometry(result);
    setSmilesStatus(
      "smiles_generated",
      { n: result.atomCount, q: result.formalCharge },
      "ok",
    );
    return result;
  } finally {
    smilesBusy = false;
    syncSmilesControls();
    syncEngineControls();
    queueMicrotask(() => void maybeRunUrlSmiles());
  }
}

async function handleManualSmiles() {
  if (smilesResourceState === "error") {
    startSmilesWorker();
    return;
  }
  try {
    await generateSmilesGeometry();
  } catch (error) {
    setSmilesStatus(
      error && error.code ? error.code : "smiles_err_unknown",
      { e: error && error.message ? error.message : "" },
      "err",
    );
  }
}

async function maybeRunUrlSmiles() {
  if (!canStartUrlSmiles({
    smiles: urlSmiles,
    started: urlSmilesStarted,
    engineState,
    smilesState: smilesResourceState,
    engineBusy,
    smilesBusy,
  })) {
    return;
  }
  urlSmilesStarted = true;
  $("smiles").value = urlSmiles;
  syncSmilesControls();
  try {
    await generateSmilesGeometry();
    setSmilesStatus("smiles_url_optimizing");
    const optimized = await withPending(() => runOptimize({
      applyFinalGeometry: true,
      throwOnFailure: true,
    }));
    if (!optimized) throw new Error("xTBloom geometry optimization failed");
    setSmilesStatus("smiles_url_done", null, "ok");
  } catch (error) {
    const detail = smilesErrorText(error);
    setSmilesStatus("smiles_url_failed", { e: detail }, "err");
    setError(t("smiles_url_failed", { e: detail }));
  }
}

$("smiles").addEventListener("input", syncSmilesControls);
$("smiles").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !$("smiles-generate").disabled) {
    event.preventDefault();
    void handleManualSmiles();
  }
});
$("smiles-generate").addEventListener("click", () => void handleManualSmiles());

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
  syncSmilesControls();
  try {
    return await fn();
  } finally {
    engineBusy = false;
    syncEngineControls();
    syncSmilesControls();
    queueMicrotask(() => void maybeRunUrlSmiles());
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

async function runOptimize({
  applyFinalGeometry = false,
  throwOnFailure = false,
} = {}) {
  const xyz = $("xyz").value;
  if (!xyz.trim()) {
    const error = new Error(t("no_xyz"));
    if (throwOnFailure) throw error;
    setError(error.message);
    return null;
  }
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
    if (!d.ok) {
      const error = new Error(errorText(d));
      error.code = d.error_code || "err_opt";
      if (throwOnFailure) throw error;
      setError(error.message);
      return null;
    }
    renderOptimize(d);
    updateMoleculeViewer(d.geometry);
    showReplay();
    $("stat-ms").textContent = fmt(dt, 1);
    $("mol-status").textContent = t("opt_done");
    if (applyFinalGeometry) {
      /* URL-triggered optimization is a complete import operation: publish
       * the converged geometry back to the canonical XYZ editor immediately. */
      $("xyz").value = d.geometry;
      updateXyzHint();
      updateMoleculeViewer(d.geometry);
      $("opt-apply").hidden = true;
    }
    return d;
  } catch (e) {
    const error = e && e.code
      ? e
      : new Error(t("engine_call_fail") + (e && e.message ? e.message : String(e)));
    if (throwOnFailure) throw error;
    setError(error.message);
    return null;
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

function currentLoadOrAbort(generation, ...signals) {
  if (generation !== engineLoadGeneration || signals.some((signal) => signal.aborted)) {
    throw new DOMException("Engine load superseded", "AbortError");
  }
}

async function loadEngineAttempt(generation, attempt, forceReload, masterSignal) {
  const attemptController = new AbortController();
  const abortAttempt = () => attemptController.abort();
  masterSignal.addEventListener("abort", abortAttempt, { once: true });
  if (masterSignal.aborted) abortAttempt();

  let candidate = null;
  let keepCandidate = false;
  try {
    const initialize = (async () => {
      const manifest = await loadEngineManifest(
        attempt,
        forceReload,
        attemptController.signal,
      );
      currentLoadOrAbort(generation, masterSignal, attemptController.signal);
      const resources = engineAssets(manifest);
      const urls = new Map(resources.map((resource) => [resource.id, resource.url]));
      let resetProgress = true;
      let lastProgress = {
        totalFiles: resources.length,
        completedFiles: 0,
        loadedBytes: 0,
        totalBytes: null,
        percent: null,
        barPercent: 0,
        complete: false,
      };

      const payloads = await fetchResourceBatch(resources, {
        signal: attemptController.signal,
        cache: forceReload || attempt > 1 ? "reload" : "default",
        onProgress: (progress) => {
          if (generation !== engineLoadGeneration || attemptController.signal.aborted) return;
          lastProgress = progress;
          updateLoader(progress, attempt, resetProgress);
          resetProgress = false;
        },
      });
      currentLoadOrAbort(generation, masterSignal, attemptController.signal);
      setLoaderInitializing(lastProgress);
      const created = await createEngineWorker(generation, {
        workerUrl: urls.get("worker"),
        moduleUrl: urls.get("module"),
        helpersUrl: urls.get("helpers"),
        wasmBinary: payloads.get("wasm"),
        dataBinary: payloads.get("data"),
      }, (createdWorker) => { candidate = createdWorker; });
      currentLoadOrAbort(generation, masterSignal, attemptController.signal);
      return created;
    })();

    const result = await withTimeout(initialize, ENGINE_ATTEMPT_TIMEOUT_MS, () => {
      abortAttempt();
      if (candidate) candidate.terminate();
    });
    currentLoadOrAbort(generation, masterSignal, attemptController.signal);
    keepCandidate = true;
    return result;
  } finally {
    masterSignal.removeEventListener("abort", abortAttempt);
    abortAttempt();
    if (candidate && !keepCandidate) candidate.terminate();
  }
}

async function startEngineLoad({ forceReload = false } = {}) {
  const generation = ++engineLoadGeneration;
  if (engineLoadController) engineLoadController.abort();
  const controller = new AbortController();
  engineLoadController = controller;
  if (worker) worker.terminate();
  worker = null;
  engineState = "loading";
  engineBusy = false;
  rejectPendingCalls(new Error("engine reloading"));
  refreshBadge();
  $("retry").hidden = true;
  setError("");
  showOverlay("overlay_loading");
  __loadingPct = 0;
  $("load-bar-wrap").hidden = true;
  $("load-bar-fill").style.width = "0%";
  let loaded = null;

  try {
    loaded = await runWithRetries(
      (attempt) => loadEngineAttempt(
        generation,
        attempt,
        forceReload,
        controller.signal,
      ),
      {
        maxAttempts: ENGINE_LOAD_MAX_ATTEMPTS,
        shouldRetry: isRetryableLoadError,
        signal: controller.signal,
        onRetry: (retry) => {
          if (generation === engineLoadGeneration && !controller.signal.aborted) {
            setLoaderRetrying(retry);
          }
        },
      },
    );
    currentLoadOrAbort(generation, controller.signal);
    publishReadyEngine(loaded.candidate, loaded.ready);
    hideOverlay();
  } catch (error) {
    if (loaded?.candidate && worker !== loaded.candidate) loaded.candidate.terminate();
    if (generation !== engineLoadGeneration || controller.signal.aborted) return;
    if (worker) worker.terminate();
    worker = null;
    engineState = "error";
    rejectPendingCalls(error instanceof Error ? error : new Error(String(error)));
    refreshBadge();
    hideOverlay();
    const message = String(error?.message || error);
    setError(message.includes("TIME_OUT") ? t("load_timeout") : t("load_fail") + message);
    $("retry").hidden = false;
  } finally {
    if (generation === engineLoadGeneration && engineLoadController === controller) {
      engineLoadController = null;
    }
  }
}

/* ---- bootstrap ---- */
(async () => {
  applyI18n();
  /* Start the optional CDN dependency immediately, but never await it here:
   * wasm32 startup and the ordinary XYZ workflow remain independent. */
  startSmilesWorker();
  try {
    urlSmiles = readSmilesQuery(window.location.href);
    if (urlSmiles) {
      $("smiles").value = urlSmiles;
      syncSmilesControls();
    }
  } catch (error) {
    urlSmilesStarted = true;
    const detail = smilesErrorText(error);
    setSmilesStatus("smiles_url_failed", { e: detail }, "err");
    setError(t("smiles_url_failed", { e: detail }));
  }
  await startEngineLoad();
})();
