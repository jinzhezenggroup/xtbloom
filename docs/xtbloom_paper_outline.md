# xTBloom 论文大纲

## 基于定稿实验方案的正文结构

## 1. 全文核心叙事

全文围绕一条可由现有实验规模闭合的证据链展开：

```text
真实 GFN2-xTB 工作负载具有尺寸异构、SCC 难度不一和部分失败的特点
→ 逐体系执行和串行参考基线不能充分利用现代多核 CPU/GPU
→ xTBloom 以原生 ragged batch、持久运行时和 CPU/CUDA 专用路径重组计算
→ 先证明数值等价与失败语义，再以 CUDA batch/crossover 为核心性能证据，并用
  单进程 CPU public-interface 对照和真实异构 workload 说明原生 batch 的端到端价值
→ 给出有效性能区间、收敛表现与边界，而不是单一最大加速比
```

正文固定回答四个问题：

1. xTBloom 在 canonical corpus、QM9 和 OMol25 上是否与 xTB/tblite 数值等价？
2. CUDA 在什么 AO 规模和 batch size 下优于 CPU，数据驻留与显存容量如何改变边界？
3. 在相同单进程和 CPU thread budget 下，xTBloom native batch 相对参考程序逐体系
   public-API loop 的端到端收益是多少？
4. 面对复杂化学空间、SCC 长尾和体系级失败，xTBloom 的收敛率与 ragged runtime 表现如何？

文章不写成功能目录。WARM、QM/MM、GFN1、第二套硬件和完整组件消融不是正文主实验；只有在补充材料中有相应证据时，才可作为次要能力或未来方向提及。

建议中心主张写成：

> xTBloom reorganizes fixed-parameter GFN2-xTB calculations around a failure-isolated ragged-batch runtime with a dedicated CUDA execution path and an accompanying multicore CPU executor, and characterizes its numerical equivalence, convergence behavior, and performance regimes on controlled and real molecular workloads.

这是一条待实验支持的主张。若真实数据中的数值门槛、收敛率或性能置信区间未通过，摘要和结论必须改为报告实际观察，不能保留预设结论。

---

## 2. 建议标题

### 推荐标题

**xTBloom: A CUDA-Accelerated Ragged-Batch Runtime for High-Throughput GFN2-xTB Calculations**

### 备选标题

**xTBloom: Native Ragged Batching for CUDA and Multicore GFN-xTB Workloads**

若后续 GFN1 只保留在 SI，正文标题优先使用更精确的 `GFN2-xTB`；只有当 GFN1 证据也足以支撑标题范围时，才采用更宽的 `GFN-xTB`。

### 标题边界

标题可以明确 CUDA 是性能主线，但必须同时保留 `ragged-batch runtime`，避免把工作
误解为只替换若干 GPU kernel 的单纯 port。多核 CPU 是同一运行时的配套执行路径，
不需要与 CUDA 在标题中占据相同权重。

不建议把 WARM、QM/MM 或 differentiable inference 放入标题，因为定稿实验并不以这些能力为正文证据主线。

---

## 3. 摘要逻辑

摘要按六步组织。

### 第一句：科学需求

构象集合评估、高通量数据生成和其他批量分子工作流需要执行大量 GFN-xTB 能量与力计算。

### 第二句：计算问题

目标 workload 通常由尺寸、AO 数和 SCC 难度不同的体系组成；逐体系执行会产生重复调用开销、负载不均衡和较低硬件利用率。

### 第三句：核心方法

介绍 xTBloom 是固定参数 GFN-xTB 的 ragged-batched runtime，不改变 GFN1/GFN2 的物理模型或参数。

### 第四句：关键设计

摘要只保留：

- variable-size ragged batch；
- molecule-level CPU scheduling；
- size-aware CUDA execution；
- caller-controlled host/device data residency；
- per-system status and failure isolation。

### 第五句：核心结果

只填入最终实验实际支持的结果：

- QM9/OMol25 上的能量、力、电荷误差和等价通过率；
- 复杂数据上的收敛率与失败类型；
- 相同单进程 CPU thread budget 下，native batch 相对逐体系 public-API loop 的
  端到端吞吐与 worker scaling；
- AO size × batch size 决定的 CPU–GPU crossover；
- 真实 OMol25 ragged workload 的 successful systems/s 与 time-to-solution。

不要只给最大 speedup，也不要把收敛失败的体系从性能分母中静默删除。

### 第六句：意义与范围

说明 xTBloom 为大量小到中型、尺寸异构的 GFN2-xTB workload 提供可嵌入执行后端；同时明确结论仅适用于实测硬件、输出合同和受支持化学空间。

---

# 4. 正文大纲

## 4.1 Introduction

Introduction 建议写成六个自然段。

### 第一段：GFN-xTB 的目标 workload

以 QM/MM molecular dynamics、增强采样和自由能计算作为最具体的应用动机。这类工作流需要在每个动力学时间步或采样窗口重新计算 QM 能量与原子力，一次科学任务因而包含大量连续且相关的 single-point evaluations。Giese 等人将 Amber、xtb 与 DeePMD-kit 集成为 QM/MM–ΔMLP 基础设施，使用 GFN2-xTB 开展核酶反应自由能面和药物样互变异构体自由能计算；该工作明确指出，大规模采样要求底层 QM 方法足够快，并表明将 xtb 直接作为库链接可以避免每个 MD step 启动外部程序和读写磁盘的开销（[Giese et al., 2024](https://doi.org/10.1021/acs.jpcb.4c01466)）。

以此引出 xTBloom 的主要动机：为重复、可嵌入的 GFN-xTB energy/force evaluation 提供低开销且可靠的执行后端。随后再扩展到相互独立的构象筛选、高通量数据生成和分子集合评估，说明目标 workload 同时包含 sequentially correlated 与 independent calculations。

这里引用 Giese et al. 支撑的是“应用需求、重复调用和库级嵌入价值”，不是 ragged batching 或 GPU 加速本身；后两项仍由本文 workload analysis 和性能实验直接证明。

### 第二段：真实 workload 的异构性

把一个 batch 表示为：

```text
B = {M1, M2, ..., MB}
```

不同体系具有不同的：

```text
Ni          atom count
Ai          AO count
Pi          pair count
Ki          SCC iteration count
Si          terminal status
```

指出尺寸差异、SCC 长尾和部分不收敛是 workload 的组成部分，而不是应在 benchmark 前清洗掉的噪声。

### 第三段：传统执行单位与目标 workload 的不匹配

传统程序通常把单个 calculation 作为执行和资源管理单位；对于大量独立且异构的体系，需要优化的是固定资源预算下的吞吐、成功率和总 time-to-solution。

### 第四段：相关软件与评价空缺

按能力类别概括 xTB/tblite 的生产型执行、dxtb 的 PyTorch-native 可微分定位、batched small operations、GPU offloading 和多体系调度。

强调同一模型不等于同一软件合同。xTB 和 tblite 是正文主要同类基线；dxtb 只在固定参数 GFN2 energy/force 的交集坐标上作为次要生态参考，不能据此评价其参数梯度或高阶自动微分能力。

### 第五段：xTBloom 的解决方案

> We reorganize fixed-parameter GFN-xTB calculations around a reusable ragged-batch runtime, with dedicated execution strategies for multicore CPUs and CUDA GPUs.

随后简述：

- flat arrays + offsets 表示 variable-size batch；
- CPU 以分子为外层调度单位；
- CUDA 按 AO/矩阵规模规划和分桶；
- public ABI 显式定义输出、内存空间和体系级 status；
- 单体系失败不抹除健康 peer 的结果。

### 第六段：贡献列表

贡献最多保留四项：

1. **Ragged batch abstraction**：支持不同原子数、AO 数和输出长度的原生 batch 表示。
2. **CUDA-centered execution**：以 GFN2 CUDA batch/bucketing 和数据驻留为性能主线，
   由单进程 CPU worker pool 提供配套路径与 crossover 参照。
3. **Explicit reliability semantics**：稳定 C ABI、完整请求验证、体系级失败隔离和可审计输出发布。
4. **Controlled and real-workload evaluation**：QM9/OMol25 数值等价性与收敛率、
   GPU crossover、native-interface CPU scaling 和真实 ragged workload。

WARM、QM/MM、GFN1 和 adapter 数量不列为正文独立贡献。

---

## 4.2 Computational Workload and Design Requirements

### 4.2.1 GFN-xTB Computational Workflow

只介绍与运行时设计相关的流程：

```text
molecular descriptor
→ geometry-dependent intermediates
→ SCC loop
   ├─ Hamiltonian construction
   ├─ eigensolver
   ├─ density/charge update
   └─ convergence check
→ energy / charges / analytic forces / status
```

解释矩阵尺寸为何由 AO 数决定、哪些中间量与体系或 topology 绑定、为什么 SCC 迭代数形成运行时间长尾、哪些阶段产生体系级数值失败，以及为什么 energy + forces 是正文计算合同。完整 GFN1/GFN2 理论引用原始方法论文。

### 4.2.2 Workload Model and Metrics

定义每个体系的 `(N_i, A_i, P_i, K_i, S_i)`，并区分：

```text
all-input time-to-solution
successful systems/s
matched-success-subset systems/s
batch wall time
p50/p95 completion latency
peak RSS/VRAM
```

主要吞吐定义为：

```text
throughput = number of successful systems / batch wall time
```

同时保留输入总数、成功率和失败类型，防止只对成功子集报告性能造成 survivorship bias。

### 4.2.3 Dataset Roles

| 数据集 | 论文中的角色 |
|---|---|
| QM9 | 小分子、平衡构型、常规闭壳层 sanity check 与小 AO 性能样本 |
| OMol25 | 电荷/自旋、金属、重元素、非平衡和大体系等复杂化学空间压力测试 |

两者不得混成一个随机池。OMol25 概率主样本和刻意过采样的压力样本也必须分开报告。

### 4.2.4 Sources of Inefficiency

- **Invocation overhead**：对象创建、接口转换、workspace 准备和结果发布。
- **CPU parallelism mismatch**：体系内线程与体系间并发可能造成 oversubscription。
- **GPU granularity mismatch**：小矩阵、短 kernel、launch 和数据移动难以由单体系摊薄。
- **Molecular heterogeneity**：AO 数、pair 数和 SCC iteration 不同导致负载不均衡。
- **Failure coupling**：若执行模型不能隔离体系级失败，少数困难体系会破坏整个 batch 的有效产出。

### 4.2.5 Design Requirements

| 编号 | Design requirement | 对应证据 |
|---|---|---|
| R1 | 无 padding 表示 variable-size molecular batch | Figure 1、Figure 4 |
| R2 | 在单进程公开接口中实现体系级并发，避免要求调用方自行编排多进程 | Figure 2 |
| R3 | 通过 batching、bucketing 和数据驻留提高 GPU 利用率 | Figure 3 |
| R4 | 完整请求在执行前验证，体系级数值失败彼此隔离 | Table 1、Figure 4 |
| R5 | 提供稳定、显式且可嵌入的内存与输出合同 | Figure 1、Methods |

Section 4.3 的设计与 Section 4.5 的结果都应回连到这些 requirement。

---

## 4.3 xTBloom Runtime Architecture

### 4.3.1 Architecture Overview

Figure 1 建议展示：

```text
Language adapters
Python / C++ / ASE / dpdata / PyTorch / Browser-WASM
                    ↓
               Stable C ABI
                    ↓
        Ragged batch descriptor
                    ↓
             Runtime planner
          ↙                     ↘
   CPU executor             CUDA executor
          ↓                     ↓
 energies / forces / charges / per-system status
```

图中区分 immutable parameters、reusable context/workspace、per-call descriptor、per-system SCC state 和 caller-owned outputs。

### 4.3.2 Native Ragged Molecular Representation

说明 flat arrays、`int64` molecule offsets、variable-length force/charge outputs、per-system charge/spin/status，以及 host、device 和 mixed memory-space tags。解释 ragged layout 如何避免按最大体系 padding，同时承认 offset lookup、planning 和 bucket formation 的固定成本。

### 4.3.3 Reusable Context and Workspace

| 对象 | 生命周期 |
|---|---|
| parameters / worker pool | context |
| reusable workspace | context 或 execution plan |
| batch descriptor | one call |
| SCC state | one system，正文默认 FRESH |
| output/status | caller-owned one call |

正文只解释持久 context 如何避免每次重建线程池和 workspace。fixed-topology plan 与 WARM state 作为次级模式简要说明，完整语义和验证移到 SI。

### 4.3.4 Multicore CPU Execution

按“问题—设计—边界”组织：molecule as outer scheduling unit、persistent worker pool、dynamic work queue、per-molecule BLAS/LAPACK single-threaded，以及固定 affinity/NUMA。说明 batch 太小或单体系过大时，外层并行收益受限。

不要在 Architecture 预先写“更快”，只提出由 Figure 2 检验的预期。

### 4.3.5 CUDA Execution

#### Work classification and bucketing

描述如何按 AO count、matrix dimension 和 pair/shell structure 规划工作，以限制同一 bucket 内的尺寸差异和 workspace 浪费。

#### Data residency

| 模式 | 输入 | 中间量 | 输出 | 正文用途 |
|---|---|---|---|---|
| host-to-host | host | device | host | 完整应用边界 |
| device-to-device | device | device | device | device-resident 上限 |
| mixed | host/device | device | contract-defined | 正确性与 SI spot check |

host-to-host 和 device-to-device 不得交叉相除形成 speedup。

#### Memory and synchronization

说明 workspace 预分配、bucket scratch、maximum resident batch、显式同步、OOM 分类，以及显存如何随 AO 数和 batch size 增长。

### 4.3.6 Failure and Publication Semantics

区分两类失败：

1. **Call-level validation/runtime failure**：完整请求在执行前验证；在 caller-output commit 前失败时，结果 flags 和 buffers 保持不变。
2. **Per-system numerical failure**：SCC 或 eigensolver failure 只影响对应体系；其请求的浮点 slices 全部填充 quiet NaN，健康 peer 保留有效结果。

invalid descriptor 属于 call-level validation failure，不与 SCC nonconvergence 混写。

### 4.3.7 Embedding Interfaces and Secondary Modes

稳定 C ABI 是唯一公共所有权边界，Python、ASE、dpdata 和 PyTorch 都调用同一 ABI。正文只给调用关系和短示例，完整字段表放 SI。

GFN1 CPU、QM/MM、fixed plan、WARM 和 mixed descriptor 只说明支持范围；除非 SI 有完成的证据，否则不得在摘要或结论中写成性能贡献。

### 4.3.8 Browser Demonstrator and Educational Access

[xTBloom 浏览器 Demo](https://xtbloom.jinzhezeng.group/) 将同一公共 C ABI 的 CPU 路径编译为客户端 wasm32，使用户无需安装软件或上传分子数据即可尝试 GFN1-xTB 和 GFN2-xTB。页面接受预设分子、XYZ 或 SMILES，展示三维结构、能量、原子电荷、可选解析力、SCC 收敛状态和迭代数，并通过上层 L-BFGS adapter 演示几何优化。

这一 demonstrator 可承担两类非性能价值：

- **可访问性与传播**：读者可以直接复现实例并观察输入、收敛状态和输出之间的关系；
- **教育用途**：用于介绍 charge/spin、GFN1/GFN2 选择、SCC convergence、energy/force 语义和几何优化过程。

必须同时说明边界：部署路径是单线程 CPU/WebAssembly；browser elapsed time 只反映交互体验，不属于 native CPU/CUDA benchmark；SMILES-to-3D 和 L-BFGS 是 Web adapter 功能；网站仅供探索和教学，不是生产科学计算环境。实现与限制以 `docs/user-guide/browser-demo.md` 为准。

---

## 4.4 Numerical Validation and Benchmark Protocol

### 4.4.1 Overall Study Design

正文实验固定为：

1. **P0：数值一致性与真实数据集等价性**；
2. **实验 1：CPU Native Batch 与公开接口端到端基线（配套实验）**；
3. **实验 2：GPU Batch Scaling 与 CPU–CUDA Crossover（核心性能实验）**；
4. **实验 3：复杂数据收敛率与 Ragged 异构 Runtime**。

正文结果载体固定为一张主数值表和三张主性能图；Figure 1 是方法架构图，不算额外实验。

### 4.4.2 Dataset Sampling and Reproducibility

三个集合互不重叠：

| 数据集 | 主样本 | 压力样本 | 独立性能样本 | 用途 |
|---|---:|---:|---:|---|
| QM9 | 10,000 | — | 2,048 | 数值等价性、基线收敛率和小体系性能 |
| OMol25 | 8,000 | 2,000 | 4,096 | 复杂体系等价性、收敛率和 ragged workload |

QM9 按非氢原子数、元素组成、AO 数分位和总原子数分层。

OMol25 主样本按数据域、charge、spin、元素类别、AO/原子数分位以及平衡/非平衡状态分层，并保留抽样权重。额外 2,000 个压力样本过采样带电、开壳层、金属/重元素、大 AO、反应/畸变/多片段和困难 SCC 体系；压力样本不得并入总体收敛率。

抽样 manifest 至少记录：数据集版本、访问来源、许可证和文件散列；稳定体系 ID、charge、spin、原子数和 AO 数；抽样层、抽样概率和集合归属；固定 hash-based seed；过滤和排除原因。

只发布可再分发的 ID、元数据、散列和脚本，不重新分发受控原始数据。

### 4.4.3 Common Computational Contract

- 正文使用 GFN2-xTB energy + analytic forces；
- 相同 geometry、charge、spin、参数、电子温度和最大 SCC iteration；
- 所有正文性能使用 FRESH，不混入 WARM；
- 位置、能量和力分别使用 bohr、Hartree 和 Hartree/bohr；
- xTB gradient 明确转换为负梯度 force；
- CPU 固定物理核心预算、affinity、NUMA 和单线程 BLAS；
- CUDA 在计时边界执行显式同步；
- setup、first-call 和 steady-state 分开；
- 每个性能坐标先通过对应正确性门槛；
- unsupported、failed 和 OOM 行必须保留。

### 4.4.4 P0: Numerical Equivalence and Dataset-Scale Validation

#### P0-A：Canonical validation

运行现有 pinned xTB/tblite conformance，覆盖 energy、forces、charges、restricted/unrestricted、CPU/CUDA、host/device/mixed、batch-vs-sequential、守恒/不变性和 peer-local failure。有限差分力为必做项。

#### P0-B：QM9/OMol25 大样本数值等价性

所有样本使用原始几何做 single point，不重新优化。主要比较：

- xTBloom CPU vs pinned xTB 6.7.1；
- xTBloom CUDA vs xTB；
- xTBloom CPU vs CUDA；
- tblite 作为第二独立参考；
- dxtb 仅在模型、输出和收敛合同可比时作为诊断参考。

记录总能量/每原子能量误差、force component RMSE、原子力向量 RMSE、最大力分量误差、atomic charge RMSE、最大电荷误差、SCC iteration 和终止状态。

OMol25 的 DFT label 只描述数据来源，不能作为“xTBloom 是否复现 xTB”的 oracle。

预注册门槛沿用现有 conformance 合同，不根据结果放宽：

| 性质 | xTBloom–oracle 门槛 |
|---|---:|
| 总能量 | `5e-7 Ha` |
| 力最大分量 | `5e-6 Ha/bohr` |
| 原子电荷最大误差 | `5e-7 e` |
| CPU–CUDA 对应性质 | `1e-6`，使用相应原子单位 |

报告 p50、p95、p99、maximum 和逐体系通过率。只有共同收敛体系全部通过预注册门槛且无未解释离群点时，才能使用 numerically equivalent。

#### P0-C：真实体系有限差分抽查

从 QM9 和 OMol25 各选 16 个分层代表体系，每个体系预先固定 8 个坐标分量；主步长为 `1e-3 bohr`，代表子集再检查 `5e-4` 和 `2e-3 bohr`，门槛为 `1e-5 Ha/bohr`。

#### P0-D：失败语义

使用 OMol25 自然非收敛体系、固定困难 SCC fixture、成功/失败混合 ragged batch、非法/非有限 descriptor 和可注入 eigensolver failure。验证健康 peer、NaN publication、call-level output preservation 和 CPU/CUDA status 一致性。

#### P0-E：退化占据与跨引擎数值压力案例

纳入 `docs/developer-guide/architecture.md` 中的 exact three-fold degeneracy public-API probe，并冻结参考软件版本、revision、电子温度和输入。该案例使用三个相距极远的 H 原子构造精确退化 one-center orbital blocks，专门检验有限温度占据、电子数可表示性和对称 publication。

重点报告 `charge=-3, uhf=0` 坐标：当前仓库证据中 xTBloom CPU 返回 finite energy 与 `SUCCESS`，xTB 6.7.1 在 Hamiltonian diagonalization 处失败，tblite 0.7 在 generalized eigensolver 处失败，dxtb 则报告 Fermi-energy convergence failure。另两个坐标用于补充边界：neutral open-shell 坐标并非 xTBloom/tblite 双失败；fractional-charge 坐标无法由 xTB integer-charge CLI 表达，因此不得记为 xTB failure。

这个实验只支持“xTBloom 的特定退化占据 policy 处理了参考实现失败的压力坐标”，不能写成 xTBloom 在一般体系上比 xTB/tblite 更稳定。正文可用一段结果或 Table 1 的 stress-case footnote，完整输入、版本、输出和 regression test 放 SI。

### 4.4.5 Experiment 1: CPU Native Batch under Public-Interface Contracts

研究问题：在相同单进程、固定 affinity 和 CPU thread budget 下，xTBloom 的 native
batch 与持久 worker pool 相对 xTB/tblite 逐体系 public-API loop 提供多少端到端收益？
这一比较有意保留 xTBloom 的原生跨体系调度，因为它是软件接口和运行时的贡献；结果
命名为 native-interface end-to-end speedup，不解释为 engine-only speedup。

```text
workers = 1 / 4 / 16 / min(32, available physical cores)
batch size = 1 / 32 / 128
AO regime = small / medium / large
property = GFN2 energy + forces
start policy = FRESH
```

正文对照为 xTB 串行 public-API loop、tblite 串行 public-API loop、xTBloom 同 context
逐体系调用、xTBloom `cpu_threads=1` ragged call 和 xTBloom full-worker ragged batch。

calculator、context 和固定 descriptor 在计时前创建；计时包含实际计算、结果发布和
必要同步。Batch 1 只提供 latency 背景，不用于多 worker 加速主张。

SI 在少量代表坐标增加 persistent xTB/tblite `N processes × 1 thread` 与 xTBloom
`1 process × N workers` 敏感性分析，固定相同物理核心与总内存限制，并报告
steady-state、aggregate RSS、CPU-seconds 和进程启动成本。它用于限定“最佳可达 CPU
吞吐”这一更强主张，不替代正文公开接口比较。

短坐标至少 10 次 warm-up 和 30 次测量；长 workload 至少 5 次独立运行。报告 median、IQR、bootstrap 95% CI、systems/s、wall time、speedup 和 parallel efficiency。

### 4.4.6 Experiment 2: GPU Batch Scaling and CPU–CUDA Crossover

研究问题：CUDA 在什么 AO 数和 batch size 下开始优于 CPU，数据传输、launch、SCC、eigensolver 和显存如何决定 crossover？

AO 档位：`1–64`、`65–128`、`129–256`、`257–512`、`513–1024`；若有样本，将 `>1024` 作为容量压力点。

Batch：`1 / 4 / 16 / 64 / 256`，继续增加到吞吐饱和或 OOM。每个坐标使用不同真实结构，不复制同一几何制造 batch。

对照：

1. xTBloom CPU；
2. xTBloom CUDA host-to-host；
3. xTBloom CUDA device-to-device；
4. dxtb CUDA 的语义可比坐标；
5. mixed memory 只做正确性和 SI spot check。

所有 CUDA 坐标固定单 GPU、单进程，使用各实现公开支持的 batch 路径。公平性由设备、
workload、输出、数据驻留和同步边界定义；不为统一 CPU 并行形态而构造多个进程共享
一张 GPU，因为多 CUDA context 和显存复制属于另一种部署问题。

已知的大 AO、batch 1 坐标必须保留，即使性能较差。报告绝对 systems/s、GPU/CPU ratio、latency、peak VRAM、maximum batch 和 OOM boundary。只选择三个 profiler 代表点：launch-bound、吞吐饱和区和大 AO eigensolver-bound。

### 4.4.7 Experiment 3: Convergence and Ragged Heterogeneous Workloads

#### 3-A：多方法收敛率

在完整 QM9/OMol25 抽样上运行 xTBloom CPU、xTBloom CUDA、xTB、tblite 和可比 dxtb。全部使用 FRESH；不同程序无法一一对应的 convergence option 必须如实记录。

状态分类：converged、SCC not converged、eigensolver/numerical failure、invalid input、unsupported、resource/OOM failure。

`unsupported` 不进入收敛率分母，但必须单独报告。数值误差在共同收敛体系上计算；收敛率在所有合法受支持体系上计算。

报告总体和分层 convergence rate、Wilson 95% CI、成对列联表、paired rate difference、SCC iteration ECDF/median/p95，以及按 charge、spin、金属、AO 数和数据域分组的失败原因。

#### 3-B：Ragged runtime 与归因消融

两个 workload：

1. 按 AO 档位均匀组合的受控宽尺寸分布；
2. 保持 manifest 固定顺序的独立 QM9/OMol25 性能样本。

xTBloom 路径：

1. 同一复用 context 下逐体系调用；
2. 单次 ragged batch，`cpu_threads=1`；
3. 单次 ragged batch，完整 CPU worker pool；
4. CUDA 当前 bucketed runtime；只有存在可审计开关时才增加 no-bucketing 对照。

不把 ragged、bucketing、dynamic scheduling 和 context reuse 合并为一个不可归因的黑盒。报告 all-input time-to-solution、successful systems/s、matched-success-subset systems/s、success rate、p50/p95 completion latency、peak RSS/VRAM 和 SCC iteration 长尾。

### 4.4.8 Statistical and Reporting Rules

- speedup 报告 bootstrap 95% CI；只有 CI 明确高于 1 才宣称更快；
- 收敛率报告 Wilson 95% CI 和成对差异；
- 原始 timing、失败行、不可用行和 OOM 行全部保留；
- 记录 hardware、compiler、CUDA、driver、BLAS、thread settings、release commit、binary hash 和 parameter hash；
- 原子数不是唯一规模轴，必须同时记录 AO 数、pair 数和 SCC iteration；
- 不从单一硬件推导普适加速结论。

---

## 4.5 Results and Discussion

结果按“正确性 → native-interface CPU 背景 → 核心 GPU 边界 → 复杂真实 workload”展开。

### 4.5.1 Numerical Equivalence and Robustness

对应 P0，正文使用 **Table 1: Numerical Equivalence and Convergence Summary**。

- Panel A：QM9/OMol25 的 energy、force、charge error 与通过率；
- Panel B：xTBloom CPU/CUDA、xTB、tblite 和可比 dxtb 的总体收敛率摘要；
- 正文补充 CPU–CUDA difference、有限差分和失败隔离结论；
- 单独说明 exact-degeneracy stress case 中 `charge=-3` 的跨引擎结果及其严格适用范围；
- 全量分布、逐体系离群点和失败矩阵进入 SI。

本节只在证据允许时得出：后续性能差异不是由放宽数值门槛、丢弃困难体系或改变物理模型造成的。

### 4.5.2 Native-Batch CPU Execution under Public-Interface Contracts

对应实验 1，使用 **Figure 2**：

- Panel A：相同单进程 thread budget 下的 native-interface systems/s；
- Panel B：xTBloom systems/s versus worker count 与 CPU utilization；
- Panel C：reference serial public-API loop 与 xTBloom native batch；
- Panel D：sequential call → native batch (`cpu_threads=1`) → full worker pool 的最小归因消融。

讨论 xTBloom 优势中有多少来自整批 API、多少来自 worker scheduling，以及 batch
size/AO 规模如何限制并行效率。明确正文 speedup 是公开接口及其原生运行时共同形成的
端到端优势；SI persistent process-pool 只回答外部应用编排能追回多少吞吐及其内存、
进程管理代价。

### 4.5.3 CUDA Scaling and CPU–GPU Crossover

这是正文核心性能结果，对应实验 2，使用 **Figure 3**：

- Panel A：AO size × batch size 的 GPU/CPU throughput heatmap；
- Panel B：CPU、CUDA host-to-host 和 device-to-device 的绝对 systems/s；
- Panel C：peak VRAM、maximum batch 和 OOM boundary；
- Panel D：三个代表 regime 的 profiler decomposition。

明确区分小体系/小 batch 的 CPU 优势区、batch 足以摊薄开销后的 GPU 优势区、大 AO eigensolver/容量受限区，以及 host-resident 与 device-resident 合同。不要暗示不存在的 GPU 全域优势。

### 4.5.4 Convergence and Ragged Heterogeneous Workloads

对应实验 3，使用 **Figure 4**：

- Panel A：QM9/OMol25 多方法收敛率和成对结果；
- Panel B：SCC iteration ECDF 与困难分层；
- Panel C：受控异构 workload 的逐体系/ragged/worker-pool 消融；
- Panel D：真实 OMol25 workload 的 throughput、time-to-solution、success rate 和内存。

讨论哪些 charge/spin/element/AO strata 产生收敛差异，失败属于 SCC、eigensolver、unsupported 还是 resource failure，all-input 与 matched-success-subset 指标是否一致，以及失败体系是否污染健康 peer 的有效产出。

### 4.5.5 Performance Regime Synthesis

这一小节不增加新实验，只综合 Figures 2–4：

| Workload | 由实验支持后可推荐的模式 |
|---|---|
| batch 1、小 AO | CPU latency path |
| 大量小体系 | CPU native batch 或 CUDA large batch，以实测 crossover 为准 |
| 中等 AO、device-resident | CUDA，以 Figure 3 实测区域为准 |
| 宽尺寸 ragged batch | ragged + runtime scheduling/bucketing |
| 含部分不收敛体系 | failure-isolated batch，并同时报告 success rate |

表中的边界必须来自数据，不得预先写死。

### 4.5.6 Scope and Limitations

- 正文主要评价 GFN2；GFN1 仅 CPU 且放 SI；
- CUDA 仅 NVIDIA，ROCm 尚未实现；
- 正文硬件数量有限，性能边界不自动迁移到其他架构；
- WARM、fixed plan、QM/MM 性能和第二套硬件不是正文完整实验；
- 无原生周期 GFN2、溶剂化、geometry optimization 或 MD；
- dxtb 比较只覆盖固定参数 energy/force 交集，不评价其完整可微分能力；
- OMol25 受支持化学空间、数据访问和 `unsupported` 行必须透明报告。

---

## 4.6 Conclusions

结论控制为三段。

### 第一段：问题与解决方案

重述 xTBloom 把大量 variable-size GFN2-xTB 计算组织为 failure-isolated ragged batch，并为多核 CPU 和 CUDA 提供专用执行路径。

### 第二段：只总结已证明的结果

按 Table 1 和 Figures 2–4 依次总结数值等价性或实际失配边界、核心 CPU–GPU
crossover 与 CUDA 有效区域、native-interface CPU batch 吞吐、复杂化学空间中的
收敛表现，以及真实 ragged workload 的 time-to-solution 和失败隔离。

不要在结论中首次引入 speedup 数字，也不要总结未做的 WARM、QM/MM 或第二套硬件性能。

### 第三段：未来工作

由实测瓶颈和范围边界推出 portable GPU backend、multi-GPU、alternative eigensolver、mixed precision、periodic/solvation support，以及在独立实验中评估 WARM 与更完整 differentiable interface。

## 4.7 Software and Data Availability

集中列出：

- 开源仓库、release/tag、源码和参数 provenance；
- benchmark scripts、抽样 manifest、raw timings 和可再分发结果；
- [客户端浏览器 Demo](https://xtbloom.jinzhezeng.group/) 及其教育/交互定位；
- 受访问控制的 OMol25 原始数据如何申请，以及论文不重新分发哪些数据；
- 复现实验所需的 compiler、BLAS、CUDA、driver、binary hash 和 parameter hash。

这里可把 Demo 描述为降低软件试用和教学门槛的 companion artifact，但不得把网站访问量、浏览器 timing 或 adapter-level geometry optimization 当作论文性能结果。

---

# 5. 主图与主表规划

## Figure 1 — Workload and Runtime Architecture

- A：逐体系调用与 ragged batch 的执行单位对比；
- B：flat arrays + offsets 的 descriptor；
- C：runtime planner 分派到 CPU/CUDA；
- D：per-system status、NaN failure slices 与健康 peer publication。

## Table 1 — Numerical Equivalence and Convergence Summary

Panel A：

| Dataset | Quantity | Comparator | p50 | p95 | p99 | Maximum | Pass rate | Threshold |
|---|---|---|---:|---:|---:|---:|---:|---:|
| QM9/OMol25 | Energy | xTB/tblite |  |  |  |  |  |  |
| QM9/OMol25 | Forces | xTB/tblite/finite difference |  |  |  |  |  |  |
| QM9/OMol25 | Charges | xTB/tblite |  |  |  |  |  |  |

Panel B：各实现总体 convergence rate、Wilson 95% CI、unsupported 和 resource-failure 数量。

## Figure 2 — CPU Native Batch under Public-Interface Contracts

展示相同单进程 thread budget 下的端到端 systems/s、xTBloom worker scaling、参考程序
逐体系 public-API loop 和最小 CPU 消融。外部 persistent process-pool 敏感性结果放 SI。

## Figure 3 — GPU Scaling and Crossover

核心图为 `(AO count, batch size) → GPU/CPU throughput ratio`，并附绝对吞吐、host/device 合同、VRAM/OOM 和三个 profiler 点。

## Figure 4 — Convergence and Ragged Real Workloads

包含多方法收敛率、SCC iteration ECDF、受控 ragged 消融和真实 OMol25 time-to-solution。

Benchmark contract、execution modes、完整硬件配置和附加表统一放 SI，正文不再增加 Table 2/3。

---

# 6. Claim–Evidence 对照矩阵

| Potential claim | 必需证据 | 允许写入摘要的条件 |
|---|---|---|
| xTBloom is numerically equivalent to xTB | P0 canonical + QM9/OMol25 errors | 所有共同收敛体系通过预注册门槛且无未解释离群点 |
| xTBloom preserves useful work under partial failure | mixed success/failure batch | 健康 peer 不变、失败 slices 为 NaN、status 可审计 |
| xTBloom resolves a documented exact-degeneracy stress coordinate | three-H public-API probe + pinned cross-engine versions | 只描述 `charge=-3` 等已测坐标，不外推为一般收敛率优势 |
| Native batch improves end-to-end CPU throughput over per-system public APIs | same-process/thread-budget public-interface comparison + CPU ablation | speedup 95% CI 高于 1，公开绝对吞吐，并明确优势包含 native batch scheduling |
| xTBloom outperforms externally parallelized CPU references | SI persistent process-pool sensitivity analysis | 同物理核心和内存合同下 CI 高于 1；否则不得使用该更强主张 |
| CUDA improves throughput in defined regimes | AO–batch crossover under identical contracts | 只描述实测优势区域，不概括为全域优势 |
| Ragged runtime handles heterogeneous workloads effectively | controlled and real ragged workloads | all-input 与 matched-success 指标、success rate 和 memory 同时报告 |
| xTBloom has characterized convergence on complex chemistry | QM9/OMol25 paired convergence analysis | unsupported 和失败原因完整公开，不把压力样本并入总体率 |
| xTBloom reduces real dataset time-to-solution | independent OMol25 performance sample | 使用全部输入、固定 manifest 和统一输出合同 |

WARM、QM/MM 性能、GFN1 CUDA、跨硬件普适性和完整可微分能力不在正文 claim 集合内。

---

# 7. Supporting Information 最低范围

- 完整 QM9/OMol25 抽样 manifest、分层定义和排除原因；
- 全量 energy/force/charge error 分布与离群体系；
- 完整有限差分和失败语义矩阵；
- exact-degeneracy probe 的完整输入、reference revisions、stdout/stderr、floating-point status 和 regression test；
- 各 OMol25 分层的误差、收敛率、SCC iteration 和失败原因；
- GFN1 CPU 数值验证；
- QM/MM energy、QM force 和 point-charge force 正确性；
- CUDA mixed descriptor 正确性；
- xTB/tblite persistent `N processes × 1 thread` 与 xTBloom
  `1 process × N workers` 的少量等核心敏感性坐标，包括 aggregate RSS、CPU-seconds
  和启动/编排成本；
- 每个 benchmark 的原始 timing、bootstrap CI、RSS/VRAM 和 profiler tables；
- build flags、compiler、CUDA、driver、BLAS、release commit、binary hash 和 parameter hash；
- benchmark scripts 和 reproducibility manifest；
- 若正文仍将 WARM 列为贡献，必须增加真实连续小位移轨迹的安全性与性能实验；否则只记录接口语义；
- 第二套硬件只复现三个代表点，属于强烈建议，不是正文硬门槛。

---

# 8. 最容易破坏文章逻辑的写法

## 8.1 用“实验只有三组”代替证据闭合

实验数量不是充分性标准。三组主实验必须共同回答数值、收敛、native-interface CPU、
GPU crossover 和 ragged workload 五类证据问题。

## 8.2 把 native-interface 对照写成最佳可达 CPU 排名

在相同单进程/thread budget 下，让 xTBloom 使用其原生跨体系 workers、让 xTB/tblite
通过各自逐体系 public API 执行 batch，是有效的端到端软件接口比较；跨体系调度正是
xTBloom 的能力，不能为了“相同并行语义”而关闭。必须避免的是把该结果写成双方所有
可能部署方式中的普遍 CPU 排名。若要声称优于外部并行的最佳参考基线，必须引用 SI
persistent process-pool 敏感性结果。

## 8.3 混用 host-to-host 与 device-to-device

两类结果可并列展示，但不能直接相除形成 speedup。

## 8.4 把 OMol25 DFT label 当作 GFN2 实现等价 oracle

DFT 与 GFN2-xTB 是不同物理模型。实现等价性应对比 pinned xTB/tblite 和 CPU–CUDA 结果。

## 8.5 合并 QM9、OMol25 主样本和压力样本的总体率

这些集合的抽样目标不同。必须分别报告；压力样本不得污染概率样本的总体收敛率。

## 8.6 只对共同成功体系报告速度和误差

误差可在共同收敛子集上计算，但收敛率必须使用全部合法受支持体系；性能同时报告 all-input 与 matched-success-subset 指标。

## 8.7 静默删除 unsupported、nonconverged 或 OOM 行

这些行是性能边界与生产可用性的一部分，必须保留并分类。

## 8.8 只用原子数作为横轴

至少同时记录 AO 数、pair 数和 SCC iteration；GPU crossover 以 AO 数和 batch size 为主轴。

## 8.9 把 WARM、QM/MM 和 adapters 升格为正文贡献

没有对应正文实验就不能进入摘要和结论。稳定 ABI 可以作为设计贡献，接口数量本身不是科学证据。

## 8.10 从单硬件结果推导普适优势

只能报告当前 CPU/GPU、软件版本、计时边界和 workload 下的实测 regime。

## 8.11 用一个最大 speedup 代替完整结果

正文同时需要绝对 wall time/systems/s、置信区间、成功率、容量边界和失败坐标。

## 8.12 把单个退化占据案例扩张成通用稳定性排名

`charge=-3` 三氢压力坐标可以证明一个明确的数值 policy 差异，但不能替代 QM9/OMol25 的总体和分层收敛率。必须同时保留 tblite 在其他坐标成功、xTB fractional-charge CLI 无法表达等限制。

## 8.13 把浏览器 Demo 当作性能 benchmark

Demo 的价值是可访问性、教学展示和 ABI/WebAssembly 可移植性。其单线程浏览器耗时受设备、缓存、网络和浏览器影响，不得与 native CPU/CUDA 结果混合。

---

# 9. 推荐的实验与写作顺序

1. 冻结正文 claim、release commit、comparators 和统一计算合同；
2. 完成 QM9/OMol25 访问、分层抽样和不可修改 manifest；
3. 完成 canonical conformance、有限差分和失败语义 P0；
4. 完成 QM9/OMol25 大样本数值等价性；
5. 优先完成 CPU–CUDA crossover、容量边界和三个 profiler 点；
6. 完成正文 native-interface CPU batch 实验和少量 SI persistent-pool 敏感性坐标；
7. 完成复杂数据收敛率与受控/真实 ragged workload；
8. 锁定 Table 1 和 Figures 2–4，检查每条 claim 是否有直接证据；
9. 绘制 Figure 1，并先写 Architecture、Methods 和 Results；
10. 根据实际结果反写 Introduction、regime synthesis 和 Limitations；
11. 最后写 Abstract 与 Conclusion，删除所有未被证据支持的形容词和扩展性主张。

最终编辑原则：

> 每一个正文主张都必须落到 P0、Figure 2、Figure 3 或 Figure 4；无法落到这四个证据载体的能力，移入 Supporting Information 或 Future Work。
