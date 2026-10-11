# xTBloom 论文实验方案

> 状态：讨论修订稿。本文档定义论文正文的最小完整证据矩阵，以及执行时不可省略的
> 数值、统计和复现合同。具体样本 ID、机器配置和软件散列在正式运行前冻结到独立
> manifest；冻结后不得根据结果重新抽样。

## 1. 目标与正文结构

论文实验压缩为 **1 个前置数值验证 + 3 个正文主实验**：

1. Numerical Validation and Dataset-Scale Equivalence；
2. Native-Batch CPU Execution under Public-Interface Contracts；
3. GPU Batch Scaling and CPU-CUDA Crossover（正文核心性能实验）；
4. Convergence and Ragged Heterogeneous Workloads。

对应的正文结果规模为：

- **Table 1**：数值等价性、CPU-CUDA 一致性和真实数据集收敛率；
- **Figure 2**：单进程 CPU native batch、逐体系 public-API 基线和归因消融；
- **Figure 3**：正文核心性能图，展示 GPU batch 饱和、CPU-CUDA crossover 和容量边界；
- **Figure 4**：复杂数据收敛率、SCC 长尾和 ragged workload 消融。

GFN2-xTB 的 CUDA 化及其 ragged-batch runtime 是正文主线。CPU 结果用于说明原生
batch 的端到端接口价值和提供 CPU-CUDA 参照，而不是承担“所有实现最佳 CPU 部署
方式”的普遍排名。GFN1、QM/MM 性能、WARM 轨迹和第二套完整硬件矩阵
不作为正文主实验。本文档定义的是论文矩阵，不会自动满足或关闭仓库中范围更大的
GitHub issues [#395](https://github.com/jinzhezenggroup/xtbloom/issues/395) 和
[#396](https://github.com/jinzhezenggroup/xtbloom/issues/396)。

## 2. 论文允许回答的问题

完成本计划后，正文可以回答：

- xTBloom 是否在真实小分子和复杂化学空间中复现同一 GFN2-xTB 数值问题；
- 在相同单进程和 CPU thread budget 下，xTBloom native batch 相对 xTB/tblite
  逐体系 public-API loop 的端到端收益是多少；
- CUDA 在哪些 AO 数和 batch size 区域内优于 xTBloom CPU；
- 面对尺寸差异、SCC 长尾和少量失败体系时，ragged runtime 是否仍能改善整体
  time-to-solution；
- xTBloom、xTB、tblite 以及可比 dxtb 坐标在真实数据中的收敛行为如何不同。

本计划不能单独支持以下结论：

- xTBloom 在所有硬件、分子和 batch size 上都更快；
- xTBloom 比 DFT 更准确，或 OMol25 的 DFT 结果是 GFN2-xTB 的数值 oracle；
- WARM、QM/MM、GFN1 CUDA、ROCm 或原生周期 GFN2 的性能优势；
- xTBloom 与 dxtb 在参数训练、高阶自动微分等非重叠能力上的整体排名。
- xTBloom 在 CPU 上必然优于用户自行构造的 persistent xTB/tblite 多进程部署；只有
  SI 敏感性对照支持时，才可讨论该更强结论。

## 3. 执行环境与证据位置

除普通快速检查和聚焦单元测试外，本文所称“实验”默认在 SSH 主机
在配置文件选定的 Slurm 集群上运行：

- 远程源码：`$PAPER_REPO_ROOT` 指向的 clean checkout；
- 所有计算负载必须通过 Slurm，以 `srun` 或 `sbatch` 提交；
- 输入、抽样 manifest、运行中间结果和最终原始结果放在
  `$PAPER_DATA_ROOT/<paper-run-id>/`，且不得位于源码树内；
- 不得把临时数据集、完整 QM9/OMol25 数据或大体积原始结果写进源码树；
- 每次正式运行记录 hostname、Slurm partition/allocation、Job ID、分支、commit、
  工作树状态、编译器、CPU/GPU、CUDA driver/toolkit、线性代数 provider、环境变量、
  完整命令和产物路径。

最终论文证据只能来自同一个干净、固定的源码 commit。开发期结果只能用于调试，
不得与正式结果拼接成同一张图表。

## 4. 数据集、抽样与许可

### 4.1 数据集角色

| 数据集 | 角色 | 主要覆盖 |
| --- | --- | --- |
| QM9 | 常规小分子基线 | 中性、闭壳层、平衡构型、H/C/N/O/F、最多 9 个非氢原子 |
| OMol25 | 复杂化学空间与压力测试 | 多数据域、可变 charge/spin、金属和重元素、非平衡与反应构型、大体系 |

QM9/OMol25 的 DFT 能量和力不是 xTBloom 数值等价性的 oracle。数值等价性比较的
对象是固定 GFN2-xTB 合同下的 xTBloom 与 pinned xTB/tblite；数据集只提供真实几何、
元素、电荷、自旋和 workload 分布。

### 4.2 固定样本规模

| 数据集 | 数值/收敛主样本 | 压力样本 | 独立性能样本 |
| --- | ---: | ---: | ---: |
| QM9 | 10,000 | 0 | 2,048 |
| OMol25 | 8,000 | 2,000 | 4,096 |

三个集合互不重叠。若正式预跑显示资源预算不足，只能在查看正式结果前统一修改样本
规模并重新冻结 manifest；不得针对失败或慢体系进行事后删选。

### 4.3 QM9 分层

按以下变量分层抽样：

- 非氢原子数；
- 元素组成；
- 总原子数；
- GFN2 AO 数分位。

各层按总体占比分配，并设置最低样本数以保留较小化学层。抽样单位是唯一 molecule
ID，不使用重复几何扩充样本。

### 4.4 OMol25 分层

使用获得授权的官方固定数据 split。主样本按以下变量分层并保存抽样权重：

- 数据域：小分子、生物分子、金属配合物、电解质；
- 电荷：负电、中性、正电；
- 自旋：闭壳层、开壳层；
- 元素：CHNOF、其他主族、过渡金属、重元素；
- 原子数和 AO 数分位；
- 平衡、非平衡/反应、多片段或显式溶剂构型。

额外 2,000 个压力样本刻意过采样带电、开壳层、金属/重元素、大 AO、反应/畸变和
多片段体系。压力样本只用于分层压力结论，不能与概率主样本合并估计 OMol25 的总体
收敛率。

### 4.5 Eligibility funnel

在运行前按公开能力把每条数据分类为：

- eligible：非周期、元素和 charge/spin 可由 GFN2-xTB 公共接口表达；
- unsupported：超出发布元素、模型、周期或适配器能力；
- malformed：缺失或矛盾的元素、几何、charge/spin 元数据。

必须报告每一步的输入数、保留数和排除原因。`unsupported` 和 `malformed` 不进入
收敛率分母，但不得静默删除或改记为计算失败。

### 4.6 抽样 manifest

manifest 至少记录：

- 数据集版本、原始文件 hash、许可证和引用；
- 稳定体系 ID、来源、charge、spin、原子数、AO 数；
- 分层标签、抽样概率和集合归属；
- hash-based 固定抽样种子及抽样脚本 hash；
- 每条排除记录和原因；
- 坐标单位转换及转换后数据 hash。

仓库仅保存许可允许的紧凑 manifest、脚本和结果摘要。任何新增数据或复制内容在进入
Git 历史前必须完成 licensing/provenance 检查。

## 5. P0：数值一致性与真实数据集等价性

### 5.1 研究问题

1. xTBloom CPU/CUDA 是否计算了与独立参考相同的 GFN2-xTB 数值问题？
2. 在 QM9 和 OMol25 抽样中，共同收敛体系的 energy、forces 和 charges 是否数值
   等价？
3. batch 中的局部数值失败是否会污染健康体系？

### 5.2 P0-A：canonical independent validation

先运行仓库现有的 pinned xTB/tblite conformance 和 invariance 工具，覆盖：

- energy、analytic forces、charges 和 SCC 状态；
- restricted/unrestricted；
- CPU、CUDA host/device/mixed descriptors；
- sequential、homogeneous ragged 和 heterogeneous ragged；
- 平移/旋转不变性、力协变、力平衡和电荷守恒；
- QM/MM 点电荷的 QM force、point-charge force 和有限差分；
- difficult/nonconvergent SCC 和 peer-local failure。

独立 golden、有限差分和 invariance 是互补证据；CPU-CUDA 自洽不能替代独立 oracle。

### 5.3 P0-B：QM9/OMol25 数据集级等价性

对冻结的主样本使用原始几何做 GFN2-xTB single point，不重新优化。主要比较：

1. xTBloom CPU vs pinned xTB 6.7.1；
2. xTBloom CUDA vs pinned xTB 6.7.1；
3. xTBloom CPU vs CUDA；
4. tblite 作为第二独立参考；
5. dxtb 只在固定 GFN2 energy/force 推理语义可比的坐标中作为次要诊断。

记录每个体系的：

- 总 Helmholtz free energy 和每原子能量差；
- force component RMSE、原子力向量 RMSE 和最大分量误差；
- atomic charge RMSE 和最大误差；
- SCC iteration、最终残差、终止状态和诊断；
- 实际模型、参数、电子温度、charge、spin 和输出定义。

### 5.4 数值门槛

沿用现有 conformance 合同，不根据抽样结果放宽：

| 性质 | 主门槛 |
| --- | ---: |
| 总能量 | `5e-7 Eh` |
| 力最大分量 | `5e-6 Eh/bohr` |
| 原子电荷最大误差 | `5e-7 e` |
| CPU-CUDA 对应性质 | `1e-6`，使用相应原子单位 |

同时报告 p50、p95、p99、最大误差和逐体系 gate pass rate。只有所有共同收敛体系
均通过对应门槛，且不存在未解释离群点时，正文才使用“numerically equivalent”。
否则报告实际通过率、置信区间和失配类别，不修改门槛。

### 5.5 P0-C：真实体系导数抽查

从 QM9 和 OMol25 各选 16 个分层代表体系。每个体系在 manifest 中预先固定 8 个
原子坐标分量：

- 主 central-difference 步长：`1e-3 bohr`；
- 代表子集附加步长：`5e-4`、`2e-3 bohr`；
- analytic force 定义为所报告 Helmholtz free energy 的负坐标导数；
- 主门槛：`1e-5 Eh/bohr`。

该抽查补充 canonical corpus 的完整有限差分，不以随机抽查替代现有门槛。

### 5.6 P0-D：失败语义

至少覆盖：

- OMol25 中自然出现的非收敛体系；
- committed `tmacl` difficult-SCC fixture；
- 健康体系与失败体系组成的真实 ragged batch；
- non-finite input 和非法 descriptor；
- 可注入 eigensolver failure。

验证：

- 单个 SCC/eigensolver 失败不改变健康 peer；
- 失败体系请求的浮点输出切片全部为 quiet NaN；
- call-level validation failure 保持所有 caller output 和 result flag 不变；
- CPU/CUDA 具有相同方程、状态分类和 publication semantics。

### 5.7 P0 输出

正文 **Table 1** 分为两个 panel：

- Panel A：QM9/OMol25 energy、force、charge 的误差分布与等价通过率；
- Panel B：xTBloom CPU/CUDA、xTB、tblite 和可比 dxtb 的收敛率摘要。

完整逐层误差、有限差分、失败矩阵和离群体系清单进入 Supporting Information。

## 6. 实验 1：CPU native batch 与公开接口端到端基线

### 6.1 研究问题

在相同单进程、固定 CPU affinity 和 thread budget 下，xTBloom 的 whole-batch API、
persistent worker pool 和分子级调度，相对 xTB/tblite 逐体系 public-API loop 带来多少
端到端吞吐收益？该实验评价的是各库公开接口及其原生运行时共同实现的工作流，不把
batch scheduling 从 xTBloom 的产品能力中人为剥离，也不声称覆盖参考程序所有可能的
应用层并行部署。

### 6.2 矩阵

| 维度 | 坐标 |
| --- | --- |
| CPU worker/core budget | `1 / 4 / 16 / min(32, allocation physical cores)` |
| Batch size | `1 / 32 / 128` |
| 系统规模 | small / medium / large AO |
| 输出 | GFN2 energy + analytic forces |
| SCC start | 正文统一 `FRESH/cold` |

规模档位由冻结性能样本的 AO 分布确定，并同时报告原子数和 SCC iteration。每个 batch
由不同真实结构组成，不复制同一几何制造吞吐。

### 6.3 正文对照与主张边界

1. xTB serial public-API loop；
2. tblite serial public-API loop；
3. xTBloom 在同一复用 context 下的逐体系调用；
4. xTBloom 单次 ragged call，`cpu_threads=1`；
5. xTBloom 单次 ragged call，完整 worker pool。

xTB/tblite 的 calculator 和 xTBloom context、固定 descriptors、caller-owned outputs
均在计时前创建；不可避免的 geometry update、single point、getter/publication 和同步
保留在计时内。不得在结果出来后减去某个实现的公开 API 成本。正文报告的 speedup
必须命名为 `native-interface end-to-end speedup`，并明确它包含 xTBloom 原生跨体系
调度相对参考程序逐体系 public API 的优势；不得简称为 engine-only speedup 或双方
最佳 CPU 部署的普遍排名。

Batch 1 只作为 latency 背景，不用于宣称多 worker batch speedup。

### 6.4 SI 多进程敏感性分析

为回应“用户可在库外自行并行”的问题，在 Supporting Information 选择
`small/medium AO × batch 128` 等少量代表坐标，增加：

1. persistent xTB multiprocessing pool，`N processes × 1 thread`；
2. persistent tblite multiprocessing pool，`N processes × 1 thread`；
3. xTBloom，`1 process × N workers`。

三条路径固定到同一组物理核心并使用相同总内存上限；进程池、calculator/context 和
IPC 队列在计时前创建，报告 steady-state wall time、CPU-seconds、peak aggregate RSS 和
进程启动成本。该对照用于量化应用层编排能追回多少吞吐及其资源代价，不替代正文的
原生接口比较。只有这一敏感性分析也支持时，才能写“xTBloom 优于参考实现的最佳外部
并行基线”。

### 6.5 统计与结果

- 短坐标：至少 10 次 warm-up、30 次测量；
- 长坐标：至少 5 次独立运行，并保留选择该替代方案的理由；
- 报告 median、IQR、bootstrap 95% CI、systems/s、speedup、parallel efficiency；
- 正文固定同一单进程 CPU affinity、NUMA、频率策略和 thread budget；xTBloom 的
  workers 与 xTB/tblite 的 OpenMP threads 都不得超出该预算，BLAS/LAPACK 避免嵌套
  oversubscription；
- 若 speedup 的置信区间没有完全高于 1，则只报告观察值，不写“更快”。

**Figure 2** 展示 native-interface systems/s、xTBloom worker scaling、CPU utilization
和最小归因消融：

```text
同一复用 context 的逐体系调用
-> 单次 ragged call, cpu_threads=1
-> 单次 ragged call, full worker pool
```

外部 persistent process-pool 结果进入 SI 独立表/图，不与正文 native-interface
speedup 混成一个未标注的比例。

## 7. 实验 2：GPU batch scaling 与 CPU-CUDA crossover

### 7.1 研究问题

CUDA 从什么 AO 规模和 batch size 开始优于 CPU？数据传输、launch、SCC 和
eigensolver 如何决定 crossover、饱和与容量边界？

### 7.2 矩阵

AO 档位：

- `1-64`；
- `65-128`；
- `129-256`；
- `257-512`；
- `513-1024`；
- 若存在，`>1024` 作为 capacity stress，而不是常规性能档位。

Batch size：`1 / 4 / 16 / 64 / 256`，随后继续增加直到吞吐饱和或 OOM。每个坐标
使用不同真实结构，并报告 SCC iteration 分布。

### 7.3 对照与计时边界

1. xTBloom CPU；
2. xTBloom CUDA host-to-host；
3. xTBloom CUDA device-to-device；
4. dxtb CUDA 的语义可比坐标。

正文 CUDA benchmark 固定为单 GPU、单进程，并使用各实现公开支持的 batch 路径。
不为“统一并行形态”而人为构造多个进程共享一张 GPU：多 CUDA context、显存复制和
进程调度属于另一种部署问题，既不是 xTBloom 的目标接口，也不是本文公平性的必要
条件。GPU 公平性由同一设备、相同 workload/输出合同、数据驻留、同步边界和正确性
门槛定义，而不是由 CPU worker/process 数定义。

Mixed descriptors 进入 P0 正确性矩阵，并在 SI 中做少量性能 spot check，不扩展正文
笛卡尔积。dxtb 只比较共享的固定参数 GFN2 energy/force inference，不把自动微分训练
能力混入整体排名。

CUDA 每个计时区间末尾显式同步。Host-to-host 与 device-to-device 分开报告；输入上传、
输出下载、结果 publication 和 correctness-only download 的边界必须逐引擎记录，不能
把 host-descriptor xTBloom 与 device-resident dxtb 直接解释为公平 GPU 排名。

已知或新发现的大 AO、batch-1 低效点必须保留，不能通过选点隐藏。

### 7.4 输出

**Figure 3** 包含：

- AO size × batch size 的 GPU/CPU throughput heatmap；
- 绝对 systems/s 与 batch-1 latency；
- host-to-host 与 device-to-device crossover；
- peak VRAM、最大可用 batch 和 OOM 边界；
- 三个 profiler 代表点：launch-bound、throughput-saturated、large-AO eigensolver-bound。

Profiler 只保留经审查的派生 CSV/text/JSON。不得把可能包含环境和凭据的原始
`.nsys-rep`、`.ncu-rep`、SQLite 等捕获文件加入仓库。

## 8. 实验 3：复杂数据收敛率与 ragged heterogeneous workloads

### 8.1 实验 3-A：多方法收敛率

在冻结的 QM9/OMol25 数值主样本和 OMol25 压力样本上运行：

- xTBloom CPU；
- xTBloom CUDA；
- xTB；
- tblite；
- dxtb 的可比坐标。

所有运行使用 FRESH、同一模型/charge/spin/电子温度和预先固定的最大 SCC iteration。
不同程序的 convergence option 不一定一一对应；必须记录各自 public controls 和最终
残差，不能因为参数名称相似就宣称阈值相同。

终止状态统一映射为：

- converged；
- SCC not converged；
- eigensolver/numerical failure；
- invalid input；
- unsupported；
- resource/OOM failure。

`unsupported` 不进入收敛率分母。数值误差只在共同收敛体系上计算；收敛率在全部
eligible 体系上计算，避免 survivorship bias。

报告：

- 主样本总体和分层 convergence rate，Wilson 95% CI；
- 配对列联表：both、xTBloom-only、reference-only、neither；
- paired convergence-rate difference；
- SCC iteration ECDF、median、p95；
- charge、spin、元素类别、AO 数、数据域和构型类别下的失败率；
- failure reason 和可复现体系 ID。

OMol25 压力样本只报告分层结果，不并入总体概率估计。

### 8.2 实验 3-B：ragged runtime 与消融

使用两个 workload：

1. **受控异构 workload**：从独立性能样本按 AO 档位均匀组合，形成预先规定的宽
   尺寸分布；
2. **真实 workload**：使用冻结的 QM9/OMol25 独立性能样本，并保持 manifest 中的
   固定顺序和真实分层比例。

xTBloom CPU 消融路径为：

1. 同一复用 context 下逐体系调用；
2. 单次 ragged batch，`cpu_threads=1`；
3. 单次 ragged batch，完整 worker pool。

若 CUDA bucketing 存在可审计、不会改变物理或错误语义的切换方式，再添加
bucketed/no-bucketing 对照；否则只报告完整公开 CUDA runtime，不构造非生产路径。

同时报告：

- all-input time-to-solution；
- successful systems/s；
- matched-success-subset systems/s；
- success rate；
- p50/p95 latency；
- peak RSS/VRAM；
- SCC iteration 长尾；
- 失败体系是否影响健康 peer。

只报告 successful systems/s 会奖励过早失败，因此它必须与 all-input time、success
rate 和 matched-success subset 同时出现。

### 8.3 输出

**Figure 4** 建议包含：

- Panel A：QM9/OMol25 多方法收敛率和配对结果；
- Panel B：SCC iteration ECDF；
- Panel C：受控 ragged workload 消融；
- Panel D：真实 OMol25 workload 的 throughput、time-to-solution 和 success rate。

## 9. 所有实验的固定合同

### 9.1 数值与物理

- 公开数值使用 IEEE binary64 和原子单位；
- 位置为 bohr，能量为 Hartree，力为 Hartree/bohr；
- xTB gradient 必须显式转换为负梯度 force；
- 电子温度是 `k_B T` 的 Hartree 能量尺度；
- 有限温度下比较的是电子 Helmholtz free energy；
- charge、spin、GFN2 参数、输出性质和最大迭代数逐坐标固定；
- 正文性能使用 FRESH/cold，不混入 WARM。

### 9.2 性能资格

- 性能点只有通过预先规定的 energy/force/charge correctness gate 后才可进入正式图；
- 失败和 unavailable 坐标必须保留；
- setup、first-call、steady-state 分开；
- CPU 正文比较使用相同单进程 thread budget、affinity、NUMA 和 BLAS 线程策略，并
  明示 xTBloom 跨体系 workers 与参考程序体系内 OpenMP threads 的执行语义不同；
- `native-interface end-to-end speedup` 可以包含 xTBloom 原生 batch scheduling；若主张
  优于参考程序的最佳可达 CPU 吞吐，则必须通过 SI persistent process-pool 对照；
- CUDA 明确 host/device/mixed descriptor，且计时结束显式同步；
- 不从单次最佳 timing、单一分子族或单一硬件推出通用排名。

### 9.3 统计

- 短坐标默认至少 30 个 measured samples；
- 长 workload 使用预先说明的多次独立运行；
- 保存所有 raw samples；
- 报告 median、IQR、p95 或 bootstrap CI，不只报告单一均值；
- 对数据集总体率使用概率样本和抽样权重；压力样本不进入总体估计。

### 9.4 复现与归档

每个正式 bundle 至少保留：

- clean source revision 和 dirty bit；
- 库/可执行文件绝对路径及 SHA-256；
- 编译器、flags、CMake cache、CUDA 和线性代数 provider；
- CPU/GPU、affinity、线程和 Slurm allocation；
- workload manifest、random/hash seed、descriptors、SCC options；
- setup、warm-up、同步、sample count 和完整 timing boundary；
- raw samples、correctness、convergence、SCC iterations 和 unavailable reason；
- `README.md`、`SHA256SUMS` 和精确复现命令。

仓库内 benchmark evidence 继续遵守单文件 1 MiB、总目录 16 MiB 上限。超出预算的
可复现原始文件不应通过手工裁剪伪装成完整证据；应保留紧凑结果、hash、命令、输入
身份和可复现说明，必要时再使用经过验证的外部不可变归档。

## 10. Supporting Information 最低范围

- GFN1 CPU energy/force/charge 数值验证；
- QM/MM energy、QM force、point-charge force 和力守恒；
- CUDA mixed descriptor 正确性；
- xTB/tblite `N processes × 1 thread` 与 xTBloom `1 process × N workers` 的少量
  等核心敏感性坐标，并同时报告 aggregate RSS 与进程启动成本；
- 完整有限差分与失败语义矩阵；
- QM9/OMol25 各分层误差、收敛和异常体系表；
- energy-only 的代表坐标，不做完整正文矩阵；
- 若摘要或贡献仍强调 warm state：增加一个小型真实轨迹的 WARM 数值安全性和性能
  sanity check；否则不做 WARM 论文 claim；
- 第二套硬件只复现三个代表点，属于强烈建议而非正文硬门槛。

## 11. 执行顺序与阶段门

1. 获得数据访问授权，完成许可检查和 eligibility funnel；
2. 生成并冻结数据抽样 manifest；
3. 在同一 clean release candidate 上构建 CPU/CUDA 和 pinned reference engines；
4. 完成 P0 canonical validation、数据集级等价性和失败语义；
5. 只有 P0 通过后，优先执行实验 2 的 GPU matrix、容量测试和三个 profiler 点；
6. 执行实验 1 的正文 native-interface CPU 对照，并完成少量 SI 多进程敏感性坐标；
7. 执行实验 3 的收敛统计和 ragged workload；
8. 校验 SHA-256、统计脚本和图表输入，形成论文表图；
9. 根据实际证据收窄或保留论文 claim，不反向修改门槛或抽样。

若独立 oracle、真实 GPU、有限差分或失败隔离证据缺失，相关结论保持未通过，不能用
CPU-CUDA agreement、绿色单元测试或较宽 benchmark compatibility gate 替代。

## 12. 参考资料

- [xTBloom architecture](developer-guide/architecture.md)
- [xTBloom performance evidence](user-guide/performance.md)
- [Conformance and oracle workflow](../tools/conformance/README.md)
- [Benchmark harnesses](../benchmarks/README.md)
- [Public cross-library matrix](../benchmarks/matrix.md)
- [Cross-engine benchmark protocol](../benchmarks/cross-engine.md)
- [FRESH/WARM benchmark protocol](../benchmarks/fresh-warm.md)
- [QM/MM user contract](user-guide/qmmm.md)
- [QM/MM theory and force conventions](theory/qmmm.md)
- [QM9: Quantum chemistry structures and properties of 134 kilo molecules](https://doi.org/10.1038/sdata.2014.22)
- [OMol25: The Open Molecules 2025 Dataset, Evaluations, and Models](https://arxiv.org/abs/2505.08762)
- [OMol25 official dataset documentation](https://github.com/facebookresearch/fairchem/blob/main/docs/molecules/datasets/omol25_elec.md)
