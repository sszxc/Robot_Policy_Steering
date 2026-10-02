# ASU Sol 超算使用参考

> 用途：交给 coding agent / 其他 AI agent 作为在 ASU Research Computing（RC）**Sol** 集群上提交、监控 GPU 任务的参考。
> 来源：逐页核对 https://docs.rc.asu.edu 全站 143 个页面（含页面中折叠的标签页内容），核对日期 2026-10-01。
> 标注约定：
> - 无标注 = 文档原文明确写明（每节末尾附来源链接）
> - **[通用 Slurm]** = 文档未写，但属于 Slurm 标准功能，应可用，首次使用时自行验证
> - **[未验证]** = 文档未提及，基于经验的建议，必须在集群上实测
> - **[文档矛盾]** = 文档不同页面说法不一致，以集群实际查询结果为准

---

## 本 repo 的现场验证（2026-10-01）

- 常用命令入口：[ASU_Sol_HPC_quick.md](ASU_Sol_HPC_quick.md)。
- `xzhan742@sol.asu.edu` 的纯公钥非交互登录成功（`BatchMode=yes`、禁用密码和键盘交互认证）；Duo 提示横幅仍会显示，但不妨碍密钥登录。
- `myaccounts -p`：默认及可用 account 为 `grp_hbenamor`，QoS 为 `debug,private,public`；未显示组内专用 `grp_` QoS。提交时显式写 `-A grp_hbenamor`。
- `public` 时限 7 天，`htc` 4 小时，`lightwork` 24 小时；A100 整卡 GRES 为 `a100`。`htc` 还显示 `a100.40gb`、`l40`、`h100`、`h200` 等，故硬件选择以 `sinfo` 为准。
- **新增实际限制：GPU 任务每节点至少申请 24000 MB 主机内存。** `--mem=4G` 被提交插件拒绝；常用 GPU 命令建议至少 `--mem=32G`。
- 登录环境可见 `interactive`、`myjobs` 和 `mamba/latest`，但 `command -v myquota` 未找到该命令，不能把文档里的 `myquota` 当成已验证可用。
- GPU smoke test：任务 `64301454`，`-A grp_hbenamor -p htc -q public -c 2 --mem=32G -G a100:1 -t 00:02:00`，在 `sg019` 运行 7 秒，`COMPLETED`、`ExitCode=0:0`，`nvidia-smi -L` 显示 `NVIDIA A100-SXM4-80GB`。日志保留在 `/scratch/xzhan742/agent-sol-smoke-64301454.out`。此测试只证明 GPU 分配和驱动查询正常。
- 仓库附带的 `slurm-job-scripts.SKILL.md` 使用通用 `general/public` 起步值；本 repo 日常应采用 `htc/public` 或 `public/public`，不要直接复制该默认组合。
- 此记录不代表已验证 Python 环境、MuJoCo/EGL、LIBERO 或文件传输。

---

## 0. 给 agent 的硬性规则（务必遵守）

1. **不要在登录节点上做计算、装包、编译、下载大文件。** 装环境用 `interactive -p lightwork` 或 `interactive -p htc`；登录节点只用于提交和查询任务。用 VS Code Remote-SSH 连登录节点也会因占用 CPU 被认定违规，应改用 VS Code Tunnel。
2. **Python 环境只用系统提供的 mamba**：`module load mamba/latest`，激活用 `source activate <env>`。**不要**用 `conda activate` / `mamba activate`，不要自己装 conda/miniconda，不要 `pip install --user`。
3. **不支持 Docker，没有 sudo。** 容器用 Apptainer（可直接拉 Docker 镜像：`apptainer pull docker://...`），GPU 容器加 `--nv`。
4. 建议在 sbatch 脚本里加 `#SBATCH --export=NONE`，避免把登录 shell 的环境变量带进任务（官方示例都这样写）。这样做之后，任务需要的环境变量都要在脚本里自己 `export`。
5. 大文件、模型缓存、checkpoint、评估输出放 `/scratch/$USER`（不是 `$HOME`，`$HOME` 只有 100 GB）。scratch **90 天未访问自动删除，且无快照、不可恢复**。
6. 不要把 `module load` 写进 `~/.bashrc`，也不要往 `~/.bashrc` 加 conda 初始化代码，否则网页门户的会话会启动失败。
7. 不要用脚本循环逐个 `sbatch` 大量任务，用 **job array**（见 §6）。

来源：[Python Envs and Mamba](https://docs.rc.asu.edu/mamba) · [Python Common Issues](https://docs.rc.asu.edu/python-common-issues) · [Building Software](https://docs.rc.asu.edu/building-software) · [VSCode](https://docs.rc.asu.edu/vscode) · [Scratch](https://docs.rc.asu.edu/scratch) · [Available Software](https://docs.rc.asu.edu/available-software) · [Troubleshooting Web Portal](https://docs.rc.asu.edu/troubleshooting-webportal)

---

## 1. 连接

| 项目 | 内容 |
|---|---|
| SSH | `ssh <asurite>@sol.asu.edu`（需要 GUI 时加 `-X`） |
| VPN | 官方要求所有连接都走 ASU Cisco VPN：`sslvpn.asu.edu/2fa` |
| 2FA | 自 2026-01-05 起，**密码登录** SSH 需要 Duo Push（不支持电话或短信）；scp/rsync 同样需要 |
| SSH 密钥 | **用密钥登录可免 Duo**，适合自动化 / agent 脚本：本地 `ssh-keygen -t ed25519` 后执行 `ssh-copy-id <asurite>@sol.asu.edu` |
| 网页门户 | https://sol.asu.edu（Open OnDemand：文件、Job Composer、Jupyter、虚拟桌面、Shell） |
| 数据传输节点 | `soldtn`（`ssh <asurite>@soldtn.sol.rc.asu.edu`），rsync 等大批量传输在这里做，**不要在登录节点跑 rsync** |
| WSL | 官方说明 WSL 自带的 ssh 与 Cisco VPN 不兼容，不受支持 |

来源：[Terminal Clients](https://docs.rc.asu.edu/terminal-clients) · [Duo](https://docs.rc.asu.edu/duo) · [SSH Keys](https://docs.rc.asu.edu/ssh-keys) · [SSL VPN](https://docs.rc.asu.edu/sslvpn) · [Transferring between Supercomputers](https://docs.rc.asu.edu/transferring-between-supercomputers)

---

## 2. Sol 硬件（公开文档列出的部分）

| 节点类型 | CPU | 内存 | 加速器 |
|---|---|---|---|
| 标准计算 | 128 核（2× EPYC 7713） | 512 GiB | — |
| 高内存 | 128 核 | 2048 GiB | — |
| GPU A100 | 48 核（2× EPYC 7413） | 512 GiB | 4× A100 80GiB |
| GPU A30 | 48 核 | 512 GiB | 3× A30 24GiB |
| GPU MIG | 48 核 | 512 GiB | 16× A100 切片（20GiB 与 10GiB） |
| Grace Hopper | 72 核 Grace（**aarch64**） | 512 GiB | 1× GH200 480GB |
| MI200 | 24 核 | 77 GiB | 2× AMD MI200（**非 CUDA**） |
| Gaudi2 | 152 核 | — | 8× Intel Gaudi2 HPU（**非 CUDA**） |
| FPGA | 多种 | — | Xilinx U280 / BittWare 520N-MX / NEC VE |

- 驱动：2026-05 维护后 NVIDIA 驱动为 `595.71.05`（支持 CUDA 13.2），Slurm 版本 `25.11.6`，系统为 Rocky Linux 8.x。

来源：[Supercomputer Hardware](https://docs.rc.asu.edu/supercomputer-hardware) · [Private Nodes](https://docs.rc.asu.edu/supercomputer-private-nodes) · [About](https://docs.rc.asu.edu/about) · [Changelog May 2026](https://docs.rc.asu.edu/changelog/may-2026) · [Building Software](https://docs.rc.asu.edu/building-software)

---

## 3. 分区（`-p`）与 QoS（`-q`）

### 分区

| 分区 | 用途 | 最长时间 | 要点 |
|---|---|---|---|
| `public` | RC 自有的全部节点（CPU+GPU），**大多数任务的默认选择** | 7 天 | 可配 QoS：`public` / `debug` / `long` / `class` |
| `htc` | ≤ 4 小时的短任务 | 4 小时 | 可以用上私有节点的空闲资源，所以通常**排队更快**；配 `public` QoS 时**不会被抢占** |
| `general` | 私有（课题组购买）节点 | 组内 30 天 | 非本组成员用 `-q private`，可能被抢占 |
| `lightwork` | 轻量任务：建 mamba 环境、编译、VS Code tunnel、批量文件操作 | 24 小时 | 每节点最多 8 核且超售；长时间满载 CPU 会被取消 |
| `highmem` | 2TB 内存节点 | 48 小时 | — |
| `arm` | Grace Hopper（aarch64） | — | x86 软件不能直接运行 |
| `fpga` | FPGA / NEC VE / **MI200** | — | — |
| `gaudi` | Intel Gaudi2 | — | 需要专用 Python 环境 |

### QoS

| QoS | 说明 |
|---|---|
| `public` | 默认，绝大多数情况都用它 |
| `debug` | 最长 **15 分钟**，排队快，用于测试脚本；**每人最多 2 个 debug 任务在队列中** |
| `private` | 在 `general` 分区使用别组的私有节点；节点所有者需要时**立即取消你的任务**；每人上限 161,280 GPU·分钟（约 16 GPU × 7 天）。建议配合 checkpoint 使用 |
| `grp_<labname>` | 课题组自有节点的 QoS，组员使用时不影响 fairshare |
| `long` | 最长 14 天，需向 RC 申请，仅限 sbatch |
| `class` | 课程账号：每任务 ≤ 32 核 / 320GB / 4 GPU / 24 小时；每人最多 2 个运行中、10 个排队；共 960 GPU·分钟 |

- 不写 `-q` 时自动用 `public`。用 `salloc -t <=240` 且不写分区时，系统会自动分到 `htc`。
- **[文档矛盾]** `lightwork` 的示例有 `-t 0-24` 和 `-t 0-12` 两种写法，文字说明为 24 小时。

来源：[Partitions and QoS](https://docs.rc.asu.edu/partitions-and-qos) · [Resource Limits](https://docs.rc.asu.edu/resource-limits) · [Changelog Fall 2024](https://docs.rc.asu.edu/changelog/fall-2024)

---

## 4. 申请资源

### CPU 与内存

```bash
-c 8                 # 8 个 CPU 核（单任务）
--mem=64G            # 总内存（推荐）；不写时每核默认 2GB
--mem-per-cpu=4G     # 每核内存（仅 OpenMP/多线程场景）
--mem=0              # 整节点内存
-N 1                 # 节点数；非 MPI 程序多节点没有收益，单节点任务建议写 -N 1
```

### GPU（`-G` 即 `--gpus`；也可写 `--gres=gpu:<type>:<n>`）

| 写法 | 得到什么 | 分区 |
|---|---|---|
| `-G 1` | **第一个可用的任意 GPU**（可能是 MIG 切片或 A30） | public / htc / general |
| `-G a100:1` | A100 **80GB** 整卡（最常见，public/htc 中有数百张） | public, htc |
| `-G a100:4` | 4× A100 80GB（一个节点上限 4 张） | public, htc |
| `-G a100.40gb:1` | A100 40GB（数量很少） | general |
| `-G a100.20gb:1` | A100 **MIG 20GB 切片**（几十个） | public, lightwork |
| `-p general -q private -G a30:2` | A30 24GB（可能被抢占） | general |
| `-p arm -G 1` | GH200（aarch64） | arm |
| `-p fpga -G mi200:1` | AMD MI200（非 CUDA） | fpga |
| `-p gaudi -G 1 -c 18` | Gaudi2（推荐每卡配 18 核） | gaudi |

- 2026-05 维护后 GPU 名称有变化：MIG 20GB 切片的名称改为 `a100.20gb`，40GB 卡改为 `a100.40gb`，`a100` 仍指 80GB。旧名称提交的排队任务需要取消后重新提交。
- 约束写法：vLLM 页面的示例用 `-G a100:1 -C a100_40` 或 `-C a100_80` 来区分显存大小。**[文档矛盾]** 这与 "40GB 卡只在 general 分区、数量很少" 的说法不一致，使用前用 §9 的命令查实际可用的 GRES 和 feature。
- MIG 切片**不支持** GPU 之间点对点通信（无法做张量并行）。**[未验证]** 按 NVIDIA 的设计，MIG 也不支持图形 API（OpenGL/EGL），**需要渲染的仿真评估不要用 `-G 1` 或 `a100.20gb`，应显式写 `-G a100:1`**。
- 私有 L40S / H100 NVL 节点的 GRES 名称文档没有写，用 `sinfo` 查（§9）。

来源：[Requesting Resources](https://docs.rc.asu.edu/requesting-resources) · [Changelog May 2026](https://docs.rc.asu.edu/changelog/may-2026) · [vLLM](https://docs.rc.asu.edu/vllm) · [Grace Hopper](https://docs.rc.asu.edu/grace-hopper)

### 时间格式

`-t 15`（分钟）、`-t 0-4`（4 小时）、`-t 0-12:00:00`、`-t 7-0`（7 天）。时间写得越短越容易被 backfill 调度提前启动；临近维护窗口时，长任务会因为 `ReqNodeNotAvail,_Reserved_for_maintenance` 一直卡在排队状态。

---

## 5. 启动任务

### 交互式

```bash
interactive                       # 等价于 salloc -c 1 -p htc -q public -t 0-4
interactive -p lightwork          # 轻量任务，几乎马上分配
interactive -p htc -t 0-2 -c 8 --mem=64G -G a100:1   # 带 GPU 的交互调试
interactive -p public -q debug -t 15   # 用 debug QoS 快速测试；若加 GPU，也须加 --mem=32G
```

`interactive` 接受所有 `salloc` 参数，两者可以互换。

### 批处理（正式运行用这种）

```bash
sbatch job.sh                        # 使用脚本内的 #SBATCH 设置
sbatch -c 4 -t 1-0 job.sh            # 命令行参数会覆盖脚本内设置
sbatch --test-only job.sh            # [通用 Slurm] 只校验脚本、不排队
scontrol update job <id> Gres=gpu:a100:2        # 修改仍在 PENDING 的任务
scontrol update job <id> QOS=private Partition=htc
scancel <jobid>
```

常见报错：`Invalid feature specification` 表示分区、QoS 或 `-C` 写错；`DOS line breaks` 用 `dos2unix job.sh` 修复；缺少 `#!/bin/bash` 首行也会报错。

来源：[Interactive Sessions](https://docs.rc.asu.edu/interactive-sessions) · [SBATCH Job Scripts](https://docs.rc.asu.edu/slurm-sbatch)

---

## 6. 机器人策略评估 sbatch 模板

```bash
#!/bin/bash
#SBATCH -J policy-eval
#SBATCH -p public                 # ≤4h 的任务改用 htc，排队更快且不会被抢占
#SBATCH -q public
#SBATCH -N 1
#SBATCH -c 8                      # 并行仿真环境多就多要；A100 节点每节点共 48 核 / 4 卡
#SBATCH --mem=64G
#SBATCH -G a100:1                 # 需要渲染时不要用 -G 1（可能分到 MIG 切片）
#SBATCH -t 0-12:00:00
#SBATCH -o logs/%x-%j.out         # 目录必须事先存在：mkdir -p logs
#SBATCH -e logs/%x-%j.err
#SBATCH --export=NONE
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user="%u@asu.edu"

module purge
module load mamba/latest
source activate my-eval-env       # 或 /scratch/$USER/envs/xxx 这样的路径

export PYTHONUNBUFFERED=1         # 让日志实时刷新
export HF_HOME=/scratch/$USER/huggingface

# ---- 无显示器渲染：以下均为 [未验证]，需先在交互会话中实测 ----
export MUJOCO_GL=egl              # MuJoCo / robosuite / LIBERO / metaworld 等
export PYOPENGL_PLATFORM=egl
# 若 EGL 不可用，可退回 CPU 软件渲染（慢）：
#   export MUJOCO_GL=osmesa  或  module load gcc-12.1.0-gcc-11.2.0 mesa-22.0.2-gcc-11.2.0

nvidia-smi
cd /scratch/$USER/my-project
python eval.py --ckpt ckpts/model.pt --num_episodes 100 --out results/$SLURM_JOB_ID
```

### 多个 checkpoint 或随机种子：用 job array

```bash
#SBATCH --array=1-20%8            # 20 个子任务，最多同时运行 8 个（%8 为 [通用 Slurm] 写法）
#SBATCH -o logs/%x-%A_%a.out      # %A=数组主任务 ID，%a=子任务编号

LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" manifest.txt)   # manifest 每行对应一组参数
python eval.py $LINE
```

- 用 job array 提交只扣一次 fairshare；逐个 sbatch 每次都扣，后提交的任务会排得越来越慢。
- 文档示例使用 Sol 自带命令 `getline $SLURM_ARRAY_TASK_ID manifest`（非标准 Linux 命令），可用 `sed -n` 替代。
- 生成参数组合的方法：`parallel -k echo {} ::: ckptA ckptB ::: $(seq 1 5) > manifest.txt`。
- 失败的子任务可以单独重跑：`sbatch -a 7 job.sh`。

### 渲染相关的已知事实与建议

- 文档说明：OpenGL/WebGL **未获官方支持**，只提供基于 Mesa 的软件实现，配合网页门户的虚拟桌面使用（至少 8 核）。Sol 上加载：`module load gcc-12.1.0-gcc-11.2.0` 和 `module load mesa-22.0.2-gcc-11.2.0`。
- 文档**没有**提到 EGL、MuJoCo、Isaac Sim/Gym 或 headless 渲染。
- `showsimg` 能列出 `carla-0.9.15.sif` 和 `carla_latest.sif`，说明 CARLA 驾驶仿真器有现成容器。
- **[未验证]** A100 是计算卡，但 NVIDIA 驱动通常带有 EGL，headless EGL 渲染一般可行。请先交互式确认：`interactive -p htc -t 0-1 -G a100:1 -c 4 --mem=32G`，然后运行 `python -c "import mujoco; ..."` 或测试 `eglinfo`。
- Isaac Sim / Isaac Lab 依赖 RTX 光追和 Vulkan，A100 **不具备 RT core**，大概率无法运行完整渲染管线；L40S（私有节点）在硬件上更合适。**[未验证]**

来源：[SBATCH Job Scripts](https://docs.rc.asu.edu/slurm-sbatch) · [Job Array Examples](https://docs.rc.asu.edu/slurm-job-array-examples) · [Fairshare](https://docs.rc.asu.edu/fairshare) · [OpenGL](https://docs.rc.asu.edu/opengl) · [X11 & GUIs](https://docs.rc.asu.edu/x11-and-guis) · [Apptainer](https://docs.rc.asu.edu/apptainer)

---

## 7. 监控与追踪

| 命令 | 作用 | 来源 |
|---|---|---|
| `myjobs` | 查看自己排队/运行中的任务（含分区/QoS、Node/Core/GPU、节点名或排队原因） | Sol 自带 |
| `sq` / `sq -u $USER` | 格式化版 `squeue` | Sol 自带 |
| `squeue -u $USER` | 标准查询 | Slurm |
| `thisjob <jobid>` | 任务详情，含**预计开始时间**（在任务内部运行可省略 ID） | Sol 自带 |
| `scontrol show job <jobid>` | 查看 `Reason=`、`StartTime=` | Slurm |
| `tail -f logs/<name>-<jobid>.out` | 实时看日志 | — |
| `ssh <节点名>` 后运行 `nvtop` / `htop` | **查看运行中任务的实时 GPU/CPU 占用**。只能 ssh 到你有任务在运行的节点，节点名在 `myjobs` 的 NODELIST 列 | 文档 |
| `srun --jobid=<id> --overlap --pty bash` | 进入运行中任务的分配，效果同上 | [通用 Slurm] |
| `seff <jobid>` | 结束后查看 CPU/内存效率（**不统计 GPU**，运行中的任务数据不准） | 文档 |
| `mysacct` | 等价于 `sacct --user=$USER --format=jobid,avecpu,maxrss,cputime,allocTRES%42,state`，默认只显示当天的任务；可加 `--starttime=YYYY-MM-DD` | Sol 自带 |
| `sacct -j <jobid> --format=jobid,state,exitcode,elapsed,maxrss` | 查看结束任务的状态和退出码 | Slurm |
| `myfairshare` / `mybalance` | 查看 fairshare 分数 | Sol 自带 |
| `myaccounts` / `myaccounts -p` | 查看自己的 Slurm 账户和可用 QoS | Sol 自带 |
| `myquota` | 查看 scratch 配额 | Sol 自带 |
| `sinfo` / `ns` | 集群节点状态（`ns` 为命令行版状态页） | Sol 自带 |

邮件通知：`#SBATCH --mail-type=ALL|END|FAIL` 配合 `--mail-user="%u@asu.edu"`。

### 常见 PENDING 原因

| Reason | 含义 |
|---|---|
| `Priority` | 正常排队，取决于 fairshare |
| `Resources` | 你是该分区优先级最高的任务，等待资源释放 |
| `ReqNodeNotAvail, May be reserved for other job` | 已排定在某个节点上，即将启动 |
| `ReqNodeNotAvail,_Reserved_for_maintenance` | 申请的时长会跨越维护窗口，**缩短 `-t`** 即可 |
| `QOSMaxCpuPerJobLimit` | 超出 QoS 的总量限制（多见于 class 账号） |
| `Nodes required for job are DOWN, DRAINED...` | 申请了稀缺或指定的节点，而该节点不可用 |
| `REQUEUEHOLD / launch_failure_limit_exceeded_requeued_held` | 启动反复失败，需联系 RC 处理 |

### 退出码

0 表示成功；1 表示一般错误；125 表示**内存不足（OOM）**；126 表示命令无法执行；127 表示命令不存在；129–192 表示被信号终止（减去 128 得到信号编号）。

来源：[Helpful Slurm Commands](https://docs.rc.asu.edu/helpful-slurm-commands) · [Job Statistics](https://docs.rc.asu.edu/job-statistics) · [Evaluating Job Performance](https://docs.rc.asu.edu/evaluating-job-performance) · [Job States](https://docs.rc.asu.edu/jobstates) · [SBATCH](https://docs.rc.asu.edu/slurm-sbatch)

---

## 8. Fairshare（影响排队速度）

- Sol 免费使用，但优先级取决于 fairshare 分数：`FairShare = 2^(-近期用量CHE / 10000)`。用量的半衰期为 1 周。
- 计费公式：`CHE = (核数 + 内存GiB/4 + 3×MIG切片 + 20×A30 + 25×A100 + 40×H100) × 运行小时数`。例如 1 核、4GiB 内存、1 张 A100 跑 4 小时，约 108 CHE。
- 所以**不要多要 GPU 和内存**。申请量多少不影响排队位置，但会决定能否被 backfill 插空提前运行，也会影响之后的 fairshare。
- 在组内私有节点上用 `grp_` QoS 运行不扣 fairshare。

来源：[Fairshare](https://docs.rc.asu.edu/fairshare) · [Slurm Accounts](https://docs.rc.asu.edu/slurm-accounts)

---

## 9. 首次使用前的现场核查命令（agent 应先运行这些，而不是依赖本文档）

```bash
myaccounts -p                                   # 我的账户和可用 QoS（是否有 grp_ / class_ / private）
sinfo -o "%P %a %l %D %G" | sort -u             # [通用 Slurm] 各分区的时限和 GRES 类型
sinfo -p public,htc,general -o "%N %G %f" | sort -u   # [通用 Slurm] 节点 GPU 型号与 feature（如 a100_80）
module avail cuda ; module avail mamba          # CUDA 和 mamba 模块的确切名称
ls /packages/envs                               # 管理员维护的公共 mamba 环境（只读）
showsimg ; ls /packages/apps/simg               # 公共 apptainer 镜像
ls /packages/public/sol-sbatch-templates        # 官方 sbatch 模板
myquota ; du -sh ~                              # 存储使用情况
```

---

## 10. 存储

| 路径 | 配额 | 备份 | 说明 |
|---|---|---|---|
| `$HOME` = `/home/$USER` | 100 GB | 每日快照，保留 28 天 | 存代码和小文件，不适合高强度 IO。写满后网页门户无法启动会话 |
| `$SCRATCH` = `/scratch/$USER` | 100 TB | **无快照** | 高速并行文件系统（BeeGFS）。**90 天未访问即删除**，第 45 天起每周发邮件提醒。禁止通过在 Sol 与 Phoenix 之间搬运数据来规避清理 |
| `/data/grp_<组名>` | 免费 100 GB，扩容 $50/TB/年 | 快照 | 组内共享，Sol 与 Phoenix 共用同一份；目录**按需挂载**，`ls /data` 可能看不到，直接 `cd` 进去即可 |
| Canyon 磁带 | $10/TB/年 | — | 冷存储，仅能通过 Globus 访问，计算节点无法直接读取 |

- **[文档矛盾]** `/data` 快照保留期：Snapshot 页写 28 天，Project Storage 页写 14 天。
- 恢复误删或误改的文件：`rewind <file>` 列出历史版本，`rewind --restore=N <file>` 恢复；也可以 `cd .snapshot` 手动查找。
- 清理 `$HOME`：用 `gdu ~` 或 `ncdu ~` 找大文件；`mamba clean --all`；`rm -rf ~/.cache/pip`；把 Hugging Face 缓存移到 scratch（`export HF_HOME=$SCRATCH/huggingface`）。
- 文件传输：大文件推荐 Globus（集合名以 ASURC 开头）。也可以在本机连 VPN 后用 `rsync -tr --progress <本地路径> <asurite>@soldtn.sol.rc.asu.edu:/scratch/<asurite>/...`（需要 Duo，或配置 SSH 密钥；传输节点先单独验证可登录）。这与 §1 使用数据传输节点的建议一致。网页门户只适合几百 MB 以内的小文件。

来源：[File System Overview](https://docs.rc.asu.edu/file-system-overview) · [Scratch](https://docs.rc.asu.edu/scratch) · [Project Storage](https://docs.rc.asu.edu/project-storage) · [Snapshot Recovery](https://docs.rc.asu.edu/snapshot-recovery) · [Home Cleanup](https://docs.rc.asu.edu/home-directory-cleanup-guide) · [Transferring Files](https://docs.rc.asu.edu/transferring-to-supercomputer) · [Storage Rates](https://docs.rc.asu.edu/storage-rates)

---

## 11. 软件环境

### 模块系统

```bash
module avail <关键词>    # 搜索模块
ml <模块名>              # 等价于 module load
module list
module purge
```

模块有两种命名格式：手动构建的为 `软件/版本`，自动构建的为 `软件-版本-编译器-版本`（例如 `cuda-12.6.1-gcc-12.1.0`）。

### mamba 环境

```bash
interactive -p lightwork          # 不要在登录节点装环境
module load mamba/latest
mamba create -n myenv -c conda-forge python=3.10 <packages>   # 默认装到 ~/.conda/envs
mamba create -p /scratch/$USER/envs/myenv -c conda-forge ...  # 或指定路径，节省 $HOME 空间
source activate myenv
pip install -e .                  # 只在已激活的环境内使用 pip
```

- 不要使用 `defaults` 频道。尽量一条命令装齐所有包，便于依赖求解。
- 公共环境位于 `/packages/envs`（只读），需要修改时先克隆再改。
- 若 `.bashrc` 被 conda 改过，运行 `remove_conda_from_bashrc` 清理。
- 包冲突时检查 `~/.local/lib/python3*`，可以改名或删除。
- 把环境注册成 Jupyter kernel：`mkjupy <env名>`。

### Apptainer

```bash
apptainer pull docker://<image>         # 在计算节点上拉取，存到 /scratch
apptainer exec --nv --bind /scratch my.sif python eval.py
SIMG=<镜像名> classic-interactive -G a100:1   # 快速进入公共镜像的交互会话
```

- 需要自己构建镜像（要 root 权限）时，在本地机器上构建后上传。
- 官方 vLLM 示例同时使用 `--export=NONE`、`apptainer run --nv --bind /scratch` 和 `HF_HOME=/scratch/$USER`，可作为 GPU 容器任务的写法参考。

来源：[Available Software](https://docs.rc.asu.edu/available-software) · [Mamba](https://docs.rc.asu.edu/mamba) · [Python Package Installation](https://docs.rc.asu.edu/python-package-installation) · [Apptainer](https://docs.rc.asu.edu/apptainer) · [Jupyter Kernels](https://docs.rc.asu.edu/jupyter-kernels) · [vLLM](https://docs.rc.asu.edu/vllm)

---

## 12. 其他可能用到的

- **端口转发**（看 TensorBoard、视频或 Web 服务）：在本机执行 `ssh -N -L 8000:<计算节点>:<端口> <asurite>@sol.asu.edu`（vLLM 页面的做法）。
- **给 agent 的官方 Skills 库**：Voyager → LLM Access → Skills，提供 OpenCode `SKILL.md` 格式的技能，包括 "Slurm batch scripts"、"Diagnose stuck and failed jobs"、"GPU right-sizing"、"Python environments on HPC"、"Containers on HPC (Apptainer)"、"Research data layout and quotas" 等，内容按 Sol 的分区、QoS 和文件系统编写。**建议下载放到项目的 `.opencode/skills/<name>/SKILL.md`，或转换给其他 agent 使用。** 官方说明这些 skill 只做只读查询，不会代替用户提交或取消任务。

来源：[VSCode](https://docs.rc.asu.edu/vscode) · [vLLM](https://docs.rc.asu.edu/vllm) · [Getting Started with AI](https://docs.rc.asu.edu/ai/getting-started) · [AI Tools](https://docs.rc.asu.edu/ai-tools) · [Kubernetes](https://docs.rc.asu.edu/kubernetes)
