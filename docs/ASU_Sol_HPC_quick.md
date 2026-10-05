# ASU Sol：agent 常用命令

！请保持这个文档精简！

账号 `xzhan742`；Slurm account `grp_hbenamor`；QoS `public/debug/private`（2026-10-01 实测）。日常用此页，更多选项见 [详细参考](ASU_Sol_HPC_reference.md)。权限变化后重新运行 `myaccounts -p`。

## 1. 连接与查权限（本机执行）

```bash
ssh xzhan742@sol.asu.edu
# agent 非交互检查：失败即退出，不等待密码或 Duo 输入
ssh -T -o BatchMode=yes -o PreferredAuthentications=publickey -o ConnectTimeout=15 \
  xzhan742@sol.asu.edu 'hostname; myaccounts -p; myjobs'
```

## 2. 申请计算节点（以下在 Sol 执行）

```bash
myaccounts -p                              # account / QoS，以现场结果为准
sinfo -p public,htc -o '%P %a %l %G'       # 分区、时限、GPU 名称
interactive -A grp_hbenamor -p lightwork -q public -c 2 --mem=8G -t 01:00:00
# GPU 调试；退出会话用 exit，释放资源
interactive -A grp_hbenamor -p htc -q public -N 1 -c 4 --mem=32G -G a100:1 -t 01:00:00
```

`htc/public` 用于 ≤4 小时任务；更长用 `public/public`（≤7 天）。此处表示 partition/QoS。`general/private` 会被抢占，勿作为默认。GPU 任务每节点至少需要 **24000 MB 主机内存**；显存是另一项资源。LIBERO 渲染显式申请 `a100:1`，避免任意 GPU 分配到 MIG。

## 3. 提交与监控

最小 GPU 检查（只查硬件；实际实验换用 `job.sh`）：

```bash
mkdir -p /scratch/xzhan742/logs
sbatch -A grp_hbenamor -p htc -q public -N 1 -n 1 -c 2 --mem=32G \
  -G a100:1 -t 00:02:00 --export=NONE -J gpu-check \
  --chdir=/scratch/xzhan742 -o /scratch/xzhan742/logs/%x-%j.out \
  --wrap='set -e; hostname; nvidia-smi -L'

sbatch --test-only job.sh                  # 校验资源请求，不执行脚本
sbatch job.sh                              # 提交正式实验
myjobs                                     # 自己的任务与排队原因
scontrol show job JOBID                    # 替换 JOBID 为实际 ID
tail -n 80 /scratch/xzhan742/logs/gpu-check-JOBID.out
sacct -j JOBID --format=JobID,State,ExitCode,Elapsed,MaxRSS
scancel JOBID                              # 只取消明确指定的自己的任务
```

正式脚本的 Python 环境需在脚本内加载：`module purge` → `module load mamba/latest` → `source activate <环境名或路径>`，然后运行程序。环境创建、装包和计算都在计算节点做。`--test-only` 成功不代表 Python/CUDA/EGL 已可运行。

## 4. 传文件（本机执行）

确认数据传输节点可登录，且远端目标目录已创建，再同步项目：

```bash
ssh -T -o BatchMode=yes -o ConnectTimeout=15 xzhan742@soldtn.sol.rc.asu.edu hostname
rsync -tr --progress ./project/ xzhan742@soldtn.sol.rc.asu.edu:/scratch/xzhan742/project/
rsync -tr --progress xzhan742@soldtn.sol.rc.asu.edu:/scratch/xzhan742/project/results/ ./results/
```

不要在登录节点做计算、装包或大批量传输；大文件、模型缓存和输出放 scratch，重要结果另行备份（scratch 90 天未访问清理、无快照）。不要使用 sudo、Docker 或自行安装 conda；环境用系统 mamba，容器用 Apptainer。

## 本项目迁移后的入口

源码仍在 `~/Coding/thesis`，数据仍在 `/scratch/xzhan742/thesis`。
项目结构见 [architecture.md](architecture.md)，环境记录见
`platforms/environments/`。现有 Python 环境无需重建或升级。

```bash
cd ~/Coding/thesis
# 在计算节点上的现有环境中执行一次；不安装依赖。
python -m pip install --no-build-isolation --no-deps -e .

sbatch --test-only platforms/hpc/slurm/smolvla_libero/eval.sbatch smoke
sbatch platforms/hpc/slurm/smolvla_libero/eval.sbatch smoke
sbatch platforms/hpc/slurm/smolvla_libero/array.sbatch
sbatch platforms/hpc/slurm/smolvla_libero/aggregate.sbatch ARRAY_JOB_ID
```

脚本从 Slurm 指定的仓库工作目录加载 `platforms/hpc/runtime.sh`。
数据/输出/环境路径在 `runtime.sh` 中设置；Slurm header
路径和资源通过 `sbatch` 参数覆盖。LIBERO 的通用入口是
`experiments/libero/run_libero10_smolvla.sh`，结果目录必须是新目录。
Object 单卡并发实验见
`experiments/libero/object_parallel/README.md`。
重汇总旧任务时可使用 `python -m research.evaluation.aggregate_eval ARRAY_JOB_ID
--output-dir NEW_DIRECTORY`，保留原结果。

## 早期平台检查记录

2026-10-01：纯密钥 SSH、account/QoS 查询、分区/GRES 查询、`mamba/latest` 可见性已验证。A100 测试任务 `64301454` 在 `sg019` 运行 7 秒，`COMPLETED / 0:0`。当时登录 shell 无 `myquota` 命令。后续环境安装和 LIBERO/EGL 结果见 [基线记录](results/smolvla_libero_baseline.md)；架构迁移验证见 [迁移记录](provenance/architecture-migration.md)。传输节点连接此前超时。
