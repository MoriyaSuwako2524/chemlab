# TDDFT 轨迹工作流与回归测试

## 安装与测试

```bash
python -m pip install -e '.[test]'
python -m pytest -q
```

pytest 只收集 `tests/`；不会执行历史计算脚本或调用真实 Q-Chem/SLURM。
覆盖轨迹读入、抽样、输入生成、能量/偶极/梯度导出、失败文件索引、批处理和原有 MECP 回归测试。
GitHub Actions 在 Python 3.11、3.13 上运行同一套测试。

## 准备输入

```bash
python -m chemlab ml_data prepare_tddft_inp \
  --file trajectory.xyz --ref ref.in --out new_inputs \
  --charge 0 --spin 1 --dataset_size 1000 --seed 42
```

支持多帧 `.xyz`、VMD XYZ 风格 `.traj`、Q-Chem AIMD `.out` 和 `.npy`。
普通 TDDFT `.out` 应交给结果导出入口。`.npz` 仍是输出数据集，不是轨迹输入。
计算方法、基组、激发态数和梯度目标态由 `ref.in` 决定；电荷/多重度需按体系设置。

NPY 的坐标必须是 `(帧数, 原子数, 3)`，类型为 `(原子数,)`，内容为整数原子序数或元素符号。
`full_coord.npy` 会寻找同目录唯一的 `full_type.npy` 或 `full_qm_type.npy`；其他名称使用
`--types atoms.npy`。不读取 pickle 对象数组。

```bash
python -m chemlab ml_data prepare_tddft_inp \
  --file positions.npy --types atoms.npy --input_distance_unit bohr \
  --ref ref.in --out new_inputs --dataset_size 1000 --seed 42
```

坐标默认按 Å 解释，XYZ/NPY 可用 `--input_distance_unit ang|bohr|nm` 指定输入单位。
AIMD 的 Standard Nuclear Orientation 固定按 Å 处理。生成的 XYZ/INP 始终使用 Å。
旧 `energy_unit/distance_unit/force_unit` 不再影响准备阶段的几何；该阶段只准备坐标，
不再生成混合单位的临时能量/梯度 NPY。

所有格式采用相同抽样规则：`--start` 跳过开头的帧（从 0 计数），
`--dataset_size 0` 或 `--mode all` 保留剩余全部帧，正数表示最多抽取多少帧。
固定 seed 可复现；抽中的帧保持原顺序。`frames.json` 和 `source_indices.npy` 保存原始帧索引。
生成文件从 `train_0000` 开始。已含 `train_*` 的目录会拒绝重新准备，避免混入旧结果。

对于 walltime 截断的 AIMD 输出，显式加入 `--allow_incomplete true` 可恢复完整几何表；
不完整末帧被丢弃并报警，内部损坏帧仍报错。恢复的是几何，不保证该帧旧能量/梯度完整。

## 批处理与失败重跑

准备一个适合所在集群的 `qchem_env.sh`，设置模块、QC、QCAUX、QCSCRATCH，并使 `qchem` 可用。
例如 Pete 模板见 `examples/tddft/pete_qchem_env.sh`；路径按安装位置调整。

```bash
# 仅生成计划，不提交。
python -m chemlab.util.tddft_batch --data new_inputs --env-setup qchem_env.sh
# 确认配置后提交；按真实文件分组，包括第 0 帧和最后不足一批的文件。
python -m chemlab.util.tddft_batch --data new_inputs --env-setup qchem_env.sh --submit
# 等前一批作业结束后，选择已有但失败/未完成的输出。
python -m chemlab.util.tddft_batch --data new_inputs --env-setup qchem_env.sh --failed-only --submit
```

默认选取缺失、失败、或输入比成功输出更新的任务；`--force` 包含成功任务。
使用 `--pattern 'train_*.inp'` 限定前缀。资源可通过 `--cores`（每个计算线程数）、
`--jobs`（每个节点并发数）、`--batch-size`、`--partition`、`--walltime`、`--qchem` 指定。
每次计划有独立目录及输入校验和；排队后改动输入会拒绝执行。
运行状态保存在 `.runner.json`，控制台输出在 `.runner.log`；旧 `.out` 在重跑前备份。
退出码和 Q-Chem 正常结束语都成功才算完成，fatal error 优先于结束语。
不要在已有批次仍运行时重复提交相同输入。

## 导出

```bash
python -m chemlab ml_data export_numpy \
  --data new_inputs --out corrected_arrays --prefix full_ \
  --state_idx 1 --ex_energy_unit ev
```

输入文件名需以 `train`、`val`、`test` 或 `frame` 开头。
跳过失败文件后，所有分组索引按实际成功记录重新生成，`source_files` 保存对应输出路径。
原子顺序和激发态编号必须一致；缺失物理量保存为 NaN。
额外训练集划分参数大于可用数据时会报警并跳过，不阻止主数据集导出。

`full_tddft.npz` schema 2 带有单位元数据：坐标 Å、激发能 eV、总能量 Hartree、
基态永久偶极 Debye、跃迁偶极 e·bohr、梯度 Hartree/bohr。
单态 NPY 根据指定单位导出；永久偶极转为原子单位，跃迁偶极的原子单位不变。
采用 `1 Debye ≈ 0.393430307 e·bohr`，与
[NIST 原子单位电偶极矩](https://physics.nist.gov/cgi-bin/cuu/Value?auedm)一致。
梯度/力缺失时使用 NaN，不再用零伪装已计算结果；力为负梯度。

## 旧数据迁移

代码修复不会自动改写历史 `.npy/.npz`。旧 `full_ex_energy.npy` 可能重复乘了
27.2113863；旧单态永久偶极数值可能是 Debye 却当作原子单位使用。
请从原始 `.out` 导出到新的目录，并核对 `source_files` 后再更新下游训练输入。
不要直接批量除以常数：不同版本的历史数据可能使用不同导出设置。
旧模板只算 S1 梯度时，S2 等态梯度仍为 NaN；修复解析器不能补出未计算的物理量。
