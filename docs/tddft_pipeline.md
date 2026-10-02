# Resumable fixed-root TDDFT pipeline

Use the checkout containing the TDDFT fixes, not another installed copy. The
entry point reuses `prepare_tddft_inp`, `tddft_batch`, and `export_numpy`.
Default action is **plan**, with no scheduler submission. For a new run use a
fresh directory; prepared files and existing datasets are never overwritten.

Pete example (one BODIPY geometry; all scientific settings come from the existing reference):

```bash
cd /path/to/chemlab
PY=python
WORK=./runs/bodipy-smoke
$PY -m chemlab.util.tddft_pipeline --work "$WORK" \
  --source ./trajectory.npy --types ./atoms.npy \
  --ref ./ref.in \
  --env-setup ./qchem_env.sh \
  --frames 1 --seed 42 --state 1 --cores 4 --memory 64G \
  --partition batch --walltime 00:30:00
$PY -m chemlab.util.tddft_pipeline --work "$WORK" --action submit
$PY -m chemlab.util.tddft_pipeline --work "$WORK" --action status
# Repeat after scheduler completion; this exports only when ALL selected frames pass.
$PY -m chemlab.util.tddft_pipeline --work "$WORK" --action resume
```

Optional one-command execution after planning (explicitly permits submission):

```bash
$PY -m chemlab.util.tddft_pipeline --work "$WORK" --action run \
  --poll-seconds 30 --wait-timeout 3600
```

`run` submits remaining required inputs once, watches queued/running jobs, and
strictly exports only after every selected frame is verified complete. If a
job is already active it attaches to that job; a completed validated run is
reused without submission. The default action is still `plan`; the original
`submit`, `status`, `export` and `resume` behaviors are unchanged. Polling is
configurable from 1 to 60 seconds, and observation timeout defaults to 3600
seconds. Increase the timeout for longer calculations.

Ctrl-C stops observation with exit code 130; timeout stops observation with
exit code 124. Neither cancels the scheduler job. Repeating `--action run`
reattaches safely. Polling releases the local lock between checks so another
status command or observer can inspect the run. An unknown accounting state
blocks submission and is watched until resolved or timeout. Failed runs still
require explicit `--retry`; even `run --retry` makes only one retry attempt and
stops if that new attempt fails. No automatic retry loop is used.

New-run parameters are frozen in `pipeline.json`. Subsequent commands use that
manifest, not new CLI parameter values. To change sampling, settings, or
resources, create a new run. `--types` and `--input-distance-unit` support NPY
sources; default is angstrom. XYZ, TRAJ and complete AIMD sources are supported.
Sampling is bounded and deterministic, and original source frame IDs are kept.
For a BODIPY reference, use PBE0, 6-31G*, five singlet roots, `jobtype force`,
`CIS_STATE_DERIV=1` and appropriate ESP settings. `MEM_TOTAL=60000` MB
requires the 64G SLURM allocation; one process uses four CPU threads, no GPU.
The pipeline rejects references whose force root differs from the selected state
or whose Q-Chem memory exceeds the requested scheduler memory.

`resume` observes existing jobs and exports after successful completion. It never
submits automatically. Failed runs are explicit in `status.json`; inspect the
SLURM logs, `.runner.log`, `.out`, and `.runner.json`, then retry deliberately:

```bash
$PY -m chemlab.util.tddft_pipeline --work "$WORK" --action submit --retry
```

Successful frames are excluded from retries. Queued/running jobs and unknown
accounting states block duplicate submissions. `squeue` and `sacct` must be
available; a missing accounting record is not treated as successful completion.
A local exclusive lock guards concurrent invocations. If the process is killed,
inspect its PID in `.pipeline.lock` and the scheduler before manually archiving
the stale lock. A submission interrupted before its ID is saved remains
`uncertain`; reconcile with the scheduler before editing the manifest. These
conservative cases intentionally require human review to avoid duplicate jobs.

If the environment setup failed, `--action repair-env --env-setup new_env.sh`
replaces only this run's environment snapshot, preserving the previous file and
recording before/after hashes. It is allowed only after a terminated failed run,
and does not submit anything; follow with an explicit `submit --retry`.
The Pete example uses the system Lmod entry point; adjust MODULE_INIT and the
compiler module for your cluster. Configure QC, QCAUX and SCRATCH_ROOT in your
private qchem_env.sh before sourcing the example inside the allocated job.

Outputs and provenance:

- `selected.xyz`, `reference.in`, `qchem_env.sh`: selected geometry and input/environment snapshots.
- `pipeline.json`: source/reference/environment/input hashes, root, resources, source frame IDs and job IDs.
- `inputs/`: generated XYZ/INP, untouched new OUT files and runner exit/checksum records.
- `plans/`: independent checksum-protected scheduler manifests and logs.
- `status.json`: every selected frame's pending/queued/running/complete/failed status.
- `arrays/full_tddft.npz` and `full_*.npy`: exported arrays; `source_frame_indices.npy` maps dataset rows to original trajectory frames.
- `arrays/availability.npz`: per-frame, per-state masks for gradients, transition dipoles and transition charges; missing values remain NaN.
- `validation.json`: output/array hashes, energy/gradient checks, shapes and units.

Export requires normal Q-Chem termination, exit code zero, matching input/output
records, unchanged prepared inputs, finite target total energy and gradient,
the exact atom order and input geometry, and
`E_target - E_ground ~= excitation_eV / 27.2113863`. Excitation energy precision
allows an absolute tolerance of 1e-5 Hartree. Missing target gradients fail the
run; missing other-state gradients remain NaN. No failed frames are silently
dropped or filled with zero. Array dimensions, file ordering, state columns,
split indices and `force=-gradient` are verified. Coordinates are angstrom,
total energies Hartree, excitations eV, gradients/forces Hartree/bohr.

Phase alignment uses the fixed target root and flips transition dipoles and
transition charges together. `full_transition_density.npy` and NPZ remain raw;
`full_transmom.npy` and `full_aligned_td.npy` are the paired aligned arrays.
This does not perform bright-state selection or electronic-character tracking.
Repeated export reuses the existing validated files only if all hashes match.
Partial or changed exports require inspection and preservation before retry.

```bash
$PY -m pytest -q tests/test_tddft_pipeline.py
$PY -m pytest -q
```

Configure paths in a private environment wrapper:

```bash
export QC=/path/to/qchem
export QCAUX=/path/to/qcaux
export SCRATCH_ROOT=/path/to/scratch
source /path/to/chemlab/examples/tddft/pete_qchem_env_current.sh
```
