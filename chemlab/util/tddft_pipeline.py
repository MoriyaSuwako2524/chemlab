"""Reproducible TDDFT prepare/plan/submit/status/export workflow (plan by default)."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np

from chemlab.scripts.ml_data.prepare_tddft_inp import PrepareTddftInp
from chemlab.scripts.ml_data.export_numpy import ExportNumpy
from chemlab.util.modify_inp import qchem_out_excite_multi
from chemlab.util.tddft_batch import make_plan, output_complete
from chemlab.util.tddft_trajectory import read_trajectory


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    temporary = Path(str(path) + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


@contextmanager
def locked(work):
    lock = work / '.pipeline.lock'
    try:
        fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        raise RuntimeError(f'Pipeline already active or interrupted; inspect {lock} before removing it')
    os.write(fd, f'{os.getpid()}\n'.encode())
    os.close(fd)
    try:
        yield
    finally:
        lock.unlink()


def rem_values(path):
    match = re.search(r'\$rem\s*\n(.*?)\$end', Path(path).read_text(), re.I | re.S)
    if not match:
        raise ValueError('Reference has no $rem section')
    values = {}
    for line in match[1].splitlines():
        words = line.split('!')[0].replace('=', ' ').split()
        if len(words) >= 2:
            values[words[0].lower()] = words[1].lower()
    return values


def prepare(args, work):
    manifest = work / 'pipeline.json'
    if manifest.exists():
        cfg = json.loads(manifest.read_text())
        for filename, expected in cfg['input_sha256'].items():
            if digest(work / 'inputs' / filename) != expected:
                raise ValueError(f'Prepared input changed: {filename}')
        if digest(work / 'reference.in') != cfg['reference_sha256'] or digest(work / 'qchem_env.sh') != cfg['env_sha256']:
            raise ValueError('Run reference/environment snapshot changed')
        return cfg
    if not args.source or not args.ref or not args.env_setup:
        raise ValueError('A new run requires --source, --ref and --env-setup')
    if args.frames < 1 or args.state < 1:
        raise ValueError('frames and state must be positive; use a small explicit sample')
    rem = rem_values(args.ref)
    if rem.get('jobtype') != 'force' or int(rem.get('cis_state_deriv', -1)) != args.state:
        raise ValueError('Reference must use jobtype force and CIS_STATE_DERIV equal to --state')
    if int(rem.get('cis_n_roots', 0)) < args.state:
        raise ValueError('Reference has insufficient CIS_N_ROOTS')
    if not re.fullmatch(r'[1-9][0-9]*[MG]', args.memory):
        raise ValueError('memory must be an explicit SLURM amount such as 64G')
    requested_mb = int(args.memory[:-1]) * (1024 if args.memory[-1] == 'G' else 1)
    if requested_mb < int(rem.get('mem_total', 0)):
        raise ValueError('SLURM memory is smaller than Q-Chem MEM_TOTAL')
    if (work / 'inputs').exists():
        raise ValueError('Interrupted preparation: inputs exist without pipeline.json; use a fresh work directory')
    coords, symbols, indices = read_trajectory(Path(args.source), types=args.types,
                                              input_distance_unit=args.input_distance_unit)
    if not 0 <= args.start < len(coords):
        raise ValueError('start outside trajectory')
    # Deterministic bounded selection; snapshot only the selected geometries.
    selected = np.arange(args.start, len(coords))
    if len(selected) > args.frames:
        selected = np.sort(np.random.default_rng(args.seed).choice(selected, args.frames, replace=False))
    sample = work / 'selected.xyz'
    sample.write_text(''.join(str(len(symbols)) + '\nsource_frame=' + str(indices[i]) + '\n' +
                      ''.join(f'{s} {x:.10f} {y:.10f} {z:.10f}\n' for s, (x, y, z) in zip(symbols, coords[i]))
                      for i in selected))
    shutil.copyfile(args.ref, work / 'reference.in')
    shutil.copyfile(args.env_setup, work / 'qchem_env.sh')
    PrepareTddftInp().run(SimpleNamespace(file=str(sample), ref=str(work / 'reference.in'),
        out=str(work / 'inputs'), charge=args.charge, spin=args.spin, dataset_size=0, mode='all',
        start=0, seed=args.seed, types='auto', allow_incomplete=False, input_distance_unit='ang'))
    records = json.loads((work / 'inputs/frames.json').read_text())['frames']
    for record, index in zip(records, selected):
        record['source_frame'] = int(indices[index])
    np.save(work / 'inputs/source_indices.npy', indices[selected])
    save(work / 'inputs/frames.json', {'source': str(Path(args.source).resolve()), 'frames': records,
                                      'coordinate_unit': 'angstrom', 'seed': args.seed})
    cfg = dict(schema_version=1, source=str(Path(args.source).resolve()), source_sha256=digest(args.source),
        reference_source=str(Path(args.ref).resolve()), reference_sha256=digest(work / 'reference.in'),
        env_source=str(Path(args.env_setup).resolve()), env_sha256=digest(work / 'qchem_env.sh'),
        state=args.state, charge=args.charge, spin=args.spin, rem=rem, frames=records, seed=args.seed,
        input_sha256={r['input']: digest(work / 'inputs' / r['input']) for r in records},
        resources=dict(cores=args.cores, jobs=1, batch_size=args.frames, partition=args.partition,
                       walltime=args.walltime, memory=args.memory), qchem=args.qchem, attempts=[],
        implementation_sha256=digest(__file__), created=datetime.now(timezone.utc).isoformat())
    save(manifest, cfg)
    return cfg


def scheduler_state(job_id):
    proc = subprocess.run(['squeue', '-h', '-j', str(job_id), '-o', '%T'], capture_output=True, text=True)
    if proc.returncode and 'Invalid job id specified' not in proc.stderr:
        raise RuntimeError(f'Cannot query squeue for {job_id}: {proc.stderr.strip()}')
    if proc.stdout.strip():
        return proc.stdout.strip().splitlines()[0]
    proc = subprocess.run(['sacct', '-n', '-X', '-j', str(job_id), '--format=State', '-P'], capture_output=True, text=True)
    if proc.returncode:
        raise RuntimeError(f'Cannot query sacct for {job_id}: {proc.stderr.strip()}')
    return proc.stdout.strip().split('|')[0].split()[0] if proc.stdout.strip() else 'UNKNOWN'


TERMINAL = {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL', 'PREEMPTED', 'BOOT_FAIL', 'DEADLINE', 'REVOKED'}


def refresh(cfg, work):
    for attempt in cfg['attempts']:
        if attempt.get('job_id'):
            attempt['scheduler_state'] = scheduler_state(attempt['job_id']).split('+')[0]
    # UNKNOWN is deliberately blocking: absent accounting is not proof the job ended.
    live = any(a.get('submission') == 'uncertain' or
               (a.get('job_id') and a['scheduler_state'] not in TERMINAL) for a in cfg['attempts'])
    rows = []
    for record in cfg['frames']:
        inp = work / 'inputs' / record['input']
        out = inp.with_suffix('.out')
        runner = inp.with_suffix('.runner.json')
        status = 'pending'
        reason = None
        if runner.exists():
            result = json.loads(runner.read_text())
            valid = (result.get('success') and result.get('returncode') == 0 and output_complete(out) and
                     result.get('input_sha256') == digest(inp) and result.get('output_mtime_ns') == out.stat().st_mtime_ns)
            status = 'complete' if valid else 'failed'
            reason = None if valid else f'Runner failure/stale result: returncode={result.get("returncode")}'
        elif out.exists():
            status = 'running' if live else 'failed'
            reason = None if live else 'Output exists without verified runner status'
        elif live:
            status = 'queued'
        elif cfg['attempts']:
            status = 'failed'
            reason = 'Submitted job ended without an output/runner record'
        rows.append(dict(record, status=status, reason=reason))
    report = {'active_or_unknown_job': live, 'frames': rows, 'attempts': cfg['attempts']}
    save(work / 'status.json', report)
    save(work / 'pipeline.json', cfg)
    return report


def plan_submit(cfg, work, report, submit=False, retry=False):
    if report['active_or_unknown_job']:
        raise RuntimeError('A submitted job is active or its state is unknown; refusing duplicate submission')
    if any(r['status'] == 'failed' for r in report['frames']) and not retry:
        raise RuntimeError('Failed frames require explicit --retry after inspecting status.json')
    files = [work / 'inputs' / r['input'] for r in report['frames'] if r['status'] != 'complete']
    if not files:
        print('All frames complete; no submission needed')
        return
    resources = {k: v for k, v in cfg['resources'].items() if k != 'memory'}
    scripts = make_plan(files, work / 'plans', env_setup=work / 'qchem_env.sh', qchem=cfg['qchem'], **resources)
    for script in scripts:
        script.write_text(script.read_text().replace('#SBATCH --nodes=1',
                          '#SBATCH --mem=' + cfg['resources']['memory'] + '\n#SBATCH --nodes=1'))
        print('Plan:', script)
        if submit:
            attempt = {'script': str(script), 'inputs': [str(p) for p in files], 'submission': 'uncertain',
                       'env_sha256': cfg['env_sha256'], 'implementation_sha256': digest(__file__)}
            cfg['attempts'].append(attempt)
            save(work / 'pipeline.json', cfg)  # interruption cannot silently resubmit
            proc = subprocess.run(['sbatch', '--parsable', str(script)], capture_output=True, text=True)
            if proc.returncode:
                attempt.update(submission='rejected', stderr=proc.stderr.strip())
                save(work / 'pipeline.json', cfg)
                raise RuntimeError('sbatch rejected: ' + proc.stderr.strip())
            job_id = proc.stdout.strip().split(';')[0]
            if not job_id.isdigit():
                raise RuntimeError('Unrecognized sbatch result; inspect scheduler before retrying: ' + proc.stdout)
            attempt.update(submission='accepted', job_id=job_id, scheduler_state='PENDING')
            save(work / 'pipeline.json', cfg)
            print('Submitted job', job_id)


def repair_env(cfg, work, report, source):
    if not source or not Path(source).is_file():
        raise ValueError('repair-env requires an existing --env-setup file')
    if report['active_or_unknown_job'] or not any(r['status'] == 'failed' for r in report['frames']):
        raise ValueError('Environment repair requires a terminated failed run')
    old = work / 'qchem_env.sh'
    backup = work / ('qchem_env.previous-' + cfg['env_sha256'] + '.sh')
    if not backup.exists():
        shutil.copyfile(old, backup)
    shutil.copyfile(source, old)
    cfg.setdefault('environment_repairs', []).append({'before_sha256': cfg['env_sha256'],
        'after_sha256': digest(old), 'source': str(Path(source).resolve()), 'backup': str(backup),
        'time': datetime.now(timezone.utc).isoformat()})
    cfg['env_source'], cfg['env_sha256'] = str(Path(source).resolve()), digest(old)
    save(work / 'pipeline.json', cfg)
    print('Environment repaired with preserved previous snapshot; no job submitted')


def validate_output(inp, state):
    out = inp.with_suffix('.out')
    reader = qchem_out_excite_multi()
    reader.read_files([str(out)])
    if len(reader.tasks) != 1:
        raise ValueError(f'Cannot parse completed output: {out}')
    task = reader.tasks[0]
    states = {s.state_idx: s for s in task.states}
    if state not in states or 0 not in states:
        raise ValueError(f'Missing state S{state}: {out}')
    target, ground = states[state], states[0]
    coords, symbols, _ = read_trajectory(inp.with_suffix('.xyz'))
    atoms = [a[0] for a in task.molecule.carti]
    xyz = np.asarray(task.molecule.carti)[:, 1:].astype(float)
    gradient = np.asarray(target.gradient, dtype=float)
    if atoms != list(symbols) or gradient.shape != (len(symbols), 3) or not np.isfinite(gradient).all():
        raise ValueError(f'Missing/invalid S{state} gradient or atom order: {out}')
    np.testing.assert_allclose(xyz, coords[0], atol=2e-5, rtol=0, err_msg='Output geometry differs from prepared input')
    energies = np.asarray([ground.total_energy, target.total_energy, target.excitation_energy], dtype=float)
    if not np.isfinite(energies).all():
        raise ValueError(f'Missing energy: {out}')
    # Printed excitation energies have ~1e-4 eV precision.
    np.testing.assert_allclose(energies[1] - energies[0], energies[2] / 27.2113863, atol=1e-5, rtol=0)
    return {'input': inp.name, 'ground_hartree': float(energies[0]), 'target_hartree': float(energies[1]),
            'excitation_eV': float(energies[2]), 'gradient_shape': list(gradient.shape), 'output_sha256': digest(out)}


def export(cfg, work, report):
    if report['active_or_unknown_job'] or any(r['status'] != 'complete' for r in report['frames']):
        raise ValueError('Export requires every selected frame to be verified complete; see status.json')
    checks = [validate_output(work / 'inputs' / r['input'], cfg['state']) for r in cfg['frames']]
    signature = hashlib.sha256(json.dumps(checks, sort_keys=True).encode()).hexdigest()
    if (work / 'validation.json').exists():
        old = json.loads((work / 'validation.json').read_text())
        if old['signature'] == signature and all(digest(work / name) == value for name, value in old['array_sha256'].items()):
            print('Existing validated export unchanged')
            return old
        raise ValueError('Export already exists with different results; preserve it and use a new run')
    arrays = work / 'arrays'
    if arrays.exists():
        raise ValueError('Partial export exists; inspect/archive arrays before retrying export')
    ExportNumpy().run(SimpleNamespace(data=str(work / 'inputs'), out=str(arrays), prefix='full_',
        state_idx=cfg['state'], energy_unit='hartree', ex_energy_unit='ev', distance_unit='ang',
        grad_unit=['hartree', 'bohr'], force_unit=['hartree', 'bohr'], align_ref='dipole',
        train_splits=[len(checks)], val_splits=0, test_splits=0))
    with np.load(arrays / 'full_tddft.npz') as data:
        for key, expected_unit in [('coordinate_unit', 'angstrom'), ('total_energy_unit', 'hartree'),
                                   ('excitation_energy_unit', 'eV'), ('gradient_unit', 'hartree/bohr')]:
            if str(data[key]) != expected_unit:
                raise ValueError(f'Unexpected array unit {key}: {data[key]}')
        states = data['state_indices'].tolist()
        expected_roots = int(cfg.get('rem', {}).get('cis_n_roots', len(states)))
        if states != list(range(1, expected_roots + 1)):
            raise ValueError(f'Calculated state indices differ from reference: {states}')
        state_column = data['state_indices'].tolist().index(cfg['state'])
        grad = np.load(arrays / 'full_grad.npy')
        np.testing.assert_allclose(grad, data['gradients'][:, state_column])
        np.testing.assert_allclose(np.load(arrays / 'full_force.npy'), -grad)
        np.testing.assert_allclose(np.load(arrays / 'full_ex_state_energy.npy'), data['total_energies'][:, state_column])
        np.testing.assert_allclose(np.load(arrays / 'full_ex_energy.npy'), data['excitation_energies'][:, state_column])
        np.testing.assert_allclose(data['total_energies'][:, state_column], [c['target_hartree'] for c in checks])
        assert data['coords'].shape[0] == len(checks)
        for name in ('gs_energies', 'total_energies', 'excitation_energies', 'gradients', 'trans_moms', 'esp_trans_density'):
            if data[name].shape[0] != len(checks):
                raise ValueError(f'Frame count mismatch: {name}')
        expected = [str((work / 'inputs' / r['input']).with_suffix('.out')) for r in cfg['frames']]
        assert data['source_files'].tolist() == expected
        for name in ('idx_train', 'idx_val', 'idx_test', 'idx_frame'):
            if name in data:
                assert np.all((data[name] >= 0) & (data[name] < len(checks)))
        # Fixed root: alignment may flip signs, never switch electronic states.
        raw_m = data['trans_moms'][:, state_column]
        raw_q = data['esp_trans_density'][:, state_column]
        aligned_m, aligned_q = ExportNumpy()._align_data(raw_m, raw_q, 'dipole')
        np.testing.assert_allclose(np.load(arrays / 'full_transmom.npy'), aligned_m, equal_nan=True)
        np.testing.assert_allclose(np.load(arrays / 'full_aligned_td.npy'), aligned_q, equal_nan=True)
        shapes = {k: list(data[k].shape) for k in ('coords', 'total_energies', 'excitation_energies', 'gradients')}
        gradient_available = np.isfinite(data['gradients']).all(axis=(2, 3))
        np.savez(arrays / 'availability.npz', state_indices=data['state_indices'],
                 gradient_available=gradient_available,
                 transition_dipole_available=np.isfinite(data['trans_moms']).all(axis=2),
                 transition_charge_available=np.isfinite(data['esp_trans_density']).all(axis=2))
    np.save(arrays / 'source_frame_indices.npy', [r['source_frame'] for r in cfg['frames']])
    result = {'signature': signature, 'state': cfg['state'], 'frames': checks, 'shapes': shapes,
              'state_indices': states, 'gradient_available': gradient_available.tolist(),
              'export_implementation_sha256': digest(__file__),
              'units': {'coordinates': 'angstrom', 'total_energy': 'hartree', 'excitation_energy': 'eV',
                        'gradient': 'hartree/bohr', 'force': 'hartree/bohr'},
              'array_sha256': {str(p.relative_to(work)): digest(p) for p in arrays.iterdir() if p.is_file()}}
    save(work / 'validation.json', result)
    print(json.dumps(result, indent=2))
    return result


def run_workflow(args, work):
    """Submit once if needed, then observe; never automatically retry a failed job."""
    deadline = time.monotonic() + args.wait_timeout
    first = True
    while True:
        if time.monotonic() >= deadline:
            raise TimeoutError('Observation timed out; submitted jobs were not cancelled. Re-run --action run to attach again.')
        # Release the lock between polls so status and another observer can run.
        # Submission decisions and exports stay protected by the same exclusive lock.
        with locked(work):
            cfg = prepare(args, work)
            report = refresh(cfg, work)
            if not report['active_or_unknown_job']:
                if all(r['status'] == 'complete' for r in report['frames']):
                    export(cfg, work, report)
                    return 0
                if not first:
                    raise RuntimeError('Job ended without all frames complete; inspect status.json. Retry requires explicit --retry.')
                plan_submit(cfg, work, report, submit=True, retry=args.retry)
            counts = {}
            for row in report['frames']:
                counts[row['status']] = counts.get(row['status'], 0) + 1
            jobs = [(a.get('job_id', 'uncertain'), a.get('scheduler_state', a.get('submission')))
                    for a in cfg['attempts']]
            print(json.dumps({'jobs': jobs, 'frames': counts}), flush=True)
        first = False
        remaining = deadline - time.monotonic()
        if remaining > 0:
            time.sleep(min(args.poll_seconds, remaining))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work', required=True)
    p.add_argument('--action', choices=['plan', 'submit', 'status', 'export', 'resume', 'repair-env', 'run'], default='plan')
    p.add_argument('--source')
    p.add_argument('--ref')
    p.add_argument('--env-setup')
    p.add_argument('--types', default='auto')
    p.add_argument('--input-distance-unit', default='ang')
    p.add_argument('--frames', type=int, default=1)
    p.add_argument('--start', type=int, default=0)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--charge', type=int, default=0)
    p.add_argument('--spin', type=int, default=1)
    p.add_argument('--state', type=int, default=1)
    p.add_argument('--cores', type=int, default=4)
    p.add_argument('--partition', default='batch')
    p.add_argument('--walltime', default='00:30:00')
    p.add_argument('--memory', default='64G')
    p.add_argument('--qchem', default='qchem')
    p.add_argument('--retry', action='store_true')
    p.add_argument('--poll-seconds', type=float, default=30,
                   help='run-mode polling interval, 1-60 seconds (default 30)')
    p.add_argument('--wait-timeout', type=float, default=3600,
                   help='run-mode observation timeout in seconds (default 3600); does not cancel jobs')
    args = p.parse_args(argv)
    if args.action == 'run' and (not 1 <= args.poll_seconds <= 60 or not 0 < args.wait_timeout < float('inf')):
        p.error('run requires --poll-seconds between 1 and 60 and a finite positive --wait-timeout')
    work = Path(args.work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    if args.action == 'run':
        try:
            return run_workflow(args, work)
        except KeyboardInterrupt:
            print('Observation interrupted; submitted jobs were not cancelled. Re-run --action run to attach again.', file=sys.stderr)
            return 130
        except TimeoutError as exc:
            print(str(exc), file=sys.stderr)
            return 124
    with locked(work):
        cfg = prepare(args, work)
        report = refresh(cfg, work)
        if args.action == 'repair-env':
            repair_env(cfg, work, report, args.env_setup)
        elif args.action == 'status':
            print(json.dumps(report, indent=2))
        elif args.action in ('plan', 'submit'):
            plan_submit(cfg, work, report, submit=args.action == 'submit', retry=args.retry)
        elif args.action == 'export':
            export(cfg, work, report)
        elif report['active_or_unknown_job']:
            print(json.dumps(report, indent=2))
        elif all(r['status'] == 'complete' for r in report['frames']):
            export(cfg, work, report)
        elif cfg['attempts'] and not args.retry:
            raise RuntimeError('Failed run; inspect status.json and use --action submit --retry explicitly')
        else:
            plan_submit(cfg, work, report, retry=args.retry)
    return 0


if __name__ == '__main__':
    sys.exit(main())
