import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from chemlab.util import tddft_pipeline as pipeline


def arguments(tmp_path, trajectories, reference):
    reference.write_text(reference.read_text().replace('jobtype sp', 'jobtype force\ncis_state_deriv 1'))
    env = tmp_path / 'env.sh'
    env.write_text('#!/bin/bash\n')
    return SimpleNamespace(source=str(trajectories['xyz']), ref=str(reference), env_setup=str(env),
        frames=2, state=1, memory='1G', types='auto', input_distance_unit='ang', start=1, seed=42,
        charge=0, spin=1, cores=1, partition='batch', walltime='00:05:00', qchem='qchem')


def prepared(tmp_path, trajectories, reference):
    work = tmp_path / 'work'
    work.mkdir()
    cfg = pipeline.prepare(arguments(tmp_path, trajectories, reference), work)
    return work, cfg


def test_prepare_resume_and_tamper(tmp_path, trajectories, reference):
    args = arguments(tmp_path, trajectories, reference)
    work = tmp_path / 'work'
    work.mkdir()
    cfg = pipeline.prepare(args, work)
    assert len(cfg['frames']) == 2
    assert all(r['source_frame'] >= 1 for r in cfg['frames'])
    assert pipeline.prepare(args, work) == cfg
    inp = work / 'inputs/train_0000.inp'
    inp.write_text(inp.read_text() + '\nchanged')
    with pytest.raises(ValueError, match='changed'):
        pipeline.prepare(args, work)


def test_wrong_gradient_root_rejected(tmp_path, trajectories, reference):
    args = arguments(tmp_path, trajectories, reference)
    args.state = 2
    work = tmp_path / 'work'
    work.mkdir()
    with pytest.raises(ValueError, match='CIS_STATE_DERIV'):
        pipeline.prepare(args, work)


def test_active_and_unknown_prevent_resubmission(tmp_path, trajectories, reference, monkeypatch):
    work, cfg = prepared(tmp_path, trajectories, reference)
    cfg['attempts'] = [{'job_id': '123', 'submission': 'accepted'}]
    monkeypatch.setattr(pipeline, 'scheduler_state', lambda _: 'UNKNOWN')
    report = pipeline.refresh(cfg, work)
    with pytest.raises(RuntimeError, match='duplicate'):
        pipeline.plan_submit(cfg, work, report, submit=True)


def test_squeue_expired_id_falls_back_to_accounting(monkeypatch):
    replies = iter([SimpleNamespace(returncode=1, stdout='', stderr='slurm_load_jobs error: Invalid job id specified'),
                    SimpleNamespace(returncode=0, stdout='COMPLETED|\n', stderr='')])
    monkeypatch.setattr(pipeline.subprocess, 'run', lambda *a, **k: next(replies))
    assert pipeline.scheduler_state('123') == 'COMPLETED'


def test_scheduler_connection_error_is_not_completion(monkeypatch):
    monkeypatch.setattr(pipeline.subprocess, 'run', lambda *a, **k:
        SimpleNamespace(returncode=1, stdout='', stderr='Unable to contact slurm controller'))
    with pytest.raises(RuntimeError, match='Cannot query'):
        pipeline.scheduler_state('123')


def test_uncertain_submission_blocks(tmp_path, trajectories, reference):
    work, cfg = prepared(tmp_path, trajectories, reference)
    cfg['attempts'] = [{'submission': 'uncertain'}]
    assert pipeline.refresh(cfg, work)['active_or_unknown_job']


def test_failed_job_requires_explicit_retry(tmp_path, trajectories, reference, monkeypatch):
    work, cfg = prepared(tmp_path, trajectories, reference)
    cfg['attempts'] = [{'job_id': '123', 'submission': 'accepted'}]
    monkeypatch.setattr(pipeline, 'scheduler_state', lambda _: 'TIMEOUT')
    report = pipeline.refresh(cfg, work)
    assert all(r['status'] == 'failed' for r in report['frames'])
    with pytest.raises(RuntimeError, match='--retry'):
        pipeline.plan_submit(cfg, work, report)
    with pytest.raises(ValueError, match='every selected frame'):
        pipeline.export(cfg, work, report)


def test_lock_blocks_concurrent_run(tmp_path):
    with pipeline.locked(tmp_path):
        with pytest.raises(RuntimeError, match='already active'):
            with pipeline.locked(tmp_path):
                pass
    assert not (tmp_path / '.pipeline.lock').exists()


def test_strict_gradient_missing_rejected(tmp_path, tddft_text):
    inp = tmp_path / 'train_0000.inp'
    inp.write_text('placeholder')
    inp.with_suffix('.xyz').write_text('2\nframe\nH 0 0 0\nH 0 0 .7\n')
    inp.with_suffix('.out').write_text(tddft_text)
    assert pipeline.validate_output(inp, 1)['gradient_shape'] == [2, 3]
    with pytest.raises(ValueError, match='gradient'):
        pipeline.validate_output(inp, 2)


def test_wrong_energy_or_geometry_rejected(tmp_path, tddft_text):
    inp = tmp_path / 'train_0000.inp'
    inp.write_text('placeholder')
    xyz = inp.with_suffix('.xyz')
    xyz.write_text('2\nframe\nH 0 0 0\nH 0 0 .7\n')
    out = inp.with_suffix('.out')
    out.write_text(tddft_text.replace('-0.875260', '-0.7'))
    with pytest.raises(AssertionError):
        pipeline.validate_output(inp, 1)
    out.write_text(tddft_text)
    xyz.write_text('2\nframe\nH 0 0 0\nH 0 0 1.0\n')
    with pytest.raises(AssertionError, match='geometry'):
        pipeline.validate_output(inp, 1)


def test_export_validation_and_idempotency(tmp_path, tddft_text, monkeypatch):
    work = tmp_path / 'work'
    inputs = work / 'inputs'
    inputs.mkdir(parents=True)
    inp = inputs / 'train_0000.inp'
    inp.write_text('placeholder')
    inp.with_suffix('.xyz').write_text('2\nsource_frame=7\nH 0 0 0\nH 0 0 .7\n')
    out = inp.with_suffix('.out')
    out.write_text(tddft_text)
    pipeline.save(inp.with_suffix('.runner.json'), dict(success=True, returncode=0,
                  input_sha256=pipeline.digest(inp), output_mtime_ns=out.stat().st_mtime_ns))
    cfg = dict(state=1, frames=[dict(input=inp.name, source_frame=7)], attempts=[])
    report = pipeline.refresh(cfg, work)
    monkeypatch.chdir(tmp_path)
    result = pipeline.export(cfg, work, report)
    assert result['shapes']['gradients'] == [1, 2, 2, 3]
    assert np.load(work / 'arrays/source_frame_indices.npy').tolist() == [7]
    assert pipeline.export(cfg, work, report) == result
    with pytest.raises(ValueError, match='different results'):
        (work / 'arrays/full_grad.npy').write_bytes(b'corrupt')
        pipeline.export(cfg, work, report)


def test_plan_and_submit_record_job_id(tmp_path, trajectories, reference, monkeypatch):
    work, cfg = prepared(tmp_path, trajectories, reference)
    report = pipeline.refresh(cfg, work)
    monkeypatch.setattr(pipeline.subprocess, 'run', lambda *a, **k: SimpleNamespace(returncode=0, stdout='123;cluster\n', stderr=''))
    pipeline.plan_submit(cfg, work, report, submit=True)
    assert cfg['attempts'][0]['job_id'] == '123'
    assert '#SBATCH --mem=1G' in Path(cfg['attempts'][0]['script']).read_text()


def test_env_repair_is_recorded_and_cannot_change_active_run(tmp_path, trajectories, reference):
    work, cfg = prepared(tmp_path, trajectories, reference)
    replacement = tmp_path / 'replacement.sh'
    replacement.write_text('# replacement\n')
    report = dict(active_or_unknown_job=True, frames=[dict(status='failed')])
    with pytest.raises(ValueError, match='terminated'):
        pipeline.repair_env(cfg, work, report, replacement)
    report['active_or_unknown_job'] = False
    previous = cfg['env_sha256']
    pipeline.repair_env(cfg, work, report, replacement)
    assert (work / ('qchem_env.previous-' + previous + '.sh')).exists()
    assert cfg['environment_repairs'][0]['before_sha256'] == previous
    assert cfg['env_sha256'] == pipeline.digest(replacement)
    assert cfg['attempts'] == []


def test_successful_resume_never_submits(tmp_path, tddft_text, monkeypatch):
    inputs = tmp_path / 'inputs'
    inputs.mkdir()
    inp = inputs / 'train_0000.inp'
    inp.write_text('placeholder')
    inp.with_suffix('.out').write_text(tddft_text)
    out = inp.with_suffix('.out')
    pipeline.save(inp.with_suffix('.runner.json'), dict(success=True, returncode=0,
                  input_sha256=pipeline.digest(inp), output_mtime_ns=out.stat().st_mtime_ns))
    cfg = dict(frames=[dict(input=inp.name, source_frame=0)], attempts=[])
    report = pipeline.refresh(cfg, tmp_path)
    monkeypatch.setattr(pipeline, 'make_plan', lambda *a, **k: pytest.fail('duplicate plan'))
    pipeline.plan_submit(cfg, tmp_path, report, submit=True)


def mock_wait(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(pipeline.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(pipeline.time, 'sleep', lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    return clock


def complete_frames(work, cfg, text):
    for row in cfg['frames']:
        inp = work / 'inputs' / row['input']
        out = inp.with_suffix('.out')
        out.write_text(text)
        pipeline.save(inp.with_suffix('.runner.json'), dict(success=True, returncode=0,
                      input_sha256=pipeline.digest(inp), output_mtime_ns=out.stat().st_mtime_ns))


def test_run_queue_running_complete_export_and_reentry(tmp_path, trajectories, reference, tddft_text, monkeypatch):
    work, cfg = prepared(tmp_path, trajectories, reference)
    mock_wait(monkeypatch)
    states = iter(['PENDING', 'RUNNING', 'COMPLETED'])
    submissions, exports = [], []
    def scheduler(job):
        state = next(states, 'COMPLETED')
        if state == 'COMPLETED':
            complete_frames(work, cfg, tddft_text)
        return state
    def sbatch(command, **kwargs):
        assert command[0] == 'sbatch'
        submissions.append(command)
        return SimpleNamespace(returncode=0, stdout='123\n', stderr='')
    def export(config, path, report):
        assert not report['active_or_unknown_job']
        assert all(r['status'] == 'complete' for r in report['frames'])
        exports.append(report)
    monkeypatch.setattr(pipeline, 'scheduler_state', scheduler)
    monkeypatch.setattr(pipeline.subprocess, 'run', sbatch)
    monkeypatch.setattr(pipeline, 'export', export)
    command = ['--work', str(work), '--action', 'run', '--poll-seconds', '1', '--wait-timeout', '10']
    assert pipeline.main(command) == 0
    assert len(submissions) == 1 and len(exports) == 1
    assert pipeline.main(command) == 0
    assert len(submissions) == 1 and len(exports) == 2
    assert not (work / '.pipeline.lock').exists()


def test_run_attaches_to_existing_job_without_submission(tmp_path, trajectories, reference, tddft_text, monkeypatch):
    work, cfg = prepared(tmp_path, trajectories, reference)
    cfg['attempts'] = [dict(job_id='123', submission='accepted')]
    pipeline.save(work / 'pipeline.json', cfg)
    mock_wait(monkeypatch)
    states = iter(['PENDING', 'RUNNING', 'COMPLETED'])
    def scheduler(job):
        state = next(states)
        if state == 'COMPLETED':
            complete_frames(work, cfg, tddft_text)
        return state
    monkeypatch.setattr(pipeline, 'scheduler_state', scheduler)
    monkeypatch.setattr(pipeline.subprocess, 'run', lambda *a, **k: pytest.fail('duplicate submission'))
    monkeypatch.setattr(pipeline, 'export', lambda c, w, r: None)
    assert pipeline.main(['--work', str(work), '--action', 'run', '--poll-seconds', '1']) == 0


def test_run_failed_requires_retry_and_never_retries_automatically(tmp_path, trajectories, reference, monkeypatch):
    work, cfg = prepared(tmp_path, trajectories, reference)
    cfg['attempts'] = [dict(job_id='122', submission='accepted')]
    pipeline.save(work / 'pipeline.json', cfg)
    mock_wait(monkeypatch)
    submissions = []
    def sbatch(command, **kwargs):
        submissions.append(command)
        return SimpleNamespace(returncode=0, stdout='123\n', stderr='')
    monkeypatch.setattr(pipeline, 'scheduler_state', lambda job: 'FAILED')
    monkeypatch.setattr(pipeline.subprocess, 'run', sbatch)
    monkeypatch.setattr(pipeline, 'export', lambda *a: pytest.fail('export failed frames'))
    command = ['--work', str(work), '--action', 'run', '--poll-seconds', '1']
    with pytest.raises(RuntimeError, match='--retry'):
        pipeline.main(command)
    assert not submissions
    with pytest.raises(RuntimeError, match='Job ended'):
        pipeline.main(command + ['--retry'])
    assert len(submissions) == 1


def test_run_timeout_does_not_cancel_unknown_job(tmp_path, trajectories, reference, monkeypatch):
    work, cfg = prepared(tmp_path, trajectories, reference)
    cfg['attempts'] = [dict(job_id='123', submission='accepted')]
    pipeline.save(work / 'pipeline.json', cfg)
    mock_wait(monkeypatch)
    monkeypatch.setattr(pipeline, 'scheduler_state', lambda job: 'UNKNOWN')
    monkeypatch.setattr(pipeline.subprocess, 'run', lambda *a, **k: pytest.fail('submit or cancel'))
    assert pipeline.main(['--work', str(work), '--action', 'run', '--poll-seconds', '1', '--wait-timeout', '2']) == 124
    assert not (work / '.pipeline.lock').exists()
    assert json.loads((work / 'pipeline.json').read_text())['attempts'][0]['job_id'] == '123'


def test_run_ctrl_c_stops_observer_only(tmp_path, trajectories, reference, monkeypatch):
    work, cfg = prepared(tmp_path, trajectories, reference)
    cfg['attempts'] = [dict(job_id='123', submission='accepted')]
    pipeline.save(work / 'pipeline.json', cfg)
    monkeypatch.setattr(pipeline, 'scheduler_state', lambda job: 'RUNNING')
    monkeypatch.setattr(pipeline.subprocess, 'run', lambda *a, **k: pytest.fail('submit or cancel'))
    def interrupt(seconds):
        assert not (work / '.pipeline.lock').exists()
        raise KeyboardInterrupt
    monkeypatch.setattr(pipeline.time, 'sleep', interrupt)
    assert pipeline.main(['--work', str(work), '--action', 'run']) == 130
    assert not (work / '.pipeline.lock').exists()


@pytest.mark.parametrize('options', [['--poll-seconds', '0'], ['--poll-seconds', '61'],
                                     ['--wait-timeout', '0'], ['--wait-timeout', 'nan']])
def test_run_invalid_wait_options_rejected(tmp_path, options):
    with pytest.raises(SystemExit) as exc:
        pipeline.main(['--work', str(tmp_path), '--action', 'run'] + options)
    assert exc.value.code == 2
