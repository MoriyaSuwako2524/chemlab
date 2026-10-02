"""Discover real input files and prepare SLURM batches without numeric ranges.

python -m chemlab.util.tddft_batch --data raw_data --env-setup qchem_env.sh
Add --submit to submit; default only writes the plan. Repeat to resume failures.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import hashlib
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile


def output_complete(path):
    if not path.is_file():
        return False
    text = path.read_text(errors="replace")
    return "Thank you very much for using Q-Chem" in text and "Q-Chem fatal error" not in text


def pending_inputs(data, pattern="*.inp", failed_only=False, force=False):
    files = sorted(Path(data).resolve().glob(pattern))
    result = []
    for inp in files:
        if not inp.is_file() or inp.suffix != ".inp":
            continue
        out = inp.with_suffix(".out")
        complete = output_complete(out)
        status_path = inp.with_suffix(".runner.json")
        if status_path.exists() and out.exists():
            status = json.loads(status_path.read_text())
            if status.get("output_mtime_ns") == out.stat().st_mtime_ns:
                complete = complete and status["success"] and status["input_sha256"] == hashlib.sha256(inp.read_bytes()).hexdigest()
        if failed_only and (not out.exists() or complete):
            continue
        if not force and complete and out.stat().st_mtime_ns >= inp.stat().st_mtime_ns:
            continue
        result.append(inp)
    return result


def run_manifest(path):
    cfg = json.loads(Path(path).read_text())
    for filename in cfg["inputs"]:
        if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != cfg["input_sha256"][filename]:
            raise ValueError(f"Input changed after planning: {filename}; prepare a new plan")

    def run_one(filename):
        inp = Path(filename)
        out = inp.with_suffix(".out")
        if out.exists():
            # Retain the previous attempt and prevent a stale success footer being reused.
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
            out.rename(out.with_name(out.name + ".previous-" + stamp))
        env = os.environ.copy()
        env.update(OMP_NUM_THREADS=str(cfg["cores"]), MKL_NUM_THREADS=str(cfg["cores"]))
        try:
            with inp.with_suffix(".runner.log").open("w") as log:
                proc = subprocess.run([cfg["qchem"], "-nt", str(cfg["cores"]), str(inp), str(out)],
                                      cwd=inp.parent, env=env, stdout=log, stderr=subprocess.STDOUT)
            success = proc.returncode == 0 and output_complete(out)
            returncode = proc.returncode
        except OSError as exc:
            print(f"Failed to launch {inp}: {exc}", file=sys.stderr)
            success, returncode = False, None
        inp.with_suffix(".runner.json").write_text(json.dumps({
            "success": success, "returncode": returncode,
            "input_sha256": cfg["input_sha256"][filename],
            "output_mtime_ns": out.stat().st_mtime_ns if out.exists() else None,
        }, indent=2) + "\n")
        return success

    with ThreadPoolExecutor(max_workers=cfg["jobs"]) as pool:
        results = list(pool.map(run_one, cfg["inputs"]))
    print(f"Completed {sum(results)}/{len(results)} inputs")
    return 0 if all(results) else 1


def make_plan(files, directory, *, batch_size=200, cores=4, jobs=8,
              partition="batch", walltime="5-00:00:00", env_setup=None, qchem="qchem"):
    if min(batch_size, cores, jobs) < 1:
        raise ValueError("batch_size, cores and jobs must be positive")
    if not re.fullmatch(r"[A-Za-z0-9_,.-]+", partition) or not re.fullmatch(r"[0-9:-]+", walltime):
        raise ValueError("Invalid SLURM partition or walltime")
    if env_setup and not Path(env_setup).is_file():
        raise FileNotFoundError(env_setup)
    if not files:
        return []
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    plan = Path(tempfile.mkdtemp(prefix="tddft-", dir=directory))
    scripts = []
    for offset in range(0, len(files), batch_size):
        number = offset // batch_size
        manifest = plan / f"batch_{number:04d}.json"
        manifest.write_text(json.dumps({"inputs": [str(Path(f).resolve()) for f in files[offset:offset + batch_size]],
                                        "input_sha256": {str(Path(f).resolve()): hashlib.sha256(Path(f).read_bytes()).hexdigest()
                                                         for f in files[offset:offset + batch_size]},
                                        "cores": cores, "jobs": jobs, "qchem": qchem}, indent=2) + "\n")
        script = plan / f"batch_{number:04d}.sh"
        setup = f"source {shlex.quote(str(Path(env_setup).resolve()))}\n" if env_setup else ""
        root = str(Path(__file__).resolve().parents[2])
        script.write_text(
            "#!/bin/bash\n"
            f"#SBATCH --partition={partition}\n#SBATCH --time={walltime}\n"
            f"#SBATCH --nodes=1\n#SBATCH --ntasks=1\n#SBATCH --cpus-per-task={cores * jobs}\n"
            f"#SBATCH --job-name=tddft_{number}\n"
            f"#SBATCH --output={shlex.quote(str(plan / (str(number) + '_%j.log')))}\n"
            "set -e\n" + setup +
            f"export PYTHONPATH={shlex.quote(root)}${{PYTHONPATH:+:$PYTHONPATH}}\n"
            f"exec {shlex.quote(sys.executable)} -m chemlab.util.tddft_batch --run-manifest {shlex.quote(str(manifest))}\n"
        )
        scripts.append(script)
    return scripts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="raw_data")
    parser.add_argument("--pattern", default="*.inp")
    parser.add_argument("--plan-dir")
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--cores", type=int, default=4)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--partition", default="batch")
    parser.add_argument("--walltime", default="5-00:00:00")
    parser.add_argument("--env-setup")
    parser.add_argument("--qchem", default="qchem")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--failed-only", action="store_true")
    mode.add_argument("--force", action="store_true")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--run-manifest", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.run_manifest:
        return run_manifest(args.run_manifest)
    if not Path(args.data).is_dir():
        parser.error(f"Input directory does not exist: {args.data}")
    files = pending_inputs(args.data, args.pattern, args.failed_only, args.force)
    scripts = make_plan(files, args.plan_dir or Path(args.data) / "slurm_plans",
                        batch_size=args.batch_size, cores=args.cores, jobs=args.jobs,
                        partition=args.partition, walltime=args.walltime,
                        env_setup=args.env_setup, qchem=args.qchem)
    print(f"Selected {len(files)} input files; prepared {len(scripts)} SLURM batches")
    for script in scripts:
        print(script)
        if args.submit:
            subprocess.run(["sbatch", str(script)], check=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
