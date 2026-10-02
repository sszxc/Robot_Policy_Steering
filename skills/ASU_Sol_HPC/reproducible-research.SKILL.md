---
name: reproducible-research
description: "Make a computational study reproducible end to end: pin environments and data versions, control randomness, structure experiments so results map to code, capture provenance for every figure and table, and assemble an artifact a reviewer can rerun."
compatibility: opencode
metadata:
  audience: "researchers"
  topic: "reproducibility"
---

## What I do

- Set up a project so results can be regenerated months later
- Add provenance capture to experiment code
- Assemble a reproducibility artifact for a paper submission

## When to use me

Use me at the start of a project, before a paper submission, or whenever
somebody cannot reproduce a previous result.

## Safety

Results are append-only: a new run gets a new run id, and I never overwrite or
delete an existing one — including the runs that failed, which are part of the
record. I do not rewrite git history, force-push, or discard uncommitted work to
"clean up" a tree.

## The rule

**Every figure, table, and number in a paper must be traceable to a commit, a
config, and the data version it was computed from.** If it is not, it is not a
result — it is an anecdote.

## Minimum viable setup

```
project/
  README.md              # one command to reproduce everything
  environment.yml
  environment.lock.txt
  configs/               # one file per experiment, versioned
  src/
  scripts/               # slurm submit scripts
  results/
    <run-id>/
      config.yaml        # the exact config used
      metrics.json
      provenance.json    # commit, date, host, job id, seeds, data hash
      figures/
```

Never edit results in place. A new run gets a new run id.

## Capture provenance automatically

```python
import hashlib, json, os, platform, subprocess, sys, time

def provenance(config: dict, data_paths: list[str]) -> dict:
    def sha256(path):
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    return {
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "python": sys.version,
        "config": config,
        "data": {p: sha256(p) for p in data_paths},
    }
```

Refuse to run a "final" experiment from a dirty working tree — record the flag
and warn loudly if it happens.

## Control randomness

```python
import os, random, numpy as np, torch

def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
```

Report results over **multiple seeds** with mean and spread. A single-seed
comparison between two methods is not evidence. Note where full determinism
was traded for speed (non-deterministic kernels, atomics, async reductions).

## Track experiments

Keep a plain `results/index.csv` (run id, commit, config hash, key metric) even
if you also use a tracking service. It survives tool changes and is greppable.

## Submission checklist

- [ ] Fresh clone + `environment.lock.txt` reproduces the main result
- [ ] `make all` or `bash scripts/reproduce.sh` regenerates every figure
- [ ] Data is deposited with a DOI, or the access procedure is documented
- [ ] Seeds, hardware, and runtime are stated in the paper
- [ ] Failed and abandoned variants are documented, not silently dropped
- [ ] The README states expected runtime and resources, so a reviewer knows what they are starting
