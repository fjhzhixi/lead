# LEAD Project Guidelines

## Architecture

LEAD is an end-to-end autonomous driving stack for CARLA Leaderboard 2.0.

| Directory    | Purpose                                                    |
| :----------- | :--------------------------------------------------------- |
| `lead/`      | Main package — model, training, inference, expert, data    |
| `3rd_party/` | Vendored deps (CARLA 0.9.15, leaderboard, scenario runner) |
| `data/`      | Route XMLs and collected sensor data                       |
| `scripts/`   | Shell launchers for training, evaluation, data collection  |
| `slurm/`     | HPC job submission templates                               |
| `notebooks/` | Jupyter notebooks for data inspection                      |

**Key packages inside `lead/`:**

- `tfv6/` — TransFuser V6 neural network (backbone + multi-task decoders)
- `training/` — Native PyTorch DDP training loop (NOT PyTorch Lightning)
- `inference/` — Open-loop and closed-loop inference; `sensor_agent.py` is the CARLA agent entry point
- `data_loader/` — Dataset classes for CARLA, NAVSIM, Waymo E2E
- `data_buckets/` — Bucket-based route collection with LZ4 caching
- `expert/` — Privileged expert policy (kinematic bicycle model) for data collection
- `carl_agent/` — CaRL RL agent integration (excluded from linting)
- `common/` — Base agent, sensors, route planner, PID controllers, constants
- `visualization/` — Debug visualizations for training/inference
- `infraction_webapp/` — Flask app for infraction analysis

## Build and Test

```bash
# Environment setup
pip install conda-lock && conda-lock install -n lead conda-lock.yml
conda activate lead
pip install uv && uv pip install -r requirements.txt && uv pip install -e .

# CARLA setup
bash scripts/setup_carla.sh  # or symlink to 3rd_party/CARLA_0915

# Training (single GPU)
python lead/training/train.py logdir=outputs/local_training/pretrain

# Training (DDP)
bash scripts/pretrain_ddp.sh   # perception pretraining
bash scripts/posttrain_ddp.sh  # planning fine-tuning

# Evaluation (requires running CARLA server)
bash scripts/start_carla.sh
python lead/leaderboard_wrapper.py \
  --checkpoint outputs/checkpoints/tfv6_resnet34 \
  --routes data/benchmark_routes/bench2drive/23687.xml \
  --bench2drive

# Linting
ruff check lead/ && ruff format --check lead/

# Tests (aspirational — no test suite yet)
pytest tests/
```

## Conventions

- **Formatting:** Ruff with 88-char line length, double quotes. Rules: E, F, I, UP, B. Run `ruff check` and `ruff format`.
- **Lint scope:** Only `lead/**/*.py` and `tests/**/*.py` are checked. `3rd_party/`, `scripts/`, `lead/carl_agent/` are excluded.
- **Type hints:** Use `jaxtyping` for tensor shape annotations and `beartype` for runtime validation.
- **Type checking:** BasedPyright in `basic` mode. Custom stubs at `3rd_party/typings/`.
- **Docstrings:** Google style (`autoDocstring.docstringFormat`).
- **Config system:** Custom `BaseConfig` with env-var override via `LEAD_TRAINING_CONFIG` and `LEAD_CLOSED_LOOP_CONFIG` (Hydra-style `key=value` strings).
- **Two-stage training:** (1) pretrain with auxiliary perception tasks, (2) posttrain with planning decoder enabled and loaded pretrain checkpoint.
- **Pre-commit hooks:** `pre-commit install` activates Ruff, nbstripout, pydoclint, trailing whitespace checks.

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `LEAD_PROJECT_ROOT` | Project root (set in `~/.bashrc`) |
| `CARLA_ROOT` | CARLA installation (`3rd_party/CARLA_0915`) |
| `LEAD_TRAINING_CONFIG` | Training config overrides (Hydra-style) |
| `LEAD_CLOSED_LOOP_CONFIG` | Closed-loop inference config overrides |
| `PYTHONPATH` | Set by `scripts/main.sh` — includes CARLA, leaderboard, scenario runner |

Always `source scripts/main.sh` (or have it in `~/.bashrc`) before running any Python code.

## Pitfalls

- **Do not start CARLA servers, SLURM jobs, or large downloads** unless the user explicitly asks.
- **Do not upgrade PyTorch.** CARLA's PythonAPI and scenario_runner depend on the pinned version. Upgrading causes silent eval failures.
- **3rd_party/ is vendored, not submodules.** Edit locally but be aware changes aren't tracked upstream.
- **CARLA GPU selection** uses `-graphicsadapter`, not `CUDA_VISIBLE_DEVICES`.
- **Stale data errors** → delete and rebuild training cache / buckets.
- **Simulator hangs** → restart CARLA; use `scripts/reset_carla_world.py` for fast map resets.
- **PYTHONPATH sensitivity:** Import errors often mean `scripts/main.sh` was not sourced.
- **Training divergence** after env changes → run a 50k-100k step test before committing to a full run.

## Further Documentation

See the [full docs site](https://ln2697.github.io/lead/docs) for data collection, training, and evaluation guides.
