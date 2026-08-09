# FrozenLake DQN experiments

Training and evaluation tooling for **[Gym `FrozenLake-v1`](https://www.gymlibrary.dev/environments/toy_text/frozen_lake/)** (deterministic maps, `is_slippery=False`) on **4×4** grids.

The agent is a **Double DQN-style** learner (current policy selects the bootstrap action on the next state; target net evaluates it) implemented as a **small CNN with a Squeeze-and-Excitation block** (`dqn_model.py`). Observations are **not raw discrete states**: a custom wrapper builds a **3×H×W** tensor (agent one-hot, encoded map tiles, normalized Manhattan distance to the goal).

## Features

- **Curriculum sampling**: During the first `CURRICULUM_EPISODES` episodes, random maps use ice probability `p` interpolated from `START_P` toward `END_P`; afterward `p` is sampled from a narrower band (~0.78–0.82).
- **Reward shaping**: Small bonuses for Manhattan improvement to the goal, penalties for revisits and per-step cost (`STEP_PENALTY`, etc.), on top of the sparse goal reward.
- **n-step returns** pushed into replay, plus **imagined transitions**: after each episode, several “planning” updates sample past states and push transitions from a deterministic internal simulator aligned with the wrapper’s observation format.
- **Periodic evaluation**: Every 3000 episodes the policy is evaluated on **all reachable** 4×4 layouts where non-terminal cells are frozen or holes and a path exists from start to goal (`utils.generate_all_valid_maps`).
- **Hyperparameter search**: `experiment.py` runs many trials with randomized hyperparameters drawn in `config.generate_experiment_config()`.
- **Fine-tuning**: `finetune.py` can resume from a checkpoint and train on maps listed in a JSON file (typically from evaluation failures).

## Requirements

Install a recent **Python** (3.9+ recommended) with:

| Package | Used for |
|--------|-----------|
| `gym` | `FrozenLake-v1`, wrappers |
| `numpy` | Arrays / map logic |
| `torch` | DQN networks and training |
| `tensorboard` | `SummaryWriter` under `runs/` |

Install from the pinned set:

```bash
pip install -r requirements.txt
```

`numpy` is pinned below 2.0 on purpose: `gym` 0.26.2 still uses the removed `np.bool8` alias
and fails on NumPy 2.x.

## Project layout

| File | Role |
|------|------|
| `config.py` | Default hyperparameters plus `generate_experiment_config()` for random search grids. |
| `dqn_model.py` | `AugmentedObservationWrapperCNN`, `SELayer`, `DQNCNN`. |
| `utils.py` | Valid-map generation, BFS connectivity check, deterministic rollout `test_on_map`. |
| `trainer.py` | Main training loop, replay, checkpoints, TensorBoard scalars, evaluation hooks. |
| `experiment.py` | Batch runner: 30 experiments → `results/` CSV + log. |
| `test.py` | Load a `.pth` checkpoint and report success rate on all valid maps; writes `failed_maps.json`. |
| `finetune.py` | Optional second-stage training on failed maps from JSON; saves `*_finetuned.pth`. |
| `fast_eval.py` | Vectorised evaluation of every valid map in one batched rollout (~0.7 s vs ~27 s for the gym path). |
| `curriculum_trainer.py` | Fine-tuning with a failure-prioritised curriculum over the enumerated map set. |

## Running training (single experiment)

The batch entry point calls `trainer.train_model` internally:

Create a **`results`** directory first (otherwise CSV logging fails), then:

```bash
python experiment.py
```

On Unix: `mkdir -p results`. In PowerShell: `New-Item -ItemType Directory -Force results`.

Creates (when the `results` directory exists):

- `results/experiment.csv` — one row per experiment with sampled hyperparameters and best rolling success metric.
- `results/test.csv` — evaluation snapshots (episode index, successes, totals, rate).
- `results/experiment.log` — text log.

Also created under the repo root:

- `runs/exp_<id>_<timestamp>/` — TensorBoard event files.
- `checkpoints/exp_<experiment_number>_checkpoint_ep<episode>.pth` — periodic full checkpoints (policy, target, optimizer, scheduler, `config`).
- `models/` — directory ensured for any model artifacts you save manually.

CUDA is used automatically when available.

## Evaluating a checkpoint

Script name is `test.py` (CLI message may still mention an older filename):

```bash
python test.py checkpoints/exp_1_checkpoint_ep3000.pth
```

Prints aggregate success rate and writes **`failed_maps.json`** with maps where cumulative reward stays below goal success.

## Fine-tuning on failed maps

```bash
python finetune.py checkpoints/exp_1_checkpoint_ep3000.pth failed_maps.json
```

Loads weights and optional embedded `config` from the checkpoint, runs additional episodes sampling from `failed_maps`, and saves **`..._finetuned.pth`** next to the input path. Logs to `finetune.log`.

## Configuration notes

- **`MAP_SIZE`** is fixed to **4** in several places (`DQNCNN` conv geometry assumes 4×4 after convs).
- After `CURRICULUM_EPISODES`, `trainer.py` applies a **“special phase”** that swaps in `NEW_LR`, exploration decay, discount, and planning-step count unless you edit the defaults.

For one-off manual runs without the experiment harness, you can import `default_config` or a dict from `generate_experiment_config()` and invoke `train_model(config, experiment_number, test_writer)` from `trainer.py` (supply a CSV writer compatible with how `trainer` logs tests, or refactor logging for interactive use).

## Solving all maps (100%)

`models/10000_curriculum_ep5200.pth` solves **3828/3828** valid 4×4 maps. Because both the
policy (greedy argmax) and the environment are deterministic, and the 3828 maps are the
complete set of valid layouts, this is an exact result rather than a sampled estimate.

Verify it either way:

```bash
python test.py models/10000_curriculum_ep5200.pth
```

```bash
python fast_eval.py models/10000_curriculum_ep5200.pth --verify 500
```

### Why plain training plateaued

`trainer.py` draws training maps from `generate_random_map(size, p)`, where `p` is the
probability that a cell is frozen. Evaluation instead sweeps all valid maps uniformly. Maps
with a high hole density are effectively absent from training at `p ∈ [0.78, 0.82]` while
carrying almost all of the residual failures — with 9 holes, 10% of maps failed, against 0%
at 3 holes or fewer. The ceiling came from that train/eval distribution mismatch, not from
model capacity.

`curriculum_trainer.py` samples from the same enumerated map set the evaluation uses and
prioritises maps that currently fail:

```bash
python curriculum_trainer.py --episodes 15000 --eval-every 200 \
    --init-from checkpoints/exp_2_checkpoint_ep30000.pth \
    --use-ckpt-config --inherit-optimizer --warmup-episodes 400 \
    --eps-start 0.05 --lr 5e-5
```

Two flags matter more than they look. `--use-ckpt-config` inherits `GAMMA`, `N_STEP` and the
shaping coefficients from the checkpoint: Q-values encode returns under one specific reward
function, so fine-tuning under a different one discards what was learned — doing that dropped
99.53% to 78% within 300 episodes. And hard maps are always mixed with uniform samples
(`HARD_FRACTION`); training on failures alone is what collapsed the earlier `finetune.py` run
to 73%.

## License / attribution

Add a license file if you redistribute; this README only describes the existing code behavior.
