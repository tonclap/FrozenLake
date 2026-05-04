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

Example:

```bash
pip install gym numpy torch tensorboard
```

There is no `requirements.txt` in this repo; pin versions locally if you need reproducibility.

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

## License / attribution

Add a license file if you redistribute; this README only describes the existing code behavior.
