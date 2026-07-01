# CartPole Q-learning

Tabular Q-learning for a custom CartPole simulation, with two environments:

* `BaseCartPoleSim`: baseline deterministic CartPole dynamics.
* `DisturbedCartPoleSim`: disturbed dynamics with wind, gusts, impulses, observation noise, and parameter jitter.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[test]"
```

If PowerShell blocks activation, use this temporary session-only bypass:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## Common commands

Train, evaluate, and visualize the base environment:

```powershell
python -m scripts.train_base
python -m scripts.evaluate base
python -m scripts.visualize_base
```

Train, evaluate, and visualize the disturbed environment:

```powershell
python -m scripts.train_disturbed
python -m scripts.evaluate disturbed
python -m scripts.visualize_disturbed
```

Run grid searches:

```powershell
python -m scripts.grid_search_base
python -m scripts.grid_search_disturbed
```

Run tests:

```powershell
python -m pytest -q
```

## Configs and outputs

The main experiment configs are:

```text
configs/base.json
configs/disturbed.json
```

Grid-search configs are:

```text
configs/base_search_space.json
configs/disturbed_search_space.json
```

Running a grid search updates the corresponding main config with the best found parameters.

Generated Q-tables, archived runs, and grid-search result logs are written under:

```text
outputs/
```
