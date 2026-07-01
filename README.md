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

```powershell

$env:PYTHONPATH="src"

python -m scripts.train_base
python -m scripts.evaluate base
python -m scripts.visualize_base

python -m scripts.train_disturbed
python -m scripts.evaluate disturbed
python -m scripts.visualize_disturbed


python -m pytest -q
```