# Agent Instructions For This Repository

## Python Environment

Use the `work` conda environment for all Python commands (tests, scripts, one-off checks).

PowerShell command pattern:

```powershell
$conda = "$env:USERPROFILE\anaconda3\Scripts\conda.exe"
& $conda run -n work <command>
```

Examples:

```powershell
& "$env:USERPROFILE\anaconda3\Scripts\conda.exe" run -n work python -m pytest tests/test_symbolic.py
& "$env:USERPROFILE\anaconda3\Scripts\conda.exe" run -n work python -m pytest
```

## Notes

- Do not assume `python`, `py`, or `pytest` are available on PATH outside conda.
- Prefer `conda run -n work ...` over interactive activation for deterministic agent runs.
