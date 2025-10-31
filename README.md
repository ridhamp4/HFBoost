# HFBoost Adaptive Spectral PINN for Burgers' Equation

This repository implements an adaptive spectral Physics-Informed Neural Network (PINN) for the 1D viscous Burgers' equation. The entry point is `main.py`, which trains an adaptive spectral PINN, evaluates it on a test set, and produces several diagnostic plots.

## Requirements

- Python 3.8+
- PyTorch (CPU or with CUDA support if you want GPU training)
- NumPy
- SciPy
- Matplotlib

A minimal `requirements.txt` is included in the repository. To install the dependencies into a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you want GPU training, install a PyTorch build with CUDA that matches your system (see https://pytorch.org for instructions) before running.

## How to run

The simplest way to run the model is:

```bash
python3 main.py
```

Behaviour when running `main.py`:

- The script looks for a dataset at `data/burgers_dataset.npz`.
- If the file exists, it will be loaded. If it does not exist, the script will generate training and test data using the exact Cole–Hopf solution and save it to `data/burgers_dataset.npz`.
- The script trains the adaptive spectral PINN (default 5000 epochs) and then evaluates and plots results.

Notes and tips:

- The repository tree in this workspace contains a `Data/` directory with `burgers_dataset.npz` (capital `D`). `main.py` expects the dataset under a lowercase `data/` directory. If you already have the dataset in `Data/`, either:

  - move it to `data/` (recommended):

    ```bash
    mkdir -p data
    mv Data/burgers_dataset.npz data/
    ```

  - or create a symlink:

    ```bash
    ln -s "$PWD/Data" "$PWD/data"
    ```

- Training for 5000 epochs can take a long time on CPU. To run a short debug session, open `main.py` and change the call:

```python
adaptive_model = train_adaptive_pinn(train_data, n_epochs=5000, lr=0.001)
```

to a smaller number of epochs, e.g. `n_epochs=200` for a quick test.

## Outputs

- During training you will see per-epoch loss/weight diagnostics printed to stdout.
- After training the script will print evaluation metrics (MSE and relative L2 error) and show plots:
  - Training history (losses, high-frequency ratios, and weight evolution)
  - Solution comparison (predicted vs. ground-truth)
  - A final plot of the adaptive loss weights over epochs

## Troubleshooting

- If you see import errors, ensure the repository root is on your PYTHONPATH or run `main.py` from the repository root directory. Example:

```bash
cd /path/to/HFBoost
python3 main.py
```

- If PyTorch cannot find a CUDA device but you expected one, verify that the PyTorch build you installed supports CUDA and that your drivers are installed.

## Quick summary

- Install dependencies: `pip install -r requirements.txt`
- Ensure dataset is at `data/burgers_dataset.npz` or let `main.py` generate it
- Run: `python3 main.py`
