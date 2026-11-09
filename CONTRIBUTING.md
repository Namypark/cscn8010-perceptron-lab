# Team Workflow

## One-time setup

```bash
git clone <our-repo-url>
cd <repo>
python3.12 -m venv .venv          # TensorFlow does not support Python 3.14 yet; use 3.10–3.12
source .venv/bin/activate         # Windows: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

In VS Code, select the `.venv` interpreter as the notebook kernel.

## Branches

| Person | Branch | Notebook |
|---|---|---|
| Nnamdi | `ws1-gradient-descent` | `workstreams/ws1_gradient_descent.ipynb` |
| Carlos | `ws2-wonderland-prostate` | `workstreams/ws2_wonderland_prostate.ipynb` |
| Davisen | `ws3-decision-boundary` | `workstreams/ws3_decision_boundary_xor.ipynb` |
| Rangeetha | `ws4-activations-gates` | `workstreams/ws4_activations_gates.ipynb` |

```bash
git checkout -b ws3-decision-boundary
# ...work...
git add workstreams/ws3_decision_boundary_xor.ipynb
git commit -m "ws3: plot NAND decision boundary and bias shift"
git push -u origin ws3-decision-boundary
```

Then open a Pull Request into `main` and ask your reviewer (see the work plan).

## Golden rule

**Only edit your own workstream notebook.** Notebooks are JSON, so two people editing the same `.ipynb` file produce merge conflicts that are painful to fix. The main notebook in `notebooks/` is only changed by `scripts/build_notebook.py` during integration.

## Before every push

1. Kernel → **Restart** → **Run All**. Every cell must succeed.
2. Save the notebook.
3. Don't commit `.venv/`, checkpoints or credentials. `.gitignore` already covers these.

## Commit messages

Start with your workstream, then say what changed:
`ws1: compare learning rates 0.01–10 with loss curves`
