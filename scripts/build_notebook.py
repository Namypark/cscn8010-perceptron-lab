"""Combine the workstream notebooks into the main workshop notebook.

python scripts/build_notebook.py            merge only
python scripts/build_notebook.py --execute  merge and run the notebook
"""

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MAIN = ROOT / "notebooks" / "SingleLayerPerceptrons_Workshop.ipynb"
WORKSTREAMS = [
    ("ws4", "Activation Functions, Logic Gates & the Movie Challenge", "Rangeetha", "ws4_activations_gates.ipynb"),
    ("ws3", "Decision Boundaries, XOR & TensorFlow Playground", "Davisen", "ws3_decision_boundary_xor.ipynb"),
    ("ws2", "Wonderland Sensitivity & Prostate Cancer ANN", "Carlos", "ws2_wonderland_prostate.ipynb"),
    ("ws1", "Gradient Descent, Backpropagation & Training", "Nnamdi", "ws1_gradient_descent.ipynb"),
]

# cells we add get this metadata key so we can strip them on the next run
MARK = "team_extensions"


def md_cell(text, tag):
    return {"cell_type": "markdown", "metadata": {MARK: tag}, "source": text}


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    nb = load(MAIN)
    original = [c for c in nb["cells"] if MARK not in c.get("metadata", {})]
    added = [md_cell("---\n# Team Extensions\n\nEverything above this point is the original workshop.", "header")]

    for key, title, owner, filename in WORKSTREAMS:
        ws = load(ROOT / "workstreams" / filename)
        # skip the task brief at the top of each workstream notebook
        body = [c for c in ws["cells"] if "brief" not in c.get("metadata", {}).get("tags", [])]
        added.append(md_cell(f"## {title}\n*Author: {owner}*", key))
        for cell in body:
            cell = copy.deepcopy(cell)
            cell.setdefault("metadata", {})[MARK] = key
            added.append(cell)
        print(f"{key}: {len(body)} cells")

    nb["cells"] = original + added
    with open(MAIN, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print(f"{len(original)} original cells + {len(added)} added")

    if args.execute:
        cmd = [sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute",
               "--inplace", "--ExecutePreprocessor.timeout=600", str(MAIN)]
        sys.exit(subprocess.call(cmd, cwd=MAIN.parent))


if __name__ == "__main__":
    main()
