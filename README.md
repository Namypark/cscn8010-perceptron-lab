# Perceptron Lab — From a Single Neuron to Its Limits

**CSCN8010 · Foundations of Machine Learning**
**Team:** Nnamdi · Carlos · Davisen · Rangeetha

In this project we take the Single-Layer Perceptrons workshop and push each of its ideas further. We test the activation functions at their extremes, check every logic gate in two frameworks, draw and move decision boundaries until they break on XOR, and check gradient descent by hand against PyTorch's autograd. Along the way we found and corrected some errors in the original material (see [Corrections](#corrections-to-the-original-material)).

> 📓 **Submission notebook:** [`notebooks/SingleLayerPerceptrons_Workshop.ipynb`](notebooks/SingleLayerPerceptrons_Workshop.ipynb)
> The original workshop content is kept unchanged. Our work is in the **Team Extensions** section at the end.

---

## What we did

| Workstream | Author | Focus |
|---|---|---|
| Activations, Logic Gates and the Movie Challenge | Rangeetha | _To be completed_ |
| Decision Boundaries, XOR and the Playground | Davisen | _To be completed_ |
| Wonderland Sensitivity and the Prostate Cancer ANN | Carlos | Flipping one Wonderland input at a time, raw score z vs prediction σ(z), and a hand-checked gradient step on the 1-2-2 Prostate Cancer network |
| Gradient Descent, Backpropagation and Training | Nnamdi | _To be completed_ |

### Key results

_To be completed. Each author adds 2–3 bullet points with the most interesting numbers or plots from their section._

**Carlos: Wonderland Sensitivity and the Prostate Cancer ANN**

- Starting from x = [1, 0, 1] (z = 6), flipping any single input never changes the decision: z only moves by that input's weight (−5, +2, −4). Across all 8 inputs the neuron says "no" only for [0,0,0] and [0,1,0].
- Raw score vs prediction: for target 1, the error on z = 6 is −5 (loss 12.5), but the error on σ(6) ≈ 0.9975 is 0.0025 (loss ≈ 3×10⁻⁶). The error must be computed on σ(z).
- The Prostate Cancer network (1 PSA input, 2 hidden neurons, 2 outputs) classifies all 15 patients correctly. Its hidden neurons cut the PSA axis at ≈ 1.32 and ≈ 2.44, and Healthy is the band in between. The weights are so large that it is saturated: the loss is below 10⁻¹⁸ for every real patient, so a gradient step changes nothing. On an illustrative borderline patient (PSA 1.3) one step with η = 0.1 takes the loss from 0.54 to nearly 0.

### Corrections to the original material

| Where | Original claim | Correction |
|---|---|---|
| Wonderland case study, slide-5 breakdown | w₁ → 5.5, w₃ → 4.5 | With L = ½(y − ŷ)², ∂L/∂wⱼ = (ŷ − y)xⱼ = +5xⱼ, so **w₁ → 4.5, w₃ → 3.5**. The original values would *increase* the output. |
| Decision-boundary section | "Above the line → z > 0 → a = 1" | Above the line, **z ≤ 0 and a = 0** (this matches the diagram). The neuron is a NAND gate. |
| A ∧ ¬B code comment | `w[0]*x + w[1]*x` | `w[0]*x[0] + w[1]*x[1]` |

---

## Repository layout

```text
.
├── notebooks/
│   ├── SingleLayerPerceptrons_Workshop.ipynb   ← submission (workshop + Team Extensions)
│   ├── Wonderland_ANN_Case_Study.ipynb         ← reference case study
│   └── images/                                 ← figures used by the notebooks
├── workstreams/                                ← one notebook per team member
├── scripts/
│   ├── build_notebook.py                       ← merges workstreams into the submission notebook
│   └── train_or_perceptron.py                  ← minimal PyTorch training loop (OR gate)
├── assets/diagram-sources/                     ← base64 sources of the embedded diagrams
├── docs/
│   ├── WORK_PLAN.md                            ← task split and acceptance criteria
│   ├── INSTRUCTOR_BRIEF.md                     ← original assignment brief
│   └── instructor_requirements_lock.txt        ← instructor's full package lock, for reference
├── CONTRIBUTING.md                             ← team git workflow
└── requirements.txt
```

We gave each member their own notebook because `.ipynb` files are JSON, and four people editing one notebook cause merge conflicts. `scripts/build_notebook.py` appends the workstreams to the end of the workshop notebook. It never modifies the original cells, and you can re-run it safely.

## Reproducing our results

Requires **Python 3.10–3.12** (TensorFlow does not support 3.14 yet).

```bash
python3.12 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

python scripts/build_notebook.py --execute   # merge and run the notebook top to bottom
```

Or open `notebooks/SingleLayerPerceptrons_Workshop.ipynb` in VS Code, select the `.venv` kernel, and choose **Run All**.

To run the standalone training loop:

```bash
python scripts/train_or_perceptron.py
```

## Acknowledgements

The workshop notebook, case study, images and training-loop script come from Prof. Espinosa's [SingleLayerPerceptrons_Workshop](https://github.com/ProfEspinosaAIML/SingleLayerPerceptrons_Workshop). The movie and decision-boundary diagrams come from the [video referenced in the workshop](https://youtu.be/OFbnpY_k7js).
