# Perceptron Lab: From a Single Neuron to Its Limits

**CSCN8010 · Foundations of Machine Learning**
**Team:** Nnamdi · Carlos · Davisen · Rangeetha

In this project we take the Single-Layer Perceptrons workshop and push each of its ideas further. We test the activation functions at their extremes, check every logic gate in two frameworks, and move decision boundaries around until they break on XOR. We also compare gradient descent worked out by hand with PyTorch's autograd. Along the way we found and corrected some errors in the original material (see [Corrections](#corrections-to-the-original-material)).

**Submission notebook:** [`notebooks/SingleLayerPerceptrons_Workshop.ipynb`](notebooks/SingleLayerPerceptrons_Workshop.ipynb). The original workshop content is unchanged. Our work is in the **Team Extensions** section at the end.

## What we did

| Section | Author | Focus |
|---|---|---|
| Activation Functions, Logic Gates and the Movie Challenge | Rangeetha | Activations over narrow and wide input ranges with their derivatives, a gate checker that tests every gate in both PyTorch and TensorFlow, how a negative weight inhibits a neuron, and our own weights for the movie decision |
| Decision Boundaries, XOR and TensorFlow Playground | Davisen | Plotting the slide neuron's boundary (it is a NAND gate), how weights rotate the line and the bias shifts it, the boundaries of AND/OR/A∧¬B, why one perceptron can't learn XOR and two ways to fix it, and Playground experiments |
| Wonderland Sensitivity and the Prostate Cancer ANN | Carlos | Flipping one Wonderland input at a time, raw score z vs prediction σ(z), and a hand-checked gradient step on the 1-2-2 Prostate Cancer network |
| Gradient Descent, Backpropagation and Training | Nnamdi | Chain rule checked against autograd, squared error vs cross-entropy, a learning-rate sweep, what `backward()`/`no_grad()`/`zero_()` do, the slide-5 correction, and a 2-2-1 network trained on XOR with backprop |

## Key results

**Rangeetha: Activation Functions, Logic Gates and the Movie Challenge**

- On [−3, 3] sigmoid and tanh are smooth curves, but on [−50, 50] they look just like the binary step. Their derivatives collapse: σ′(10) ≈ 4.5 × 10⁻⁵ and tanh′(10) ≈ 8.2 × 10⁻⁹. Our hand-written derivatives match both PyTorch autograd and TensorFlow's `GradientTape` to within about 10⁻⁷.
- `verify_gate` checks every input combination in both frameworks and stops with an error if anything is wrong. C = A, AND, OR and A AND NOT B all pass, and a deliberately wrong truth table is caught.
- In A AND NOT B, switching B on drops z by exactly 1, from +0.5 to −0.5. B therefore acts as a veto. Our movie neuron (w = [3, 2, 1], b = −3.5) computes "good weather AND (company OR cinema nearby)", which neither a plain AND nor a plain OR can express.

**Davisen: Decision Boundaries, XOR and TensorFlow Playground**

- The slide neuron z = −2x₁ − 2x₂ + 3 is a NAND gate. Changing w₁ from −2 to −4 pivots the line around (0, 1.5) and turns the neuron into NOT x₁. Changing only the bias (1, 3, 5) shifts the line in parallel, turning the neuron into NOR, NAND and "always 1".
- XOR is impossible for one perceptron, shown three ways:
  - an algebraic proof;
  - the perceptron learning rule never gets below 2 errors on XOR, while it learns AND by epoch 9;
  - a search over 1,030,301 weight/bias combinations found 21,563 AND neurons and 0 XOR neurons.

  Adding the feature x₁x₂, or a hidden layer built as AND(OR, NAND), solves it.
- In TensorFlow Playground, XOR with no hidden layer stays at a test loss of about 0.498. One hidden layer of 4 neurons reaches 0.011 after 10,000 epochs. Adding the x₁x₂ input with no hidden layer reaches 0.014 within about 50 epochs.

**Carlos: Wonderland Sensitivity and the Prostate Cancer ANN**

- Starting from x = [1, 0, 1] (z = 6), flipping any single input never changes the decision: z only moves by that input's weight (−5, +2, −4). Across all 8 inputs the neuron says "no" only for [0,0,0] and [0,1,0].
- Raw score vs prediction: for target 1, the error on z = 6 is −5 (loss 12.5), but the error on σ(6) ≈ 0.9975 is 0.0025 (loss ≈ 3×10⁻⁶). The error must be computed on σ(z).
- The Prostate Cancer network (1 PSA input, 2 hidden neurons, 2 outputs) classifies all 15 patients correctly. Its hidden neurons cut the PSA axis at ≈ 1.32 and ≈ 2.44, and Healthy is the band in between. The weights are so large that it is saturated: the loss is below 10⁻¹⁸ for every real patient, so a gradient step changes nothing. On an illustrative borderline patient (PSA 1.3), one step with η = 0.1 takes the loss from 0.54 to nearly 0.

**Nnamdi: Gradient Descent, Backpropagation and Training**

- At the Wonderland point, one gradient step moves w₁ by only 1.2 × 10⁻⁶, because σ′(6) ≈ 0.0025. Cross-entropy cancels that slope and pushes 203 times harder. When the neuron is confidently wrong (z = −6), squared error barely reacts (|dL/dz| ≈ 0.005), while cross-entropy pushes at almost full strength (≈ 1).
- On the OR gate, η = 0.01 is still wrong on one input after 100 epochs, η = 1 to 10 converge well, and η = 100 pushes the neuron into saturation, where it predicts 1 for everything and the loss stays at 1.0. Forgetting `grad.zero_()` causes the same failure.
- A 2-2-1 network trained with backprop solves XOR, with its hidden neurons acting as NAND and NOR. But 3 of 6 random starts get stuck in a local minimum at loss 0.348. The instructor's training script, at its default 20 epochs, still misclassifies (0, 1).

## Corrections to the original material

| Where | Original claim | Correction |
|---|---|---|
| Wonderland case study, slide-5 breakdown | w₁ → 5.5, w₃ → 4.5 | With L = ½(y − ŷ)², ∂L/∂wⱼ = (ŷ − y)xⱼ = +5xⱼ, so **w₁ → 4.5, w₃ → 3.5**. The original values would *increase* the output. |
| Decision-boundary section | "Above the line → z > 0 → a = 1" | Above the line, **z ≤ 0 and a = 0**, which matches the diagram. The neuron is a NAND gate. |
| A ∧ ¬B code comment | `w[0]*x + w[1]*x` | `w[0]*x[0] + w[1]*x[1]` |

## Repository layout

```text
.
├── notebooks/
│   ├── SingleLayerPerceptrons_Workshop.ipynb   ← submission (workshop + Team Extensions)
│   ├── Wonderland_ANN_Case_Study.ipynb         ← reference case study
│   └── images/                                 ← figures used by the notebooks
├── workstreams/                                ← each member's section as a separate notebook
├── scripts/
│   ├── build_notebook.py                       ← merges the sections into the submission notebook
│   └── train_or_perceptron.py                  ← minimal PyTorch training loop (OR gate)
├── requirements.txt                            ← pip dependencies (Python 3.12)
└── pyproject.toml, uv.lock                     ← the same dependencies for uv users
```

Each of us wrote our section in a separate notebook, because `.ipynb` files are JSON and four people editing one notebook would cause merge conflicts. `scripts/build_notebook.py` appends the sections to the end of the workshop notebook. It never changes the original cells, and it can be re-run safely.

## Reproducing our results

Requires **Python 3.12**. TensorFlow 2.21 has no wheels for 3.13 or newer, and numpy 2.5 needs at least 3.12.

```bash
python3.12 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

python scripts/build_notebook.py --execute   # merge and run the notebook top to bottom
```

If you use [uv](https://docs.astral.sh/uv/), `uv sync` creates the same environment from `pyproject.toml` and `uv.lock`.

You can also open `notebooks/SingleLayerPerceptrons_Workshop.ipynb` in VS Code, select the `.venv` kernel and choose **Run All**.

To run the standalone training loop:

```bash
python scripts/train_or_perceptron.py                 # instructor defaults: lr 0.1, 20 epochs
python scripts/train_or_perceptron.py --lr 1 --epochs 50
```

## Acknowledgements

The workshop notebook, case study, images and training-loop script come from Prof. Espinosa's [SingleLayerPerceptrons_Workshop](https://github.com/ProfEspinosaAIML/SingleLayerPerceptrons_Workshop). The movie and decision-boundary diagrams come from the [video referenced in the workshop](https://youtu.be/OFbnpY_k7js).
