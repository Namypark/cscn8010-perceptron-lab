# Work Plan — Single-Layer Perceptrons Workshop

**Team:** Nnamdi · Carlos · Davisen · Rangeetha
**Course:** CSCN8010 — Foundations of Machine Learning

The instructor's brief is in [`INSTRUCTOR_BRIEF.md`](INSTRUCTOR_BRIEF.md). This page breaks the technical work (the coding challenges) into four workstreams that can be done in parallel without anyone editing the same file.

---

## What the assignment requires

| # | Requirement (from the brief and notebook) | Owner |
|---|---|---|
| C1 | **Activation functions:** change the input range or activation and explain how the output changes | Rangeetha |
| C2 | **Logic gates:** verify all 4 input combinations, explain how a negative weight inhibits activation | Rangeetha |
| — | **Movie challenge:** choose your own weights and bias, add a talking point | Rangeetha |
| C3 | **Decision boundary:** change one weight or the bias and explain how the line moves | Davisen |
| P3 | **Linear limits:** explain why one perceptron cannot solve XOR | Davisen |
| P5 | **TensorFlow Playground:** record what you observed | Davisen |
| C4 | **Wonderland:** change one input at a time and compare raw `z`, `sigmoid(z)` and the decision | Carlos |
| — | **Prostate Cancer ANN** (last cell of the notebook): Markdown, code and **3 talking points** | Carlos |
| C5 | **Gradient descent:** compare learning rates; explain `loss.backward()`, the update step and the chain rule | Nnamdi |
| — | **Homework question:** how does the update rule relate to backpropagation? | Nnamdi |
| — | **`scripts/train_or_perceptron.py` TODO:** run it, step through it in the debugger | Nnamdi |
| — | Repo, README, `requirements.txt`, integration and a final top-to-bottom run | Nnamdi (everyone writes their own README section) |

All explanations go in Markdown cells, with LaTeX for the maths. The results and maths must **agree with the code** (this is on the marking checklist).

---

## Workstream 1 — Gradient Descent, Backprop and Training · **Nnamdi**

File: `workstreams/ws1_gradient_descent.ipynb`

1. **Chain rule by hand:** for the Wonderland neuron (sigmoid + squared error) derive
   $\frac{\partial L}{\partial w_j} = \frac{\partial L}{\partial \hat y}\cdot\frac{\partial \hat y}{\partial z}\cdot\frac{\partial z}{\partial w_j} = 2(\hat y - y)\,\sigma(z)(1-\sigma(z))\,x_j$
   and show numerically that it matches PyTorch's `weights.grad`.
2. **Learning-rate comparison:** train the same neuron with η ∈ {0.01, 0.1, 1, 10}. Plot the loss curves, tabulate the weights after training, and explain what you see (too slow, good, overshoot or oscillation).
3. **What the training-loop lines do:** demonstrate `loss.backward()`, `torch.no_grad()` and `grad.zero_()`. Show what happens if you *forget* `zero_()`: the gradients accumulate.
4. **Saturation insight:** at z = 6, σ′(z) ≈ 0.0025, so one update changes w₁ by only about 1.2 × 10⁻⁶. Explain why, and compare this with the binary cross-entropy gradient `(ŷ − y)·x`, which has no σ′ factor.
5. **Erratum:** `Wonderland_ANN_Case_Study.ipynb`, slide-5 breakdown. With L = ½(y − ŷ)², ∂L/∂w = (ŷ − y)·x = **+5**·x, so the correct updates are **w₁ = 4.5, w₂ = 2, w₃ = 3.5**. The notebook says 5.5 / 2 / 4.5, which would *increase* ŷ and contradicts its own sentence "These new weights will produce a smaller output".
6. **Homework answer:** relate the single-neuron update to backprop through multiple layers (the chain rule applied layer by layer).
7. **Script:** add `--lr` and `--epochs` arguments to `scripts/train_or_perceptron.py`, run it under the VS Code debugger, and describe what you saw.

## Workstream 2 — Wonderland Sensitivity and Prostate Cancer ANN · **Carlos**

File: `workstreams/ws2_wonderland_prostate.ipynb`

1. **One-at-a-time changes:** starting from x = [1, 0, 1], flip each input in turn. Then build the full 8-row table of x → z → σ(z) → decision (threshold 0.5).
2. **Raw score vs prediction:** explain clearly why z = 6 and σ(6) ≈ 0.9975 are not interchangeable when you compute error. The brief puts this sentence in **bold**, so the marker will look for it.
3. **Prostate Cancer ANN:** take the inputs, weights, bias and target from the *Deep Learning Fundamentals* class notes. Rebuild them in PyTorch following the Wonderland pattern: forward pass → error → one gradient step. Add Markdown and **exactly three talking points**.
   - ⚠️ The class notes are **not** in this repo. Get the slide from the course page before you start.

## Workstream 3 — Decision Boundaries, XOR and the Playground · **Davisen**

File: `workstreams/ws3_decision_boundary_xor.ipynb`

1. **Plot the boundary** z = −2x₁ − 2x₂ + 3: shade both regions and plot the four binary points. Show that this neuron is actually a **NAND** gate.
2. **Move the line:** change one weight, then the bias, and overlay the lines. Explain with the line equation x₂ = −(w₁/w₂)x₁ − b/w₂: the bias shifts the line in parallel, and the weights rotate it.
3. **Plot the gate boundaries** for the notebook's AND, OR and A ∧ ¬B perceptrons.
4. **XOR:** show that no single line separates XOR. For example, a perceptron training loop never reaches 0 errors, or a grid search over (w₁, w₂, b) finds no solution. Then solve it either with an extra feature x₁·x₂ or with two layers (OR, NAND → AND).
5. **Erratum:** the notebook says "above the line → z > 0 → a = 1". The diagram and the maths both say the region **above** the line has **z ≤ 0, a = 0**. Check: (0, 0) gives z = 3 > 0, so the point *below* the line fires.
6. **TensorFlow Playground:** compare XOR with no hidden layer against XOR with one hidden layer, and also try the x₁x₂ feature. Write up what you observed. Take screenshots if useful, and save them in `notebooks/images/`.

## Workstream 4 — Activation Functions, Logic Gates and the Movie Challenge · **Rangeetha**

File: `workstreams/ws4_activations_gates.ipynb`

1. **Activations:** re-plot over a narrow range ([−3, 3]) and a wide one ([−50, 50]). Plot each function's **derivative**, and explain vanishing gradients (sigmoid/tanh) and the zero region of ReLU.
2. **Gate verifier:** write `verify_gate(weights, bias, expected)`. It should run all 4 input combinations in **both** PyTorch and TensorFlow, `assert` that they match, and print a truth table. Use it on AND, OR and A ∧ ¬B.
3. **Inhibition:** using the z values, show how the weight −1 on B pushes z below 0 whenever B = 1.
4. **Movie challenge:** choose your own weights and bias, for example "good weather plus at least one other condition". Print the 8-row truth table and write the talking point.
5. **Minor erratum:** in the A ∧ ¬B cell, the comment `w[0]*x + w[1]*x` should be `w[0]*x[0] + w[1]*x[1]`.

---

## Rules for everyone's notebook

- **It must run on its own from a fresh kernel.** Import and define everything you use, and don't rely on variables from the workshop notebook.
- **Don't reference image files** from a workstream notebook (the relative paths would break when we merge). Inline matplotlib plots are fine.
- The first cell (the brief) is tagged `brief` and is **not** copied into the final notebook. Everything below it is.
- Before you push: **Kernel → Restart & Run All**, then save with the outputs.

## Integration

`python scripts/build_notebook.py --execute` appends every workstream (in the order ws4 → ws3 → ws2 → ws1) to the end of `notebooks/SingleLayerPerceptrons_Workshop.ipynb`, under a "Team Extensions" heading. It then runs the whole notebook top to bottom. It is safe to re-run, because it replaces the previous Team Extensions section rather than duplicating it. The instructor's original cells are never modified.

## Milestones

1. Everyone: environment set up and your own notebook running.
2. First draft of your workstream pushed to your branch.
3. Peer review: each person reviews one other workstream (1→2→3→4→1).
4. Integrate, write the README, run top to bottom, submit the URL.
