# Single-Layer Perceptrons Workshop

**Course:** CSCN8010 — Foundations of Machine Learning  
**Topic:** Introduction to Artificial Neural Networks (ANNs) and Single-Layer Perceptrons  
**Environment:** Python, Jupyter Notebook / Visual Studio Code, PyTorch, TensorFlow

## Workshop overview

This workshop introduces artificial neural networks through the biological-neuron analogy and builds toward implementing and interpreting single-layer perceptrons. You will examine how inputs, weights, biases, and activation functions produce predictions, and how gradient descent adjusts model parameters during training.

You will use the workshop notebook, **`SingleLayerPerceptrons_Workshop.ipynb`** (or the version provided in class), to run examples, explain their outputs, and complete coding challenges.

## Learning outcomes

By the end of the workshop, you should be able to:

1. Describe the relationship—and important differences—between biological neurons and artificial neurons.
2. Calculate a weighted sum, add a bias, and apply an activation function.
3. Compare identity, binary step, sigmoid, tanh, and ReLU activations.
4. Implement and test simple perceptrons using PyTorch and TensorFlow.
5. Explain how perceptrons implement binary logic, including AND, OR, and AND-NOT.
6. Interpret a linear decision boundary and recognize the limits of a single-layer perceptron.
7. Compute a prediction and loss, then explain gradient descent, learning rate, and backpropagation.

## Workshop activities

### Part 1 — Foundations of ANNs

Explore the biological-neuron analogy, electrical impulses, the mathematical neuron model, and activation functions. Run the activation-function visualization and compare their outputs.

### Part 2 — Single-layer perceptrons

Work through the notebook's examples of weighted inputs and bias. Test all binary input combinations for the logic-gate exercises, including inhibitory connections (`A AND NOT B`). Compare the PyTorch and TensorFlow implementations and explain the results.

### Part 3 — Linear classification

Explain how the equation `z = w·x + b` defines a decision boundary at `z = 0`. Relate the boundary to the perceptron's classification outputs and discuss why a single-layer perceptron cannot solve XOR without additional features or layers.

### Part 4 — Canada's Wonderland prediction

Use three binary inputs—weather, ticket queue, and health—to model a decision about visiting Canada's Wonderland. With the example inputs `x = [1, 0, 1]`, weights `w = [5, 2, 4]`, and bias `b = -3`, calculate the raw score and sigmoid output using PyTorch.

Compare the predicted output with an observed target, calculate a loss, and inspect how gradients update the weights. **Distinguish the raw score `z = 6` from the activated prediction `sigmoid(6) ≈ 0.9975`: they are not interchangeable when computing classification error.**

### Part 5 — Interactive exploration

Experiment with [TensorFlow Playground](https://playground.tensorflow.org/) to see how network architecture, activation functions, input features, and learning rate affect classification.

## Coding challenges

Complete the challenges identified in the notebook. At minimum, document your findings for these investigations:

1. **Activation functions:** Change the input range or activation function and explain how the output changes.
2. **Logic gates:** Verify all four binary input combinations and explain how a negative weight inhibits activation.
3. **Decision boundary:** Change one weight or the bias, and explain how the separating line moves.
4. **Wonderland prediction:** Change one input at a time and compare the raw score, sigmoid prediction, and final decision.
5. **Gradient descent:** Compare weight updates using different learning rates. Explain the roles of `loss.backward()` and the parameter update step, and how backpropagation uses the chain rule.

Keep your code, results, and short explanations together in the notebook. Where a challenge asks for a mathematical explanation, use readable LaTeX in Markdown cells.

## Set up your environment

Use a supported Python version compatible with the selected PyTorch and TensorFlow releases. Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
```

**Windows (PowerShell):**

```powershell
.venv\Scripts\Activate.ps1
```

**macOS / Linux:**

```bash
source .venv/bin/activate
```

Install packages:

```bash
python -m pip install -r requirements.txt
```

A suitable starting `requirements.txt` is:

```text
numpy
matplotlib
torch
tensorflow
notebook
ipykernel
```

Register the environment as a Jupyter kernel if needed:

```bash
python -m ipykernel install --user --name slp-workshop --display-name "Python (SLP Workshop)"
```

Open the notebook in Visual Studio Code with the Python and Jupyter extensions, select the environment's kernel, and run all cells in order. If a framework is unavailable on your platform, document the limitation and the steps you attempted.

## Repository structure

Create **your own GitHub repository**. A suggested structure is:

```text
SingleLayerPerceptrons_Workshop/
├── README.md
├── requirements.txt
├── .gitignore
├── SingleLayerPerceptrons_Workshop.ipynb
└── images/
    ├── ANN_1.png
    ├── ANN_2.png
    ├── ANN_3.png
    ├── ANN_4.png
    ├── ANN_5.png
    └── ANN_6.png
```

Include any other images referenced by the notebook. Preserve relative paths such as `./images/ANN_1.png` so the figures display correctly when the repository is cloned. If your supplied notebook has a version suffix, you may keep that filename and document it here.

Your `README.md` must be **your own**, adapted to describe your work rather than merely copying these instructions. Include the workshop objective, completed activities, noteworthy results, challenges attempted, and instructions for reproducing the notebook.

Your `requirements.txt` must list the Python packages required to run your submitted notebook. Do not commit your virtual environment, secrets, or large temporary files. Include `.venv/`, `__pycache__/`, and `.ipynb_checkpoints/` in `.gitignore`.

## Submission instructions

1. Create a new repository under **your own GitHub account** (suggested name: `SingleLayerPerceptrons_Workshop`).
2. Add your completed notebook, your own `README.md`, `requirements.txt`, and the images or other assets the notebook uses.
3. Run the notebook **from top to bottom** using a fresh kernel. Fix errors and save the notebook with its outputs where practical.
4. Commit and push your work to GitHub.
5. Confirm that your repository displays the README and that the notebook and images are accessible.
6. Submit the **GitHub repository URL** through the submission channel specified in class.

### Final submission checklist

- [ ] Repository belongs to my GitHub account and is accessible for review.
- [ ] Completed Jupyter notebook is included.
- [ ] `README.md` describes my work and how to reproduce it.
- [ ] `requirements.txt` lists all required packages.
- [ ] Referenced images are included and relative paths work.
- [ ] Code cells run in order without avoidable errors.
- [ ] Coding challenges and talking-point explanations are completed.
- [ ] Results and mathematical explanations agree with the code.
- [ ] No credentials, virtual environment folders, or unnecessary generated files are committed.

**Submission:** Your GitHub repository URL. Follow the deadline and any additional requirements announced in class.
