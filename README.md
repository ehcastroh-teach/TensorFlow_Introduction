# An Introduction to TensorFlow V.2

TensorFlow (TF) is an open-source library for dataflow programming, differentiable computation, and machine learning. This repository introduces TF V.2 from the ground up - covering tensors, operations, computation graphs, and the `@tf.function` decorator - so that readers without a machine learning background can build confidence with the framework before applying it to real models.

---

## Learning Objectives

By working through this repository you will be able to:

- Create and manipulate `tf.constant` and `tf.Variable` tensors across scalar, vector, matrix, and higher-rank shapes
- Perform arithmetic, linear algebra, gradient, and slicing operations on tensors
- Understand eager execution and why it simplifies debugging compared to TF 1.x graph execution
- Set up TensorBoard and interpret its training curves
- Decorate Python functions with `@tf.function` to compile them into portable TensorFlow graphs
- Recognize when to use `@tf.function` and when to stay in eager mode
- Handle special tensor types: ragged, string, and sparse

---

## File Dictionary

| File / Folder | Description |
|---|---|
| `TensorFlow_Introduction.ipynb` | Main notebook - four parts covering setup, tensors, operations, and graphs |
| `assets/homeworks/m410_homework-1_tensorflow.ipynb` | Practice exercises to reinforce the material |
| `assets/content/images/` | Diagrams and logos used inside the notebook |
| `requirements.txt` | Python package dependencies |
| `logs/` | TensorBoard event files generated when you run the TensorBoard demo cells |

---

## Workflow Diagram

```
Clone repo
    |
    v
Install dependencies (requirements.txt)
    |
    v
Open TensorFlow_Introduction.ipynb
    |
    +---> Part 0: About and Motivation
    |
    +---> Part 1: Tensors and Operations
    |         |
    |         +-- 1.1 TensorFlow Setup
    |         +-- 1.2 TensorBoard Setup
    |         +-- 1.3 TF Tensors (constant, Variable, shapes)
    |         +-- 1.4 TF Operations (arithmetic, gradients, slicing, reshape)
    |
    +---> Part 2: Graphs and Executions
    |         |
    |         +-- 2.1 @tf.function, side effects, input signatures
    |
    +---> Part 3: Wrap-Up and Next Steps
    |
    +---> Appendix: Installation guide, references
    |
    v
Work through the homework notebook
```

---

## Step-by-Step Walkthrough

### Part 0 - About and Motivation

TF V.2 ships with eager execution on by default. This means tensor operations run immediately and you can inspect values with a plain `print()` - no `Session.run()` needed. The notebook opens by verifying this and demonstrating why it matters for interactive learning.

### Part 1.1 - TensorFlow Setup

Installing TF V.2 and confirming the version number. The reason this comes first is that many errors in downstream cells trace back to version mismatches between TF, Keras, and CUDA.

### Part 1.2 - TensorBoard Setup

TensorBoard is TF's built-in visualization dashboard. The demo here trains a tiny MNIST model only to show how to log and display training curves - not to teach classification. Understanding where logs are written (the `logs/` directory) and how to load the extension (`%load_ext tensorboard`) saves time when you scale up to real training runs.

### Part 1.3 - TF Tensors

Tensors are the core data structure. The notebook walks through:

- `tf.constant` - immutable tensors initialized from Python values
- `tf.Variable` - mutable tensors that represent model weights
- Scalar (rank-0), vector (rank-1), matrix (rank-2), and higher-rank shapes
- Building tensors from distributions (`tf.random.normal`, `tf.random.uniform`)
- Optional: variable placement across CPU and GPU devices

### Part 1.4 - TF Operations

Operations in TF follow NumPy conventions closely. The notebook covers:

- Arithmetic: `tf.add`, `tf.multiply`, `tf.matmul`
- Gradients: `tf.GradientTape` for computing derivatives of any differentiable function
- Indexing and slicing: integer indexing removes a dimension; range indexing preserves it
- Reshape: any reshape is valid as long as the total element count stays the same
- Optional: broadcasting, ragged tensors, string tensors, sparse tensors

### Part 2.1 - `@tf.function`

Eager execution is great for debugging but slow for production. `@tf.function` traces a Python function once and compiles it into a TF computation graph that runs faster and can be exported. The section covers tracing behavior, side effects (why `print()` only fires once), polymorphism (one decorated function, multiple graphs for different dtypes), and input signatures that lock a function to a specific shape and dtype.

---

## How to Run

```bash
# 1. Clone and enter the repo
git clone https://github.com/ehcastroh-teach/TensorFlow_Introduction.git
cd TensorFlow_Introduction

# 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Launch Jupyter
jupyter notebook TensorFlow_Introduction.ipynb
```

Run all cells top-to-bottom (Kernel - Restart & Run All). TensorBoard cells will write logs to `logs/` and display inline.

---

## Key Concepts Glossary

| Term | Plain-language definition |
|---|---|
| Tensor | A multidimensional array - the basic unit of data in TensorFlow |
| Rank | The number of dimensions of a tensor (scalar = rank 0, vector = rank 1, matrix = rank 2) |
| Eager execution | Running TF operations immediately, returning concrete values instead of building a graph first |
| `tf.constant` | An immutable tensor; its value cannot change after creation |
| `tf.Variable` | A mutable tensor; typically used for model weights that are updated during training |
| `tf.GradientTape` | A context manager that records operations to enable automatic differentiation |
| Gradient | The rate of change of a function with respect to its inputs - the core signal for training a model |
| `@tf.function` | A decorator that compiles a Python function into a TensorFlow computation graph |
| Graph execution | Running a precompiled computation graph rather than interpreted Python - faster for production |
| Tracing | The process by which `@tf.function` reads through a function once and records the TF operations |
| Broadcasting | Automatically expanding a smaller tensor to match the shape of a larger one during operations |
| TensorBoard | TensorFlow's built-in dashboard for visualizing training metrics, graphs, and profiling data |
| Ragged tensor | A tensor where rows can have different lengths (e.g., variable-length sequences) |

---

## Further Reading

- TensorFlow Guide
- TensorFlow Tutorials
- Deep Learning with Python (Francois Chollet)
- NumPy Array Broadcasting

---

## Credits and Acknowledgements

Notebook content originally developed for an applied machine learning curriculum. Textbook reference: *Deep Learning with Python* by Francois Chollet (Manning Publications). TF logo and diagram assets copyright TensorFlow / Google.

---

## Contact

<div align="center">
  <img src="images/thumbnails/ehcastroh_teach_banner_flower.png" alt="ehcastroh" width="90" style="border-radius: 50%;" />

  <sub>ehcastroh</sub>

  <a href="https://github.com/ehcastroh">GitHub</a> · <a href="https://www.linkedin.com/in/ehcastroh/">LinkedIn</a>
</div>
