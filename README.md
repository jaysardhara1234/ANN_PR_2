
# 💼 Adult Income Prediction Using Artificial Neural Networks (ANN)

A binary classification project that uses a **Multilayer Perceptron (MLP)** to predict whether an individual's annual income is **above $50K** or **at or below $50K**, based on demographic and employment-related attributes from the Adult Income dataset.

The notebook explores the complete ANN workflow: data cleaning, exploratory data analysis, preprocessing, model construction, controlled experiments with neural-network components, and final model evaluation.

---

## 🎯 Project Objective

The objective is to build and evaluate an Artificial Neural Network for predicting the income class:

- `0` — Income is **≤50K**
- `1` — Income is **>50K**

The project also investigates how activation functions, weight initialization, loss functions, batch normalization, and optimizers affect model training and classification performance.

## 📂 Dataset

The notebook reads the dataset from:

```text
../Dataset/adult.csv
```

The input file should be named **`adult.csv`** and placed in a folder named `Dataset` one directory level above the notebook's working directory.

The target column is `income`. The notebook uses the following dataset fields after removing two columns:

- **Numerical:** `age`, `educational-num`, `capital-gain`, `capital-loss`, `hours-per-week`
- **Categorical:** `workclass`, `marital-status`, `occupation`, `relationship`, `race`, `gender`, `native-country`

The notebook drops `fnlwgt` and `education` during its cleaning stage. It does not bundle the dataset file, so you must obtain and place `adult.csv` separately.

### Expected project structure

```text
Project/
│
├── Dataset/
│   └── adult.csv
│
├── Notebook/
│   └── Adult_Income_prediction.ipynb
│
└── README.md
```

> **Path note:** The notebook uses a relative path (`../Dataset/adult.csv`). Keep the folder arrangement and notebook working directory consistent with that path. If your notebook is stored somewhere else, update the `pd.read_csv()` path accordingly.

## 🧹 Data Cleaning & Preprocessing

The notebook performs the following steps:

1. Loads the CSV file using Pandas and inspects its shape, data types, and first few records.
2. Removes leading and trailing whitespace from categorical text values.
3. Replaces `?` entries with missing values and removes rows containing missing values.
4. Drops the `fnlwgt` and `education` columns.
5. Converts the `income` target into binary values (`0` and `1`).
6. Explores the target distribution and selected feature relationships using visualizations and a numerical correlation heatmap.
7. Applies one-hot encoding to categorical input features using `pd.get_dummies(drop_first=True)`.
8. Splits the data into training and testing sets using an 80:20 split, `random_state=42`, and stratification on the target.
9. Standardizes the numerical columns using `StandardScaler`. The scaler is fitted on the training data and then applied to the test data.

## 🧠 ANN Architecture

The notebook defines a reusable function named `build_ann()` to create and compile neural-network models.

**Default architecture:**

```text
Input Layer
    ↓
Dense Layer — 128 neurons
    ↓
Dense Layer — 64 neurons
    ↓
Output Layer — 1 neuron (Sigmoid)
```

- Hidden layers use ReLU by default.
- The output layer uses a sigmoid activation for binary classification.
- The default loss function is binary cross-entropy.
- The default optimizer is Adam.
- The model tracks accuracy, precision, recall, and F1-score.

The model is trained using a batch size of 64. Most experiments use 50 epochs and a 10% validation split; individual experiments may specify different epoch counts.

## 🧪 Experiments Covered

### 1. Activation Function Comparison

Trains separate ANN models using:
- ReLU
- Tanh
- Sigmoid
- ELU

Compares training and validation performance and examines ReLU hidden-layer activations and gradient flow in the sigmoid-hidden-layer experiment.

### 2. Weight Initialization Techniques

Compares:
- Glorot Uniform
- Glorot Normal
- He Uniform
- He Normal
- Zeros

The notebook examines convergence behaviour, demonstrates the effect of zero initialization, and visualizes weight distributions before and after training for selected initializers.

### 3. Loss Function Experiments

Investigates:
- Binary Cross-Entropy (BCE) as the baseline classification loss
- Mean Squared Error (MSE) as an alternative-loss experiment
- Class-weighted training using class weights
- Focal Loss, with `gamma=2.0` and `alpha=0.25`

The class-weighted experiment uses weights computed from the training target distribution.

### 4. Batch Normalization

Builds an ANN with Batch Normalization and compares its training dynamics with the baseline ANN. It also compares Batch Normalization placed after activation with the canonical arrangement before activation, and inspects learned `gamma` and `beta` parameters.

### 5. Optimizer Comparison

Compares five optimizer configurations:
- Vanilla SGD
- SGD with Momentum
- RMSprop
- Adam (default)
- Adam with explicitly specified parameters

A separate learning-rate experiment compares several learning rates for SGD and Adam.

### 6. Final Combined ANN

Trains a final ANN using:
- Hidden units: `[128, 64]`
- Activation: ReLU
- Weight initializer: He Normal
- Batch Normalization: Enabled
- Optimizer: Adam
- Loss: Binary Cross-Entropy
- Epochs: 80
- Batch size: 64
- Validation split: 10%

In the notebook, `use_weighted_bce = False`, so the final model uses **ordinary binary cross-entropy without class weights**. Change this setting only if you intentionally want the final model to use the computed class weights.

## 📊 Model Evaluation

The notebook evaluates models using:

- **Accuracy** — overall proportion of correct predictions
- **Precision (Class 1)** — proportion of predicted `>50K` cases that are actually `>50K`
- **Recall (Class 1)** — proportion of actual `>50K` cases correctly identified
- **F1-score (Class 1)** — harmonic mean of class-1 precision and recall
- **ROC-AUC** — ability to rank positive cases above negative cases

It also includes a classification report, confusion matrix, training/validation plots, a model comparison table, and ROC curves for selected models.

The final numerical results are generated when the notebook is executed. They are intentionally not hard-coded in this README because results depend on the dataset file and the actual training run.

## 🛠️ Technologies & Libraries

- Python
- Jupyter Notebook
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- TensorFlow / Keras

## ⚙️ Installation & How to Run

### 1. Clone or download the project

Place the notebook and dataset in the folder arrangement described above.

### 2. Install the required libraries

Run this in a terminal or notebook environment:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow jupyter
```

### 3. Start Jupyter Notebook

```bash
jupyter notebook
```

### 4. Open and run the notebook

Open `Adult_Income_prediction.ipynb` and execute the cells **from top to bottom**. Ensure that `adult.csv` is available at the path expected by the notebook.

> **Important:** The notebook contains multiple model-training experiments. Running all cells can take a considerable amount of time, depending on your system and TensorFlow setup. Run each training cell once unless you deliberately intend to train that model again.

## 📌 Key Learning Outcomes

By completing this project, you can gain practical experience with:

- Preparing mixed numerical and categorical tabular data for ANN training
- Building reusable neural-network architectures with TensorFlow/Keras
- Understanding activation functions and weight initialization
- Comparing classification loss functions and handling class imbalance
- Applying Batch Normalization
- Comparing optimizers and learning-rate choices
- Evaluating binary classifiers beyond accuracy
- Interpreting training curves, confusion matrices, and ROC curves

## 👤 Project Note

This repository/notebook is an educational machine-learning project based on the Adult Income prediction task. Review the notebook's outputs after execution before reporting any performance figures or drawing conclusions about which experiment performed best.
