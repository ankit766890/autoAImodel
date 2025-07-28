# 🤖 AutoAI Model Builder

An intelligent AutoML project built using Python that automatically loads a dataset, cleans it, trains multiple machine learning models, evaluates them, and selects the best model based on performance. This project aims to reduce manual work in building ML models and help users quickly analyze datasets without writing complex code.

---

## 📌 Table of Contents

- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [How to Use](#-how-to-use)
- [Dataset Guidelines](#-dataset-guidelines)
- [License](#-license)
- [Author](#-author)

---
## Author 
ANKIT
## Licence
Free to use.

## 🚀 Features

- 📥 **Auto Data Loading** — Load dataset from CSV.
- 🧹 **Data Cleaning** — Handle missing values and encode categorical features.
- 🤖 **Auto Model Training** — Train 3 pre-defined ML models:
  - Random Forest
  - Support Vector Machine (SVM)
  - XGBoost
- 📊 **Evaluation** — Print accuracy of all models.
- 🏆 **Best Model Selection** — Automatically select the best-performing model.

---

## 🛠 Tech Stack

| Tool        | Purpose                        |
|-------------|--------------------------------|
| Python      | Programming Language           |
| Pandas      | Data Loading & Cleaning        |
| NumPy       | Numerical Operations           |
| Scikit-learn| ML Models & Evaluation Metrics |
| XGBoost     | Gradient Boosting Model        |
| VS Code     | Code Editor                    |

---

## 📁 Project Structure

 AutoAI_Model_Builder/
│
├── main.py # Main script to run the pipeline

├── utils.py # Utility functions (load, preprocess, train, evaluate)

├── dataset.csv # Input dataset file

├── requirements.txt # All required libraries

└── README.md # Project description and guide


---

💻 How to Use
Place your dataset as dataset.csv in the root folder.

Make sure the target/output column is the last column in the file.

Run the project using:

python main.py

---

## ⚙️ Installation

1. Clone the repository.
2. Install Python (3.10+)
3. Open terminal in the project folder and install dependencies:

```bash
pip install -r requirements.txt
