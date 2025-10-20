# DoS Attack Detection using Transformer

This project implements and compares different machine learning models for detecting Denial of Service (DoS) attacks from the UNSW-NB15 dataset.

## Features

*   **Two Machine Learning Models:**
    *   Random Forest
    *   Decision Tree
*   **Data Preprocessing:** Scripts to filter and prepare the UNSW-NB15 dataset for training.
*   **Model Evaluation:** Detailed evaluation of the models, including accuracy, precision, recall, and F1-score.
*   **Result Visualization:** Generates and saves plots for:
    *   Confusion Matrix
    *   Feature Importance
    *   Classification Report

## Future Work

*   **Transformer Model:** A Transformer-based model will be implemented and compared with the other models.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <your-repository-name>
    ```
2.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

You can run the project from the command line and choose the model you want to use.

### Running a Model

*   **Random Forest (default):**
    ```bash
    python3 main.py
    ```
    or
    ```bash
    python3 main.py --model random_forest
    ```
*   **Decision Tree:**
    ```bash
    python3 main.py --model decision_tree
    ```

The evaluation results will be printed to the console, and the visualization plots will be saved in the root directory of the project.

## Project Structure

```
.
├── config/
│   └── config.py
├── data/
│   ├── UNSW_NB15_DoS_train_data.csv
│   └── UNSW_NB15_DoS_test_data.csv
├── data_loader/
│   └── data_loader.py
├── models/
│   ├── decision_tree.py
│   └── random_forest.py
├── preprocessor/
│   └── preprocessor.py
├── utils/
│   └── visualizer.py
├── main.py
├── prepare_data.py
└── requirements.txt
```
