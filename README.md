# Machine Learning Case Studies from Bosscoder Academy

Welcome to my collection of machine learning case studies completed as part of the curriculum at Bosscoder Academy. This repository showcases my ability to tackle various data science problems, from data preprocessing and exploratory analysis to model building and evaluation.

Each directory contains a specific case study, including the Jupyter Notebook (`main.ipynb`), the dataset (`.csv`), and a detailed PDF report summarizing the findings.

---

## 📂 Case Studies Overview

Here is a summary of the projects included in this repository:

1.  **[Sentiment Analysis on Movie Reviews](./1_ShoeSize_from_age_price_sex/)**
    * **Objective:** To classify movie reviews as either 'positive' or 'negative'.
    * **Techniques:** Text preprocessing (stopword removal, tokenization), feature extraction using TF-IDF, and classification using Naive Bayes and Support Vector Machines (SVM).
    * **Outcome:** All tested models achieved 100% accuracy on this dataset, demonstrating proficiency in handling and modeling text data for classification tasks.

2.  **[Predicting Shoe Size with Regression](./2_Credit_Card_Fraud_Detection/)**
    * **Objective:** To predict a person's shoe size based on their age, sex, and the price of their shoes.
    * **Techniques:** Simple Linear Regression, Multiple Linear Regression, and Polynomial Regression. This project also involved handling multicollinearity using Ridge (L2) and Lasso (L1) regularization.
    * **Outcome:** The Multiple Linear Regression and Ridge Regression models performed the best, achieving the lowest Mean Squared Error (MSE). This study highlights the process of model selection and regularization to handle common data issues.

3.  **[Credit Card Fraud Detection](./3_Sentiment_analysis_on_movie_reviews/)**
    * **Objective:** To detect fraudulent credit card transactions from a highly imbalanced dataset.
    * **Techniques:** This project focuses heavily on handling class imbalance using various resampling techniques, including Random Under-Sampling (RUS), Random Over-Sampling (ROS), SMOTE, and Tomek Links. Logistic Regression and K-Nearest Neighbors (KNN) models were evaluated.
    * **Outcome:** The KNN model trained on data cleaned with Tomek Links provided the best balance between precision and recall, achieving the highest F1-Score. This demonstrates a deep understanding of evaluation metrics and strategies for imbalanced classification.

---

## 🛠️ Tech Stack

The following tools and libraries were used across these projects:

* **Python 3**
* **Pandas:** For data manipulation and analysis.
* **NumPy:** For numerical operations.
* **Matplotlib & Seaborn:** For data visualization.
* **Scikit-learn:** For data preprocessing, modeling, and evaluation.
* **Imbalanced-learn:** For resampling techniques in the fraud detection case study.
* **Jupyter Notebook:** As the primary development environment.

---

## 🚀 How to Use

To run these case studies on your local machine, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/PrashikBhimte/ML-Case-Studies.git](https://github.com/PrashikBhimte/ML-Case-Studies.git)
    cd ML-Case-Studies
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**
    *(You should create a `requirements.txt` file for this step)*
    ```bash
    pip install -r requirements.txt
    ```

4.  **Launch Jupyter Notebook:**
    ```bash
    jupyter notebook
    ```

5.  Navigate to one of the project directories (e.g., `sentiment-analysis/`) and open the `main.ipynb` file.
