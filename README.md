# Placement Prediction using Machine Learning

## Project Overview
This project focuses on building a machine learning model to predict whether a student will be placed in a company based on academic and demographic features. The model leverages various data preprocessing techniques, including handling imbalanced data, to ensure accurate predictions for student placement status.

## Tech Stack
- **Python**: Programming language for developing the machine learning pipeline.
- **Pandas**: For data manipulation and preprocessing.
- **Scikit-learn**: For implementing machine learning models and evaluation metrics.
- **Matplotlib/Seaborn**: For data visualization and exploratory data analysis (EDA).
- **Tomek Links**: To handle class imbalance and improve model performance by removing ambiguous samples.
- **SVM (Support Vector Machine)** & **Logistic Regression**: Machine learning algorithms used for classification.

## Problem Statement
The goal of this project is to predict whether a student will get placed based on various features such as:
- **CGPA**
- **SSC% (Secondary School Percentage)**
- **HSC% (Higher Secondary School Percentage)**
- **Gender**
- **Work Experience**
- **Employability Test Scores**

The model helps in identifying students at risk of not being placed, allowing institutions to provide targeted training and interventions.

## Data Preprocessing
- **Missing Value Handling**: Missing values were dropped or imputed appropriately.
- **Categorical Encoding**: Categorical variables like gender and work experience were encoded using Label Encoding.
- **Feature Scaling**: Numerical features were standardized using StandardScaler to normalize the data.
- **Handling Imbalanced Data**: Applied **Tomek Links** undersampling to balance the dataset and improve the performance of the model.

## Machine Learning Models
- **Support Vector Machine (SVM)**: Used for classification to identify the optimal decision boundary.
- **Logistic Regression**: A simple, interpretable model for baseline classification.

## Model Evaluation
The models were evaluated using the following metrics:
- **Accuracy**: Overall accuracy of the predictions.
- **Precision**: Precision score to evaluate the quality of the positive predictions.
- **Recall**: Recall score to assess how well the model identifies positive class instances.
- **Confusion Matrix**: Used to show the true positives, true negatives, false positives, and false negatives.

## Key Insights
- Higher CGPA, SSC%, HSC%, and employability test scores were strongly correlated with higher placement chances.
- The Tomek Links technique helped improve the model’s fairness and accuracy by addressing the class imbalance in the dataset.

## Impact
The project demonstrated how machine learning can assist educational institutions in identifying students who need additional support. By improving prediction accuracy on imbalanced datasets, it enables early identification of at-risk students for targeted interventions, ultimately improving student placement rates.

## Future Enhancements
- **Hyperparameter Tuning**: Implementing GridSearchCV to optimize model parameters.
- **Model Expansion**: Trying ensemble methods such as Random Forest and XGBoost for better performance.
- **Deployment**: Deploying the model as a web application using Flask or Streamlit for real-time predictions.

## Installation & Setup
1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/placement-prediction.git
    ```
2. Install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the project:
    ```bash
    jupyter notebook placement_prediction.ipynb
    ```

