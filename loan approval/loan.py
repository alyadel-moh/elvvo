import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, RocCurveDisplay, roc_auc_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import warnings
import matplotlib.pyplot as plt  # Required for plotting
import json  # Used for parsing the classification report

# Suppress minor warnings for clean terminal output
warnings.filterwarnings('ignore')

TRAIN_FILE = 'train.csv'
TEST_FILE = 'test.csv'


def get_preprocessing_pipeline(X):
    """
    Creates and returns the ColumnTransformer for preprocessing steps.
    This fulfills the requirement: 'Handle missing values and encode categorical features'.
    """
    # Identify feature types
    numerical_features = X.select_dtypes(include=np.number).columns
    categorical_features = X.select_dtypes(include='object').columns

    # Pipeline for Numerical Features: Impute with Median (Handles missing values)
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median'))
    ])

    # Pipeline for Categorical Features: Impute with Most Frequent, then One-Hot Encode (Encodes categorical features)
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine transformers using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )
    return preprocessor


def plot_evaluation_metrics(model_pipeline, X_eval, y_eval):
    """
    Generates and saves Confusion Matrix and ROC Curve plots as PNG files.
    """

    # --- Confusion Matrix ---
    cm = confusion_matrix(y_eval, model_pipeline.predict(X_eval))

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()

    classes = ['Rejected (0)', 'Approved (1)']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()
    print("Graph saved: 'confusion_matrix.png'")

    # --- ROC Curve ---
    # Predict probabilities for ROC curve
    y_proba = model_pipeline.predict_proba(X_eval)[:, 1]
    auc = roc_auc_score(y_eval, y_proba)

    plt.figure(figsize=(8, 6))
    RocCurveDisplay.from_predictions(y_eval, y_proba, ax=plt.gca(), name=f"Logistic Regression (AUC = {auc:.4f})")

    # Add diagonal line for random classifier
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')

    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig('roc_curve.png')
    plt.close()
    print("Graph saved: 'roc_curve.png'")


def run_full_pipeline():
    """
    Executes the entire workflow for Loan Approval Prediction: Train, Evaluate, and Predict.
    """
    # --- 1. Load Training Data ---
    print("--- 1. Loading Training Data and Initial Preparation ---")
    try:
        df_train = pd.read_csv(TRAIN_FILE)
        df_train = df_train.drop('id', axis=1)

        X_train_full = df_train.drop('loan_status', axis=1)
        y_train_full = df_train['loan_status']

        print(f"Total Training Samples: {len(df_train)}")
        print(f"Target Class Distribution (Imbalance Check):\n{y_train_full.value_counts()}")

    except FileNotFoundError:
        print(f"Error: Training file '{TRAIN_FILE}' not found.")
        return
    except KeyError:
        print("Error: 'loan_status' column not found in the training data.")
        return

    # --- 2. Define and Train the Full Pipeline ---
    # Split for evaluation (80% train / 20% evaluation)
    X_train, X_eval, y_train, y_eval = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
    )

    preprocessor = get_preprocessing_pipeline(X_train_full)

    # Define the full pipeline: Preprocessor -> Model
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        # Train a classification model (Logistic Regression)
        ('classifier', LogisticRegression(solver='liblinear',
                                          random_state=42,
                                          # CRITICAL: This handles the requirement for 'imbalanced data'
                                          class_weight='balanced',
                                          max_iter=1000))
    ])

    print("\n--- 2. Training Classification Model (with class balancing) ---")
    # Train on the entire training feature set (X_train_full) for the final model
    model_pipeline.fit(X_train_full, y_train_full)
    print("Training complete.")

    # --- 3. Evaluate Performance (on held-out data from the original train set) ---
    y_pred_eval = model_pipeline.predict(X_eval)

    print("\n" + "=" * 80)
    print("✨ Evaluation on Held-out Data (Focus: Precision, Recall, F1-score) ✨")
    print("=" * 80)

    # CRITICAL: classification_report focuses on 'precision, recall, and F1-score'
    report = classification_report(y_eval, y_pred_eval, output_dict=True)

    # Print the full report
    print("\nFull Classification Report:\n")
    print(classification_report(y_eval, y_pred_eval,
                                target_names=['Loan Rejected (0)', 'Loan Approved (1)']))

    # NEW: Extract and display core metrics for the minority class (1: Approved)
    # This fulfills the requirement to "focus on precision, recall, and F1-score"
    minority_class = '1'  # '1' is the key used by sklearn's report for the positive class (Loan Approved)

    precision = report[minority_class]['precision']
    recall = report[minority_class]['recall']
    f1_score = report[minority_class]['f1-score']

    print("\n--- Core Imbalanced Metrics (Loan Approved Class) ---")
    print(f"Precision (Approved): {precision:.4f}")
    print(f"Recall (Approved):    {recall:.4f}")
    print(f"F1-Score (Approved):  {f1_score:.4f}")
    print("-------------------------------------------------------")
    print("\n" + "=" * 80)

    # Generate and save classification graphs
    print("\n--- Generating Classification Graphs (PNG files) ---")
    plot_evaluation_metrics(model_pipeline, X_eval, y_eval)

    # --- 4. Predict on Test Data ---
    print("\n--- 4. Loading Test Data and Generating Predictions ---")
    try:
        df_test = pd.read_csv(TEST_FILE)
        loan_ids = df_test['id']
        X_test = df_test.drop('id', axis=1)
    except FileNotFoundError:
        print(f"Error: Test file '{TEST_FILE}' not found. Skipping final prediction.")
        return

    # Generate predictions using the trained pipeline
    predictions = model_pipeline.predict(X_test)

    # --- 5. Display Results to Console (No File Output) ---
    results = pd.DataFrame({
        'id': loan_ids,
        'loan_status_predicted': predictions
    })

    print("\n" + "=" * 80)
    print(f"✅ Prediction Complete! Results for '{TEST_FILE}' displayed below.")
    print("Prediction counts (0=Rejected, 1=Approved):")
    print(results['loan_status_predicted'].value_counts())
run_full_pipeline()
