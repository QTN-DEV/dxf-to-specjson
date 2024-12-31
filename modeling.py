import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Models
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
from catboost import CatBoostClassifier
train_df= pd.read_csv("./train_panels_clean.csv")
test_df= pd.read_csv("./test_panels_clean.csv")
dev_df= pd.read_csv("./dev_panels_clean.csv")
# ================== BEGIN LOGGING SETUP ===================
log_file_path = "log_experiment.txt"

# Ensure confusion-image directory exists
if not os.path.exists("confussion_image"):
    os.makedirs("confussion_image")

# Open a file in write mode for logging
# (Use 'a' instead of 'w' if you want to append to the file)
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# Example of generating and saving a confusion matrix with vertical x-axis labels
def plot_and_save_confusion_matrix(cm, display_labels, title, save_path):
    """
    cm: confusion matrix (2D array)
    display_labels: list of label names for x/y axis
    title: str, figure title
    save_path: str, path to save the figure
    """
    # Create a bigger figure (width=12, height=10, for example).
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Optionally set a smaller global font size
    plt.rcParams.update({'font.size': 9})
    
    # Plot confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    
    # We can pass 'values_format' to control how numbers are displayed 
    # e.g. as integers ('d'), or with decimals, etc.
    disp.plot(ax=ax, cmap=plt.cm.Blues, values_format='d')
    
    # Rotate x-axis labels vertically
    plt.setp(ax.get_xticklabels(), rotation=90, ha="center")

    # Title
    ax.set_title(title)
    
    # Force a nicely-fitted layout so nothing is cut off
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(save_path)
    
    # Close the figure so it doesn't pop up
    plt.close()

# =========================
# Inside your code snippet:
# (Once you have confusion matrix, e.g. mlp_test_cm, and predicted classes)
# =========================

# Example usage for the MLP Test set:
# Define label-to-integer mapping
label_to_int = {
    "Lfront": 0, "Lhood": 1, "Rfront": 2, "Rhood": 3, "back": 4, "front": 5,
    "front_left": 6, "front_right": 7, "left": 8, "left_0": 9,
    "lfsleeve": 10, "rfsleeve": 11, "right": 12, "right_0": 13,
    "skirt_front": 14, "top_back": 15, "top_front": 16, "up_back": 17,
    "up_front": 18, "wb_front": 19
}
# Reverse mapping for decoding predictions
int_to_label = {v: k for k, v in label_to_int.items()}



with open(log_file_path, "w") as log_file:

    def sp(*args):
        """
        sp = 'save and print'
        Prints the given arguments to console AND writes them to the log file.
        """
        print(*args)
        print(*args, file=log_file)

    # ================ 1. Fill NaNs ================
    train_df = train_df.fillna(0)
    dev_df = dev_df.fillna(0)
    test_df = test_df.fillna(0)

    # ================ 2. Define features & label ================
    FEATURES = [
        'vertices_x0', 'vertices_y0', 'curves_x0', 'curves_y0', 'vertices_x1',
        'vertices_y1', 'curves_x1', 'curves_y1', 'vertices_x2', 'vertices_y2',
        'curves_x2', 'curves_y2', 'vertices_x3', 'vertices_y3', 'curves_x3',
        'curves_y3', 'vertices_x4', 'vertices_y4', 'curves_x4', 'curves_y4',
        'vertices_x5', 'vertices_y5', 'curves_x5', 'curves_y5', 'vertices_x6',
        'vertices_y6', 'curves_x6', 'curves_y6', 'vertices_x7', 'vertices_y7',
        'curves_x7', 'curves_y7', 'vertices_x8', 'vertices_y8', 'curves_x8',
        'curves_y8', 'vertices_x9', 'vertices_y9', 'curves_x9', 'curves_y9'
    ]
    LABEL = "sewing_name"

    X_train = train_df[FEATURES]
    y_train = train_df[LABEL]

    X_dev = dev_df[FEATURES]
    y_dev = dev_df[LABEL]

    X_test = test_df[FEATURES]
    y_test = test_df[LABEL]

    # Encode labels using label_to_int
    y_train_encoded = y_train.map(label_to_int)
    y_dev_encoded = y_dev.map(label_to_int)
    y_test_encoded = y_test.map(label_to_int)


    # ===========================================================
    # ================ 3. MLPClassifier ================
    # ===========================================================
    mlp = MLPClassifier(random_state=42)

    mlp_param_grid = {
        'learning_rate_init': [ 0.0005,0.001],
        'max_iter': [300, 500]
    }

    mlp_gridsearch = GridSearchCV(
        estimator=mlp,
        param_grid=mlp_param_grid,
        scoring='accuracy',
        cv=5,
        n_jobs=-1
    )
    mlp_gridsearch.fit(X_train, y_train)

    sp("===== MLP Best Params =====")
    sp(mlp_gridsearch.best_params_)
    sp(f"MLP Best CV Accuracy: {mlp_gridsearch.best_score_:.4f}")

    # Evaluate on Dev set
    mlp_best = mlp_gridsearch.best_estimator_
    mlp_dev_preds = mlp_best.predict(X_dev)

    sp("\n===== MLP - Classification Report (Dev) =====")
    mlp_dev_report = classification_report(y_dev, mlp_dev_preds)
    sp(mlp_dev_report)

    # Plot & save confusion matrix (Dev)

    mlp_dev_cm = confusion_matrix(y_dev, mlp_dev_preds)
    plot_and_save_confusion_matrix(
        cm=mlp_dev_cm,
        display_labels=mlp_best.classes_,
        title="MLP - Confusion Matrix (Dev)",
        save_path="confussion_image/MLP_confusion_matrix_dev.png"
    )

    # Evaluate on Test set
    mlp_test_preds = mlp_best.predict(X_test)
    sp("\n===== MLP - Classification Report (Test) =====")
    mlp_test_report = classification_report(y_test, mlp_test_preds, digits=4)
    sp(mlp_test_report)

    # Plot & save confusion matrix (Test)

    mlp_test_cm = confusion_matrix(y_test, mlp_test_preds)
    plot_and_save_confusion_matrix(
        cm=mlp_test_cm,
        display_labels=mlp_best.classes_,
        title="MLP - Confusion Matrix (Test)",
        save_path="confussion_image/MLP_confusion_matrix_test.png"
    )

    # ===========================================================
    # ================ XGBoostClassifier ================
    # ===========================================================
    xgb_model = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')

    xgb_param_grid = {
        'max_depth': [12, 7],
        'learning_rate': [ 0.01,0.001],
        'n_estimators': [200,100]
    }

    xgb_gridsearch = GridSearchCV(
        estimator=xgb_model,
        param_grid=xgb_param_grid,
        scoring='accuracy',
        cv=5,
        n_jobs=-1
    )
    xgb_gridsearch.fit(X_train, y_train_encoded)

    sp("\n===== XGBoost Best Params =====")
    sp(xgb_gridsearch.best_params_)
    sp(f"XGBoost Best CV Accuracy: {xgb_gridsearch.best_score_:.4f}")

    # Evaluate on Dev set
    xgb_best = xgb_gridsearch.best_estimator_
    xgb_dev_preds = xgb_best.predict(X_dev)
    # xgb_dev_preds = xgb_dev_preds.map(int_to_label)
    xgb_dev_preds = [int_to_label[pred] for pred in xgb_dev_preds]

    sp("\n===== XGBoost - Classification Report (Dev) =====")
    xgb_dev_report = classification_report(y_dev, xgb_dev_preds)
    sp(xgb_dev_report)

    # Plot & save confusion matrix (Dev)
    xgb_dev_cm = confusion_matrix(y_dev, xgb_dev_preds)
    plot_and_save_confusion_matrix(
        cm=xgb_dev_cm,
        display_labels=xgb_best.classes_,
        title="XGBoost - Confusion Matrix (Dev)",
        save_path="confussion_image/XGBoost_confusion_matrix_dev.png"
    )

    # Evaluate on Test set
    xgb_test_preds = xgb_best.predict(X_test)
    # xgb_test_preds = xgb_test_preds.map(int_to_label)
    xgb_test_preds = [int_to_label[pred] for pred in xgb_test_preds]

    xgb_test_cm = confusion_matrix(y_test, xgb_test_preds)
    plot_and_save_confusion_matrix(
        cm=xgb_test_cm,
        display_labels=xgb_best.classes_,
        title="XGBoost - Confusion Matrix (Test)",
        save_path="confussion_image/XGBoost_confusion_matrix_test.png"
    )

    sp("\n===== XGBoost - Classification Report (Test) =====")
    xgb_test_report = classification_report(y_test, xgb_test_preds, digits=4)
    sp(xgb_test_report)


    # ===========================================================
    # ================ CatBoostClassifier ================
    # ===========================================================
    cat_model = CatBoostClassifier(verbose=False, random_state=42)

    cat_param_grid = {
        'depth': [3, 5],
        'learning_rate': [ 0.01],
        'iterations': [100, 200]
    }

    cat_gridsearch = GridSearchCV(
        estimator=cat_model,
        param_grid=cat_param_grid,
        scoring='accuracy',
        cv=5,
        n_jobs=-1
    )
    cat_gridsearch.fit(X_train, y_train)

    sp("\n===== CatBoost Best Params =====")
    sp(cat_gridsearch.best_params_)
    sp(f"CatBoost Best CV Accuracy: {cat_gridsearch.best_score_:.4f}")

    # Evaluate on Dev set
    cat_best = cat_gridsearch.best_estimator_
    cat_dev_preds = cat_best.predict(X_dev)

    sp("\n===== CatBoost - Classification Report (Dev) =====")
    cat_dev_report = classification_report(y_dev, cat_dev_preds)
    sp(cat_dev_report)

    # Plot & save confusion matrix (Dev)
    cat_dev_preds = cat_best.predict(X_dev)
    cat_dev_cm = confusion_matrix(y_dev, cat_dev_preds)
    plot_and_save_confusion_matrix(
        cm=xgb_dev_cm,
        display_labels=xgb_best.classes_,
        title="CatBoost - Confusion Matrix (Dev)",
        save_path="confussion_image/CatBoost_confusion_matrix_dev.png"
    )

    # Evaluate on Test set
    cat_test_preds = cat_best.predict(X_test)
    sp("\n===== CatBoost - Classification Report (Test) =====")
    cat_test_report = classification_report(y_test, cat_test_preds, digits=4)
    sp(cat_test_report)

    cat_dev_preds = cat_best.predict(X_test)
    cat_dev_cm = confusion_matrix(y_test, cat_test_preds)
    plot_and_save_confusion_matrix(
        cm=cat_dev_cm,
        display_labels=cat_best.classes_,
        title="CatBoost - Confusion Matrix (Test)",
        save_path="confussion_image/CatBoost_confusion_matrix_test.png"
    )
    # Plot & save confusion matrix (Test)
    # cat_test_cm = confusion_matrix(y_test, cat_test_preds)
    # disp_test = ConfusionMatrixDisplay(confusion_matrix=cat_test_cm, display_labels=cat_best.classes_)
    # disp_test.plot(cmap=plt.cm.Blues)
    # plt.title("CatBoost - Confusion Matrix (Test)")
    # plt.savefig("confussion_image/CatBoost_confusion_matrix_test.png")
    # plt.close()
