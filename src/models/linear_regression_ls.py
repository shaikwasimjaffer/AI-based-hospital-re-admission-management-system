import os
import sys
import numpy as np
import pandas as pd


# -------------------------------------------------------------
# 1. SET PROJECT PATH
# -------------------------------------------------------------

# Gets the path of the current file
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Gets the src folder
SRC_DIR = os.path.abspath(
    os.path.join(CURRENT_DIR, "..")
)

# Adds src to Python's search path
sys.path.insert(0, SRC_DIR)

# Imports the validated ingestion function
from data.ingest import load_and_validate_data


# -------------------------------------------------------------
# 2. TRAIN LINEAR REGRESSION
# -------------------------------------------------------------

def train_linear_regression_ls():
    # Defines a function to train Linear Regression
    # using the Standard Least Squares Method


    # ---------------------------------------------------------
    # 3. LOAD DATA
    # ---------------------------------------------------------

    DATA_PATH = os.path.join(
        SRC_DIR,
        "data",
        "hospital_readmission_dataset (1).csv"
    )
    # Creates the path to the hospital readmission dataset

    df = load_and_validate_data(DATA_PATH)
    # Loads the dataset using the validation function


    # ---------------------------------------------------------
    # 4. SELECT INPUT FEATURES AND TARGET
    # ---------------------------------------------------------

    feature_cols = [
        'age',
        'comorbidities_count',
        'length_of_stay',
        'medications_count',
        'followup_visits_last_year',
        'prev_readmissions'
    ]
    # Selects six numerical patient features

    target_col = 'readmission_risk_score'
    # Selects readmission risk score as the target variable


    # ---------------------------------------------------------
    # 5. REMOVE MISSING VALUES
    # ---------------------------------------------------------

    df_clean = df.dropna(
        subset=feature_cols + [target_col]
    ).copy()
    # Removes rows containing missing values
    # in the selected features or target


    print("\n" + "=" * 60)
    print("--- LINEAR REGRESSION DATA ---")
    print("=" * 60)

    print(
        f"Total valid patient records: {df_clean.shape[0]}"
    )

    print(
        f"Number of input features: {len(feature_cols)}"
    )

    print(
        f"Target variable: {target_col}"
    )


    # ---------------------------------------------------------
    # 6. PREPARE INPUT X AND OUTPUT y
    # ---------------------------------------------------------

    X_raw = df_clean[feature_cols].values
    # Stores the six input features as a NumPy array

    y = df_clean[target_col].values.reshape(-1, 1)
    # Stores the readmission risk score as the target

    N = X_raw.shape[0]
    # Stores the number of patient records


    # ---------------------------------------------------------
    # 7. CREATE DESIGN MATRIX
    # ---------------------------------------------------------

    X_design = np.hstack(
        [
            np.ones((N, 1)),
            X_raw
        ]
    )
    # Adds a column of 1s for the intercept
    #
    # Design matrix:
    # [1, age, comorbidities_count,
    #  length_of_stay, medications_count,
    #  followup_visits_last_year, prev_readmissions]


    print(
        f"Design matrix shape: {X_design.shape}"
    )


    # ---------------------------------------------------------
    # 8. STANDARD LEAST SQUARES METHOD
    # ---------------------------------------------------------

    # Normal Equation:
    #
    # w = (X^T X)^-1 X^T y

    XT_X = np.dot(
        X_design.T,
        X_design
    )
    # Calculates X transpose multiplied by X

    XT_y = np.dot(
        X_design.T,
        y
    )
    # Calculates X transpose multiplied by y


    try:

        XT_X_inv = np.linalg.inv(XT_X)
        # Calculates the inverse of X transpose X

    except np.linalg.LinAlgError:

        print(
            "Matrix is not invertible. "
            "Using pseudo-inverse."
        )

        XT_X_inv = np.linalg.pinv(XT_X)
        # Uses pseudo-inverse if the matrix is singular


    w_optimal = np.dot(
        XT_X_inv,
        XT_y
    )
    # Calculates the optimal model weights


    # ---------------------------------------------------------
    # 9. DISPLAY MODEL PARAMETERS
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("--- OPTIMAL MODEL PARAMETERS ---")
    print("=" * 60)

    print(
        f"Intercept (w0): "
        f"{w_optimal[0, 0]:.6f}"
    )

    for i, feature in enumerate(feature_cols):

        print(
            f"Coefficient for {feature}: "
            f"{w_optimal[i + 1, 0]:.6f}"
        )


    # ---------------------------------------------------------
    # 10. MAKE PREDICTIONS
    # ---------------------------------------------------------

    y_pred = np.dot(
        X_design,
        w_optimal
    )
    # Calculates predicted readmission risk scores


    # ---------------------------------------------------------
    # 11. CALCULATE SUM OF SQUARED ERROR
    # ---------------------------------------------------------

    E_w = 0.5 * np.sum(
        (y_pred - y) ** 2
    )
    # Calculates the Sum of Squared Errors


    print("\n" + "=" * 60)
    print("--- MODEL ERROR ---")
    print("=" * 60)

    print(
        f"Sum of Squared Errors (E_w): "
        f"{E_w:.6f}"
    )


    # ---------------------------------------------------------
    # 12. CALCULATE MEAN SQUARED ERROR
    # ---------------------------------------------------------

    mse = np.mean(
        (y_pred - y) ** 2
    )
    # Calculates the average squared prediction error

    print(
        f"Mean Squared Error (MSE): "
        f"{mse:.6f}"
    )


    # ---------------------------------------------------------
    # 13. CALCULATE R-SQUARED
    # ---------------------------------------------------------

    ss_total = np.sum(
        (y - np.mean(y)) ** 2
    )
    # Calculates total variation in the target

    ss_residual = np.sum(
        (y - y_pred) ** 2
    )
    # Calculates remaining prediction error

    r_squared = 1 - (
        ss_residual / ss_total
    )
    # Calculates R-squared

    print(
        f"R-squared (R²): "
        f"{r_squared:.6f}"
    )


    # ---------------------------------------------------------
    # 14. DISPLAY SAMPLE PREDICTIONS
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("--- SAMPLE PREDICTIONS ---")
    print("=" * 60)

    results = pd.DataFrame({
        "Actual Risk Score": y.flatten()[:10],
        "Predicted Risk Score": y_pred.flatten()[:10]
    })

    print(results)


    # ---------------------------------------------------------
    # 15. CREATE REPORTS FOLDER
    # ---------------------------------------------------------

    PROJECT_ROOT = os.path.abspath(
        os.path.join(SRC_DIR, "..")
    )

    REPORTS_DIR = os.path.join(
        PROJECT_ROOT,
        "reports"
    )

    os.makedirs(
        REPORTS_DIR,
        exist_ok=True
    )
    # Creates the reports folder if it does not exist


    # ---------------------------------------------------------
    # 16. SAVE PREDICTIONS
    # ---------------------------------------------------------

    predictions_path = os.path.join(
        REPORTS_DIR,
        "linear_regression_predictions.csv"
    )

    results.to_csv(
        predictions_path,
        index=False
    )
    # Saves sample predictions

    print(
        "\n-> Saved predictions to "
        f"{predictions_path}"
    )


    # ---------------------------------------------------------
    # 17. SAVE MODEL WEIGHTS
    # ---------------------------------------------------------

    weights = {
        "intercept": w_optimal[0, 0]
    }

    for i, feature in enumerate(feature_cols):

        weights[feature] = w_optimal[i + 1, 0]
        # Stores the learned coefficient for each feature


    weights_df = pd.DataFrame(
        [weights]
    )

    weights_path = os.path.join(
        REPORTS_DIR,
        "linear_regression_weights.csv"
    )

    weights_df.to_csv(
        weights_path,
        index=False
    )
    # Saves the learned model parameters

    print(
        "-> Saved model weights to "
        f"{weights_path}"
    )


    # ---------------------------------------------------------
    # 18. COMPLETION MESSAGE
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("Linear Regression training completed successfully.")
    print("=" * 60)


# -------------------------------------------------------------
# MAIN PROGRAM
# -------------------------------------------------------------

if __name__ == "__main__":
    # Executes the function when this file is run directly

    train_linear_regression_ls()
    # Starts the Linear Regression training