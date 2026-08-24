import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression as SklearnLinearRegression


# -------------------------------------------------------------
# ADD SRC DIRECTORY TO PYTHON PATH
# -------------------------------------------------------------

SRC_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

sys.path.insert(
    0,
    SRC_PATH
)

# Now Python can find src/data/ingest.py
from data.ingest import load_and_validate_data


# -------------------------------------------------------------
# 1. COMPUTE COST FUNCTION
# -------------------------------------------------------------

def compute_cost(X, y, w):
    # Calculates the Mean Squared Error cost

    m = len(y)
    # Stores the number of training records

    predictions = np.dot(X, w)
    # Calculates predictions using the current weights

    errors = predictions - y
    # Calculates the difference between predicted and actual values

    cost = (1 / (2 * m)) * np.sum(errors ** 2)
    # Calculates the cost value

    return cost
    # Returns the cost


# -------------------------------------------------------------
# 2. GRADIENT DESCENT
# -------------------------------------------------------------

def gradient_descent(X, y, w, alpha, num_iters):
    # Trains Linear Regression using Gradient Descent

    m = len(y)
    # Stores the number of training records

    cost_history = []
    # Stores the cost after every iteration

    for i in range(num_iters):

        predictions = np.dot(X, w)
        # Calculates predictions

        errors = predictions - y
        # Calculates prediction errors

        gradient = (1 / m) * np.dot(
            X.T,
            errors
        )
        # Calculates the gradient for every weight

        w = w - alpha * gradient
        # Updates the model weights

        cost = compute_cost(
            X,
            y,
            w
        )
        # Calculates the new cost

        cost_history.append(cost)
        # Stores the cost for this iteration

    return w, cost_history
    # Returns final weights and cost history


# -------------------------------------------------------------
# 3. MAIN GRADIENT DESCENT EXPERIMENT
# -------------------------------------------------------------

def run_gradient_descent_experiment():

    print("\n" + "=" * 60)
    print("--- HOSPITAL READMISSION GRADIENT DESCENT ---")
    print("=" * 60)


    # ---------------------------------------------------------
    # 4. LOAD DATA
    # ---------------------------------------------------------

    PROJECT_ROOT = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            ".."
        )
    )
    # Finds the main project folder

    DATA_PATH = os.path.join(
        PROJECT_ROOT,
        "src",
        "data",
        "hospital_readmission_dataset (1).csv"
    )
    # Creates the complete path to the dataset

    df = load_and_validate_data(DATA_PATH)
    # Loads and validates the dataset using ingest.py


    # ---------------------------------------------------------
    # 5. SELECT FEATURES AND TARGET
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
    # Selects readmission risk score as the target


    print(
        f"Total records: {df.shape[0]}"
    )

    print(
        f"Input features: {len(feature_cols)}"
    )

    print(
        f"Target: {target_col}"
    )


    # ---------------------------------------------------------
    # 6. REMOVE MISSING VALUES
    # ---------------------------------------------------------

    df_clean = df.dropna(
        subset=feature_cols + [target_col]
    ).copy()
    # Removes rows containing missing values


    print(
        f"Valid records after cleaning: "
        f"{df_clean.shape[0]}"
    )


    # ---------------------------------------------------------
    # 7. PREPARE X AND y
    # ---------------------------------------------------------

    X_raw = df_clean[
        feature_cols
    ].values
    # Stores the input features

    y_raw = df_clean[
        target_col
    ].values.reshape(-1, 1)
    # Stores the target values


    # ---------------------------------------------------------
    # 8. SPLIT DATA INTO TRAINING AND TESTING
    # ---------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X_raw,
        y_raw,
        test_size=0.20,
        random_state=42
    )
    # Uses 80% of the data for training
    # and 20% for testing


    print(
        f"Training records: {X_train.shape[0]}"
    )

    print(
        f"Testing records: {X_test.shape[0]}"
    )


    # ---------------------------------------------------------
    # 9. STANDARDIZE FEATURES AND TARGET
    # ---------------------------------------------------------

    scaler_x = StandardScaler()
    # Creates a scaler for input features

    scaler_y = StandardScaler()
    # Creates a scaler for the target


    X_train_scaled = scaler_x.fit_transform(
        X_train
    )
    # Learns feature scaling from training data

    X_test_scaled = scaler_x.transform(
        X_test
    )
    # Applies the same scaling to test data

    y_train_scaled = scaler_y.fit_transform(
        y_train
    )
    # Scales the training target


    # ---------------------------------------------------------
    # 10. CREATE DESIGN MATRIX
    # ---------------------------------------------------------

    X_train_design = np.hstack(
        [
            np.ones(
                (X_train_scaled.shape[0], 1)
            ),
            X_train_scaled
        ]
    )
    # Adds a column of 1s for the intercept


    print(
        f"Design matrix shape: "
        f"{X_train_design.shape}"
    )


    # ---------------------------------------------------------
    # 11. EXPERIMENT WITH LEARNING RATES
    # ---------------------------------------------------------

    learning_rates = [
        0.001,
        0.01,
        0.1,
        0.5
    ]
    # Defines different learning rates

    num_iterations = 1000
    # Defines the number of training iterations


    REPORTS_PATH = os.path.join(
        PROJECT_ROOT,
        "reports"
    )

    FIGURES_PATH = os.path.join(
        REPORTS_PATH,
        "figures"
    )

    os.makedirs(
        FIGURES_PATH,
        exist_ok=True
    )
    # Creates the reports/figures folder


    results = {}
    # Stores results for each learning rate


    plt.figure(
        figsize=(10, 6)
    )


    for alpha in learning_rates:

        print(
            f"Training with learning rate: {alpha}"
        )


        w_init = np.zeros(
            (X_train_design.shape[1], 1)
        )
        # Starts all weights at zero


        w_opt, cost_history = gradient_descent(
            X_train_design,
            y_train_scaled,
            w_init,
            alpha,
            num_iterations
        )
        # Trains the model using Gradient Descent


        results[alpha] = {
            'weights': w_opt,
            'history': cost_history
        }
        # Stores the weights and cost history


        plt.plot(
            cost_history,
            label=f"Learning Rate = {alpha}"
        )
        # Adds the cost history to the graph


    plt.xlabel(
        "Iterations"
    )

    plt.ylabel(
        "Cost Function"
    )

    plt.title(
        "Gradient Descent Learning Rate Comparison"
    )

    plt.legend()

    plt.grid(True)

    plt.tight_layout()


    learning_rate_graph = os.path.join(
        FIGURES_PATH,
        "gd_learning_rates_comparison.png"
    )

    plt.savefig(
        learning_rate_graph
    )
    # Saves the graph

    plt.close()


    print(
        "\n-> Saved learning rate graph to "
        f"{learning_rate_graph}"
    )


    # ---------------------------------------------------------
    # 12. SELECT BEST LEARNING RATE
    # ---------------------------------------------------------

    best_alpha = 0.1
    # Selects the learning rate used for final training


    final_w = results[
        best_alpha
    ]['weights']
    # Gets the final weights


    print("\n" + "=" * 60)
    print(
        f"--- CUSTOM GRADIENT DESCENT PARAMETERS "
        f"(Learning Rate = {best_alpha}) ---"
    )
    print("=" * 60)


    print(
        f"Intercept (w0): "
        f"{final_w[0, 0]:.6f}"
    )


    for i, feature in enumerate(feature_cols):

        print(
            f"Coefficient for {feature}: "
            f"{final_w[i + 1, 0]:.6f}"
        )


    # ---------------------------------------------------------
    # 13. CREATE TEST DESIGN MATRIX
    # ---------------------------------------------------------

    X_test_design = np.hstack(
        [
            np.ones(
                (X_test_scaled.shape[0], 1)
            ),
            X_test_scaled
        ]
    )
    # Adds intercept column to test data


    # ---------------------------------------------------------
    # 14. MAKE TEST PREDICTIONS
    # ---------------------------------------------------------

    y_test_pred_scaled = np.dot(
        X_test_design,
        final_w
    )
    # Calculates predictions in scaled form


    y_test_pred = scaler_y.inverse_transform(
        y_test_pred_scaled
    )
    # Converts predictions back to original risk-score scale


    # ---------------------------------------------------------
    # 15. CALCULATE TEST MSE
    # ---------------------------------------------------------

    test_mse = np.mean(
        (y_test_pred - y_test) ** 2
    )
    # Calculates Mean Squared Error


    print("\n" + "=" * 60)
    print("--- TEST PERFORMANCE ---")
    print("=" * 60)


    print(
        f"Test Mean Squared Error: "
        f"{test_mse:.6f}"
    )


    # ---------------------------------------------------------
    # 16. SAMPLE PREDICTIONS
    # ---------------------------------------------------------

    results_df = pd.DataFrame({
        "Actual Risk Score":
            y_test.flatten()[:10],

        "Predicted Risk Score":
            y_test_pred.flatten()[:10]
    })
    # Creates a table containing sample predictions


    print("\n--- SAMPLE TEST PREDICTIONS ---")

    print(
        results_df
    )


    # ---------------------------------------------------------
    # 17. SAVE PREDICTIONS
    # ---------------------------------------------------------

    predictions_path = os.path.join(
        REPORTS_PATH,
        "gradient_descent_predictions.csv"
    )

    results_df.to_csv(
        predictions_path,
        index=False
    )
    # Saves predictions to CSV


    print(
        "\n-> Saved predictions to "
        f"{predictions_path}"
    )


    # ---------------------------------------------------------
    # 18. COMPARE WITH SCIKIT-LEARN
    # ---------------------------------------------------------

    sklearn_model = SklearnLinearRegression()
    # Creates Scikit-learn Linear Regression model


    sklearn_model.fit(
        X_train_scaled,
        y_train_scaled
    )
    # Trains Scikit-learn model


    print("\n" + "=" * 60)
    print("--- SCIKIT-LEARN COMPARISON ---")
    print("=" * 60)


    print(
        f"Scikit-learn Intercept: "
        f"{sklearn_model.intercept_[0]:.6f}"
    )


    for i, feature in enumerate(feature_cols):

        print(
            f"Scikit-learn Coefficient for "
            f"{feature}: "
            f"{sklearn_model.coef_[0, i]:.6f}"
        )


    # ---------------------------------------------------------
    # 19. SAVE GRADIENT DESCENT WEIGHTS
    # ---------------------------------------------------------

    weights = {
        "intercept": final_w[0, 0]
    }


    for i, feature in enumerate(feature_cols):

        weights[feature] = final_w[
            i + 1,
            0
        ]
        # Stores each learned coefficient


    weights_df = pd.DataFrame(
        [weights]
    )


    weights_path = os.path.join(
        REPORTS_PATH,
        "gradient_descent_weights.csv"
    )

    weights_df.to_csv(
        weights_path,
        index=False
    )
    # Saves learned model weights


    print(
        "-> Saved model weights to "
        f"{weights_path}"
    )


    print("\n" + "=" * 60)
    print("--- GRADIENT DESCENT TRAINING COMPLETED ---")
    print("=" * 60)


# -------------------------------------------------------------
# MAIN PROGRAM
# -------------------------------------------------------------

if __name__ == "__main__":
    # Runs the training when this file is executed directly

    run_gradient_descent_experiment()
    # Starts Gradient Descent training