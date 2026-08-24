import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Imports the validated ingestion function from ingest.py
from ingest import load_and_validate_data


def perform_eda():
    # Defines a function to perform Exploratory Data Analysis

    DATA_PATH = os.path.join(
        "src",
        "data",
        "hospital_readmission_dataset (1).csv"
    )
    # Creates the path to the hospital readmission dataset

    df = load_and_validate_data(DATA_PATH)
    # Loads the dataset using the validation function from ingest.py


    print("\n" + "=" * 50)
    print("--- 1. DATASET DIMENSIONS ---")

    print(f"Total Rows (Patients): {df.shape[0]}")
    # Displays the total number of patient records

    print(f"Total Columns (Features): {df.shape[1]}")
    # Displays the total number of columns


    print("\n" + "=" * 50)
    print("--- 2. FEATURE NAMES & DATA TYPES ---")

    print(df.dtypes)
    # Displays the name and data type of every column


    print("\n" + "=" * 50)
    print("--- 3. MISSING VALUES & DUPLICATES ---")

    missing_vals = df.isnull().sum()
    # Counts missing values in every column

    if missing_vals.sum() > 0:
        print("Missing Values per Column:")
        print(missing_vals[missing_vals > 0])
    else:
        print("No missing values found.")
    # Displays missing values only if they exist

    duplicates = df.duplicated().sum()
    # Counts duplicate records

    print(f"Duplicate Records Count: {duplicates}")
    # Displays the number of duplicate records


    print("\n" + "=" * 50)
    print("--- 4. SUMMARY STATISTICS ---")

    print(df.describe())
    # Displays statistical information for numerical columns


    print("\n" + "=" * 50)
    print("--- 5. CLASS IMBALANCE ANALYSIS ---")

    if 'label' in df.columns:

        class_counts = df['label'].value_counts()
        # Counts the number of records in each readmission class

        class_percentages = (
            df['label'].value_counts(normalize=True) * 100
        )
        # Calculates the percentage of each readmission class

        print("Readmission Label Counts:")
        print(class_counts)

        print("\nReadmission Label Percentages:")
        print(class_percentages)

    else:
        print("Target column 'label' not found.")
        # Displays a message if the target column is missing


    print("\n" + "=" * 50)
    print("--- 6. GENERATING VISUALIZATIONS ---")

    sns.set_theme(style="whitegrid")
    # Sets the visual style for the graphs

    os.makedirs("reports/figures", exist_ok=True)
    # Creates the reports/figures folder if it does not exist


    # A. Correlation Matrix Heatmap

    numerical_df = df.select_dtypes(include=[np.number])
    # Selects only numerical columns

    corr_matrix = numerical_df.corr()
    # Calculates the correlation between numerical features

    plt.figure(figsize=(10, 8))

    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        linewidths=0.5
    )
    # Creates the correlation heatmap

    plt.title("Hospital Readmission Feature Correlation Matrix")
    # Adds a title to the heatmap

    plt.tight_layout()

    plt.savefig(
        "reports/figures/correlation_heatmap.png"
    )
    # Saves the heatmap as an image

    plt.close()

    print(
        "-> Saved correlation heatmap to "
        "reports/figures/correlation_heatmap.png"
    )


    # B. Scatter Plot

    if (
        'age' in df.columns
        and 'readmission_risk_score' in df.columns
        and 'label' in df.columns
    ):

        plt.figure(figsize=(8, 6))

        sns.scatterplot(
            data=df,
            x='age',
            y='readmission_risk_score',
            hue='label',
            alpha=0.7
        )
        # Shows the relationship between age and readmission risk score

        plt.title(
            "Age vs Readmission Risk Score"
        )
        # Adds a title to the scatter plot

        plt.tight_layout()

        plt.savefig(
            "reports/figures/scatter_age_risk.png"
        )
        # Saves the scatter plot as an image

        plt.close()

        print(
            "-> Saved scatter plot to "
            "reports/figures/scatter_age_risk.png"
        )


    # C. Pair Plot

    try:

        pairplot_cols = [
            'age',
            'comorbidities_count',
            'length_of_stay',
            'medications_count',
            'prev_readmissions'
        ]
        # Selects important numerical hospital features

        valid_pair_cols = [
            col for col in pairplot_cols
            if col in df.columns
        ]
        # Checks which selected columns actually exist

        if len(valid_pair_cols) > 1:

            pp = sns.pairplot(
                df[valid_pair_cols],
                diag_kind='hist',
                corner=True
            )
            # Creates pairwise relationship plots

            pp.fig.suptitle(
                "Pairwise Relationships of Hospital Features",
                y=1.02
            )
            # Adds a title to the pair plot

            pp.savefig(
                "reports/figures/pairplot_features.png"
            )
            # Saves the pair plot as an image

            plt.close()

            print(
                "-> Saved pair plot to "
                "reports/figures/pairplot_features.png"
            )

    except Exception as e:

        print(
            f"Skipping pair plot due to error: {e}"
        )
        # Continues the program if the pair plot cannot be generated


    # D. Outlier Detection using Boxplots

    plt.figure(figsize=(12, 6))

    sns.boxplot(
        data=numerical_df,
        orient="h"
    )
    # Creates horizontal boxplots for numerical features

    plt.title(
        "Outlier Identification in Hospital Dataset"
    )
    # Adds a title to the boxplot

    plt.tight_layout()

    plt.savefig(
        "reports/figures/outliers_boxplot.png"
    )
    # Saves the boxplot as an image

    plt.close()

    print(
        "-> Saved outlier boxplot to "
        "reports/figures/outliers_boxplot.png"
    )


    print(
        "\nEDA Execution Complete. "
        "Visualizations stored in 'reports/figures/'."
    )
    # Displays the completion message


if __name__ == "__main__":
    # Executes the function only when this file is run directly

    perform_eda()
    # Starts the EDA process