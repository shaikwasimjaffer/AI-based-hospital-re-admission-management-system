import os
# Imports Python's built-in Operating System module

import pandas as pd
# Imports the Pandas library


def load_and_validate_data(file_path: str) -> pd.DataFrame:
    # Defines a reusable function that loads and validates the dataset

    if not os.path.exists(file_path):
        # Checks whether the given file exists

        raise FileNotFoundError(
            f"Critical Error: Targeted data footprint not discovered at {file_path}"
        )
        # Stops execution if the file is not found

    print(f"Executing secure data extraction from: {file_path}")
    # Displays a message showing that data extraction has started

    df = pd.read_csv(file_path)
    # Reads the CSV file and stores it as a Pandas DataFrame

    required_columns = [
        'patient_id',
        'admission_date',
        'season',
        'age',
        'gender',
        'region',
        'primary_diagnosis',
        'comorbidities_count',
        'length_of_stay',
        'treatment_type',
        'medications_count',
        'followup_visits_last_year',
        'prev_readmissions',
        'insurance_type',
        'discharge_disposition',
        'readmission_risk_score',
        'label'
    ]
    # Defines the columns required in the hospital readmission dataset

    missing_cols = [
        col for col in required_columns
        if col not in df.columns
    ]
    # Finds the required columns that are missing from the dataset

    if missing_cols:
        # Checks whether the missing columns list is not empty

        raise ValueError(
            f"Schema Validation Failure: Missing essential columns: {missing_cols}"
        )
        # Stops processing if required columns are missing

    print(
        f"Data ingestion resolved successfully. "
        f"Dimensions captured: {df.shape[0]} samples, {df.shape[1]} metrics."
    )
    # Displays the number of rows and columns in the dataset

    return df
    # Returns the validated DataFrame to the caller


if __name__ == "__main__":
    # Executes this section only when the file is run directly

    DATA_PATH = os.path.join(
        "src",
        "data",
        "hospital_readmission_dataset (1).csv"
    )
    # Creates the path to the raw hospital readmission dataset

    try:
        # Starts exception handling

        raw_data = load_and_validate_data(DATA_PATH)
        # Calls the function to load and validate the dataset

    except Exception as e:
        # Catches errors that occur during the ingestion process

        print(f"Ingestion lifecycle termination: {str(e)}")
        # Displays the error message