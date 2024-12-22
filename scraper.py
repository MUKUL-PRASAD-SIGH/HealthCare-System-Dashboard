import pandas as pd

def scrape_medical_record(filepath):
    # Simplified example: Assume records are CSVs
    data = pd.read_csv(filepath)
    important_data = data[['Diagnosis', 'Medications']]  # Example columns
    return important_data.to_dict()
