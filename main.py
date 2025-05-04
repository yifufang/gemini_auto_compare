import eva_res as eva
import os
import json
from Deepseek import Deepseek
import pandas as pd

def load_test_cases(file_path):
    """ load .csv  get the input and expected output column from the file """
    df = pd.read_csv(file_path)
    if 'Input' not in df.columns or 'Expected Output' not in df.columns:
        raise ValueError("CSV file must contain 'input' and 'expected_output' columns.")
    df = df[['Input', 'Expected Output']].dropna()
    df.columns = ['Input', 'Expected Output']
    return df.to_dict(orient='records')

def main():
    # Load environment variables
    api_key = os.getenv("DEEPSEEK_API_KEY")
    deepseek = Deepseek(api_key)
    data = load_test_cases("./test_cases.csv")
    print("Loaded test cases:", data)
    print(len(data), "test cases loaded")

if __name__ == "__main__":
    main()