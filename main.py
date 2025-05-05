import eva_res as eva
import os
import json
from Deepseek import Deepseek
from GPT import GPT
import pandas as pd
from dotenv import load_dotenv
import sys

# Load environment variables from .env file
load_dotenv()

def load_test_cases(file_path):
    """ load .csv  get the input and expected output column from the file """
    df = pd.read_csv(file_path)
    # Ensure column names are correctly handled, making them case-insensitive
    df.columns = df.columns.str.strip().str.lower()
    if 'input' not in df.columns or 'expected output' not in df.columns:
        raise ValueError("CSV file must contain 'input' and 'expected output' columns (case-insensitive).")

    # Select and rename columns, drop rows with missing values in these columns
    df = df[['input', 'expected output']].dropna()
    df.columns = ['prompt', 'expected_output'] # Rename columns for the desired JSON output

    # Convert to dictionary list
    return df.to_dict(orient='records')

def main(test_mode=False, output_filename="results.json"):
    # =============== Deepseek initiation======================
    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")

    if not deepseek_api_key:
        print("Error: DEEPSEEK_API_KEY environment variable not set.")
        return

    try:
        deepseek = Deepseek(deepseek_api_key)
    except Exception as e:
        print(f"Error initializing Deepseek: {e}")
        return
    # ========================================================
    
    # =================GPT initiation===========================
    gpt_api_key = os.getenv("OPENAI_API_KEY")

    if not gpt_api_key:
        print("Error: OPENAI_API_KEY environment variable not set.")
        return

    try:
        gpt = GPT(gpt_api_key)
    except Exception as e:
        print(f"Error initializing Deepseek: {e}")
        return
    
    # =============== Load Test Cases ========================
    test_cases_file = "./test_cases.csv"
    if not os.path.exists(test_cases_file):
        print(f"Error: Test cases file not found at {test_cases_file}")
        return

    try:
        data = load_test_cases(test_cases_file)
        print(f"Loaded {len(data)} test cases from {test_cases_file}")
        if test_mode and len(data) > 0:
            print("Running in test mode, processing only the first test case.")
        elif test_mode and len(data) == 0:
             print("Running in test mode, but no test cases loaded.")
             # Output an empty results file if no data was loaded and test mode is on
             try:
                 with open(output_filename, 'w', encoding='utf-8') as f:
                     json.dump([], f, indent=2)
                 print(f"Saved empty results to {output_filename}")
             except IOError as e:
                 print(f"Error saving empty results to {output_filename}: {e}")
             return
    except ValueError as e:
        print(f"Error loading test cases: {e}")
        return
    except FileNotFoundError:
         print(f"Error: The file {test_cases_file} was not found.")
         return
    except pd.errors.EmptyDataError:
         print(f"Error: The file {test_cases_file} is empty.")
         return
    except Exception as e:
         print(f"An unexpected error occurred while loading the CSV: {e}")
         return

    results = []

    cases_to_process = data if not test_mode else data[:1]

    if not cases_to_process:
        print("No test cases to process.")
        try:
            with open(output_filename, 'w', encoding='utf-8') as f:
                json.dump([], f, indent=2)
            print(f"Saved empty results to {output_filename}")
        except IOError as e:
             print(f"Error saving empty results to {output_filename}: {e}")
        return
    # =======================================================================
    
    # =================Process each test case with both models==================
    for case in cases_to_process:
        prompt = case['prompt']
        expected_output = case['expected_output']

        # Print the prompt being processed
        print(f"\n--- Processing Prompt ---")
        print(f"Prompt: {prompt[:200]}{'...' if len(prompt) > 200 else ''}") # Truncate long prompts for printing


        # =========================Get actual output from Deepseek model=================================
        actual_output_deepseek = "Error generating response." # Default error message
        try:
            actual_output_deepseek = deepseek.generate_response(prompt)
            # Print the received response (truncated)
            print(f"Response received: {actual_output_deepseek[:300]}{'...' if len(actual_output_deepseek) > 300 else ''}") # Truncate long responses for printing
        except Exception as e:
            actual_output_deepseek = f"Error generating response: {e}"
            print(f"Error getting Deepseek response: {e}")
        
        # ============================Get actual output from GPT model=================================
        actual_output_gpt = "Error generating response." # Default error message
        try:
            actual_output_gpt = gpt.generate_response(prompt)
            # Print the received response (truncated)
            print(f"Response received: {actual_output_gpt[:300]}{'...' if len(actual_output_gpt) > 300 else ''}") # Truncate long responses for printing
        except Exception as e:
            actual_output_gpt = f"Error generating response: {e}"
            print(f"Error getting GPT response: {e}")

        # ===========================Append results in the desired format==============================
        results.append({
            "prompt": prompt,
            "expected_output": expected_output,
            "actual_output_deepseek": actual_output_deepseek,
            "actual_output_gpt": actual_output_gpt
        })

    # --- Save the results to a JSON file ---
    try:
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n--- Results Saved ---")
        print(f"Results successfully saved to {output_filename}")
    except IOError as e:
        print(f"\n--- Error Saving Results ---")
        print(f"Error saving results to {output_filename}: {e}")
    # --- End Save ---


if __name__ == "__main__":
    run_test_mode = '--test' in sys.argv or '-t' in sys.argv
    main(test_mode=run_test_mode, output_filename="test_results.json")