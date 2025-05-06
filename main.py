import eva_res as eva # Import the eva_res module
import os
import json
from Deepseek import Deepseek
from GPT import GPT
import pandas as pd
from dotenv import load_dotenv
import sys
import time # Import time for delays if needed during evaluation
from typing import List, Dict, Any # Import typing hints

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

def perform_evaluation(results: List[Dict[str, Any]]):
    """ Performs evaluation on the results list using the evaluation model from eva_res.py. """
    print("\n--- Starting Automated Response Evaluation ---")

    if not results:
        print("No results to evaluate.")
        return

    # Initialize the evaluation model using logic from eva_res.py
    # We assume eva_res.py handles its own API key loading and genai configuration
    # However, we still need to check the GOOGLE_API_KEY in main for clarity
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        print("Error: GOOGLE_API_KEY environment variable not set for evaluation.")
        print("Evaluation cannot proceed.")
        return # Exit evaluation if API key is missing

    try:
        # Use the GenerativeModel class directly from the imported eva_res module's genai
        # Using gemini-1.5-pro-latest as specified in eva_res.py
        evaluator_model = eva.genai.GenerativeModel(model_name="gemini-1.5-pro-latest")
        print(f"Using evaluation model: {'gemini-1.5-pro-latest'}")
    except Exception as e:
        print(f"Error initializing evaluation model: {e}")
        print("Please check the model name and your GOOGLE_API_KEY.")
        return # Exit evaluation if model initialization fails

    for i, test_case in enumerate(results):
        prompt = test_case.get("prompt")
        expected = test_case.get("expected_output")
        actual_chatgpt = test_case.get("actual_output_gpt")
        actual_deepseek = test_case.get("actual_output_deepseek")

        # Ensure all necessary data is present for this test case before evaluating
        if not all([prompt, expected, actual_chatgpt is not None, actual_deepseek is not None]):
            print(f"Skipping evaluation for case {i+1} due to missing data: {test_case}")
            continue

        print(f"\n--- Evaluating Case {i + 1}/{len(results)} ---")
        print(f"Prompt: {prompt[:80]}...")  # Print truncated prompt

        # Evaluate GPT's response using the function from eva_res.py
        print("  Evaluating GPT response...")
        gpt_evaluation_raw = eva.get_evaluation_result(
            evaluator_model, prompt, expected, actual_chatgpt
        )
        print(f"    GPT Evaluation Result: {gpt_evaluation_raw}")

        # Evaluate DeepSeek's response using the function from eva_res.py
        print("  Evaluating DeepSeek response...")
        deepseek_evaluation_raw = eva.get_evaluation_result(
            evaluator_model, prompt, expected, actual_deepseek
        )
        print(f"    DeepSeek Evaluation Result: {deepseek_evaluation_raw}")

        # Store the evaluation results within the existing test case structure in the results list
        results[i]['evaluation_gpt'] = gpt_evaluation_raw
        results[i]['evaluation_deepseek'] = deepseek_evaluation_raw

        # Add a small delay between evaluation calls to avoid hitting rate limits
        time.sleep(1)

    print("\n--- Evaluation Complete ---")
    # The 'results' list is now updated with evaluation outcomes.
    # You might want to save this updated list to a new JSON file or overwrite the old one.
    # For now, the evaluation results are stored in the 'results' list in memory.


def main(num_test_cases=None, output_filename="results.json"):
    # =============== Deepseek initiation======================
    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")

    if not deepseek_api_key:
        print("Error: DEEPSEEK_API_KEY environment variable not set.")
        # Exit if API key is missing, as we need Deepseek responses
        sys.exit(1)
    try:
        deepseek = Deepseek(deepseek_api_key)
    except Exception as e:
        print(f"Error initializing Deepseek: {e}")
        sys.exit(1) # Exit on initialization error
    # ========================================================

    # =================GPT initiation===========================
    gpt_api_key = os.getenv("OPENAI_API_KEY")

    if not gpt_api_key:
        print("Error: OPENAI_API_KEY environment variable not set.")
        # Exit if API key is missing, as we need GPT responses
        sys.exit(1)
    try:
        gpt = GPT(gpt_api_key)
    except Exception as e:
        print(f"Error initializing GPT: {e}")
        sys.exit(1) # Exit on initialization error
    # ========================================================

    # =============== Load Test Cases ========================
    test_cases_file = "./test_cases.csv"
    if not os.path.exists(test_cases_file):
        print(f"Error: Test cases file not found at {test_cases_file}")
        sys.exit(1) # Exit if test cases file is missing

    try:
        data = load_test_cases(test_cases_file)
        print(f"Loaded {len(data)} test cases from {test_cases_file}")
    except (ValueError, FileNotFoundError, pd.errors.EmptyDataError) as e:
        print(f"Error loading test cases: {e}")
        sys.exit(1) # Exit on loading errors
    except Exception as e:
         print(f"An unexpected error occurred while loading the CSV: {e}")
         sys.exit(1) # Exit on unexpected errors

    results = []

    # Apply slicing based on num_test_cases
    cases_to_process = data
    if num_test_cases is not None:
        # Ensure we don't try to process more cases than available
        cases_to_process = data[:min(num_test_cases, len(data))]
        print(f"Processing the first {len(cases_to_process)} test case(s) based on input arguments.")


    if not cases_to_process:
        print("No test cases to process based on the provided arguments or file content.")
        # Optionally save an empty results file if no cases were processed
        # try:
        #     with open(output_filename, 'w', encoding='utf-8') as f:
        #         json.dump([], f, indent=2)
        #     print(f"Saved empty results to {output_filename}")
        # except IOError as e:
        #      print(f"Error saving empty results to {output_filename}: {e}")
        return # Exit main if no cases to process
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
            print(f"Deepseek Response: {actual_output_deepseek[:300]}{'...' if len(actual_output_deepseek) > 300 else ''}") # Truncate long responses for printing
        except Exception as e:
            actual_output_deepseek = f"Error generating response: {e}"
            print(f"Error getting Deepseek response: {e}")

        # ============================Get actual output from GPT model=================================
        actual_output_gpt = "Error generating response." # Default error message
        try:
            actual_output_gpt = gpt.generate_response(prompt)
            # Print the received response (truncated)
            print(f"GPT Response: {actual_output_gpt[:300]}{'...' if len(actual_output_gpt) > 300 else ''}") # Truncate long responses for printing
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
    # This saves the results *before* evaluation is added to the JSON file
    try:
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n--- Results Saved (before evaluation) ---")
        print(f"Results successfully saved to {output_filename}")
    except IOError as e:
        print(f"\n--- Error Saving Results ---")
        print(f"Error saving results to {output_filename}: {e}")
    # --- End Save ---

    # --- Perform Evaluation ---
    # This function will modify the 'results' list in place to add evaluation outcomes
    perform_evaluation(results)
    # --- End Evaluation ---

    # --- Optionally Save Results with Evaluation ---
    # If you want the evaluation results included in the JSON file, uncomment the block below
    # Note: This will overwrite the previously saved file or save to a new one if filename is changed
    try:
        output_filename_with_eval = output_filename.replace(".json", "_with_eval.json")
        with open(output_filename_with_eval, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n--- Results Saved (with evaluation) ---")
        print(f"Results successfully saved to {output_filename_with_eval}")
    except IOError as e:
        print(f"\n--- Error Saving Results with Evaluation ---")
        print(f"Error saving results with evaluation to {output_filename_with_eval}: {e}")
    # --- End Save with Evaluation ---


if __name__ == "__main__":
    num_test_cases_to_run = None # Default to running all test cases

    # Check for the test flag first
    if '--test' in sys.argv or '-t' in sys.argv:
        num_test_cases_to_run = 1
        print("'-t' or '--test' flag detected. Running only the first test case.")
    # If test flag is not present, check for the -n flag
    elif '-n' in sys.argv:
        try:
            n_index = sys.argv.index('-n')
            # Check if there is an argument after -n
            if n_index + 1 < len(sys.argv):
                num_test_cases_to_run_str = sys.argv[n_index + 1]
                # Attempt to convert the argument to an integer
                num_test_cases_to_run = int(num_test_cases_to_run_str)
                # Validate the number of test cases
                if num_test_cases_to_run < 1:
                    raise ValueError("Number of test cases must be at least 1.")
                print(f"'-n' flag detected. Running the first {num_test_cases_to_run} test case(s).")
            else:
                # If no number is provided after -n
                print("Error: '-n' flag requires a number of test cases.")
                sys.exit(1) # Exit if argument is missing
        except ValueError as e:
            # Handle cases where the argument is not a valid integer or is less than 1
            print(f"Error: Invalid number provided for '-n': {e}")
            sys.exit(1) # Exit on invalid number
        except Exception as e:
            # Catch any other unexpected errors during parsing
            print(f"An unexpected error occurred while parsing '-n': {e}")
            sys.exit(1) # Exit on unexpected parsing error
    else:
        # If neither flag is present, run all test cases (num_test_cases_to_run remains None)
        print("No test case limit flags ('-t' or '-n') detected. Running all test cases.")

    # Call the main function with the determined number of test cases
    main(num_test_cases=num_test_cases_to_run, output_filename="results.json")
