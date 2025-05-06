import eva_res as eva # Import the eva_res module
import os
import json
# Assuming Deepseek and GPT classes are defined in Deepseek.py and GPT.py respectively
from Deepseek import Deepseek
from GPT import GPT
import pandas as pd
from dotenv import load_dotenv
import sys
import time # Import time for delays if needed during evaluation
from typing import List, Dict, Any # Import typing hints
import statistics # Import statistics for calculating average

# Load environment variables from .env file
load_dotenv()

def load_test_cases(file_path):
    """ load .csv get the input and expected output column from the file """
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

# Modify perform_evaluation to accept num_retries and handle lists of responses
def perform_evaluation(results: List[Dict[str, Any]], num_retries: int):
    """ Performs evaluation on the results list including consistency and overall accuracy checks. """
    print("\n--- Starting Automated Response Evaluation ---")

    if not results:
        print("No results to evaluate.")
        return # Exit evaluation if no results

    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        print("Error: GOOGLE_API_KEY environment variable not set for evaluation.")
        print("Evaluation cannot proceed.")
        return # Exit evaluation if API key is missing

    try:
        # Use the GenerativeModel class directly from the imported eva_res module's genai
        # Using gemini-1.5-pro-latest as specified in eva_res.py
        # Ensure eva.genai is available (it should be if eva_res configured correctly)
        if not hasattr(eva, 'genai') or not hasattr(eva.genai, 'GenerativeModel'):
             print("Error: Google Generative AI not configured in eva_res.py.")
             print("Evaluation cannot proceed.")
             return

        evaluator_model = eva.genai.GenerativeModel(model_name="gemini-1.5-pro-latest")
        print(f"Using evaluation model: {'gemini-1.5-pro-latest'}")
    except Exception as e:
        print(f"Error initializing evaluation model: {e}")
        print("Please check the model name and your GOOGLE_API_KEY.")
        return # Exit evaluation if model initialization fails

    # Initialize overall scores and counts
    total_cases = len(results)
    gpt_consistency_scores = []
    deepseek_consistency_scores = []

    # New counters for overall valid/invalid responses across all retries
    gpt_valid_count_overall = 0
    deepseek_valid_count_overall = 0
    total_possible_evaluations = total_cases * num_retries # Total number of individual responses to evaluate for validity

    for i, test_case in enumerate(results):
        prompt = test_case.get("prompt")
        expected = test_case.get("expected_output")
        actual_outputs_gpt = test_case.get("actual_outputs_gpt") # Now a list
        actual_outputs_deepseek = test_case.get("actual_outputs_deepseek") # Now a list

        # Ensure all necessary data is present for this test case before evaluating
        # Check if the lists of outputs exist and have content if num_retries > 0
        # If num_retries is 0, actual_outputs will be empty lists, which is fine for skipping evaluation
        if not all([prompt, expected]):
             print(f"Skipping evaluation for case {i+1} due to missing prompt or expected output: {test_case}")
             continue # Skip this case if fundamental data is missing

        # Ensure actual_outputs lists are not None, even if empty (e.g., num_retries=0 or API failed)
        actual_outputs_gpt = actual_outputs_gpt if actual_outputs_gpt is not None else []
        actual_outputs_deepseek = actual_outputs_deepseek if actual_outputs_deepseek is not None else []


        print(f"\n--- Evaluating Case {i + 1}/{total_cases} ---")
        print(f"Prompt: {prompt[:80]}...")  # Print truncated prompt

        # --- Evaluate Consistency ---
        if num_retries > 1:
            print("  Evaluating Consistency...")
            # Evaluate GPT Consistency
            print("    Evaluating GPT consistency...")
            if actual_outputs_gpt: # Only evaluate if there are responses
                try:
                    gpt_consistency_raw = eva.get_consistency_score(
                        evaluator_model, prompt, expected, actual_outputs_gpt
                    )
                    gpt_consistency_score = eva.parse_evaluation_score(gpt_consistency_raw)
                    if gpt_consistency_score is not None:
                        gpt_consistency_scores.append(gpt_consistency_score)
                    results[i]['consistency_evaluation_gpt'] = gpt_consistency_raw
                    print(f"      GPT Consistency Result: {gpt_consistency_raw} (Parsed Score: {gpt_consistency_score})")
                except Exception as e:
                     print(f"      Error evaluating GPT consistency: {e}")
                     results[i]['consistency_evaluation_gpt'] = f"Error: {e}"
            else:
                 print("    GPT consistency evaluation skipped (no responses).")


            # Evaluate Deepseek Consistency
            print("    Evaluating Deepseek consistency...")
            if actual_outputs_deepseek: # Only evaluate if there are responses
                try:
                    deepseek_consistency_raw = eva.get_consistency_score(
                        evaluator_model, prompt, expected, actual_outputs_deepseek
                    )
                    deepseek_consistency_score = eva.parse_evaluation_score(deepseek_consistency_raw)
                    if deepseek_consistency_score is not None:
                        deepseek_consistency_scores.append(deepseek_consistency_score)
                    results[i]['consistency_evaluation_deepseek'] = deepseek_consistency_raw
                    print(f"      Deepseek Consistency Result: {deepseek_consistency_raw} (Parsed Score: {deepseek_consistency_score})")
                except Exception as e:
                     print(f"      Error evaluating Deepseek consistency: {e}")
                     results[i]['consistency_evaluation_deepseek'] = f"Error: {e}"
            else:
                 print("    Deepseek consistency evaluation skipped (no responses).")
        else:
            print("    Consistency evaluation skipped as num_retries is 1.")


        # --- Perform Overall Accuracy (Valid/Invalid) Evaluation for ALL responses ---
        print("  Evaluating overall accuracy (Valid/Invalid) for all responses...")
        results[i]['individual_evaluations_gpt'] = [] # Store individual evaluation results
        results[i]['individual_evaluations_deepseek'] = [] # Store individual evaluation results

        # Evaluate GPT's responses
        print("    Evaluating GPT responses...")
        for j, response in enumerate(actual_outputs_gpt):
            try:
                gpt_evaluation_raw = eva.get_evaluation_result(
                    evaluator_model, prompt, expected, response
                )
                results[i]['individual_evaluations_gpt'].append(gpt_evaluation_raw)
                gpt_passed = eva.parse_pass_fail(gpt_evaluation_raw)
                if gpt_passed:
                    gpt_valid_count_overall += 1
                print(f"      GPT Response {j+1}/{num_retries} Evaluation: {gpt_evaluation_raw} (Passed: {gpt_passed})")
            except Exception as e:
                 eval_error_msg = f"Error evaluating GPT response {j+1}: {e}"
                 results[i]['individual_evaluations_gpt'].append(eval_error_msg)
                 print(f"      {eval_error_msg}")
                 # Error in evaluation counts as a failure for pass/fail purposes


        # Evaluate Deepseek's responses
        print("    Evaluating Deepseek responses...")
        for j, response in enumerate(actual_outputs_deepseek):
            try:
                deepseek_evaluation_raw = eva.get_evaluation_result(
                    evaluator_model, prompt, expected, response
                )
                results[i]['individual_evaluations_deepseek'].append(deepseek_evaluation_raw)
                deepseek_passed = eva.parse_pass_fail(deepseek_evaluation_raw)
                if deepseek_passed:
                    deepseek_valid_count_overall += 1
                print(f"      Deepseek Response {j+1}/{num_retries} Evaluation: {deepseek_evaluation_raw} (Passed: {deepseek_passed})")
            except Exception as e:
                 eval_error_msg = f"Error evaluating Deepseek response {j+1}: {e}"
                 results[i]['individual_evaluations_deepseek'].append(eval_error_msg)
                 print(f"      {eval_error_msg}")
                 # Error in evaluation counts as a failure for pass/fail purposes

        # Add a small delay between test cases
        time.sleep(1) # Adjust delay as needed


    print("\n--- Evaluation Complete ---")

    # --- Calculate and Report Overall Scores ---
    print("\n--- Overall Results ---")
    if total_cases > 0:
        # Report Overall Accuracy (Valid/Invalid) across all retries
        print("\nOverall Accuracy (Valid/Invalid) across all responses:")
        # Calculate percentages based on total number of responses generated
        gpt_valid_percent_overall = (gpt_valid_count_overall / total_possible_evaluations) * 100 if total_possible_evaluations > 0 else 0
        deepseek_valid_percent_overall = (deepseek_valid_count_overall / total_possible_evaluations) * 100 if total_possible_evaluations > 0 else 0

        print(f"GPT: Valid Responses: {gpt_valid_count_overall}/{total_possible_evaluations} ({gpt_valid_percent_overall:.2f}%) Invalid Responses: {total_possible_evaluations - gpt_valid_count_overall}/{total_possible_evaluations} ({100 - gpt_valid_percent_overall:.2f}%)")
        print(f"Deepseek: Valid Responses: {deepseek_valid_count_overall}/{total_possible_evaluations} ({deepseek_valid_percent_overall:.2f}%) Invalid Responses: {total_possible_evaluations - deepseek_valid_count_overall}/{total_possible_evaluations} ({100 - deepseek_valid_percent_overall:.2f}%)")


        # Report Consistency (only if num_retries > 1)
        if num_retries > 1:
            print("\nConsistency (Average Score):")
            # Handle cases where no consistency scores were recorded (e.g., all had errors or parsing failed)
            avg_gpt_consistency = statistics.mean(gpt_consistency_scores) if gpt_consistency_scores else 0
            avg_deepseek_consistency = statistics.mean(deepseek_consistency_scores) if deepseek_consistency_scores else 0
            # Assuming score is out of 5, state the scale clearly
            print(f"GPT: Average Consistency Score = {avg_gpt_consistency:.2f} (Scale 1-5)") # Assuming 1-5 scale from parsing
            print(f"Deepseek: Average Consistency Score = {avg_deepseek_consistency:.2f} (Scale 1-5)") # Assuming 1-5 scale from parsing
        else:
             print("\nConsistency evaluation was not performed (num_retries = 1).")

    else:
        print("No test cases were processed.")

    # The 'results' list is now updated with evaluation outcomes.
    # The save logic is after the perform_evaluation call in main.


# Add num_retries parameter with a default value of 1
def main(num_test_cases=None, num_retries=1, output_filename="results.json"):
    # =============== Deepseek initiation======================
    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")

    if not deepseek_api_key and num_retries > 0:
        print("Error: DEEPSEEK_API_KEY environment variable not set.")
        print("Cannot generate Deepseek responses.")
        # Don't exit, allow running without Deepseek if GPT is available
        deepseek = None
    elif deepseek_api_key:
        try:
            deepseek = Deepseek(deepseek_api_key)
        except Exception as e:
            print(f"Error initializing Deepseek: {e}")
            print("Cannot generate Deepseek responses.")
            deepseek = None # Set to None if initialization fails
    else: # num_retries is 0, no need for API key
        deepseek = None


    # ========================================================

    # =================GPT initiation===========================
    gpt_api_key = os.getenv("OPENAI_API_KEY")

    if not gpt_api_key and num_retries > 0:
        print("Error: OPENAI_API_KEY environment variable not set.")
        print("Cannot generate GPT responses.")
        # Don't exit, allow running without GPT if Deepseek is available
        gpt = None
    elif gpt_api_key:
        try:
            gpt = GPT(gpt_api_key)
        except Exception as e:
            print(f"Error initializing GPT: {e}")
            print("Cannot generate GPT responses.")
            gpt = None # Set to None if initialization fails
    else: # num_retries is 0, no need for API key
        gpt = None
    # ========================================================

    # Exit if neither model can be initialized and we need responses
    if num_retries > 0 and not deepseek and not gpt:
        print("\nError: Neither Deepseek nor GPT could be initialized and num_retries > 0.")
        print("Please check your API keys and network connection.")
        sys.exit(1)


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
        # Allow proceeding even with no cases if just testing setup, but print message


    if num_retries > 0:
        print(f"Each test case will be run {num_retries} time(s) for consistency check.")
    else:
        print("Running with 0 retries. No chatbot responses will be generated.")

    # =================Process each test case with both models==================
    for case in cases_to_process:
        prompt = case['prompt']
        expected_output = case['expected_output']

        print(f"\n--- Processing Prompt ---")
        print(f"Prompt: {prompt[:200]}{'...' if len(prompt) > 200 else ''}")


        # =========================Get actual outputs from Deepseek model=================================
        actual_outputs_deepseek = [] # Changed to a list
        if deepseek and num_retries > 0: # Only call if deepseek is initialized and retries > 0
            print(f"Getting {num_retries} responses from Deepseek...")
            for i in range(num_retries): # Loop for retries
                output = "Error generating response." # Default error message
                try:
                    output = deepseek.generate_response(prompt)
                    print(f"  Deepseek Response {i+1}/{num_retries}: {output[:200]}{'...' if len(output) > 200 else ''}") # Truncate long responses for printing
                except Exception as e:
                    output = f"Error generating response: {e}"
                    print(f"  Error getting Deepseek response {i+1}/{num_retries}: {e}")
                actual_outputs_deepseek.append(output)
                time.sleep(0.5) # Small delay between retries for one model
        elif num_retries > 0: # num_retries > 0 but deepseek is None
            print("Deepseek skipped (API key missing or initialization failed).")


        # ============================Get actual outputs from GPT model=================================
        actual_outputs_gpt = [] # Changed to a list
        if gpt and num_retries > 0: # Only call if gpt is initialized and retries > 0
            print(f"Getting {num_retries} responses from GPT...")
            for i in range(num_retries): # Loop for retries
                output = "Error generating response." # Default error message
                try:
                    output = gpt.generate_response(prompt)
                    print(f"  GPT Response {i+1}/{num_retries}: {output[:200]}{'...' if len(output) > 200 else ''}") # Truncate long responses for printing
                except Exception as e:
                    output = f"Error generating response: {e}"
                    print(f"  Error getting GPT response {i+1}/{num_retries}: {e}")
                actual_outputs_gpt.append(output)
                time.sleep(0.5) # Small delay between retries for one model
        elif num_retries > 0: # num_retries > 0 but gpt is None
            print("GPT skipped (API key missing or initialization failed).")


        # ===========================Append results in the desired format==============================
        results.append({
            "prompt": prompt,
            "expected_output": expected_output,
            "actual_outputs_deepseek": actual_outputs_deepseek, # Store list
            "actual_outputs_gpt": actual_outputs_gpt # Store list
        })

        # Add a small delay between test cases if both models were called
        if (deepseek or gpt) and num_retries > 0:
             time.sleep(1) # Delay between test cases


    # --- Save the results to a JSON file (before evaluation) ---
    try:
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n--- Raw Results Saved (before evaluation) ---")
        print(f"Raw results successfully saved to {output_filename}")
    except IOError as e:
        print(f"\n--- Error Saving Raw Results ---")
        print(f"Error saving raw results to {output_filename}: {e}")
    # --- End Save ---

    # --- Perform Evaluation ---
    # Pass num_retries to the evaluation function
    # Only attempt evaluation if there were cases loaded AND at least one model was called with retries > 0
    if cases_to_process and ((deepseek or gpt) and num_retries > 0):
        perform_evaluation(results, num_retries)
    elif cases_to_process:
        print("\nSkipping evaluation as no chatbot responses were generated (num_retries is 0 or models failed to initialize).")
    else:
        print("\nNo test cases processed, skipping evaluation.")

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
    num_retries_to_run = 1 # Default to 1 retry

    # Parse arguments
    args = sys.argv[1:] # Get arguments excluding script name

    # Check for the test flag first
    if '--test' in args or '-t' in args:
        num_test_cases_to_run = 1
        print("'-t' or '--test' flag detected. Running only the first test case.")
        # Remove test flag so it doesn't interfere with subsequent parsing
        args = [arg for arg in args if arg not in ['--test', '-t']]

    # Check for the -n flag
    if '-n' in args:
        try:
            n_index = args.index('-n')
            # Check if there is an argument after -n
            if n_index + 1 < len(args):
                num_test_cases_to_run_str = args[n_index + 1]
                num_test_cases_to_run = int(num_test_cases_to_run_str)
                if num_test_cases_to_run < 0: # Allow -n 0 to load 0 cases
                    raise ValueError("Number of test cases must be non-negative.")
                print(f"'-n' flag detected. Running the first {num_test_cases_to_run} test case(s).")
                # Remove -n flag and its value
                args = args[:n_index] + args[n_index + 2:]
            else:
                print("Error: '-n' flag requires a number of test cases.")
                sys.exit(1)
        except ValueError as e:
            print(f"Error: Invalid number provided for '-n': {e}")
            sys.exit(1)
        except Exception as e:
             print(f"An unexpected error occurred while parsing '-n': {e}")
             sys.exit(1)

    # Check for the -retest flag
    if '-retest' in args:
         try:
            retest_index = args.index('-retest')
            # Check if there is an argument after -retest
            if retest_index + 1 < len(args):
                num_retries_str = args[retest_index + 1]
                num_retries_to_run = int(num_retries_str)
                if num_retries_to_run < 0:
                    raise ValueError("Number of retries must be non-negative.")
                print(f"'-retest' flag detected. Running each test case {num_retries_to_run} time(s).")
                 # Remove -retest flag and its value
                args = args[:retest_index] + args[retest_index + 2:]
            else:
                print("Error: '-retest' flag requires a number of retries.")
                sys.exit(1)
         except ValueError as e:
            print(f"Error: Invalid number provided for '-retest': {e}")
            sys.exit(1)
         except Exception as e:
             print(f"An unexpected error occurred while parsing '-retest': {e}")
             sys.exit(1)

    # If there are any remaining unhandled arguments, print a warning
    if args:
        print(f"Warning: Unrecognized arguments ignored: {args}")

    # Call the main function with the determined parameters
    main(num_test_cases=num_test_cases_to_run, num_retries=num_retries_to_run, output_filename="results.json")
