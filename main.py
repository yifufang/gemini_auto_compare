import eva_res as eva
import os
import json
from Deepseek import Deepseek
from GPT import GPT
import pandas as pd
from dotenv import load_dotenv
import sys
import time
from typing import List, Dict, Any
import statistics

load_dotenv()

def load_test_cases(file_path):
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip().str.lower()
    if 'input' not in df.columns or 'expected output' not in df.columns:
        raise ValueError("CSV file must contain 'input' and 'expected output' columns (case-insensitive).")

    df = df[['input', 'expected output']].dropna()
    df.columns = ['prompt', 'expected_output']

    return df.to_dict(orient='records')

def perform_evaluation(results: List[Dict[str, Any]], num_retries: int):
    print("\n--- Starting Automated Response Evaluation ---")

    if not results:
        print("No results to evaluate.")
        return

    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        print("Error: GOOGLE_API_KEY environment variable not set for evaluation.")
        print("Evaluation cannot proceed.")
        return

    try:
        if not hasattr(eva, 'genai') or not hasattr(eva.genai, 'GenerativeModel'):
             print("Error: Google Generative AI not configured in eva_res.py.")
             print("Evaluation cannot proceed.")
             return

        evaluator_model = eva.genai.GenerativeModel(model_name="gemini-1.5-pro-latest")
        print(f"Using evaluation model: {'gemini-1.5-pro-latest'}")
    except Exception as e:
        print(f"Error initializing evaluation model: {e}")
        print("Please check the model name and your GOOGLE_API_KEY.")
        return

    total_cases = len(results)
    gpt_consistency_scores = []
    deepseek_consistency_scores = []

    gpt_valid_count_overall = 0
    deepseek_valid_count_overall = 0
    total_possible_evaluations = total_cases * num_retries

    for i, test_case in enumerate(results):
        prompt = test_case.get("prompt")
        expected = test_case.get("expected_output")
        actual_outputs_gpt = test_case.get("actual_outputs_gpt")
        actual_outputs_deepseek = test_case.get("actual_outputs_deepseek")

        if not all([prompt, expected]):
             print(f"Skipping evaluation for case {i+1} due to missing prompt or expected output: {test_case}")
             continue

        actual_outputs_gpt = actual_outputs_gpt if actual_outputs_gpt is not None else []
        actual_outputs_deepseek = actual_outputs_deepseek if actual_outputs_deepseek is not None else []


        print(f"\n--- Evaluating Case {i + 1}/{total_cases} ---")
        print(f"Prompt: {prompt[:80]}...")

        if num_retries > 1:
            print("  Evaluating Consistency...")
            print("    Evaluating GPT consistency...")
            if actual_outputs_gpt:
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

            print("    Evaluating Deepseek consistency...")
            if actual_outputs_deepseek:
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

        print("  Evaluating overall accuracy (Valid/Invalid) for all responses...")
        results[i]['individual_evaluations_gpt'] = []
        results[i]['individual_evaluations_deepseek'] = []

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

        time.sleep(1)

    print("\n--- Evaluation Complete ---")

    print("\n--- Overall Results ---")
    if total_cases > 0:
        print("\nOverall Accuracy (Valid/Invalid) across all responses:")
        gpt_valid_percent_overall = (gpt_valid_count_overall / total_possible_evaluations) * 100 if total_possible_evaluations > 0 else 0
        deepseek_valid_percent_overall = (deepseek_valid_count_overall / total_possible_evaluations) * 100 if total_possible_evaluations > 0 else 0

        print(f"GPT: Valid Responses: {gpt_valid_count_overall}/{total_possible_evaluations} ({gpt_valid_percent_overall:.2f}%) Invalid Responses: {total_possible_evaluations - gpt_valid_count_overall}/{total_possible_evaluations} ({100 - gpt_valid_percent_overall:.2f}%)")
        print(f"Deepseek: Valid Responses: {deepseek_valid_count_overall}/{total_possible_evaluations} ({deepseek_valid_percent_overall:.2f}%) Invalid Responses: {total_possible_evaluations - deepseek_valid_count_overall}/{total_possible_evaluations} ({100 - deepseek_valid_percent_overall:.2f}%)")

        if num_retries > 1:
            print("\nConsistency (Average Score):")
            avg_gpt_consistency = statistics.mean(gpt_consistency_scores) if gpt_consistency_scores else 0
            avg_deepseek_consistency = statistics.mean(deepseek_consistency_scores) if deepseek_consistency_scores else 0
            print(f"GPT: Average Consistency Score = {avg_gpt_consistency:.2f} (Scale 1-5)")
            print(f"Deepseek: Average Consistency Score = {avg_deepseek_consistency:.2f} (Scale 1-5)")
        else:
             print("\nConsistency evaluation was not performed (num_retries = 1).")

    else:
        print("No test cases were processed.")


def main(num_test_cases=None, num_retries=1, output_filename="results.json"):
    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")

    if not deepseek_api_key and num_retries > 0:
        print("Error: DEEPSEEK_API_KEY environment variable not set.")
        print("Cannot generate Deepseek responses.")
        deepseek = None
    elif deepseek_api_key:
        try:
            deepseek = Deepseek(deepseek_api_key)
        except Exception as e:
            print(f"Error initializing Deepseek: {e}")
            print("Cannot generate Deepseek responses.")
            deepseek = None
    else:
        deepseek = None

    gpt_api_key = os.getenv("OPENAI_API_KEY")

    if not gpt_api_key and num_retries > 0:
        print("Error: OPENAI_API_KEY environment variable not set.")
        print("Cannot generate GPT responses.")
        gpt = None
    elif gpt_api_key:
        try:
            gpt = GPT(gpt_api_key)
        except Exception as e:
            print(f"Error initializing GPT: {e}")
            print("Cannot generate GPT responses.")
            gpt = None
    else:
        gpt = None

    if num_retries > 0 and not deepseek and not gpt:
        print("\nError: Neither Deepseek nor GPT could be initialized and num_retries > 0.")
        print("Please check your API keys and network connection.")
        sys.exit(1)

    test_cases_file = "./test_cases.csv"
    if not os.path.exists(test_cases_file):
        print(f"Error: Test cases file not found at {test_cases_file}")
        sys.exit(1)

    try:
        data = load_test_cases(test_cases_file)
        print(f"Loaded {len(data)} test cases from {test_cases_file}")
    except (ValueError, FileNotFoundError, pd.errors.EmptyDataError) as e:
        print(f"Error loading test cases: {e}")
        sys.exit(1)
    except Exception as e:
         print(f"An unexpected error occurred while loading the CSV: {e}")
         sys.exit(1)

    results = []

    cases_to_process = data
    if num_test_cases is not None:
        cases_to_process = data[:min(num_test_cases, len(data))]
        print(f"Processing the first {len(cases_to_process)} test case(s) based on input arguments.")

    if not cases_to_process:
        print("No test cases to process based on the provided arguments or file content.")

    if num_retries > 0:
        print(f"Each test case will be run {num_retries} time(s) for consistency check.")
    else:
        print("Running with 0 retries. No chatbot responses will be generated.")

    for case in cases_to_process:
        prompt = case['prompt']
        expected_output = case['expected_output']

        print(f"\n--- Processing Prompt ---")
        print(f"Prompt: {prompt[:200]}{'...' if len(prompt) > 200 else ''}")

        actual_outputs_deepseek = []
        if deepseek and num_retries > 0:
            print(f"Getting {num_retries} responses from Deepseek...")
            for i in range(num_retries):
                output = "Error generating response."
                try:
                    output = deepseek.generate_response(prompt)
                    print(f"  Deepseek Response {i+1}/{num_retries}: {output[:200]}{'...' if len(output) > 200 else ''}")
                except Exception as e:
                    output = f"Error generating response: {e}"
                    print(f"  Error getting Deepseek response {i+1}/{num_retries}: {e}")
                actual_outputs_deepseek.append(output)
                time.sleep(0.5)
        elif num_retries > 0:
            print("Deepseek skipped (API key missing or initialization failed).")

        actual_outputs_gpt = []
        if gpt and num_retries > 0:
            print(f"Getting {num_retries} responses from GPT...")
            for i in range(num_retries):
                output = "Error generating response."
                try:
                    output = gpt.generate_response(prompt)
                    print(f"  GPT Response {i+1}/{num_retries}: {output[:200]}{'...' if len(output) > 200 else ''}")
                except Exception as e:
                    output = f"Error generating response: {e}"
                    print(f"  Error getting GPT response {i+1}/{num_retries}: {e}")
                actual_outputs_gpt.append(output)
                time.sleep(0.5)
        elif num_retries > 0:
            print("GPT skipped (API key missing or initialization failed).")

        results.append({
            "prompt": prompt,
            "expected_output": expected_output,
            "actual_outputs_deepseek": actual_outputs_deepseek,
            "actual_outputs_gpt": actual_outputs_gpt
        })

        if (deepseek or gpt) and num_retries > 0:
             time.sleep(1)

    try:
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n--- Raw Results Saved (before evaluation) ---")
        print(f"Raw results successfully saved to {output_filename}")
    except IOError as e:
        print(f"\n--- Error Saving Raw Results ---")
        print(f"Error saving raw results to {output_filename}: {e}")

    if cases_to_process and ((deepseek or gpt) and num_retries > 0):
        perform_evaluation(results, num_retries)
    elif cases_to_process:
        print("\nSkipping evaluation as no chatbot responses were generated (num_retries is 0 or models failed to initialize).")
    else:
        print("\nNo test cases processed, skipping evaluation.")

    try:
        output_filename_with_eval = output_filename.replace(".json", "_with_eval.json")
        with open(output_filename_with_eval, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n--- Results Saved (with evaluation) ---")
        print(f"Results successfully saved to {output_filename_with_eval}")
    except IOError as e:
        print(f"\n--- Error Saving Results with Evaluation ---")
        print(f"Error saving results with evaluation to {output_filename_with_eval}: {e}")


if __name__ == "__main__":
    num_test_cases_to_run = None
    num_retries_to_run = 1

    args = sys.argv[1:]

    if '--test' in args or '-t' in args:
        num_test_cases_to_run = 1
        print("'-t' or '--test' flag detected. Running only the first test case.")
        args = [arg for arg in args if arg not in ['--test', '-t']]

    if '-n' in args:
        try:
            n_index = args.index('-n')
            if n_index + 1 < len(args):
                num_test_cases_to_run_str = args[n_index + 1]
                num_test_cases_to_run = int(num_test_cases_to_run_str)
                if num_test_cases_to_run < 0:
                    raise ValueError("Number of test cases must be non-negative.")
                print(f"'-n' flag detected. Running the first {num_test_cases_to_run} test case(s).")
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

    if '-retest' in args:
         try:
            retest_index = args.index('-retest')
            if retest_index + 1 < len(args):
                num_retries_str = args[retest_index + 1]
                num_retries_to_run = int(num_retries_str)
                if num_retries_to_run < 0:
                    raise ValueError("Number of retries must be non-negative.")
                print(f"'-retest' flag detected. Running each test case {num_retries_to_run} time(s).")
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

    if args:
        print(f"Warning: Unrecognized arguments ignored: {args}")

    main(num_test_cases=num_test_cases_to_run, num_retries=num_retries_to_run, output_filename="results.json")