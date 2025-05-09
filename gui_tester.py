import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
import threading
import os
import json
import pandas as pd
import statistics
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from dotenv import load_dotenv
import time
from typing import List, Dict, Any
import io
import base64
import numpy as np
import re # Added for keyword extraction
from collections import Counter # Added for keyword extraction

try:
    from ttkthemes import ThemedStyle
except ImportError:
    print("Warning: 'ttkthemes' not installed. Install with 'pip install ttkthemes' for more themes.")
    ThemedStyle = None

try:
    from Deepseek import Deepseek
    from GPT import GPT
    import eva_res as eva
except ImportError as e:
    print(f"Error importing custom modules: {e}")
    print("Please ensure Deepseek.py, GPT.py, and eva_res.py are in the same directory.")

load_dotenv()

class APITesterGUI:
    def __init__(self, master):
        self.master = master
        master.title("LLM API Automation Tester")
        master.geometry("1600x1000")
        master.minsize(800, 600)

        if ThemedStyle:
            style = ThemedStyle(master)
            try:
                style.theme_use('adapta')
                print("Using 'adapta' theme from ttkthemes.")
            except tk.TclError:
                print("Warning: 'adapta' theme not found in ttkthemes. Trying another...")
                try:
                    style.theme_use('arc')
                    print("Using 'arc' theme from ttkthemes.")
                except tk.TclError:
                    print("Warning: 'arc' theme not found in ttkthemes. Falling back to built-in 'alt'.")
                    style = ttk.Style()
                    try:
                        style.theme_use('alt')
                    except tk.TclError:
                        print("Warning: Built-in 'alt' theme not found. Falling back to default.")
                        style.use('default')
                        print("Using built-in 'default' theme.")
        else:
            style = ttk.Style()
            try:
                style.theme_use('alt')
                print("Using built-in 'alt' theme.")
            except tk.TclError:
                print("Warning: Built-in 'alt' theme not found. Falling back to default.")
                style.use('default') # Changed style.theme_use to style.use for default
                print("Using built-in 'default' theme.")


        self.deepseek = None
        self.gpt = None
        self.evaluator_model = None
        self._initialize_apis_and_evaluator()

        self.all_test_cases = []
        self.case_checkboxes = []
        self.selected_test_cases = []
        self.results = []
        self.overall_stats = {}
        self.is_running = False
        self.fig = None
        self.overall_coverage = 0
        self.overall_avg_complexity = 0
        self.common_keywords = [] # Added for storing common keywords

        self.create_widgets()

    def _initialize_apis_and_evaluator(self):
        deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
        gpt_api_key = os.getenv("OPENAI_API_KEY")
        google_api_key = os.getenv("GOOGLE_API_KEY")

        if deepseek_api_key:
            try:
                self.deepseek = Deepseek(deepseek_api_key)
                print("Deepseek API initialized.")
            except Exception as e:
                print(f"Error initializing Deepseek API: {e}")
                self.deepseek = None
        else:
            print("DEEPSEEK_API_KEY not found. Deepseek tests will be skipped.")

        if gpt_api_key:
            try:
                self.gpt = GPT(gpt_api_key)
                print("GPT API initialized.")
            except Exception as e:
                print(f"Error initializing GPT API: {e}")
                self.gpt = None
        else:
            print("OPENAI_API_KEY not found. GPT tests will be skipped.")

        if google_api_key:
            try:
                if hasattr(eva, 'genai') and hasattr(eva.genai, 'GenerativeModel'):
                     eva.genai.configure(api_key=google_api_key)
                     self.evaluator_model = eva.genai.GenerativeModel(model_name="gemini-1.5-pro-latest")
                     print(f"Gemini evaluation model initialized: {'gemini-1.5-pro-latest'}")
                else:
                     print("Error: Google Generative AI not configured in eva_res.py.")
                     self.evaluator_model = None
            except Exception as e:
                print(f"Error initializing Gemini evaluation model: {e}")
                self.evaluator_model = None
        else:
            print("GOOGLE_API_KEY not found. Evaluation will be skipped.")

    def create_widgets(self):
        config_frame = ttk.LabelFrame(self.master, text="Configuration", padding="10")
        config_frame.pack(pady=10, padx=10, fill="x")

        ttk.Label(config_frame, text="Test Cases CSV:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.file_path_entry = ttk.Entry(config_frame, width=50)
        self.file_path_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        self.browse_button = ttk.Button(config_frame, text="Browse", command=self.browse_file)
        self.browse_button.grid(row=0, column=2, padx=5, pady=5)

        ttk.Label(config_frame, text="Number of Retries:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.num_retries_entry = ttk.Spinbox(config_frame, from_=0, to=10, width=10)
        self.num_retries_entry.set("1")
        self.num_retries_entry.grid(row=1, column=1, padx=5, pady=5, sticky="w")

        self.start_button = ttk.Button(config_frame, text="Start Test", command=self.start_testing)
        self.start_button.grid(row=0, column=3, rowspan=2, padx=10, pady=5, sticky="nsew")

        self.download_button = ttk.Button(config_frame, text="Download Report", command=self.download_report)
        self.download_button.grid(row=0, column=4, rowspan=2, padx=10, pady=5, sticky="nsew")
        self.download_button.config(state=tk.DISABLED)

        self.quit_button = ttk.Button(config_frame, text="Quit", command=self.master.quit)
        self.quit_button.grid(row=0, column=5, rowspan=2, padx=10, pady=5, sticky="nsew")

        config_frame.columnconfigure(1, weight=1)

        status_frame = ttk.LabelFrame(self.master, text="Status", padding="10")
        status_frame.pack(pady=5, padx=10, fill="x")

        self.status_label = ttk.Label(status_frame, text="Idle", font=('TkDefaultFont', 10, 'bold'), anchor="center")
        self.status_label.pack(pady=5, fill="x")

        # Added Test Coverage and Complexity Labels
        metrics_frame = ttk.Frame(status_frame)
        metrics_frame.pack(pady=5, fill="x")
        self.coverage_label = ttk.Label(metrics_frame, text="Coverage: N/A", anchor="w")
        self.coverage_label.pack(side="left", expand=True, padx=10)
        self.complexity_label = ttk.Label(metrics_frame, text="Avg Complexity: N/A", anchor="e")
        self.complexity_label.pack(side="right", expand=True, padx=10)


        loaded_cases_frame = ttk.LabelFrame(self.master, text="Loaded Test Cases", padding="10")
        loaded_cases_frame.pack(pady=5, padx=10, fill="both", expand=True)

        self.case_canvas = tk.Canvas(loaded_cases_frame, borderwidth=0)
        self.case_scrollbar = ttk.Scrollbar(loaded_cases_frame, orient="vertical", command=self.case_canvas.yview)
        self.case_scrollable_frame_inner = ttk.Frame(self.case_canvas)

        self.case_scrollable_frame_inner.bind(
            "<Configure>",
            lambda e: self.case_canvas.configure(
                scrollregion = self.case_canvas.bbox("all")
            )
        )

        self.case_canvas.create_window((0, 0), window=self.case_scrollable_frame_inner, anchor="nw")
        self.case_canvas.configure(yscrollcommand=self.case_scrollbar.set)

        self.case_scrollbar.pack(side="right", fill="y")
        self.case_canvas.pack(side="left", fill="both", expand=True)

        select_buttons_frame = ttk.Frame(loaded_cases_frame)
        select_buttons_frame.pack(pady=5)
        self.select_all_button = ttk.Button(select_buttons_frame, text="Select All", command=self.select_all_cases)
        self.select_all_button.pack(side="left", padx=5)
        self.deselect_all_button = ttk.Button(select_buttons_frame, text="Deselect All", command=self.deselect_all_cases)
        self.deselect_all_button.pack(side="left", padx=5)

        test_case_frame = ttk.LabelFrame(self.master, text="Current Test Case", padding="10")
        test_case_frame.pack(pady=5, padx=10, fill="x")

        prompt_expected_frame = ttk.Frame(test_case_frame)
        prompt_expected_frame.pack(fill="both", expand=True)

        ttk.Label(prompt_expected_frame, text="Prompt:").grid(row=0, column=0, padx=5, pady=2, sticky="w")
        ttk.Label(prompt_expected_frame, text="Expected Output:").grid(row=0, column=1, padx=5, pady=2, sticky="w")

        self.prompt_text = scrolledtext.ScrolledText(prompt_expected_frame, height=8, wrap=tk.WORD)
        self.prompt_text.grid(row=1, column=0, padx=5, pady=2, sticky="nsew")
        self.prompt_text.config(state=tk.DISABLED)

        self.expected_output_text = scrolledtext.ScrolledText(prompt_expected_frame, height=5, wrap=tk.WORD)
        self.expected_output_text.grid(row=1, column=1, padx=5, pady=2, sticky="nsew")
        self.expected_output_text.config(state=tk.DISABLED)

        prompt_expected_frame.columnconfigure(0, weight=1)
        prompt_expected_frame.columnconfigure(1, weight=1)
        prompt_expected_frame.rowconfigure(1, weight=1)

        results_pane = ttk.PanedWindow(self.master, orient=tk.HORIZONTAL)
        results_pane.pack(pady=5, padx=10, fill="both", expand=True)

        deepseek_frame = ttk.LabelFrame(results_pane, text="Deepseek Results", padding="10")
        results_pane.add(deepseek_frame, weight=1)

        self.deepseek_responses_text = scrolledtext.ScrolledText(deepseek_frame, height=10, wrap=tk.WORD)
        self.deepseek_responses_text.pack(padx=5, pady=2, fill="both", expand=True)
        self.deepseek_responses_text.config(state=tk.DISABLED)

        ttk.Label(deepseek_frame, text="Evaluation:").pack(padx=5, pady=2, anchor="w")
        self.deepseek_eval_text = scrolledtext.ScrolledText(deepseek_frame, height=4, wrap=tk.WORD)
        self.deepseek_eval_text.pack(padx=5, pady=2, fill="x")
        self.deepseek_eval_text.config(state=tk.DISABLED)

        gpt_frame = ttk.LabelFrame(results_pane, text="GPT Results", padding="10")
        results_pane.add(gpt_frame, weight=1)

        self.gpt_responses_text = scrolledtext.ScrolledText(gpt_frame, height=10, wrap=tk.WORD)
        self.gpt_responses_text.pack(padx=5, pady=2, fill="both", expand=True)
        self.gpt_responses_text.config(state=tk.DISABLED)

        ttk.Label(gpt_frame, text="Evaluation:").pack(padx=5, pady=2, anchor="w")
        self.gpt_eval_text = scrolledtext.ScrolledText(gpt_frame, height=4, wrap=tk.WORD)
        self.gpt_eval_text.pack(padx=5, pady=2, fill="x")
        self.gpt_eval_text.config(state=tk.DISABLED)

        # Frame for the graph display in the GUI
        self.graph_display_frame = ttk.LabelFrame(self.master, text="Overall Performance Graph", padding="10")
        self.graph_display_frame.pack(pady=5, padx=10, fill="both", expand=True)
        self.graph_canvas_widget = None # Widget to hold the matplotlib canvas

    def browse_file(self):
        filename = filedialog.askopenfilename(
            initialdir=".",
            title="Select Test Cases CSV",
            filetypes=(("CSV Files", "*.csv"), ("All Files", "*.*"))
        )
        if filename:
            self.file_path_entry.delete(0, tk.END)
            self.file_path_entry.insert(0, filename)
            self.load_and_display_test_cases(filename)

    def load_and_display_test_cases(self, file_path: str):
        self.update_status("Loading test cases...")
        for widget in self.case_scrollable_frame_inner.winfo_children():
            widget.destroy()
        self.case_checkboxes = []
        self.all_test_cases = []
        self.overall_coverage = 0
        self.overall_avg_complexity = 0
        self.common_keywords = [] # Clear keywords on new file load
        self.update_metrics_display() # Clear metrics display
        self.hide_graph() # Hide graph when new file is loaded

        try:
            self.all_test_cases = self.load_test_cases(file_path)
            self.update_status(f"Loaded {len(self.all_test_cases)} test cases.")

            if self.all_test_cases:
                # Calculate and display initial average complexity based on all loaded cases
                complexities = [case.get('complexity', 0) for case in self.all_test_cases]
                self.overall_avg_complexity = statistics.mean(complexities) if complexities else 0
                self.update_metrics_display() # Update display after calculating initial average complexity

                for i, case in enumerate(self.all_test_cases):
                    prompt_snippet = case['prompt'][:100].replace('\n', ' ') + ('...' if len(case['prompt']) > 100 else '')
                    expected_snippet = case['expected_output'][:100].replace('\n', ' ') + ('...' if len(case['expected_output']) > 100 else '')
                    # Display complexity for each case
                    text = f"Case {i+1} (Complexity: {case.get('complexity', 'N/A'):.2f}) | Prompt: {prompt_snippet} | Expected: {expected_snippet}"

                    var = tk.BooleanVar(value=True)
                    cb = ttk.Checkbutton(self.case_scrollable_frame_inner, text=text, variable=var)
                    cb.pack(anchor="w", padx=5, pady=1)
                    self.case_checkboxes.append((var, case))

                # Call update_coverage after creating and setting initial state of checkboxes
                self.update_coverage()

                # Bind the update_coverage method to checkbox state changes
                for var, _ in self.case_checkboxes:
                    var.trace_add('write', lambda *args: self.update_coverage())

            else:
                ttk.Label(self.case_scrollable_frame_inner, text="No test cases found in the selected file.").pack(padx=5, pady=5)

        except (ValueError, FileNotFoundError, pd.errors.EmptyDataError) as e:
            self.update_status(f"Error loading test cases: {e}")
            messagebox.showerror("Error", f"Error loading test cases: {e}")
            ttk.Label(self.case_scrollable_frame_inner, text=f"Error loading test cases: {e}").pack(padx=5, pady=5)
            self.all_test_cases = []
            self.case_checkboxes = []
        except Exception as e:
             self.update_status(f"An unexpected error occurred while loading the CSV: {e}")
             messagebox.showerror("Error", f"An unexpected error occurred while loading the CSV: {e}")
             ttk.Label(self.case_scrollable_frame_inner, text=f"An unexpected error occurred: {e}").pack(padx=5, pady=5)
             self.all_test_cases = []
             self.case_checkboxes = []

    def calculate_test_case_complexity(self, prompt: str, expected_output: str) -> float:
        """
        Calculates a simple complexity score based on the length of prompt and expected output.
        This can be customized for more sophisticated complexity metrics.
        """
        return len(prompt) + len(expected_output)


    def update_coverage(self):
        """Updates the displayed coverage percentage based on selected checkboxes."""
        if not self.all_test_cases:
            self.overall_coverage = 0
        else:
            total_cases = len(self.all_test_cases)
            selected_cases_count = sum(var.get() for var, _ in self.case_checkboxes)
            self.overall_coverage = (selected_cases_count / total_cases) * 100 if total_cases > 0 else 0
        self.update_metrics_display()

    def update_metrics_display(self):
        """Updates the GUI labels for coverage and complexity."""
        self.master.after(0, lambda: self.coverage_label.config(text=f"Coverage: {self.overall_coverage:.2f}%"))
        self.master.after(0, lambda: self.complexity_label.config(text=f"Avg Complexity: {self.overall_avg_complexity:.2f}"))


    def select_all_cases(self):
        for var, _ in self.case_checkboxes:
            var.set(True)
        self.update_coverage()

    def deselect_all_cases(self):
        for var, _ in self.case_checkboxes:
            var.set(False)
        self.update_coverage()

    def extract_keywords(self, text_list: List[str], num_keywords: int = 10, n_gram: int = 1) -> List[tuple[str, int]]:
        """
        Extracts the most common keywords (words or n-grams) from a list of texts.
        """
        if not text_list:
            return []

        # Simple list of English stop words
        stop_words = set([
            "a", "an", "the", "is", "are", "and", "or", "in", "on", "at", "for", "with",
            "about", "as", "by", "from", "into", "like", "of", "off", "out", "over",
            "through", "to", "up", "with", "as", "by", "for", "from", "in", "into",
            "of", "off", "on", "out", "over", "through", "to", "under", "up", "with",
            "be", "been", "being", "have", "has", "had", "having", "do", "does",
            "did", "doing", "i", "me", "my", "myself", "we", "our", "ours", "ourselves",
            "you", "your", "yours", "yourself", "yourselves", "he", "him", "his",
            "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they",
            "them", "their", "theirs", "themselves", "what", "which", "who", "whom",
            "this", "that", "these", "those", "am", "is", "are", "was", "were", "be",
            "been", "being", "have", "has", "had", "having", "do", "does", "did",
            "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as",
            "until", "while", "of", "at", "by", "for", "with", "about", "against",
            "between", "into", "through", "during", "before", "after", "above",
            "below", "to", "from", "up", "down", "in", "out", "on", "off", "over",
            "under", "again", "further", "then", "once", "here", "there", "when",
            "where", "why", "how", "all", "any", "both", "each", "few", "more",
            "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same",
            "so", "than", "too", "very", "s", "t", "can", "will", "just", "don",
            "should", "now"
        ])

        combined_text = " ".join(text_list).lower()
        # Remove punctuation and non-alphanumeric characters (except spaces)
        combined_text = re.sub(r'[^a-z0-9\s]', '', combined_text)
        words = combined_text.split()

        # Remove stop words
        filtered_words = [word for word in words if word not in stop_words]

        if n_gram > 1:
            # Create n-grams
            ngrams = zip(*[filtered_words[i:] for i in range(n_gram)])
            items = [" ".join(ngram) for ngram in ngrams]
        else:
            items = filtered_words

        # Count frequencies
        item_counts = Counter(items)

        # Get the most common items
        most_common_items = item_counts.most_common(num_keywords)

        return most_common_items


    def start_testing(self):
        if self.is_running:
            messagebox.showinfo("Info", "Test is already running.")
            return

        if not self.all_test_cases:
            messagebox.showwarning("Warning", "No test cases loaded. Please load a CSV file first.")
            return

        self.selected_test_cases = [case for var, case in self.case_checkboxes if var.get()]

        if not self.selected_test_cases:
            messagebox.showwarning("Warning", "No test cases selected to run.")
            return

        try:
            num_retries = int(self.num_retries_entry.get())
            if num_retries < 0:
                 raise ValueError("Number of retries cannot be negative.")
        except ValueError as e:
            messagebox.showwarning("Warning", f"Invalid input for number of retries: {e}")
            return

        if num_retries > 0 and not (self.deepseek or self.gpt):
             messagebox.showwarning("Warning", "Neither Deepseek nor GPT API is initialized. Cannot run tests with retries > 0.")
             return

        if num_retries > 0 and not self.evaluator_model:
             messagebox.showwarning("Warning", "Gemini evaluation model not initialized. Evaluation results will not be available.")


        self.is_running = True
        self.start_button.config(state=tk.DISABLED)
        self.download_button.config(state=tk.DISABLED)
        self.clear_results_display()
        self.hide_graph() # Hide graph before starting a new run
        self.update_status(f"Running {len(self.selected_test_cases)} selected test cases.")

        # Recalculate coverage based on selected cases before starting
        self.update_coverage()

        self.test_thread = threading.Thread(
            target=self.run_tests_in_thread,
            args=(self.selected_test_cases, num_retries)
        )
        self.test_thread.start()

    def run_tests_in_thread(self, test_cases_to_run: List[Dict[str, Any]], num_retries: int):
        try:
            self.results = []
            total_cases = len(test_cases_to_run)
            run_complexities = [] # To calculate average complexity of run cases
            run_prompts = [] # To extract keywords from run cases

            for i, case in enumerate(test_cases_to_run):
                prompt = case['prompt']
                expected_output = case['expected_output']
                complexity = case.get('complexity', 0) # Get complexity

                run_complexities.append(complexity) # Add complexity to list for average calculation
                run_prompts.append(prompt) # Add prompt to list for keyword extraction

                try:
                    original_index = self.all_test_cases.index(case) + 1
                except ValueError:
                    original_index = i + 1

                self.update_status(f"Processing Case {original_index}/{len(self.all_test_cases)} (Run {i+1}/{total_cases})...")
                self.update_test_case_display(prompt, expected_output)
                self.clear_individual_results_display()

                actual_outputs_deepseek = []
                if self.deepseek and num_retries > 0:
                    self.update_status(f"Case {original_index}: Getting {num_retries} responses from Deepseek...")
                    for j in range(num_retries):
                        self.update_status(f"Case {original_index}: Deepseek Response {j+1}/{num_retries} - Generating...")
                        output = "Error generating response."
                        try:
                            output = self.deepseek.generate_response(prompt)
                            self.update_status(f"Case {original_index}: Deepseek Response {j+1}/{num_retries} - Received.")
                        except Exception as e:
                            output = f"Error generating response: {e}"
                            self.update_status(f"Case {original_index}: Error getting Deepseek response {j+1}/{num_retries}: {e}")
                        actual_outputs_deepseek.append(output)
                        self.update_responses_display("deepseek", actual_outputs_deepseek)
                        time.sleep(0.1)

                actual_outputs_gpt = []
                if self.gpt and num_retries > 0:
                    self.update_status(f"Case {original_index}: Getting {num_retries} responses from GPT...")
                    for j in range(num_retries):
                        self.update_status(f"Case {original_index}: GPT Response {j+1}/{num_retries} - Generating...")
                        output = "Error generating response."
                        try:
                            output = self.gpt.generate_response(prompt)
                            self.update_status(f"Case {original_index}: GPT Response {j+1}/{num_retries} - Received.")
                        except Exception as e:
                            output = f"Error generating response: {e}"
                            self.update_status(f"Case {original_index}: Error getting GPT response {j+1}/{num_retries}: {e}")
                        actual_outputs_gpt.append(output)
                        self.update_responses_display("gpt", actual_outputs_gpt)
                        time.sleep(0.1)

                case_results = {
                    "original_index": original_index,
                    "prompt": prompt,
                    "expected_output": expected_output,
                    "complexity": complexity, # Include complexity in results
                    "actual_outputs_deepseek": actual_outputs_deepseek,
                    "actual_outputs_gpt": actual_outputs_gpt,
                    "individual_evaluations_deepseek": [],
                    "individual_evaluations_gpt": [],
                    "consistency_evaluation_deepseek": "N/A",
                    "consistency_evaluation_gpt": "N/A"
                }

                if self.evaluator_model and num_retries > 0:
                    self.update_status(f"Case {original_index}: Starting Evaluation...")

                    self.update_status(f"Case {original_index}: Evaluating Deepseek individual responses...")
                    individual_evals_deepseek = []
                    for j, response in enumerate(actual_outputs_deepseek):
                        self.update_status(f"Case {original_index}: Evaluating Deepseek response {j+1}/{num_retries}...")
                        eval_result = eva.get_evaluation_result(
                            self.evaluator_model, prompt, expected_output, response
                        )
                        individual_evals_deepseek.append(eval_result)
                        self.update_evaluation_display("deepseek", individual_evals_deepseek)
                        time.sleep(0.1)

                    self.update_status(f"Case {original_index}: Evaluating GPT individual responses...")
                    individual_evals_gpt = []
                    for j, response in enumerate(actual_outputs_gpt):
                        self.update_status(f"Case {original_index}: Evaluating GPT response {j+1}/{num_retries}...")
                        eval_result = eva.get_evaluation_result(
                            self.evaluator_model, prompt, expected_output, response
                        )
                        individual_evals_gpt.append(eval_result)
                        self.update_evaluation_display("gpt", individual_evals_gpt)
                        time.sleep(0.1)

                    case_results["individual_evaluations_deepseek"] = individual_evals_deepseek
                    case_results["individual_evaluations_gpt"] = individual_evals_gpt

                    if num_retries > 1:
                        self.update_status(f"Case {original_index}: Evaluating Consistency...")
                        if actual_outputs_deepseek:
                            self.update_status(f"Case {original_index}: Evaluating Deepseek consistency...")
                            consistency_deepseek = eva.get_consistency_score(
                                self.evaluator_model, prompt, expected_output, actual_outputs_deepseek
                            )
                            case_results["consistency_evaluation_deepseek"] = consistency_deepseek
                            self.update_evaluation_display("deepseek", individual_evals_deepseek + [f"\nConsistency: {consistency_deepseek}"])
                            time.sleep(0.1)

                        if actual_outputs_gpt:
                            self.update_status(f"Case {original_index}: Evaluating GPT consistency...")
                            consistency_gpt = eva.get_consistency_score(
                                self.evaluator_model, prompt, expected_output, actual_outputs_gpt
                            )
                            case_results["consistency_evaluation_gpt"] = consistency_gpt
                            self.update_evaluation_display("gpt", individual_evals_gpt + [f"\nConsistency: {consistency_gpt}"])
                            time.sleep(0.1)

                else:
                    self.update_status(f"Case {original_index}: Evaluation skipped (Gemini model not initialized or retries is 0).")

                self.results.append(case_results)
                time.sleep(0.5)

            # Calculate average complexity of the run cases
            self.overall_avg_complexity = statistics.mean(run_complexities) if run_complexities else 0
            # Extract common keywords from run prompts
            self.common_keywords = self.extract_keywords(run_prompts, num_keywords=15, n_gram=1) # Extract top 15 single words
            self.update_metrics_display() # Update GUI with final average complexity


            self.update_status("All selected test cases processed. Calculating overall results...")
            self.calculate_and_show_overall_results()

        except Exception as e:
            self.update_status(f"An unexpected error occurred during testing: {e}")
            messagebox.showerror("Error", f"An unexpected error occurred during testing: {e}")
        finally:
            self.testing_complete()

    def load_test_cases(self, file_path):
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip().str.lower()
        if 'input' not in df.columns or 'expected output' not in df.columns:
            raise ValueError("CSV file must contain 'input' and 'expected output' columns (case-insensitive).")

        df = df[['input', 'expected output']].dropna()
        df.columns = ['prompt', 'expected_output']

        # Calculate complexity for each test case
        test_cases = df.to_dict(orient='records')
        for case in test_cases:
            case['complexity'] = self.calculate_test_case_complexity(case['prompt'], case['expected_output'])

        return test_cases

    def update_status(self, message: str):
        self.master.after(0, lambda: self.status_label.config(text=f"Status: {message}"))

    def update_test_case_display(self, prompt: str, expected_output: str):
        def update():
            self.prompt_text.config(state=tk.NORMAL)
            self.prompt_text.delete('1.0', tk.END)
            self.prompt_text.insert(tk.END, prompt)
            self.prompt_text.config(state=tk.DISABLED)

            self.expected_output_text.config(state=tk.NORMAL)
            self.expected_output_text.delete('1.0', tk.END)
            self.expected_output_text.insert(tk.END, expected_output)
            self.expected_output_text.config(state=tk.DISABLED)

        self.master.after(0, update)

    def clear_individual_results_display(self):
        def clear():
            self.deepseek_responses_text.config(state=tk.NORMAL)
            self.deepseek_responses_text.delete('1.0', tk.END)
            self.deepseek_responses_text.config(state=tk.DISABLED)

            self.deepseek_eval_text.config(state=tk.NORMAL)
            self.deepseek_eval_text.delete('1.0', tk.END)
            self.deepseek_eval_text.config(state=tk.DISABLED)

            self.gpt_responses_text.config(state=tk.NORMAL)
            self.gpt_responses_text.delete('1.0', tk.END)
            self.gpt_responses_text.config(state=tk.DISABLED)

            self.gpt_eval_text.config(state=tk.NORMAL)
            self.gpt_eval_text.delete('1.0', tk.END)
            self.gpt_eval_text.config(state=tk.DISABLED)

        self.master.after(0, clear)

    def update_responses_display(self, model_name: str, responses: List[str]):
        def update():
            text_widget = self.deepseek_responses_text if model_name == "deepseek" else self.gpt_responses_text
            text_widget.config(state=tk.NORMAL)
            text_widget.delete('1.0', tk.END)
            for i, response in enumerate(responses):
                text_widget.insert(tk.END, f"Response {i+1}:\n{response}\n---\n")
            text_widget.config(state=tk.DISABLED)

        self.master.after(0, update)

    def update_evaluation_display(self, model_name: str, evaluations: List[str]):
        def update():
            text_widget = self.deepseek_eval_text if model_name == "deepseek" else self.gpt_eval_text
            text_widget.config(state=tk.NORMAL)
            text_widget.delete('1.0', tk.END)
            for eval_result in evaluations:
                 text_widget.insert(tk.END, f"{eval_result}\n")
            text_widget.config(state=tk.DISABLED)

        self.master.after(0, update)

    def calculate_and_show_overall_results(self):
        num_retries = int(self.num_retries_entry.get())
        total_possible_evaluations = len(self.selected_test_cases) * num_retries
        gpt_valid_count_overall = 0
        deepseek_valid_count_overall = 0
        gpt_consistency_scores = []
        deepseek_consistency_scores = []
        run_complexities = [case.get('complexity', 0) for case in self.results]

        for case in self.results:
            for eval_result in case.get("individual_evaluations_gpt", []):
                if eva.parse_pass_fail(eval_result):
                    gpt_valid_count_overall += 1
            for eval_result in case.get("individual_evaluations_deepseek", []):
                 if eva.parse_pass_fail(eval_result):
                    deepseek_valid_count_overall += 1

            gpt_consistency_raw = case.get("consistency_evaluation_gpt")
            if gpt_consistency_raw and gpt_consistency_raw != "N/A":
                score = eva.parse_evaluation_score(gpt_consistency_raw)
                if score is not None:
                    gpt_consistency_scores.append(score)

            deepseek_consistency_raw = case.get("consistency_evaluation_deepseek")
            if deepseek_consistency_raw and deepseek_consistency_raw != "N/A":
                score = eva.parse_evaluation_score(deepseek_consistency_raw)
                if score is not None:
                    deepseek_consistency_scores.append(score)

        gpt_valid_percent_overall = (gpt_valid_count_overall / total_possible_evaluations) * 100 if total_possible_evaluations > 0 else 0
        deepseek_valid_percent_overall = (deepseek_valid_count_overall / total_possible_evaluations) * 100 if total_possible_evaluations > 0 else 0

        avg_gpt_consistency = statistics.mean(gpt_consistency_scores) if gpt_consistency_scores else 0
        avg_deepseek_consistency = statistics.mean(deepseek_consistency_scores) if deepseek_consistency_scores else 0
        avg_run_complexity = statistics.mean(run_complexities) if run_complexities else 0

        self.overall_stats = {
            "total_cases_loaded": len(self.all_test_cases),
            "total_cases_run": len(self.selected_test_cases),
            "num_retries": num_retries,
            "total_possible_evaluations": total_possible_evaluations,
            "gpt_valid_count_overall": gpt_valid_count_overall,
            "deepseek_valid_count_overall": deepseek_valid_count_overall,
            "gpt_valid_percent_overall": gpt_valid_percent_overall,
            "deepseek_valid_percent_overall": deepseek_valid_percent_overall,
            "avg_gpt_consistency": avg_gpt_consistency,
            "avg_deepseek_consistency": avg_deepseek_consistency,
            "show_consistency": num_retries > 1,
            "overall_coverage": self.overall_coverage,
            "overall_avg_complexity": avg_run_complexity, # This will now be the average of RUN cases
            "common_keywords": self.common_keywords # Include common keywords
        }

        overall_summary = (
            "\n--- Overall Results ---\n"
            f"Total Test Cases Loaded: {self.overall_stats.get('total_cases_loaded', 'N/A')}\n"
            f"Total Test Cases Run: {self.overall_stats.get('total_cases_run', 'N/A')}\n"
            f"Test Coverage: {self.overall_stats.get('overall_coverage', 0):.2f}%\n"
            f"Average Test Complexity (Run Cases): {self.overall_stats.get('overall_avg_complexity', 0):.2f}\n" # Displaying average of run cases
            f"Number of Retries per Case: {self.overall_stats.get('num_retries', 'N/A')}\n"
            f"Total Possible Evaluations: {self.overall_stats.get('total_possible_evaluations', 'N/A')}\n"
            f"GPT Accuracy: {self.overall_stats.get('gpt_valid_count_overall', 'N/A')}/{self.overall_stats.get('total_possible_evaluations', 'N/A')} ({self.overall_stats.get('gpt_valid_percent_overall', 0):.2f}% Valid)\n"
            f"Deepseek Accuracy: {self.overall_stats.get('deepseek_valid_count_overall', 'N/A')}/{self.overall_stats.get('total_possible_evaluations', 'N/A')} ({self.overall_stats.get('deepseek_valid_percent_overall', 0):.2f}% Valid)\n"
        )
        if self.overall_stats.get('show_consistency', False):
             overall_summary += (
                f"GPT Consistency (Avg Score 1-5): {self.overall_stats.get('avg_gpt_consistency', 0):.2f}\n"
                f"Deepseek Consistency (Avg Score 1-5): {self.overall_stats.get('avg_deepseek_consistency', 0):.2f}\n"
             )
        else:
            overall_summary += "\nConsistency evaluation skipped (retries = 1).\n"

        overall_summary += "\nCommon Keywords (from run prompts):\n"
        if self.common_keywords:
            for keyword, count in self.common_keywords:
                overall_summary += f"- {keyword} ({count})\n"
        else:
            overall_summary += "No keywords found.\n"


        self.update_status("Overall results calculated.")
        print(overall_summary)

        # Show the graph in the GUI
        self.show_results_graph_in_gui(
            self.overall_stats.get('gpt_valid_percent_overall', 0),
            self.overall_stats.get('deepseek_valid_percent_overall', 0),
            self.overall_stats.get('avg_gpt_consistency', 0),
            self.overall_stats.get('avg_deepseek_consistency', 0),
            self.overall_stats.get('show_consistency', False)
        )

        self.master.after(0, lambda: self.download_button.config(state=tk.NORMAL))

    def show_results_graph_in_gui(self, gpt_accuracy, deepseek_accuracy, gpt_consistency, deepseek_consistency, show_consistency):
        """Creates and displays the performance graph in the GUI."""
        if self.graph_canvas_widget:
            self.graph_canvas_widget.destroy()
            self.graph_canvas_widget = None
            plt.close(self.fig)
            self.fig = None

        self.fig, ax1 = plt.subplots(figsize=(8, 4))

        models = ['GPT', 'Deepseek']
        accuracy_scores = [gpt_accuracy, deepseek_accuracy]

        bars1 = ax1.bar(models, accuracy_scores, color=['skyblue', 'lightcoral'], width=0.4, label='Accuracy (%)')
        ax1.set_ylabel('Accuracy (%)', color='black')
        ax1.tick_params(axis='y', labelcolor='black')
        ax1.set_ylim(0, 100)

        for bar in bars1:
            yval = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2, yval + 2, f'{yval:.2f}%', ha='center', va='bottom')

        if show_consistency:
            ax2 = ax1.twinx()
            consistency_scores = [gpt_consistency, deepseek_consistency]
            lines = ax2.plot(models, consistency_scores, color='green', marker='o', linestyle='-', linewidth=2, label='Consistency (1-5)')
            ax2.set_ylabel('Consistency Score (1-5)', color='green')
            ax2.tick_params(axis='y', labelcolor='green')
            ax2.set_ylim(0, 5)

            for i, score in enumerate(consistency_scores):
                 ax2.text(i, score + 0.1, f'{score:.2f}', ha='left', va='bottom', color='green')

            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        else:
             ax1.legend(loc='upper left')

        plt.title('Overall ChatBot Performance')
        plt.tight_layout()

        # Embed the matplotlib figure in the Tkinter GUI
        self.graph_canvas = FigureCanvasTkAgg(self.fig, master=self.graph_display_frame)
        self.graph_canvas.draw()
        self.graph_canvas_widget = self.graph_canvas.get_tk_widget()
        self.graph_canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    # Removed the old show_results_graph function

    def hide_graph(self):
        """Hides the graph from the GUI."""
        if self.graph_canvas_widget:
            self.graph_canvas_widget.destroy()
            self.graph_canvas_widget = None
            if self.fig: # Close the matplotlib figure to free up memory
                plt.close(self.fig)
                self.fig = None


    def clear_results_display(self):
        self.update_test_case_display("", "")
        self.clear_individual_results_display()
        self.hide_graph() # Also hide the graph when clearing results


    def testing_complete(self):
        self.is_running = False
        self.master.after(0, lambda: self.start_button.config(state=tk.NORMAL))

    def download_report(self):
        if not self.results:
            messagebox.showinfo("Info", "No results available to download.")
            return

        filetypes = ( ("HTML Report", "*.html"), ("All Files", "*.*"))

        file_path = filedialog.asksaveasfilename(
            initialdir=".",
            title="Save Test Report",
            defaultextension=".html",
            filetypes=filetypes
        )

        if file_path:
            try:
                lower_file_path = file_path.lower()

                if lower_file_path.endswith('.html'):
                    self.generate_html_report(file_path)
                    messagebox.showinfo("Success", f"HTML report saved successfully to {file_path}")
                elif lower_file_path.endswith('.json'):
                     with open(file_path, 'w', encoding='utf-8') as f:
                        json.dump(self.results, f, indent=4, ensure_ascii=False)
                     messagebox.showinfo("Success", f"JSON report saved successfully to {file_path}")
                else:
                     messagebox.showwarning(
                         "Saving Issue",
                         f"The file was saved without a .json or .html extension: {file_path}\n\n"
                         "Please ensure you select the desired 'Files of type' (.json or .html) "
                         "in the save dialog before clicking 'Save'."
                     )

            except IOError as e:
                messagebox.showerror("Error", f"Error saving report: {e}")
            except Exception as e:
                 messagebox.showerror("Error", f"An unexpected error occurred while saving the report: {e}")

    def generate_html_report(self, file_path: str):
        if not self.overall_stats:
            messagebox.showwarning("Warning", "Overall results not available for HTML report.")
            return

        # Generate the graph image for the HTML report
        fig, ax1 = plt.subplots(figsize=(8, 4))

        models = ['GPT', 'Deepseek']
        accuracy_scores = [
            self.overall_stats.get('gpt_valid_percent_overall', 0),
            self.overall_stats.get('deepseek_valid_percent_overall', 0)
        ]

        bars1 = ax1.bar(models, accuracy_scores, color=['skyblue', 'lightcoral'], width=0.4, label='Accuracy (%)')
        ax1.set_ylabel('Accuracy (%)', color='black')
        ax1.tick_params(axis='y', labelcolor='black')
        ax1.set_ylim(0, 100)

        for bar in bars1:
            yval = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2, yval + 2, f'{yval:.2f}%', ha='center', va='bottom')

        if self.overall_stats.get('show_consistency', False):
            ax2 = ax1.twinx()
            consistency_scores = [
                self.overall_stats.get('avg_gpt_consistency', 0),
                self.overall_stats.get('avg_deepseek_consistency', 0)
            ]
            lines = ax2.plot(models, consistency_scores, color='green', marker='o', linestyle='-', linewidth=2, label='Consistency (1-5)')
            ax2.set_ylabel('Consistency Score (1-5)', color='green')
            ax2.tick_params(axis='y', labelcolor='green')
            ax2.set_ylim(0, 5)

            for i, score in enumerate(consistency_scores):
                ax2.text(i, score + 0.1, f'{score:.2f}', ha='left', va='bottom', color='green')

            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        else:
            ax1.legend(loc='upper left')

        plt.title('Overall API Performance')
        plt.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        plt.close(fig) # Close the figure after saving to buffer
        data = base64.b64encode(buf.getbuffer()).decode("ascii")
        img_html = f'<img src="data:image/png;base64,{data}" alt="Overall Performance Graph" style="max-width: 100%; height: auto;">'

        # HTML content with table for overall stats and keyword section
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>ChatGpt vs DeepSeek Travel Topic Evaluation Report</title>
            <style>
                body {{ font-family: 'Arial', sans-serif; margin: 0; padding: 0; background-color: #f4f7f6; color: #333; }}
                .container {{ max-width: 1000px; margin: 20px auto; background: #fff; padding: 30px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }}
                h1 {{ text-align: center; color: #0056b3; margin-bottom: 20px; padding-bottom: 10px; border-bottom: 2px solid #0056b3; }}
                h2 {{ color: #007bff; margin-top: 20px; margin-bottom: 10px; }}
                h3 {{ color: #17a2b8; margin-top: 15px; margin-bottom: 8px; }}
                .stats-box {{ background-color: #e9ecef; padding: 20px; margin-bottom: 20px; border-radius: 8px; border: 1px solid #ced4da; }}
                .stats-table {{ width: 100%; border-collapse: collapse; margin-top: 15px; }}
                .stats-table th, .stats-table td {{ border: 1px solid #ced4da; padding: 10px; text-align: left; }}
                .stats-table th {{ background-color: #d0d9e2; font-weight: bold; }}
                .graph-container {{ text-align: center; margin-bottom: 30px; padding: 20px; background-color: #e9ecef; border-radius: 8px; border: 1px solid #ced4da; }}
                .test-case {{ border: 1px solid #dee2e6; padding: 20px; margin-bottom: 25px; border-radius: 8px; background-color: #f8f9fa; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }}
                .test-case h3 {{ border-bottom: 1px solid #17a2b8; padding-bottom: 5px; }}
                .test-case pre {{ white-space: pre-wrap; word-wrap: break-word; background-color: #e9ecef; padding: 15px; border: 1px solid #ced4da; border-radius: 5px; overflow-x: auto; margin-top: 10px; }}
                .evaluation {{ font-style: italic; font-weight: bold; margin-top: 5px; padding-left: 15px; border-left: 3px solid; }}
                .evaluation.fail {{ color: #dc3545; border-left-color: #dc3545; }}
                .evaluation.pass {{ color: #28a745; border-left-color: #28a745; }}
                .evaluation.consistency {{ color: #0000CC; border-left-color: #0000CC; }}
                strong {{ color: #555; }}
                .keywords-box {{ background-color: #e9ecef; padding: 20px; margin-top: 20px; border-radius: 8px; border: 1px solid #ced4da; }}
                .keywords-list {{ list-style: none; padding: 0; }}
                .keywords-list li {{ background-color: #d0d9e2; margin: 5px 0; padding: 8px; border-radius: 4px; display: inline-block; margin-right: 10px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>ChatGpt vs DeepSeek Travel Topic Evaluation Report</h1>

                <div class="stats-box">
                    <h2>Overall Results</h2>
                    <table class="stats-table">
                        <tr>
                            <th>Metric</th>
                            <th>Value</th>
                        </tr>
                        <tr>
                            <td>Total Test Cases Loaded</td>
                            <td>{self.overall_stats.get('total_cases_loaded', 'N/A')}</td>
                        </tr>
                        <tr>
                            <td>Total Test Cases Run</td>
                            <td>{self.overall_stats.get('total_cases_run', 'N/A')}</td>
                        </tr>
                        <tr>
                            <td>Test Coverage</td>
                            <td>{self.overall_stats.get('overall_coverage', 0):.2f}%</td>
                        </tr>
                        <tr>
                            <td>Average Test Complexity (Run Cases)</td>
                            <td>{self.overall_stats.get('overall_avg_complexity', 0):.2f}</td>
                        </tr>
                        <tr>
                            <td>Number of Retries per Case</td>
                            <td>{self.overall_stats.get('num_retries', 'N/A')}</td>
                        </tr>
                        <tr>
                            <td>Total Possible Evaluations</td>
                            <td>{self.overall_stats.get('total_possible_evaluations', 'N/A')}</td>
                        </tr>
                        <tr>
                            <td>GPT Accuracy</td>
                            <td>{self.overall_stats.get('gpt_valid_count_overall', 'N/A')}/{self.overall_stats.get('total_possible_evaluations', 'N/A')} ({self.overall_stats.get('gpt_valid_percent_overall', 0):.2f}% Valid)</td>
                        </tr>
                        <tr>
                            <td>Deepseek Accuracy</td>
                            <td>{self.overall_stats.get('deepseek_valid_count_overall', 'N/A')}/{self.overall_stats.get('total_possible_evaluations', 'N/A')} ({self.overall_stats.get('deepseek_valid_percent_overall', 0):.2f}% Valid)</td>
                        </tr>
                        """
        if self.overall_stats.get('show_consistency', False):
             html_content += f"""
                        <tr>
                            <td>GPT Consistency (Avg Score 1-5)</td>
                            <td>{self.overall_stats.get('avg_gpt_consistency', 0):.2f}</td>
                        </tr>
                        <tr>
                            <td>Deepseek Consistency (Avg Score 1-5)</td>
                            <td>{self.overall_stats.get('avg_deepseek_consistency', 0):.2f}</td>
                        </tr>
             """
        html_content += f"""
                    </table>
                </div>

                <div class="graph-container">
                    <h2>Performance Graph</h2>
                    {img_html}
                </div>

                <div class="keywords-box">
                    <h2>Common Keywords (from Run Prompts)</h2>
                    <ul class="keywords-list">
        """
        if self.overall_stats.get('common_keywords'):
            for keyword, count in self.overall_stats['common_keywords']:
                html_content += f"<li>{keyword} ({count})</li>"
        else:
            html_content += "<li>No keywords found.</li>"

        html_content += """
                    </ul>
                </div>

                <h2>Detailed Test Case Results</h2>
        """

        if self.results:
            for i, case in enumerate(self.results):
                html_content += f"""
                <div class="test-case">
                    <h3>Test Case (Original Index: {case.get('original_index', 'N/A')})</h3>
                    <p><strong>Complexity:</strong> {case.get('complexity', 'N/A'):.2f}</p>
                    <p><strong>Prompt:</strong></p>
                    <pre>{case.get('prompt', 'N/A')}</pre>
                    <p><strong>Expected Output:</strong></p>
                    <pre>{case.get('expected_output', 'N/A')}</pre>

                    <h4>Deepseek Responses & Evaluation:</h4>
                    """
                if case.get('actual_outputs_deepseek'):
                    for j, response in enumerate(case['actual_outputs_deepseek']):
                        html_content += f"<p><strong>Response {j+1}:</strong></p><pre>{response}</pre>"
                    if case.get('individual_evaluations_deepseek'):
                         html_content += "<p><strong>Individual Evaluations:</strong></p>"
                         for eval_res in case['individual_evaluations_deepseek']:
                             eval_class = 'pass' if eva.parse_pass_fail(eval_res) else 'fail'
                             html_content += f"<p class='evaluation {eval_class}'>- {eval_res}</p>"
                    if case.get('consistency_evaluation_deepseek') and case['consistency_evaluation_deepseek'] != "N/A":
                         html_content += f"<p class='evaluation consistency'><strong>Consistency Evaluation:</strong> {case['consistency_evaluation_deepseek']}</p>"
                else:
                    html_content += "<p>No Deepseek responses or evaluation available.</p>"

                html_content += """
                    <h4>GPT Responses & Evaluation:</h4>
                    """
                if case.get('actual_outputs_gpt'):
                    for j, response in enumerate(case['actual_outputs_gpt']):
                        html_content += f"<p><strong>Response {j+1}:</strong></p><pre>{response}</pre>"
                    if case.get('individual_evaluations_gpt'):
                         html_content += "<p><strong>Individual Evaluations:</strong></p>"
                         for eval_res in case['individual_evaluations_gpt']:
                             eval_class = 'pass' if eva.parse_pass_fail(eval_res) else 'fail'
                             html_content += f"<p class='evaluation {eval_class}'>- {eval_res}</p>"
                    if case.get('consistency_evaluation_gpt') and case['consistency_evaluation_gpt'] != "N/A":
                         html_content += f"<p class='evaluation consistency'><strong>Consistency Evaluation:</strong> {case['consistency_evaluation_gpt']}</p>"""
                else:
                    html_content += "<p>No GPT responses or evaluation available.</p>"

                html_content += "</div>"

        html_content += """
            </div> </body>
        </html>
        """

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(html_content)


    def hide_graph(self):
        """Hides the graph from the GUI."""
        if self.graph_canvas_widget:
            self.graph_canvas_widget.destroy()
            self.graph_canvas_widget = None
            if self.fig: # Close the matplotlib figure to free up memory
                plt.close(self.fig)
                self.fig = None


    def clear_results_display(self):
        self.update_test_case_display("", "")
        self.clear_individual_results_display()
        self.hide_graph() # Also hide the graph when clearing results


    def testing_complete(self):
        self.is_running = False
        self.master.after(0, lambda: self.start_button.config(state=tk.NORMAL))

if __name__ == "__main__":
    root = tk.Tk()
    app = APITesterGUI(root)
    root.mainloop()

