import re
import json
import time
from typing import List, Optional, Dict, Any, Union
from datasets import Dataset, load_dataset
from vllm import LLM, SamplingParams
from rapidfuzz import fuzz
from datetime import datetime
from rapidfuzz import fuzz

# --- Configuration ---
DEFAULT_MAX_RETRIES = 3
# DEFAULT_RETRY_DELAY_SECONDS = 1

import ast
import math

PROMPT = """Given the following function description and inputs, predict and return only the output value. Do not provide any explanation or code.

Function Description:
{function_desc}

{function_code}

Inputs:
{function_run}

Output:"""

def is_valid_literal(s):
    try:
        ast.literal_eval(s)
        return True
    except (ValueError, SyntaxError):
        return False

def parse_literal(s):
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return s.strip()

def deep_equal_str_outputs(str1, str2, float_tol=1e-9):
    a = parse_literal(str1)
    b = parse_literal(str2)
    return deep_equal(a, b, float_tol)

def deep_equal(a, b, float_tol=1e-9):
    if type(a) != type(b):
        return False

    if isinstance(a, float):
        return math.isclose(a, b, abs_tol=float_tol)

    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(deep_equal(x, y, float_tol) for x, y in zip(a, b))

    if isinstance(a, dict):
        return (a.keys() == b.keys() and
                all(deep_equal(a[k], b[k], float_tol) for k in a))

    return a == b

class VLLMInferenceWithRetry:
    def __init__(
        self,
        model_name_or_path: str,
        sampling_params: Optional[SamplingParams] = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        # retry_delay: float = DEFAULT_RETRY_DELAY_SECONDS,
        vllm_init_kwargs: Optional[Dict[str, Any]] = None,
        verbose: bool = True
    ):
        """
        Initializes the inference handler.

        Args:
            model_name_or_path: Path or Hugging Face identifier for the vLLM model.
            sampling_params: vLLM SamplingParams object. If None, default ones are used.
            max_retries: Maximum number of retry attempts for a single prompt.
            retry_delay: Seconds to wait between retry attempts.
            vllm_init_kwargs: Optional dictionary of keyword arguments passed directly
                              to the vLLM LLM class initializer (e.g., tensor_parallel_size).
            verbose: If True, print information about retries and failures.
        """
        if vllm_init_kwargs is None:
            vllm_init_kwargs = {}
        try:
            self.llm = LLM(model=model_name_or_path, **vllm_init_kwargs)
            print(f"Successfully loaded model: {model_name_or_path}")
        except Exception as e:
            print(f"Error loading vLLM model '{model_name_or_path}': {e}")
            raise e

        self.sampling_params = sampling_params if sampling_params else SamplingParams(
            # temperature=0.6, 
            max_tokens=1024
        )
        self.max_retries = max_retries
        # self.retry_delay = retry_delay
        self.verbose = verbose
       
        self.boxed_pattern = re.compile(r"\\boxed\{([^}]+)\}")
        
    def get_answer(self, text:str):
        answer = text.split('</think>')[-1]
        return answer.strip()

    def _parse_and_validate(self, texts: List[str], current_answers:List[str]) -> Optional[str]:

        for text in texts:
            if not is_valid_literal(text):
                continue
            for answer in current_answers:
                if deep_equal(text, answer):
                    break
            else:
                current_answers.append(text)
        return current_answers

    def infer_prompt(self, prompt: str, org_answer: str, prompt_id: Optional[Union[int, str]] = None) -> str:
        
        log_prefix = f"Prompt {prompt_id}: " if prompt_id is not None else "Prompt: "
        if self.verbose:
            print(f"{log_prefix}Starting inference...")

        current_options = [org_answer]

        for attempt in range(self.max_retries + 1):
            if len(current_options) > 3:
                break
            try:
                outputs = self.llm.chat([
                    {"role":"user", "content":prompt}
                    ],
                    self.sampling_params
                    )
                
                generated_texts = [output.text for output in outputs[0].outputs]

                if self.verbose:
                    print(f"{log_prefix}Attempt {attempt+1}/{self.max_retries+1} - Raw Output: '{generated_texts}...'") # Log snippet

                current_options = self._parse_and_validate(generated_texts, current_options)

            except Exception as e:
                if self.verbose:
                    print(f"{log_prefix}Attempt {attempt+1} - Error during vLLM generation: {e}")

        if len(current_options)<4:        
            if self.verbose:
                print(f"{log_prefix}All {self.max_retries + 1} attempts failed.")
        
        return current_options[1:] # remove org answer from list

    def process_dataset(self, samples: List[dict[str, Any]]) -> List[str]:
        """
        Processes a list of prompts, performing inference with retries for each.

        Args:
            prompts: A list of input prompt strings.

        Returns:
            A list containing the inference result (validated answer or
            FAILURE_MARKER) for each corresponding prompt.
        """
        new_samples = []
        total_prompts = len(samples)
        if self.verbose:
            print(f"\nStarting dataset processing for {total_prompts} prompts...")

        for i, sample in enumerate(samples):
            parsed_test = parse_literal(sample['test'])[0]
            sample['input'] = parsed_test['input'].replace("\n", ", ")
            sample['output'] = parsed_test['output']
            sample['test_type'] = parsed_test['testtype']

            prompt = PROMPT.format(function_desc=sample['question_content'],
                                   function_code=sample['starter_code'], 
                                   function_run=sample['input']
                                   )
            if self.verbose and i == 0:
                print("PROMPT:", prompt)
            result = self.infer_prompt(prompt, prompt_id=i, org_answer = sample['output'])
           
            sample['alternate_options'] = result
            new_samples.append(sample)
            if self.verbose:
                print(f"--- Processed prompt {i+1}/{total_prompts} ---")
            # break

        if self.verbose:
            print("\nDataset processing complete.")
            
        return new_samples

if __name__ == "__main__":    
    dataset_id = 'livecodebench/test_generation'

    dataset = load_dataset(dataset_id, split='test')

    # for easier checking
    # filtered_dataset = dataset.filter(lambda x: len(x['answer'])<35) 

    # MODEL_ID = "/data/rishabh/tej/hub/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/0cb88a4f764b7a12671c53f0838cd831a0843b95"
    MODEL_ID = "/data/rishabh/tej/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
    
    print("--- Running with default sampling parameters ---")
    try:
        vllm_kwargs = {"tensor_parallel_size": 1}
        custom_sampling_params = SamplingParams(
            temperature=1.0,
            top_p=0.9,
            n=5,
            max_tokens=1024
        )
        inferencer = VLLMInferenceWithRetry(
            model_name_or_path=MODEL_ID,
            max_retries=2,
            sampling_params=custom_sampling_params,
            verbose=True,
            vllm_init_kwargs=vllm_kwargs
        )

        processed_samples = inferencer.process_dataset(dataset)

        processed_hf_dataset = Dataset.from_list(processed_samples)

        print(processed_hf_dataset[0])

        processed_hf_dataset.to_json('/home/rishabh/Tej/MAS/data/processed/livebench_test_pred_8B.json')
        processed_hf_dataset.save_to_disk('/home/rishabh/Tej/MAS/data/processed/livebench_test_pred_8B')


    except Exception as e:
         print(f"\n--- Error during example run ---")
         print(f"An exception occurred: {e}")
         print("Please ensure vLLM is installed, a compatible GPU is available,")
         print(f"and the model '{MODEL_ID}' can be loaded.")
         raise e