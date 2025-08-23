import re
from VLLM_inference_abstract import VLLMInferenceWithRetry
from vllm import SamplingParams
from typing import List, Optional

from datasets import load_dataset, Dataset
from math_verify import parse, verify

MATH500_PROMPT= """You are an expert mathematician. Solve the problem step by step, but do not use \\boxed\{\} until the very end. At the end, on its own line, output **only** the final answer as:

\\boxed\{…\}

Problem:
"""

def remove_size_macros(s: str) -> str:
    """
    Delete all \left and \right tokens so that
    parentheses remain plain and parseable by sympy.
    """
    return re.sub(r'\\left|\\right', '', s)

class Math500InferenceWithRetry(VLLMInferenceWithRetry):

    def _parse_and_validate(self, texts: List[str], current_answers:List[str]) -> Optional[str]:
        """
        Adds new unseen answers from texts to current answer list.

        Args:
            texts: completions from LLM 
            current_answers: Existing list of answer.

        Output:
            updated_answers: returns an updated list of current answers with new answers from texts 
        """
        # breakpoint()
        for text in texts:
            if "\\boxed{" not in text:
                continue

            parsed_text = parse(text)
            
            if len(parsed_text) == 0:
                continue

            for existing_answer in current_answers:
                parsed_answer = parse("$"+existing_answer+"$")
                
                if verify(parsed_answer, parsed_text):
                    break
            else:
                current_answers.append(parsed_text[-1])

        if self.verbose:
            print("Current Answers Found:", current_answers)
        return current_answers

    def get_prompt(self, sample:dict) -> str:
        """
        Formats prompt to VLLM based on sample

        Args:
            sample: an instance of a hf dataset

        Output:
            prompt 
        """
        return MATH500_PROMPT + sample['problem']

    def get_answer(self, sample:dict) -> str:
        """
        gets answer to verify against llm completions

        Args:
            sample: an instance of a hf dataset

        Output:
            Answer
        """
        return remove_size_macros(sample['answer'])

if __name__ == "__main__":
    dataset_id = 'HuggingFaceH4/MATH-500'
    dataset = load_dataset(dataset_id, split='test')
    
    json_save_path="/data/rishabh/tej/conformity/processed/Math_500.json"
    disk_save_path="/data/rishabh/tej/conformity/processed/Math_500"

    MODEL_ID = "/data/rishabh/tej/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
    
    print("--- Running with default sampling parameters ---")
    try:
        vllm_kwargs = {"tensor_parallel_size": 4}
        custom_sampling_params = SamplingParams(
            temperature=1.2,
            top_p=0.9,
            n=20,
            max_tokens=1024
        )
        inferencer = Math500InferenceWithRetry(
            model_name_or_path=MODEL_ID,
            max_retries=1,
            sampling_params=custom_sampling_params,
            verbose=True,
            vllm_init_kwargs=vllm_kwargs
        )

        processed_samples = inferencer.process_dataset(dataset)

        processed_hf_dataset = Dataset.from_list(processed_samples)

        print(processed_hf_dataset[0])

        processed_hf_dataset.to_json(json_save_path)
        processed_hf_dataset.save_to_disk(disk_save_path)


    except Exception as e:
         print(f"\n--- Error during example run ---")
         print(f"An exception occurred: {e}")
         print("Please ensure vLLM is installed, a compatible GPU is available,")
         print(f"and the model '{MODEL_ID}' can be loaded.")
         raise e