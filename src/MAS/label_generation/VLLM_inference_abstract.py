from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Union
from datasets import Dataset, load_dataset
from vllm import LLM, SamplingParams


DEFAULT_MAX_RETRIES = 3

class VLLMInferenceWithRetry(ABC):
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
       
        # self.boxed_pattern = re.compile(r"\\boxed\{([^}]+)\}")

    @abstractmethod
    def _parse_and_validate(self, texts: List[str], current_answers:List[str]) -> Optional[str]:
        """
        Adds new unseen answers from texts to current answer list.

        Args:
            texts: completions from LLM 
            current_answers: Existing list of answer.

        Output:
            updated_answers: returns an updated list of current answers with new answers from texts 
        """
        pass

    @abstractmethod
    def get_prompt(self, sample:dict) -> str:
        """
        Formats prompt to VLLM based on sample

        Args:
            sample: an instance of a hf dataset

        Output:
            prompt 
        """
        pass

    @abstractmethod
    def get_answer(self, sample:dict) -> str:
        """
        gets answer to verify against llm completions

        Args:
            sample: an instance of a hf dataset

        Output:
            Answer
        """
        pass


    def infer_prompt(self, prompt: str, org_answer: str, prompt_id: Optional[Union[int, str]] = None) -> str:
        
        log_prefix = f"Prompt {prompt_id}: " if prompt_id is not None else "Prompt: "
        if self.verbose:
            print(f"{log_prefix}Starting inference...")

        current_options = [org_answer]

        for attempt in range(self.max_retries + 1):
            if len(current_options) > 4:
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

        if len(current_options)<5:        
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
            prompt = self.get_prompt(sample)
            answer = self.get_answer(sample)
            if self.verbose and i == 0:
                print("PROMPT:", prompt)
            result = self.infer_prompt(prompt, prompt_id=i, org_answer = answer)
           
            sample['alternate_options'] = result
            new_samples.append(sample)
            if self.verbose:
                print(f"--- Processed prompt {i+1}/{total_prompts} ---")
            # break

        if self.verbose:
            print("\nDataset processing complete.")
            
        return new_samples

if __name__ == "__main__":    
    dataset_id = None
    json_save_path=None
    disk_save_path=None

    dataset = load_dataset(dataset_id, split='test')

    MODEL_ID = None
    
    print("--- Running with default sampling parameters ---")
    try:
        vllm_kwargs = {"tensor_parallel_size": 1}
        custom_sampling_params = SamplingParams(
            temperature=1.0,
            top_p=0.9,
            n=5
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

        processed_hf_dataset.to_json(json_save_path)
        processed_hf_dataset.save_to_disk(disk_save_path)


    except Exception as e:
         print(f"\n--- Error during example run ---")
         print(f"An exception occurred: {e}")
         print("Please ensure vLLM is installed, a compatible GPU is available,")
         print(f"and the model '{MODEL_ID}' can be loaded.")
         raise e