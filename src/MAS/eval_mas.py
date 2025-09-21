# import debugpy; debugpy.connect(("localhost", 9501))
import argparse
import json
import math
import os
import random
import re
import statistics
import traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from string import ascii_uppercase
from time import time
from typing import Any, Dict, List, Optional

from datasets import Dataset, load_from_disk
from scipy.stats import entropy
from tqdm import tqdm, trange

from MAS.utils import set_seed
from MAS.utils.eval_utils import (CONFIDENCE_PROTOCOLS, PROMPT_TEMPLATE,
                                  RAW_PROMPT_TEMPLATE, RESPONSE_PROTOCOLS,
                                  EvalConfig, build_example, extract_answer,
                                  generate_llm_chat, generate_llm_reflection,
                                  get_llm_client, get_option_content)
from MAS.utils.logging_utils import setup_logger
from dotenv import load_dotenv

# Set up logging
logger = setup_logger()
# Set seed for reproducibility
set_seed(42)
load_dotenv()

N_AGENTS = 6
MAX_WORKERS = int(os.environ.get('MAX_WORKERS_NUM', 4))  # Default max workers for thread pool


class EvalManager:
    def __init__(self, args: argparse.Namespace):
        self.ans_map = {i: letter for i, letter in enumerate(ascii_uppercase)}
        self.model_names = args.models  # Qwen/Qwen3-0.6B
        self.clients = {
            model: get_llm_client(model, ip=ip, port_number=port)
            for model, ip, port in zip(self.model_names, args.ips, args.port_numbers)
        }
        self.save_root = args.save_root
        self.dataset_path = args.dataset_path
        self.mode = args.mode
        self.tag = args.tag
        self.testing = args.testing
        self.temperature = args.temperature

    def save_results(self, config: EvalConfig, outputs: Dict, acc: int, failed_idx: set):
        parts = config.model.split('/')
        model_dir = "_".join(parts[1:]) if len(parts) > 1 else parts[0]
        results_path = os.path.join(self.save_root, model_dir, config.fname)
        try:
            with open(results_path, 'w', encoding="utf-8") as f:
                json.dump({
                    'config': config.__dict__,
                    'outputs': outputs,
                    'correct_num': acc,
                    'failed_idx': list(failed_idx),
                }, f, indent=2)
        except IOError as e:
            logger.error(f"Failed to save results to {results_path}: {e}")

    def build_prompts(self, config: EvalConfig, eval_data: Dataset, raw_results: Optional[Dict[str, Any]] = None) -> List[str]:
        if config.tag == 'RAW':
            formatted_prompts = [RAW_PROMPT_TEMPLATE.format(current=data) for data in eval_data['formatted_question']]
            
        else:
            formatted_prompts = []
            for idx, sample in enumerate(eval_data):
                history = sample['history']
                predefined_action = {"y_pred": raw_results['y_pred'][idx], "hard_oppose": raw_results['hard_oppose'][idx], "easy_oppose": raw_results['easy_oppose'][idx]}
                current = build_example(sample, N_AGENTS, sample['protocol'], add_yourself=False, predefined_action=predefined_action)
                formatted_prompts.append(PROMPT_TEMPLATE.format(number=N_AGENTS, history=history, current=current, idx=idx))
        
        return formatted_prompts

    
    def evaluate_single_config(self, config: EvalConfig, eval_data: Dataset, raw_results: Optional[Dict[str, Any]] = None, is_failed_example_loop: bool = False) -> Dict:
        try:        
            outputs = defaultdict(lambda: [None for _ in range(len(eval_data))])
            idx_list = range(len(eval_data))
            futures = {}  # Initialize futures dictionary
            
            # Determine which examples to go over
            if is_failed_example_loop:

                with open(f'{self.save_root}/{config.fname}','r') as f:
                    results = json.load(f)
                
                # Load up `outputs` with the results from the completed examples
                outputs.update(results['outputs'])

                idx_list = results['failed_idx'] 
                logger.info('Going over these examples:', idx_list)
            
            formatted_prompts = self.build_prompts(config, eval_data, raw_results)    
            failed_idx = set()
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                for idx in idx_list:
                    futures[executor.submit(self.process_single_example, idx, formatted_prompts, eval_data, config, failed_idx, raw_results)] = idx
                
                for cnt, future in enumerate(tqdm(as_completed(futures), total=len(futures), desc=f"Processing {config.model}-{config.mode}-{config.tag}")):
                    idx = futures[future]
                    result_dict = future.result()
                    for key, val in result_dict.items():
                        outputs[key][idx] = val
                    
                    if cnt % 100 == 0 or cnt + 1 == len(idx_list):
                        logger.info(f'=== PROGRESS: {cnt + 1}/{len(idx_list)} ===')
                        
                        # Compute metrics
                        acc = sum([int(is_correct) for is_correct in outputs['is_correct'] if is_correct is not None])
                        logger.info(f'Acc (%): {acc / len(idx_list) * 100:.2f}')
                        logger.info(f'Num failed: {len(failed_idx)}')
                        
                        self.save_results(config, outputs, acc, failed_idx)
                
            # update protocol with uncertainty labels
            median_uncertainty = statistics.mean(outputs['uncertainty'].copy())
            for idx in idx_list:
                if outputs['uncertainty'][idx] <= median_uncertainty:
                    outputs['protocol'][idx] = f"{outputs['protocol'][idx]}_{CONFIDENCE_PROTOCOLS[1]}"  # 0_SUPPORT_CORRECT_HIGH
                else:
                    outputs['protocol'][idx] = f"{outputs['protocol'][idx]}_{CONFIDENCE_PROTOCOLS[0]}"
            
            # save final results again
            self.save_results(config, outputs, acc, failed_idx)
                
                            
        except KeyboardInterrupt:
            if 'futures' in locals():
                for t in futures:
                    t.cancel()
        except Exception as e:
            logger.error(traceback.format_exc())
            if 'futures' in locals():
                for t in futures:
                    t.cancel()

        return outputs
        
    
    def process_single_example(self, idx: int, formatted_prompts: List[str], eval_data: Dataset, config: EvalConfig, failed_idx: set, raw_results: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process a single example with both raw and protocol-based evaluation"""
        
        responses = generate_llm_chat(
            self.clients[config.model],
            config.model,
            formatted_prompts[idx],
            {
                "n": 8 if config.tag == 'RAW' else 1,
                "temperature": 1.0 if config.tag == 'RAW' else self.temperature,
                "top_p": 0.9,
                "top_k": 50,
                "min_p": 0.02,
                "seed": 42,
            },
            self.mode,
            max_retries=3
        )
        
        preds = [extract_answer(response) for response in responses]
        
        # find uncertainty
        num_options = len(eval_data[idx]['wrong_options']) + 1
        pred_counts = {self.ans_map[i]: 0 for i in range(num_options)}
        for pred in preds:
            if pred in pred_counts:
                pred_counts[pred] += 1
            else:
                # Catch failures
                logger.warning(f"Invalid {idx}th prediction ##{pred}## for options {pred_counts.keys()}")
        
        majority_pred = max(pred_counts.items(), key=lambda x: x[1])[0]
        majority_pred_content = get_option_content(majority_pred, eval_data[idx]['wrong_options'], eval_data[idx]['gt_option'])
        is_correct = int(majority_pred_content == eval_data[idx]['gt_option'])
        uncertainty = entropy(list(pred_counts.values()))

        if config.tag == 'RAW':
            if is_correct:
                # For correct predictions:
                # Hard oppose: wrong answer with highest probability
                wrong_probs = {opt: pred_counts[opt] for opt in pred_counts if get_option_content(opt, eval_data[idx]['wrong_options'], eval_data[idx]['gt_option']) != eval_data[idx]['gt_option']}
                hard_oppose = max(wrong_probs.items(), key=lambda x: x[1])[0]
                hard_oppose = get_option_content(hard_oppose, eval_data[idx]['wrong_options'], eval_data[idx]['gt_option'])
                
                # Easy oppose: wrong answer with lowest probability
                easy_oppose = min(wrong_probs.items(), key=lambda x: x[1])[0]
                easy_oppose = get_option_content(easy_oppose, eval_data[idx]['wrong_options'], eval_data[idx]['gt_option'])
            else:
                # For wrong predictions:
                # Hard oppose: wrong answer with highest probability (excluding the predicted wrong answer)
                wrong_probs = {opt: pred_counts[opt] for opt in pred_counts if get_option_content(opt, eval_data[idx]['wrong_options'], eval_data[idx]['gt_option']) != eval_data[idx]['gt_option'] and opt != majority_pred}
                if not wrong_probs:  # Handle case with only two options
                    wrong_probs = {opt: pred_counts[opt] for opt in pred_counts if opt != majority_pred}
                hard_oppose = max(wrong_probs.items(), key=lambda x: x[1])[0]
                hard_oppose = get_option_content(hard_oppose, eval_data[idx]['wrong_options'], eval_data[idx]['gt_option'])
                
                # Easy oppose: correct answer
                easy_oppose = eval_data[idx]['gt_option']
                
            # update protocol with correctness
            new_protocol = f"{eval_data[idx]['protocol']}_{RESPONSE_PROTOCOLS[is_correct]}"
        else:
            assert raw_results is not None, f"do raw protocol first before running **{config.tag}** protocol"
            hard_oppose = raw_results['hard_oppose'][idx]
            easy_oppose = raw_results['easy_oppose'][idx]
            new_protocol = f"{raw_results['protocol'][idx]}_{RESPONSE_PROTOCOLS[is_correct]}"
        
        # edge case: all predictions are wrongly formatted
        if all(count == 0 for count in pred_counts.values()):
            failed_idx.add(idx)
            new_protocol = "_".join(new_protocol.split('_')[:-1]) + f"_{RESPONSE_PROTOCOLS[0]}"
            response = responses[0]
            uncertainty = 0
            hard_oppose = "Error"
            easy_oppose = "Error"
            majority_pred_content = "Error"
            is_correct = 0
        else:
            response = responses[preds.index(majority_pred)]
        
        return {
            'model': config.model,
            'dataset': eval_data[idx]['dataset'],
            'protocol': new_protocol,  # 0_SUPPORT_CORRECT
            'input': formatted_prompts[idx],
            'response': response,
            'pred_counts': pred_counts,
            'uncertainty': uncertainty,
            'hard_oppose': hard_oppose,
            'easy_oppose': easy_oppose,
            'y_pred': majority_pred_content,
            'y_true': eval_data[idx]['gt_option'],
            'is_correct': is_correct,
        }
    
    
    def load_eval_data(self, dataset_path: str) -> Dataset:

        dataset = load_from_disk(dataset_path)

        # If in testing mode, keep only a small subset (first 5 samples)
        if self.testing:
            try:
                dataset = dataset.select(random.sample(range(len(dataset)), 100))
            except Exception:
                # Fallback to basic slicing for iterable/other dataset types
                dataset = dataset[:100]

        return dataset
    
    
    def estimate_raw_results(self, eval_data: Dataset) -> List[Dict]:
        # estimate raw results
        raw_configs: List[EvalConfig] = [
            EvalConfig(
                model=model,
                mode=self.mode,
                tag="RAW"
            )
            for model in self.model_names
        ]
        
        all_raw_results = []
        for raw_cfg in raw_configs:
            parts = raw_cfg.model.split('/')
            model_dir = "_".join(parts[1:]) if len(parts) > 1 else parts[0]
            cache_path = os.path.join(self.save_root, model_dir, raw_cfg.fname)
            
            if os.path.exists(cache_path):
                logger.info(f"Skipping {raw_cfg.model} because it already exists")
                try:
                    with open(cache_path, "r", encoding="utf-8") as f:
                        cached_ouputs = json.load(f)['outputs']
                        all_raw_results.append({
                            'model': cached_ouputs['model'],
                            'protocol': cached_ouputs['protocol'],
                            'y_pred': cached_ouputs['y_pred'],
                            'hard_oppose': cached_ouputs['hard_oppose'],
                            'easy_oppose': cached_ouputs['easy_oppose']
                        })
                    
                    logger.info(f"Loaded cached raw results for {raw_cfg.model} "
                                f"from {cache_path}")
                except Exception as e:
                    logger.warning(f"Failed to load cache {cache_path}: {e}. "
                                   f"Re-computing raw results…")
                continue

            try:
                logger.info(f"Estimating raw accuracy for model "
                            f"{raw_cfg.model} …")
                cached_ouputs = self.evaluate_single_config(raw_cfg, eval_data, is_failed_example_loop=False)
                all_raw_results.append({
                    'model': cached_ouputs['model'],
                    'protocol': cached_ouputs['protocol'],
                    'y_pred': cached_ouputs['y_pred'],
                    'hard_oppose': cached_ouputs['hard_oppose'],
                    'easy_oppose': cached_ouputs['easy_oppose']
                })
                logger.info(f"Cached raw results for {raw_cfg.model} at "
                            f"{cache_path}")
            except Exception as e:
                logger.error(f"Raw estimation failed for model "
                             f"{raw_cfg.model}: {e}")
                logger.error(traceback.format_exc())
                
        return all_raw_results


    def run(self, configs_to_resolve: List[str] = None) -> None:
        """
        1. load eval data
        2. format examples and estimate raw results
        3. setup eval configs
        4. format examples and run eval
        5. run reflection
        6. save results
        """
        
        start_time = time()
        
        # load eval data
        eval_data = self.load_eval_data(self.dataset_path)
        
        # estimate raw results
        all_raw_results = self.estimate_raw_results(eval_data)
        
        # create evaluation configs
        eval_configs: List[EvalConfig] = []
        if configs_to_resolve:
            is_failed_example_loop = True
            logger.warning('CONFIGS TO RESOLVE FOR FAILED CASES')
            for con in configs_to_resolve:
                newcon = EvalConfig()
                with open(con,'r') as f:
                    newcon.__dict__ = json.load(f)["config"]
                eval_configs.append(newcon)
        else:
            is_failed_example_loop = False
            for model, raw_results in zip(self.model_names, all_raw_results):
                if model == raw_results['model'][0]:
                    eval_configs.append(EvalConfig(model=model, mode=self.mode, tag=self.tag))
                else:
                    raise ValueError(f"Model {model} not found in all_raw_results")
             
        # run evaluation
        for config, raw_results in zip(eval_configs, all_raw_results):
            
            logger.info('\n\n\nNew config')
            logger.info(config.__dict__)

            parts = config.model.split('/')
            model_dir = "_".join(parts[1:]) if len(parts) > 1 else parts[0]
            cache_path = os.path.join(self.save_root, model_dir, config.fname)
            if os.path.exists(cache_path):
                logger.info(f"Skipping {config.model} because it already exists")
                try:
                    with open(cache_path, 'r', encoding="utf-8") as f:
                        outputs = json.load(f)['outputs']
                    logger.info(f"Loaded cached protocol outputs for {config.model} from {cache_path}")
                except Exception as e:
                    logger.error(f"Failed to load cached protocol outputs for {config.model}: {e}")
                    logger.error(traceback.format_exc())
            else:
                try:
                    logger.info(f"Starting evaluation for {config.model} in {config.mode} mode...")
                    outputs = self.evaluate_single_config(config, eval_data, raw_results, is_failed_example_loop)
                    logger.info(f"Eval on {config.model} completed in {round(time() - start_time)} seconds")
                except Exception as e:
                    logger.info(f"Error processing {config.model}: {str(e)}")
                    logger.error(traceback.format_exc())
                
            if "reflection" in self.mode:
                logger.info('Running reflection on all results...')
                outputs = self.run_reflection_on_results(outputs, eval_data)
                
        logger.info(f"Evaluation completed in {round(time() - start_time)} seconds")


    def run_reflection_on_results(self, outputs: Dict[str, Any], eval_data: Dataset) -> Dict[str, Any]:
        outputs_reflected = []
        y_pred_reflected = []
        is_correct_reflected = []
        
        for idx in trange(len(outputs['input'])):
            context = outputs['input'][idx]
            outputs_text = outputs['response'][idx]
            y_true = outputs['y_true'][idx]
            
            try:
                # Extract the choice from the original output
                choice = re.findall(r'"(.*?)"', outputs_text)
                if len(choice) == 0:
                    match = re.search(r'\((.*)', outputs_text)
                    if match is None:
                        choice = f'"{outputs_text}"'
                    else:
                        choice = f'"{match.group(0)}"'
                else:
                    choice = f'"{choice[0]}"'
                
                # Generate reflection using the model
                reflection = generate_llm_reflection(
                    self.clients[outputs['model'][0]],
                    outputs['model'][0],
                    context,
                    choice,
                    {
                        "n": 1,
                        "temperature": self.temperature,
                        "top_p": 0.9,
                        "top_k": 50,
                        "min_p": 0.02,
                        "seed": 42,
                    },
                )[0]
                
                outputs_reflected.append(reflection)
                
                # Extract and process the reflected answer
                y_pred = extract_answer(reflection)
                y_pred = get_option_content(y_pred, eval_data[idx]['wrong_options'], eval_data[idx]['gt_option'])
                y_pred_reflected.append(y_pred)
                
                # Compute correctness
                is_correct = int(y_pred == y_true)
                is_correct_reflected.append(is_correct)
                
                logger.info(f"Processed reflection for example {idx+1}/{len(outputs['input'])}")
                
            except Exception as e:
                logger.error(f"Error processing reflection for example {idx}: {e}")
                outputs_reflected.append(f"Error: {str(e)}")
                y_pred_reflected.append("Error")
                is_correct_reflected.append(False)
        
        # Add reflected results to outputs
        outputs['response_reflected'] = outputs_reflected
        outputs['y_pred_reflected'] = y_pred_reflected
        outputs['is_correct_reflected'] = is_correct_reflected
        
        # Save the updated results
        config = EvalConfig(
            model=outputs['model'][0],
            mode=self.mode,
            tag=f"{self.tag}_reflected"
        )
        self.save_results(config, outputs, sum(is_correct_reflected), set())
        
        logger.info(f"Reflection completed. Accuracy: {sum(is_correct_reflected)/len(is_correct_reflected)*100:.2f}%")
        
        return outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', type=str, required=True, nargs='+', help='List of Model names, e.g. gpt-3.5-turbo gpt-4o')
    parser.add_argument('--ips', type=str, required=True, nargs='+', help='List of Model ips, e.g. 127.0.0.1 127.0.0.2')
    parser.add_argument('--port_numbers', type=int, required=True, nargs='+', help='List of Model port numbers, e.g. 8080 8081')
    parser.add_argument('--temperature', default=0.7, type=float, help='Sampling temperature for LLM generation')
    parser.add_argument('--save_root', type=str, required=True, help='Path to save results')
    parser.add_argument('--dataset_path', default='data/final_test', type=str, help='dataset name, e.g. bbh other')
    parser.add_argument('--mode', default=['normal'], type=str, choices=['normal', 'empowered', 'reflection'], help='Mode(s) of LLM: normal, empowered, reflection')
    parser.add_argument('--testing', action='store_true', help='Run on small subset of data for testing')
    parser.add_argument('--tag', type=str, default='first', help='Tag for the evaluation')
    args = parser.parse_args()
    
    # check if there is any model named the same
    if len(set(args.models)) != len(args.models):
        raise ValueError('There are models with the same name')
    
    if not os.path.exists(args.save_root):
        os.makedirs(args.save_root, exist_ok=True)
    
    # use this to retry examples that previously failed
    # List paths to the json files for the results you want to retry
    configs_to_resolve = []
    
    manager = EvalManager(args)
    manager.run(configs_to_resolve)
    
    logger.info("Execution completed.")

if __name__ == '__main__':
    main()