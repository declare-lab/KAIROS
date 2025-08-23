# import debugpy;debugpy.connect(('localhost', 9501))
import random
from accelerate.state import PartialState
from transformers.trainer_utils import get_last_checkpoint, set_seed
from trl import GRPOConfig, ModelConfig, get_peft_config
from trl.trainer.utils import empty_cache

from MAS.trainer.grpo_trainer_log import GRPOTrainerWithLog
from MAS.rewards import combined_reward, format_reward_normal, accuracy_reward_normal
from MAS.utils.arguments import H4ArgumentParser, ScriptArguments
from MAS.utils.logging_utils import setup_logger
from MAS.utils.train_utils import get_datasets, load_model_and_tokenizer
from MAS.utils.eval_utils import SYSTEM_PROMPT_DEBATE, SYSTEM_PROMPT_NORMAL

REWARD_FUNCS = []

def main():
    parser = H4ArgumentParser((ScriptArguments, ModelConfig, GRPOConfig))
    script_args, model_args, training_args = parser.parse()

    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Setup logging
    ###############
    logger = setup_logger(training_args, script_args)
    
    # Log on each process a small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.bf16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Data parameters {script_args}")
    logger.info(f"Training/evaluation parameters {training_args}")
    
    
    ########################################
    # Model & Tokenizer & Reward functions
    ########################################
    logger.info("*** Loading pretrained model and tokenizer ***")
    
    model, model_kwargs, tokenizer = load_model_and_tokenizer(script_args, model_args, training_args)

    # check if it is debating grpo or normal grpo
    global REWARD_FUNCS
    if "-DR" in training_args.output_dir:
        REWARD_FUNCS = [combined_reward]
        training_args.reward_weights = [1.0]
    elif "-OR" in training_args.output_dir:
        REWARD_FUNCS = [format_reward_normal, accuracy_reward_normal]
        training_args.reward_weights = [1.0, 1.0]
    elif "-ER" in training_args.output_dir:
        raise NotImplementedError("ER is not implemented yet")
    else:
        raise ValueError(f"Invalid output directory with reward not defined: {training_args.output_dir}")
    
    ################
    # Dataset
    ################
    logger.info("*** Loading datasets ***") 
   
    raw_datasets = get_datasets(
        script_args,
        splits=script_args.dataset_splits,
        configs=script_args.dataset_configs,
        columns_to_keep=None,
    )
    
    # split the dataset into train and test if requires evaluation
    if training_args.do_eval and "test" not in raw_datasets:
        raw_datasets = raw_datasets.train_test_split(test_size=0.1)
    
    logger.info(
        f"Training on the following splits: {[split + ' : ' + str(dset.num_rows) for split, dset in raw_datasets.items()]}"
    )

    if training_args.debug:
        for key in raw_datasets:
            raw_datasets[key] = raw_datasets[key].select(range(30))
    
    # Handle system prompt if needed
    with PartialState().main_process_first():
        if script_args.system_prompt is not None:
            logger.info(f"Using system prompt: {script_args.system_prompt}")
            if script_args.system_prompt == "SYSTEM_PROMPT_DEBATE":
                raw_datasets = raw_datasets.map(lambda x: {"prompt": [{"role": "system", "content": SYSTEM_PROMPT_DEBATE}, x["prompt"][-1]]}, num_proc=script_args.preprocessing_num_workers)
            elif script_args.system_prompt == "SYSTEM_PROMPT_NORMAL":
                raw_datasets = raw_datasets.map(lambda x: {"prompt": [{"role": "system", "content": SYSTEM_PROMPT_NORMAL}, x["prompt"][-1]]}, num_proc=script_args.preprocessing_num_workers)
            else:
                raise ValueError(f"Invalid system prompt: {script_args.system_prompt}")
            
    # Log a few random samples from the training set:
    if PartialState().is_main_process:
        print(raw_datasets)
        for index in random.sample(range(len(raw_datasets["train"])), 2):
            logger.info(f"Prompt sample {index} of the raw training set:\n\n{raw_datasets['train'][index]['prompt']}")
  
    train_dataset = raw_datasets.get("train", None)
    eval_dataset = raw_datasets.get("test", None)


    ################
    # Instantiate GRPO trainer
    ################
    if training_args.model_init_kwargs is None:
        training_args.model_init_kwargs = model_kwargs
    else:
        training_args.model_init_kwargs.update(model_kwargs)
    
    trainer = GRPOTrainerWithLog(
        model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        reward_funcs=REWARD_FUNCS,
        peft_config=get_peft_config(model_args)
    )


    ###############
    # Training loop
    ###############
    logger.info("*** Training ***")
    checkpoint = None
    # Check for last checkpoint
    if training_args.resume_from_checkpoint is not None:
        checkpoint = get_last_checkpoint(training_args.output_dir) if isinstance(training_args.resume_from_checkpoint, bool) else training_args.resume_from_checkpoint
        if checkpoint is not None:
            logger.warning(f"Checkpoint detected, resuming training at {checkpoint=}.")
        else:
            logger.error(f"Failed to load last checkpoint at {checkpoint=}. Start training from scratch")
    
    train_result = trainer.train(resume_from_checkpoint=checkpoint)        
    metrics = train_result.metrics
    metrics["train_samples"] = len(raw_datasets["train"])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    logger.info("*** Training complete ***")
    
    
    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Saving model ***")
    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    # Restore k,v cache for fast inference
    trainer.model.config.use_cache = True
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")
    
    
if __name__ == "__main__":
    main()