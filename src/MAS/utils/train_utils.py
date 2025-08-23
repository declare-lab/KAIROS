import os
from typing import List, Optional

import colorlog
from datasets import (DatasetDict, concatenate_datasets, load_dataset,
                      load_from_disk)
from datasets.exceptions import DatasetGenerationError
from huggingface_hub import list_repo_files
from huggingface_hub.errors import RepositoryNotFoundError
from huggingface_hub.utils._validators import HFValidationError
from peft import PeftConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import get_kbit_device_map, get_quantization_config

from MAS.utils.arguments import ScriptArguments

logger = colorlog.getLogger(__name__)

INTERNET_CONNECTION = True


def load_model_and_tokenizer(script_args, model_args, training_args):
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        torch_dtype=model_args.torch_dtype,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    
    model = model_args.model_name_or_path
    if is_adapter_model(model, model_args.model_revision):
        logger.info(f"Loading SFT adapter for {model_args.model_name_or_path=}")
        peft_config = PeftConfig.from_pretrained(model_args.model_name_or_path, revision=model_args.model_revision)
        base_model = AutoModelForCausalLM.from_pretrained(
            peft_config.base_model_name_or_path,
            **model_kwargs,
        )
        model = get_peft_model(base_model, peft_config)
        # Use a standalone base model
        model = model.merge_and_unload()
        model_kwargs = None
        
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        padding_side=script_args.padding_side if script_args.padding_side is not None else "left",  # for generation
        truncation_side=script_args.truncation_side if script_args.truncation_side is not None else "left",  # Truncate from left to ensure we don't lose labels in final turn
    )
    if hasattr(tokenizer, "pad_token_id") and tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.unk_token_id
    if script_args.chat_template is not None:
        tokenizer.chat_template = script_args.chat_template
    
    return model, model_kwargs, tokenizer


def is_adapter_model(model_name_or_path: str, revision: str = "main") -> bool:
    if INTERNET_CONNECTION:
        try:
            
            # Try first if model on a Hub repo
            repo_files = list_repo_files(model_name_or_path, revision=revision)
        except (HFValidationError, RepositoryNotFoundError):
            # If not, check local repo
            repo_files = os.listdir(model_name_or_path)
    else:
        repo_files = os.listdir(model_name_or_path)
        
    return "adapter_model.safetensors" in repo_files or "adapter_model.bin" in repo_files


def get_datasets(
    data_config: ScriptArguments | dict,
    splits: Optional[List[str]] = None,
    configs: Optional[List[str]] = None,
    columns_to_keep: Optional[List[str]] = None,
    shuffle: bool = True,
) -> DatasetDict:
    """
    Loads one or more datasets with varying training set proportions.

    Args:
        data_config (`ScriptArguments` or `dict`):
            Dataset configuration and split proportions.
        splits (`List[str]`, *optional*, defaults to `['train', 'test']`):
            Dataset splits to load and mix. Assumes the splits exist in all datasets and have a `train_` or `test_` prefix.
        configs (Optional[List[str]], *optional*, defaults to `None`):
            List of dataset config names. If given must be the same length as 'data_config' keys.
        columns_to_keep (Optional[List[str]], *optional*, defaults to `None`):
            Column names to keep in the dataset. Useful in the datamixer to avoid schema conflicts,
            and for cpt this should be (at least) the text column.
        shuffle (`bool`, *optional*, defaults to `True`):
            Whether to shuffle the training and testing/validation data.

    Returns
        [`DatasetDict`]: The dataset dictionary containing the loaded datasets.
    """
    if type(data_config) is ScriptArguments:
        # Structure of the config to read the datasets and their mix
        # datasets_mixer:
        #     - 'dataset1': 0.5
        #     - 'dataset2': 0.3
        #     - 'dataset3': 0.2
        dataset_mixer = data_config.dataset_mixer
    elif isinstance(data_config, dict):
        # Structure of the input is:
        #     dataset_mixer = {
        #             "dataset1": 0.5,
        #             "dataset1": 0.3,
        #             "dataset1": 0.2,
        #         }
        dataset_mixer = data_config
    else:
        raise ValueError(f"Data config {data_config} not recognized.")

    raw_datasets = mix_datasets(
        dataset_mixer,
        splits=splits,
        configs=configs,
        columns_to_keep=columns_to_keep,
        shuffle=shuffle,
    )
    return raw_datasets


def mix_datasets(
    dataset_mixer: dict,
    splits: Optional[List[str]] = None,
    configs: Optional[List[str]] = None,
    columns_to_keep: Optional[List[str]] = None,
    shuffle=True,
) -> DatasetDict:
    """
    Loads and mixes datasets according to proportions specified in `dataset_mixer`.

    Args:
        dataset_mixer (`dict`):
            Dictionary containing the dataset names and their training proportions. By default, all test proportions are 1.
        splits (Optional[List[str]], *optional*, defaults to `None`):
            Dataset splits to load and mix. Assumes the splits exist in all datasets and have a `train_` or `test_` prefix.
        configs (Optional[List[str]], *optional*, defaults to `None`):
            List of dataset config names. If given must be the same length as 'dataset_mixer' keys.
        columns_to_keep (Optional[List[str]], *optional*, defaults to `None`):
            Column names to keep in the dataset. Useful in the datamixer to avoid schema conflicts,
            and for cpt this should be (at least) the text column.
        shuffle (`bool`, *optional*, defaults to `True`):
            Whether to shuffle the training and testing/validation data.
    """
    splits = ["train", "test"] if splits is None else splits
    configs = [None] * len(dataset_mixer) if not configs else configs

    if configs is not None and len(configs) != len(dataset_mixer):
        raise ValueError("The number of given dataset config names must be the same as the given number of datasets.")

    raw_datasets = DatasetDict()
    raw_train_datasets = []
    raw_val_datasets = []
    fracs = []
    for (ds, frac), ds_config in zip(dataset_mixer.items(), configs):
        fracs.append(frac)
        for split in splits:
            try:
                dataset = load_from_disk(os.path.join(ds, split))
            except FileNotFoundError:
                dataset = load_dataset(ds, ds_config, split=split)

            # Remove redundant columns to avoid schema conflicts on load
            if columns_to_keep is not None:
                dataset = dataset.remove_columns([col for col in dataset.column_names if col not in columns_to_keep])
            if "train" in split:
                raw_train_datasets.append(dataset)
            elif "test" in split:
                raw_val_datasets.append(dataset)
            else:
                raise ValueError(f"Split type {split} not recognized as one of test or train.")

    if any(frac < 0 for frac in fracs):
        raise ValueError("Dataset fractions cannot be negative.")

    if len(raw_train_datasets) > 0:
        train_subsets = []
        for dataset, frac in zip(raw_train_datasets, fracs):
            train_subset = dataset.select(range(int(frac * len(dataset))))
            train_subsets.append(train_subset)
        if shuffle:
            raw_datasets["train"] = concatenate_datasets(train_subsets).shuffle(seed=42)
        else:
            raw_datasets["train"] = concatenate_datasets(train_subsets)
    # No subsampling for test datasets to enable fair comparison across models
    if len(raw_val_datasets) > 0:
        if shuffle:
            raw_datasets["test"] = concatenate_datasets(raw_val_datasets).shuffle(seed=42)
        else:
            raw_datasets["test"] = concatenate_datasets(raw_val_datasets)

    if len(raw_datasets) == 0:
        raise ValueError(
            f"Dataset {dataset_mixer} not recognized with splits {splits}. Check the dataset has been correctly formatted."
        )

    return raw_datasets