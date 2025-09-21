# KAIROS: An LLM Eval Technique to Evaluate Multi-Agent Social Interactions

Large language models (LLMs) are increasingly deployed in multi-agent systems (MAS) as components of collaborative intelligence, where peer interactions dynamically shape individual decision-making. Although prior work has focused on conformity bias, we extend the analysis to examine how LLMs form trust from previous impressions, resist misinformation, and integrate peer input during interaction, key factors for achieving collective intelligence under complex social dynamics. We present \benchmark, a benchmark simulating quiz contests with peer agents of varying reliability, offering fine-grained control over conditions such as expert–novice roles, noisy crowds, and adversarial peers. LLMs receive both historical interactions and current peer responses, allowing systematic investigation into how trust, peer action, and self-confidence influence decisions. As for mitigation strategies, we evaluate prompting, supervised fine-tuning, and reinforcement learning—Group Relative Policy Optimization (GRPO)—across multiple models. Our results reveal that GRPO with multi-agent context combined with outcome-based rewards and unconstrained reasoning achieves the best overall performance, but also decreases the robustness to social influence compared to Base models.


## 🛠️ Setup

### 1. Clone the Repository

```bash
git clone https://github.com/declare-lab/KAIROS
cd KAIROS
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Install the Package

```bash
pip install -e .
```

### 4. Environment Setup

Create a `.env` file with your API keys (if using external APIs for evaluation):

```bash
# OpenAI API (optional, for evaluation)
OPENAI_API_KEY=your_openai_api_key

# Azure OpenAI (optional, for evaluation)
AZURE_OPENAI_API_KEY=your_azure_openai_api_key
AZURE_OPENAI_ENDPOINT=your_azure_endpoint

# Weights & Biases (optional, for logging)
WANDB_API_KEY=your_wandb_api_key
```


## 📚 Dataset

This dataset contains question-answering tasks in multi-agent social interaction scenarios, supporting evaluation of trust, peer influence, robustness to misinformation, and more. For detailed data format and usage instructions, please refer to the [KAIROS_EVAL](https://huggingface.co/datasets/declare-lab/KAIROS_EVAL).



## 🎯 Training

### Supervised Fine-tuning (SFT) & Group Relative Policy Optimization (GRPO)

There are two ways to train:

1. Cross-server training: Use `scripts/grpo_train.sh` to launch training on one machine while automatically starting the vLLM server on a remote server. This is suitable for distributed or multi-machine scenarios.

2. Single-machine training: Use `scripts/train.sh` to run all processes locally on one machine.

Just change variable `CONFIGS` inside the script to the `yaml` file inside `KAIROS/recipes/train_configs`

``` bash
CONFIGS=(
    grpo-qwen25-14b-MAS-NS-OR-LCorr.yaml
)
```


### Custom Training

For custom training configurations, modify the YAML files in `recipes/train_configs/` and run:

```bash
bash scripts/train.sh
```

The training script will:
1. Create necessary directories (`saved`, `logs`)
2. Start VLLM server (for GRPO training)
3. Launch distributed training with appropriate DeepSpeed configuration
4. Save checkpoints and logs
5. Clean up VLLM processes

### Training Configuration

Training configurations are stored in `recipes/train_configs/`. Key parameters include:

- **Model Settings**: Model path, torch dtype, attention implementation
- **Data Settings**: Dataset path, system prompt type, preprocessing options
- **Training Settings**: Learning rate, batch size, epochs, optimization strategy
- **GRPO Settings**: Reward functions, generation parameters, VLLM configuration

Example configuration structure:
```yaml
# Model arguments
model_name_or_path: Qwen/Qwen2.5-3B-Instruct
torch_dtype: bfloat16
trust_remote_code: true

# Data training arguments
system_prompt: SYSTEM_PROMPT_DEBATE
dataset_mixer:
  "data/final_train_dict_MAS": 1.0

# GRPO trainer config
output_dir: saved/grpo-qwen25-3b-MAS-DS-DR
learning_rate: 3.0e-06
per_device_train_batch_size: 16
num_train_epochs: 1
```

## 📊 Evaluation

### Standard Evaluation

To evaluate trained models, you must first launch the VLLM server for the subject model, and then run the evaluation script:

``` bash 
bash scripts/eval_vllm.sh
# wait until server setup
bash scripts/eval_mas.sh
```

### Custom Evaluation

For specific model evaluation:

```bash
python src/MAS/eval_mas.py \
  --models saved/your-model-checkpoint \
  --ips 0.0.0.0 \
  --port_numbers 9090 \
  --temperature 0.7 \
  --save_root eval_results \
  --dataset_path data/your_test_data \
  --mode reflection \
  --tag your_experiment_tag
```

### Evaluation Analysis

Generate analysis reports:

```bash
python src/MAS/eval_analysis.py --input_dir eval_results/model_directory
```

## 🏗️ Architecture

### Multi-Agent Debate System

The system implements an internal debate mechanism where multiple AI "voices" engage in reasoning:

```
<think>
Curious voice: This question asks about X, let me consider...
Skeptical voice: Wait, we should also consider Y because...
Analytical voice: Looking at the evidence, Z seems most likely...
</think>

<answer>
Based on the internal debate, the answer is...
</answer>
```

### Training Pipeline

1. **Data Processing**: Load and preprocess datasets with appropriate formatting
2. **Model Loading**: Initialize base models with proper configurations
3. **Reward Functions**: Apply task-specific reward functions for GRPO training
4. **Multi-GPU Training**: Distributed training with DeepSpeed optimization
5. **Evaluation**: Comprehensive evaluation with multiple metrics

### Key Components

- `src/MAS/sft.py`: Supervised fine-tuning implementation
- `src/MAS/grpo.py`: Group Relative Policy Optimization training
- `src/MAS/eval_mas.py`: Evaluation framework
- `src/MAS/rewards.py`: Reward function implementations
- `src/MAS/trainer/`: Custom trainer implementations

## 🔧 Configuration Options

### Model Variants

The system supports various model configurations:
- **MAS vs Non-MAS**: With or without multi-agent debate
- **DS vs NS**: Different system prompt (Debate Prompt vs Normal Prompt)
- **DR vs OR**: Different reward functions (Debate Reward vs Outcome Reward)
- **LConf vs LCorr**: Low Confidence-based vs Low Correctness-based learning

### Training Types

- **SFT**: Standard supervised fine-tuning
- **GRPO**: Group Relative Policy Optimization with multi-agent debate
- **GRPO-MAS**: GRPO with Multi-Agent System enhancements

## 📁 Directory Structure

```
KAIROS/
├── src/MAS/                    # Main source code
│   ├── grpo.py                # GRPO training implementation
│   ├── sft.py                 # SFT training implementation
│   ├── eval_mas.py            # Evaluation framework
│   ├── rewards.py             # Reward functions
│   ├── trainer/               # Custom trainers
│   ├── utils/                 # Utility functions
│   └── label_generation/      # Data generation tools
├── recipes/                   # Configuration files
│   ├── train_configs/         # Training configurations
│   └── accelerate_configs/    # DeepSpeed configurations
├── scripts/                   # Training and evaluation scripts
├── requirements.txt           # Python dependencies
└── setup.py                   # Package setup
```
