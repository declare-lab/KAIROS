# import debugpy; debugpy.connect(("localhost", 9501))
import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd

from MAS.utils.eval_utils import (
    analyze_transitions, calculate_accuracies,
    load_eval_data, plot_transition_analysis, 
    calculate_utility_resistance, save_transition_xlsx
)


def main(args: argparse.Namespace):
    """Main function to run the analysis."""
    input_path = Path(args.input_dir)
    if not input_path.is_dir():
        raise FileNotFoundError(f"Directory not found at {input_path}")

    files = sorted(list(input_path.glob("*.json")))
    models = defaultdict(lambda: {"protocols": []})
    
    # load all files into models
    for f in files:
        model_name = f.stem.rsplit('-', 1)[0]
        protocol_name = f.stem.rsplit('-', 1)[1]
        if protocol_name == "RAW":
            models[model_name]["RAW"] = f
        else:
            models[model_name]["protocols"].append((protocol_name, f))

    # analyze each model
    for model_name, model_files in models.items():
        print(f"\n{'='*20} Analysis for Model: {model_name} {'='*20}")
        
        raw_file = model_files.get('RAW', None)
        if not raw_file:
            print(f"RAW file not found for model {model_name}. Skipping.")
            continue

        raw_data = load_eval_data(raw_file)
        raw_data = raw_data.get("outputs", {})
        if not raw_data:
            print(f"Warning: 'outputs' key not found in {raw_file.name}. Cannot determine RAW accuracy.")
            continue
        
        raw_correctness= raw_data.get("is_correct", [])
        if not raw_correctness:
            print(f"Warning: 'is_correct' key not found in {raw_file.name}. Cannot determine RAW accuracy.")
            continue
        
        datasets = raw_data.get("dataset", [])
        if not datasets:
            print(f"Warning: 'dataset' key not found in {raw_file.name}. Cannot perform dataset-level analysis.")
        
        # calculate raw accuracy
        raw_overall_acc, raw_dataset_accs = calculate_accuracies(raw_data, from_reflection=False)
        
        # Prepare data for combined table
        acc_data = []
        dataset_data = []
        transition_data = []
        
        row_acc_data = {"RAW Acc": f"{raw_overall_acc:.2%}"}
        row_dataset_data_dict = {ds: {"Dataset": ds, "RAW Acc": f"{raw_dataset_accs.get(ds, 0):.2%}"} for ds in sorted(raw_dataset_accs.keys())}
        for proto_name, proto_file in model_files["protocols"]:
            base_name = proto_name.replace("_reflected", "")
            proto_data = load_eval_data(proto_file)
            proto_data = proto_data.get("outputs", {})
            if not proto_data:
                print(f"Warning: 'outputs' key not found in {proto_file.name}. Cannot determine {proto_name} accuracy.")
                continue
            
            is_reflected = "reflected" in proto_name
            proto_correctness = proto_data.get("is_correct_reflected" if is_reflected else "is_correct", [])
            if not proto_correctness:
                print(f"Warning: 'is_correct_reflected' / 'is_correct' key not found in {proto_file.name}. Cannot determine {proto_name} accuracy.")
                continue
            
            proto_overall_acc, proto_dataset_accs = calculate_accuracies(proto_data, from_reflection=is_reflected)
            
            change = proto_overall_acc - raw_overall_acc
            
            (utility_overall, resistance_overall, f1_overall), utility_resistance_f1_dataset = calculate_utility_resistance(raw_correctness, proto_correctness, datasets)
            
            row_acc_data[f"{proto_name} Acc"] = f"{proto_overall_acc:.2%}"
            row_acc_data[f"{proto_name} Change"] = f"{change:+.2%}"
            row_acc_data[f"{proto_name} Utility"] = f"{utility_overall:.2%}"
            row_acc_data[f"{proto_name} Resistance"] = f"{resistance_overall:.2%}"
            row_acc_data[f"{proto_name} F1"] = f"{f1_overall:.2%}"
            
            # add dataset data
            for ds in sorted(raw_dataset_accs.keys()):
                proto_acc = proto_dataset_accs.get(ds, 0)
                change = proto_acc - raw_dataset_accs.get(ds, 0)
                proto_utility, proto_resistance, proto_f1 = utility_resistance_f1_dataset.get(ds, (0, 0, 0))
                row_dataset_data_dict[ds][f"{proto_name} Acc"] = f"{proto_acc:.2%}"
                row_dataset_data_dict[ds][f"{proto_name} Change"] = f"{change:+.2%}"
                row_dataset_data_dict[ds][f"{proto_name} Utility"] = f"{proto_utility:.2%}"
                row_dataset_data_dict[ds][f"{proto_name} Resistance"] = f"{proto_resistance:.2%}"
                row_dataset_data_dict[ds][f"{proto_name} F1"] = f"{proto_f1:.2%}"
            
            if proto_name == base_name:
                trans_analysis = analyze_transitions(proto_data)
                transition_data.append({
                    "Model": model_name,
                    "Protocol": base_name,
                    "Transition Analysis": trans_analysis,
                })
        
        acc_data.append(row_acc_data)
        for row_dataset_data in sorted(row_dataset_data_dict.values(), key=lambda x: x["Dataset"]):
            dataset_data.append(row_dataset_data)

        # Report final analysis
        print("\n1. Raw vs. Protocol Acc")
        if acc_data:
            df = pd.DataFrame(acc_data)
            print(df.to_string(index=False))
        else:
            print("No other protocols found or processed.")

        print("\n2. Per-Dataset Accuracy Breakdown:")
        if dataset_data:
            df_ds = pd.DataFrame(dataset_data)
            print(df_ds.to_string(index=False))
            
            # Category-based analysis
            print("\n3. Category-based Dataset Metrics:")
            categories = {
                "reasoning": ["bbh", "livecode", "math500"],
                "knowledge": ["truthfulqa", "mmlupro"], 
                "common_sense": ["siqa", "commonsenseqa"],
                "creativity": ["macgyver", "brainteaser"]
            }
            
            category_data = []
            for category, datasets_in_category in categories.items():
                row = {"Category": category}
                
                # Find datasets that exist in our data and belong to this category
                existing_datasets = [ds for ds in datasets_in_category if any(d["Dataset"] == ds for d in dataset_data)]
                
                if existing_datasets:
                    # Calculate averages for each metric across datasets in this category
                    for col in df_ds.columns:
                        if col != "Dataset":
                            # Extract values for datasets in this category
                            values = []
                            for ds_name in existing_datasets:
                                ds_row = next((d for d in dataset_data if d["Dataset"] == ds_name), None)
                                if ds_row and col in ds_row:
                                    val_str = ds_row[col]
                                    # Parse percentage values
                                    if "%" in val_str:
                                        val = float(val_str.replace("%", "").replace("+", ""))
                                        values.append(val)
                            
                            if values:
                                avg_val = sum(values) / len(values)
                                if "Change" in col:
                                    row[col] = f"{avg_val:+.2f}%"
                                else:
                                    row[col] = f"{avg_val:.2f}%"
                            else:
                                row[col] = "N/A"
                else:
                    # No datasets found for this category
                    for col in df_ds.columns:
                        if col != "Dataset":
                            row[col] = "N/A"
                
                category_data.append(row)
            
            if category_data:
                df_cat = pd.DataFrame(category_data)
                print(df_cat.to_string(index=False))
        else:
            print("No other protocols found or processed.")

        # Transition analysis
        print("\n4. Transition Analysis") 
        save_transition_xlsx(transition_data, args.input_dir, normalize=False)    
        plot_transition_analysis(transition_data, args.input_dir)
        print(f"\nTransition analysis plots have been saved in '{args.input_dir}/plots/'")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Analyze model evaluation results.")
    parser.add_argument(
        "--input_dir", 
        type=str, 
        default="mas_eval",
        help="Directory containing the .json evaluation files."
    )
    args = parser.parse_args()
    main(args) 