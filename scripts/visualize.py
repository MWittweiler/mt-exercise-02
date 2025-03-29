import argparse
import os
import re
import matplotlib.pyplot as plt
import pandas as pd

def extract_perplexities(file_path):
    training_ppls = []
    validation_ppls = []
    test_ppl = None
    
    with open(file_path, 'r') as f:
        for line in f:
            train_match = re.search(r'Training Perplexity: ([\d\.]+) at epoch: (\d+)', line)
            val_match = re.search(r'Validation Perplexity: ([\d\.]+) at epoch: (\d+)', line)
            test_match = re.search(r'Test Perplexity: ([\d\.]+)', line)
            
            if train_match:
                training_ppls.append((int(train_match.group(2)), float(train_match.group(1))))
            if val_match:
                validation_ppls.append((int(val_match.group(2)), float(val_match.group(1))))
            if test_match:
                test_ppl = float(test_match.group(1))
    
    return training_ppls, validation_ppls, test_ppl

def plot_perplexities(perplexities, title, filename):
    plt.figure()
    for model, values in perplexities.items():
        epochs, ppls = zip(*values)
        plt.plot(epochs, ppls, label=model)
    
    plt.xlabel("Epoch")
    plt.ylabel("Perplexity")
    plt.title(title)
    plt.legend()
    plt.grid()
    
    os.makedirs("../plots", exist_ok=True)
    plt.savefig(f"../plots/{filename}")
    plt.close()

def extract_numeric_value(model_name):
    match = re.search(r'([\d\.]+)$', model_name)
    return float(match.group(1)) if match else float('inf')

def save_table_as_image(df, filename, title):
    df = df.reindex(sorted(df.columns, key=extract_numeric_value), axis=1)
    fig, ax = plt.subplots(figsize=(max(8, len(df.columns) * 1.2), max(4, len(df) * 0.5)))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=df.values, colLabels=df.columns, rowLabels=df.index, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.2, 1.2)
    plt.title(title)
    plt.savefig(f"../plots/{filename}", bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Parse perplexity logs and generate plots.")
    parser.add_argument("--path", type=str, default="../logs", help="Path to directory containing log files.")
    args = parser.parse_args()
    
    log_dir = args.path
    
    models = {}
    test_perplexities = {}
    
    for file_name in os.listdir(log_dir):
        if file_name.endswith(".log"):
            model_name = file_name.replace(".log", "")
            file_path = os.path.join(log_dir, file_name)
            train_ppls, val_ppls, test_ppl = extract_perplexities(file_path)
            models[model_name] = {"train": train_ppls, "val": val_ppls}
            test_perplexities[model_name] = test_ppl
    
    train_perplexities = {model: data["train"] for model, data in models.items()}
    val_perplexities = {model: data["val"] for model, data in models.items()}
    
    plot_perplexities(train_perplexities, "Training Perplexity Over Epochs", "training_perplexity.png")
    plot_perplexities(val_perplexities, "Validation Perplexity Over Epochs", "validation_perplexity.png")
    
    test_df = pd.DataFrame([test_perplexities])
    test_df.to_csv("../plots/test_perplexities.csv", index=False)
    save_table_as_image(test_df, "test_perplexities.png", "Test Perplexities")
    
    val_df = pd.DataFrame({model: dict(data["val"]) for model, data in models.items()})
    val_df.to_csv("../plots/validation_perplexities.csv", index=True)
    save_table_as_image(val_df, "validation_perplexities.png", "Validation Perplexities")
    
    train_df = pd.DataFrame({model: dict(data["train"]) for model, data in models.items()})
    train_df.to_csv("../plots/training_perplexities.csv", index=True)
    save_table_as_image(train_df, "training_perplexities.png", "Training Perplexities")

if __name__ == "__main__":
    main()