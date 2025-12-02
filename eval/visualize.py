import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

def plot_confusion_matrix(cm, dataset_name, output_dir):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix - {dataset_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'confusion_matrix_{dataset_name}.png'))
    plt.close()

def plot_metrics_comparison(results, output_dir):
    # Prepare data for plotting
    data = []
    for dataset_name, metrics in results.items():
        data.append({
            'Dataset': dataset_name,
            'Metric': 'F1 Score',
            'Value': metrics['f1']
        })
        data.append({
            'Dataset': dataset_name,
            'Metric': 'Precision',
            'Value': metrics['precision']
        })
        data.append({
            'Dataset': dataset_name,
            'Metric': 'Recall',
            'Value': metrics['recall']
        })
    
    df = pd.DataFrame(data)
    
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x='Dataset', y='Value', hue='Metric', data=df)
    plt.title('Model Performance Comparison')
    plt.ylim(0, 1.1)  # Increased limit to make room for labels
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels
    for container in ax.containers:
        ax.bar_label(container, fmt='%.4f', padding=3)
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_comparison.png'))
    plt.close()

def plot_metrics_comparison_zoomed(results, output_dir):
    # Prepare data for plotting
    data = []
    for dataset_name, metrics in results.items():
        data.append({
            'Dataset': dataset_name,
            'Metric': 'F1 Score',
            'Value': metrics['f1']
        })
        data.append({
            'Dataset': dataset_name,
            'Metric': 'Precision',
            'Value': metrics['precision']
        })
        data.append({
            'Dataset': dataset_name,
            'Metric': 'Recall',
            'Value': metrics['recall']
        })
    
    df = pd.DataFrame(data)
    
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x='Dataset', y='Value', hue='Metric', data=df)
    plt.title('Model Performance Comparison (Zoomed 0.8 - 1.0)')
    plt.ylim(0.8, 1.0)  # Zoomed scale
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels
    for container in ax.containers:
        ax.bar_label(container, fmt='%.4f', padding=3)
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_comparison_zoomed.png'))
    plt.close()

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    eval_dir = os.path.join(base_dir, 'eval')
    json_path = os.path.join(eval_dir, 'results.json')
    
    if not os.path.exists(json_path):
        print(f"Results file not found: {json_path}")
        return
        
    with open(json_path, 'r') as f:
        results = json.load(f)
        
    for dataset_name, metrics in results.items():
        cm = np.array(metrics['confusion_matrix'])
        plot_confusion_matrix(cm, dataset_name, eval_dir)
        
    plot_metrics_comparison(results, eval_dir)
    plot_metrics_comparison_zoomed(results, eval_dir)
    print(f"Visualizations saved to {eval_dir}")

if __name__ == "__main__":
    main()
