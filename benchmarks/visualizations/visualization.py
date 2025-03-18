"""
Visualization utilities for DeepHermes benchmark results.

This module provides functions for generating visualizations of benchmark results,
including performance comparisons across models and datasets.
"""

from typing import Dict, List, Any, Optional
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes


def create_accuracy_comparison(results: Dict[str, Dict[str, Any]], output_dir: str) -> None:
    """
    Create accuracy comparison chart for all models and datasets.
    
    Args:
        results: Benchmark results
        output_dir: Directory to save visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract model names and datasets
    model_names = list(results.keys())
    datasets = set()
    for model_id in model_names:
        for dataset_name in results[model_id]["datasets"].keys():
            datasets.add(dataset_name)
    datasets = sorted(list(datasets))
    
    # Extract accuracy data
    accuracy_data = {}
    for model_id in model_names:
        model_name = model_id.split('/')[-1]
        accuracy_data[model_name] = []
        
        for dataset_name in datasets:
            if dataset_name in results[model_id]["datasets"]:
                accuracy = results[model_id]["datasets"][dataset_name]["metrics"].get("accuracy", 0)
                accuracy_data[model_name].append(accuracy * 100)  # Convert to percentage
            else:
                accuracy_data[model_name].append(0)
    
    # Create bar chart
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x = np.arange(len(datasets))
    width = 0.8 / len(model_names)
    
    for i, model_name in enumerate(accuracy_data.keys()):
        offset = (i - len(model_names) / 2 + 0.5) * width
        bars = ax.bar(x + offset, accuracy_data[model_name], width, label=model_name)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
    
    # Add labels and legend
    ax.set_xlabel('Dataset')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Accuracy Comparison Across Models and Datasets')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_comparison.png'), dpi=300)
    plt.close()


def create_inference_time_comparison(results: Dict[str, Dict[str, Any]], output_dir: str) -> None:
    """
    Create inference time comparison chart for all models and datasets.
    
    Args:
        results: Benchmark results
        output_dir: Directory to save visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract model names and datasets
    model_names = list(results.keys())
    datasets = set()
    for model_id in model_names:
        for dataset_name in results[model_id]["datasets"].keys():
            datasets.add(dataset_name)
    datasets = sorted(list(datasets))
    
    # Extract inference time data
    time_data = {}
    for model_id in model_names:
        model_name = model_id.split('/')[-1]
        time_data[model_name] = []
        
        for dataset_name in datasets:
            if dataset_name in results[model_id]["datasets"]:
                time = results[model_id]["datasets"][dataset_name]["metrics"].get("inference_time_avg", 0)
                time_data[model_name].append(time)
            else:
                time_data[model_name].append(0)
    
    # Create bar chart
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x = np.arange(len(datasets))
    width = 0.8 / len(model_names)
    
    for i, model_name in enumerate(time_data.keys()):
        offset = (i - len(model_names) / 2 + 0.5) * width
        bars = ax.bar(x + offset, time_data[model_name], width, label=model_name)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}s',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
    
    # Add labels and legend
    ax.set_xlabel('Dataset')
    ax.set_ylabel('Average Inference Time (seconds)')
    ax.set_title('Inference Time Comparison Across Models and Datasets')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'inference_time_comparison.png'), dpi=300)
    plt.close()


def create_memory_usage_comparison(results: Dict[str, Dict[str, Any]], output_dir: str) -> None:
    """
    Create memory usage comparison chart for all models.
    
    Args:
        results: Benchmark results
        output_dir: Directory to save visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract model names and memory usage
    model_names = []
    memory_usage = []
    
    for model_id in results.keys():
        model_name = model_id.split('/')[-1]
        model_names.append(model_name)
        
        # Get memory usage from the first dataset (should be similar across datasets)
        dataset_name = next(iter(results[model_id]["datasets"].keys()))
        memory_mb = results[model_id]["datasets"][dataset_name].get("memory_usage_mb", 0)
        memory_usage.append(memory_mb)
    
    # Create bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(model_names, memory_usage)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f} MB',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    # Add labels
    ax.set_xlabel('Model')
    ax.set_ylabel('Memory Usage (MB)')
    ax.set_title('Memory Usage Comparison Across Models')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'memory_usage_comparison.png'), dpi=300)
    plt.close()


def create_reasoning_length_comparison(results: Dict[str, Dict[str, Any]], output_dir: str) -> None:
    """
    Create reasoning length comparison chart for all models and datasets.
    
    Args:
        results: Benchmark results
        output_dir: Directory to save visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract model names and datasets
    model_names = list(results.keys())
    datasets = set()
    for model_id in model_names:
        for dataset_name in results[model_id]["datasets"].keys():
            datasets.add(dataset_name)
    datasets = sorted(list(datasets))
    
    # Extract reasoning length data
    length_data = {}
    for model_id in model_names:
        model_name = model_id.split('/')[-1]
        length_data[model_name] = []
        
        for dataset_name in datasets:
            if dataset_name in results[model_id]["datasets"]:
                length = results[model_id]["datasets"][dataset_name]["metrics"].get("reasoning_length_avg", 0)
                length_data[model_name].append(length)
            else:
                length_data[model_name].append(0)
    
    # Create bar chart
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x = np.arange(len(datasets))
    width = 0.8 / len(model_names)
    
    for i, model_name in enumerate(length_data.keys()):
        offset = (i - len(model_names) / 2 + 0.5) * width
        bars = ax.bar(x + offset, length_data[model_name], width, label=model_name)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.0f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
    
    # Add labels and legend
    ax.set_xlabel('Dataset')
    ax.set_ylabel('Average Reasoning Length (words)')
    ax.set_title('Reasoning Length Comparison Across Models and Datasets')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'reasoning_length_comparison.png'), dpi=300)
    plt.close()


def create_summary_table(results: Dict[str, Dict[str, Any]], output_dir: str) -> None:
    """
    Create a summary table of all metrics.
    
    Args:
        results: Benchmark results
        output_dir: Directory to save visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract model names and datasets
    model_names = list(results.keys())
    datasets = set()
    for model_id in model_names:
        for dataset_name in results[model_id]["datasets"].keys():
            datasets.add(dataset_name)
    datasets = sorted(list(datasets))
    
    # Create summary data
    summary_data = []
    
    for model_id in model_names:
        model_name = model_id.split('/')[-1]
        
        for dataset_name in datasets:
            if dataset_name in results[model_id]["datasets"]:
                metrics = results[model_id]["datasets"][dataset_name]["metrics"]
                memory_usage = results[model_id]["datasets"][dataset_name].get("memory_usage_mb", 0)
                total_time = results[model_id]["datasets"][dataset_name].get("total_time_seconds", 0)
                
                summary_data.append({
                    "Model": model_name,
                    "Dataset": dataset_name,
                    "Accuracy (%)": f"{metrics.get('accuracy', 0) * 100:.1f}%",
                    "Avg. Inference Time (s)": f"{metrics.get('inference_time_avg', 0):.2f}",
                    "Memory Usage (MB)": f"{memory_usage:.1f}",
                    "Reasoning Rate (%)": f"{metrics.get('reasoning_rate', 0) * 100:.1f}%",
                    "Avg. Reasoning Length": f"{metrics.get('reasoning_length_avg', 0):.0f}",
                    "Total Time (s)": f"{total_time:.1f}"
                })
    
    # Save summary as JSON
    with open(os.path.join(output_dir, 'summary.json'), 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    # Create a simple HTML table
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>DeepHermes Benchmark Summary</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            table { border-collapse: collapse; width: 100%; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
            tr:nth-child(even) { background-color: #f9f9f9; }
            h1 { color: #333; }
        </style>
    </head>
    <body>
        <h1>DeepHermes Benchmark Summary</h1>
        <table>
            <tr>
                <th>Model</th>
                <th>Dataset</th>
                <th>Accuracy (%)</th>
                <th>Avg. Inference Time (s)</th>
                <th>Memory Usage (MB)</th>
                <th>Reasoning Rate (%)</th>
                <th>Avg. Reasoning Length</th>
                <th>Total Time (s)</th>
            </tr>
    """
    
    for row in summary_data:
        html += f"""
            <tr>
                <td>{row['Model']}</td>
                <td>{row['Dataset']}</td>
                <td>{row['Accuracy (%)']}</td>
                <td>{row['Avg. Inference Time (s)']}</td>
                <td>{row['Memory Usage (MB)']}</td>
                <td>{row['Reasoning Rate (%)']}</td>
                <td>{row['Avg. Reasoning Length']}</td>
                <td>{row['Total Time (s)']}</td>
            </tr>
        """
    
    html += """
        </table>
    </body>
    </html>
    """
    
    with open(os.path.join(output_dir, 'summary.html'), 'w') as f:
        f.write(html)


def generate_visualizations(results: Dict[str, Dict[str, Any]], output_dir: str) -> None:
    """
    Generate visualizations for benchmark results.
    
    Args:
        results: Benchmark results
        output_dir: Directory to save visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create visualizations
    create_accuracy_comparison(results, output_dir)
    create_inference_time_comparison(results, output_dir)
    create_memory_usage_comparison(results, output_dir)
    create_reasoning_length_comparison(results, output_dir)
    create_summary_table(results, output_dir)
    
    print(f"Visualizations saved to {output_dir}")
