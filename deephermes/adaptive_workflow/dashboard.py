"""
Dashboard for visualizing hardware capabilities and performance metrics.

This module provides a visual dashboard for the adaptive workflow,
displaying hardware information and performance metrics.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import time
import threading
import os
import sys
from .hardware_profiles import detect_hardware, AppleSiliconProfile
from ..model_selector.hardware_detection import get_hardware_info, get_available_memory

def create_hardware_plot(hardware_profile: AppleSiliconProfile, ax: plt.Axes) -> None:
    """
    Create a radar chart of hardware capabilities.
    
    Args:
        hardware_profile: Hardware profile to visualize
        ax: Matplotlib axes to plot on
    """
    # Define the categories and values
    categories = [
        'CPU Cores', 
        'GPU Cores',
        'Neural Engine',
        'Memory (GB)',
        'Memory BW',
        'Compute Power'
    ]
    
    # Normalize values to 0-1 scale based on known max values
    max_values = {
        'CPU Cores': 24,  # M3 Max has 16 cores
        'GPU Cores': 76,  # M3 Max has 40 cores
        'Neural Engine': 32,  # 16-core is standard
        'Memory (GB)': 128,  # Max supported
        'Memory BW': 1000,  # GB/s (approximate max)
        'Compute Power': 100  # Arbitrary scale
    }
    
    values = [
        hardware_profile.cpu_cores / max_values['CPU Cores'],
        hardware_profile.gpu_cores / max_values['GPU Cores'],
        hardware_profile.neural_engine_cores / max_values['Neural Engine'],
        hardware_profile.memory_gb / max_values['Memory (GB)'],
        hardware_profile.memory_bandwidth_gbps / max_values['Memory BW'],
        hardware_profile.total_compute_power / max_values['Compute Power']
    ]
    
    # Number of variables
    N = len(categories)
    
    # Create angles for each category (divide the plot into equal parts)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Add values to complete the loop
    values += values[:1]
    
    # Draw the plot
    ax.plot(angles, values, linewidth=2, linestyle='solid', label=f"{hardware_profile.chip_family} {hardware_profile.chip_variant}")
    ax.fill(angles, values, alpha=0.25)
    
    # Add category labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    
    # Add radial labels (0-100%)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'])
    
    # Add title
    ax.set_title('Hardware Capabilities', size=14, fontweight='bold')
    
    # Add legend
    ax.legend(loc='upper right')

def create_memory_plot(ax: plt.Axes) -> Tuple[List[float], List[float]]:
    """
    Create a memory usage plot.
    
    Args:
        ax: Matplotlib axes to plot on
        
    Returns:
        Tuple of memory usage data and time data for updating
    """
    # Initialize data
    memory_data = []
    time_data = []
    start_time = time.time()
    
    # Get initial memory usage
    total_memory, available_memory = get_available_memory()
    used_memory = total_memory - available_memory
    memory_data.append(used_memory / total_memory * 100)  # As percentage
    time_data.append(0)  # Start at 0
    
    # Set up the plot
    line, = ax.plot(time_data, memory_data, 'b-', label='Memory Usage')
    ax.set_xlim(0, 60)  # Show 60 seconds
    ax.set_ylim(0, 100)  # 0-100%
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Memory Usage (%)')
    ax.set_title('Memory Usage Over Time', size=14, fontweight='bold')
    ax.grid(True)
    ax.legend()
    
    return memory_data, time_data, line

def update_memory_plot(memory_data: List[float], time_data: List[float], 
                      line: plt.Line2D, ax: plt.Axes, 
                      stop_event: threading.Event) -> None:
    """
    Update the memory usage plot in real-time.
    
    Args:
        memory_data: List to store memory usage data
        time_data: List to store time data
        line: Matplotlib line object to update
        ax: Matplotlib axes to update
        stop_event: Threading event to signal when to stop updating
    """
    start_time = time.time()
    
    while not stop_event.is_set():
        # Get memory usage
        total_memory, available_memory = get_available_memory()
        used_memory = total_memory - available_memory
        memory_data.append(used_memory / total_memory * 100)  # As percentage
        
        # Update time
        current_time = time.time() - start_time
        time_data.append(current_time)
        
        # Keep only the last 60 seconds of data
        if current_time > 60:
            # Find the index of the oldest data point to keep
            cutoff_index = 0
            for i, t in enumerate(time_data):
                if t > current_time - 60:
                    cutoff_index = i
                    break
            
            # Trim the data
            memory_data = memory_data[cutoff_index:]
            time_data = time_data[cutoff_index:]
            
            # Adjust time values
            time_data = [t - time_data[0] for t in time_data]
        
        # Update the plot
        line.set_data(time_data, memory_data)
        ax.set_xlim(0, max(60, current_time))
        
        # Pause to reduce CPU usage
        time.sleep(1)

def create_model_comparison_plot(ax: plt.Axes, hardware_profile: AppleSiliconProfile) -> None:
    """
    Create a bar chart comparing model performance on the current hardware.
    
    Args:
        ax: Matplotlib axes to plot on
        hardware_profile: Hardware profile to use for comparison
    """
    # Define models to compare
    models = [
        "DeepHermes-3-Llama-3-8B",
        "DeepHermes-3-Llama-3-8B-4bit",
        "DeepHermes-3-Llama-3-70B",
        "DeepHermes-3-Llama-3-70B-4bit"
    ]
    
    # Estimated tokens per second based on hardware profile
    # These are approximate values and would need to be calibrated
    compute_factor = hardware_profile.total_compute_power / 50.0  # Normalize to a baseline
    
    tokens_per_second = [
        15 * compute_factor,  # 8B full precision
        30 * compute_factor,  # 8B 4-bit quantized
        2 * compute_factor,   # 70B full precision
        5 * compute_factor    # 70B 4-bit quantized
    ]
    
    # Memory requirements (GB)
    memory_required = [
        16,   # 8B full precision
        6,    # 8B 4-bit quantized
        140,  # 70B full precision
        40    # 70B 4-bit quantized
    ]
    
    # Create the bar chart for tokens per second
    x = np.arange(len(models))
    width = 0.35
    
    # Color bars based on whether they fit in available memory
    total_memory = hardware_profile.memory_gb
    colors = ['green' if mem <= total_memory * 0.8 else 'red' for mem in memory_required]
    
    bars = ax.bar(x, tokens_per_second, width, color=colors)
    
    # Add labels and title
    ax.set_xlabel('Model')
    ax.set_ylabel('Tokens per Second (estimated)')
    ax.set_title('Model Performance Comparison', size=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    
    # Add memory requirement annotations
    for i, bar in enumerate(bars):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{memory_required[i]} GB",
                ha='center', va='bottom', rotation=0)
    
    # Add a note about memory
    memory_note = f"Available Memory: {total_memory} GB"
    ax.text(0.5, 0.95, memory_note, transform=ax.transAxes,
            ha='center', va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.grid(True, axis='y')

def show_dashboard() -> None:
    """Show the hardware and performance dashboard."""
    try:
        # Detect hardware
        hardware_profile = detect_hardware()
        
        # Create figure with subplots
        fig = plt.figure(figsize=(15, 10))
        fig.suptitle(f"DeepHermes Adaptive Workflow Dashboard - {hardware_profile.chip_family} {hardware_profile.chip_variant}", 
                     fontsize=16, fontweight='bold')
        
        # Create subplots
        ax1 = fig.add_subplot(2, 2, 1, polar=True)  # Hardware radar chart
        ax2 = fig.add_subplot(2, 2, 2)              # Memory usage over time
        ax3 = fig.add_subplot(2, 2, 3)              # Model comparison
        ax4 = fig.add_subplot(2, 2, 4)              # Reserved for future use
        
        # Create hardware radar chart
        create_hardware_plot(hardware_profile, ax1)
        
        # Create memory usage plot
        memory_data, time_data, line = create_memory_plot(ax2)
        
        # Create model comparison plot
        create_model_comparison_plot(ax3, hardware_profile)
        
        # Add a placeholder for the fourth plot
        ax4.text(0.5, 0.5, "Future Metrics\n(Coming Soon)", 
                 ha='center', va='center', fontsize=14)
        ax4.set_title("Additional Metrics", size=14, fontweight='bold')
        ax4.axis('off')
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Create a stop event for the memory update thread
        stop_event = threading.Event()
        
        # Start memory update thread
        update_thread = threading.Thread(
            target=update_memory_plot,
            args=(memory_data, time_data, line, ax2, stop_event)
        )
        update_thread.daemon = True
        update_thread.start()
        
        # Show the plot (this blocks until the window is closed)
        plt.show()
        
        # Stop the update thread when the plot is closed
        stop_event.set()
        update_thread.join(timeout=1)
        
    except Exception as e:
        print(f"Error showing dashboard: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    show_dashboard()
