"""
This script processes data collected from robot simulation experiments.
It contains functions to clean, transform, and prepare data for subsequent analysis.
SOLER PROJECT.
Author: Gabriel Madroñero
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import logging
from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import MaxNLocator
import sys
from scipy import stats
import seaborn as sns
import math
from matplotlib.patches import Patch

logger = logging.getLogger(__name__)

# === Global parameters (adjust if changed in the controller) ===
T_TICKS_PER_SEC = 10.0   # Should match T_TICKS_PER_SEC in Lua controller
W_STD_SEC = 60.0#180.0        # w_std from paper (for reference axis)
W_MIN_SEC = 7.0#24.0         # w_min from paper (for reference axis)
MIN_TIMESTEP = 0         # Adjusted to 0 to capture all new data
MAX_TIMESTEP = 10000   # Maximum timestep to consider

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def thousands_formatter(x, pos):
    """
    Formatter to display Y-axis values in thousands (x10^3).
    Example: 120000 -> 120
    """
    return f"{x/1000:.1f}"

class RobotDataProcessor:
    """Class to process robot simulation experiment data."""
    
    def __init__(self, csv_path: str, output_dir: str, ticks_per_sec: float = 10.0):
        self.csv_path = Path(csv_path)
        self.output_dir = Path(output_dir)
        self.ticks_per_sec = ticks_per_sec
        self.raw_data = None
        self.data_selective = None
        self.data_greedy = None
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_data(self) -> pd.DataFrame:
        try:
            if not self.csv_path.exists():
                raise FileNotFoundError(f"CSV file not found: {self.csv_path}")
            self.raw_data = pd.read_csv(self.csv_path)
            logger.info(f"Successfully loaded data with {len(self.raw_data)} rows")
            return self.raw_data
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        required_columns = ['tick', 'greedy', 'robot', 'm', 'p_x', 'planned_wticks', 'task', 'x', 'y', 'seed']
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return False
        if not pd.api.types.is_numeric_dtype(df['tick']):
            logger.warning("'tick' column should be numeric")
        if df.empty:
            logger.warning("DataFrame is empty")
            return False
        logger.info("Data validation passed")
        return True
    
    def preprocess_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        df_clean = df.copy()
        
        # Filtrar exclusivamente para el robot q0 <---
        logger.info("Filtrando datos exclusivamente para el robot 'q0'")
        df_clean = df_clean[df_clean['robot'] == 'q0']
                
        logger.info(f"Filtering data: timestep {MIN_TIMESTEP} to {MAX_TIMESTEP}")
        df_clean = df_clean[(df_clean['tick'] >= MIN_TIMESTEP) & (df_clean['tick'] <= MAX_TIMESTEP)]
        
        if df_clean['greedy'].dtype == object:
            df_clean['greedy'] = df_clean['greedy'].astype(str).str.lower().map({'true': True, 'false': False})
        
        data_selective = df_clean[df_clean['greedy'] == False].copy()
        data_greedy = df_clean[df_clean['greedy'] == True].copy()
        
        for data in [data_selective, data_greedy]:
            if not data.empty:
                data['time_seconds'] = data['tick'] / self.ticks_per_sec
                data['w_sec'] = data['planned_wticks'] / self.ticks_per_sec
        
        return data_selective, data_greedy
    
    def get_basic_stats(self, df: pd.DataFrame, strategy_name: str) -> Dict[str, Any]:
        if df.empty: return {}
        stats = {
            'strategy': strategy_name,
            'total_entries': len(df),
            'unique_robots': df['robot'].nunique(),
            'task_distribution': df['task'].value_counts().to_dict(),
            'avg_p_x': df['p_x'].mean(),
            'avg_m': df['m'].mean()
        }
        if 'w_sec' in df.columns:
            stats.update({'avg_completion_time': df['w_sec'].mean()})
        return stats
        
    def print_comparison(self):
        stats_selective = self.get_basic_stats(self.data_selective, "Selective")
        stats_greedy = self.get_basic_stats(self.data_greedy, "Greedy")
        print("\n" + "="*50)
        print("STRATEGY COMPARISON")
        print("="*50)
        for stats in [stats_selective, stats_greedy]:
            if stats:
                print(f"\n--- {stats['strategy']} Strategy ---")
                print(f"Total tasks: {stats['total_entries']}")
                print(f"Average p_x: {stats['avg_p_x']:.3f}")
                print(f"Average m: {stats['avg_m']:.3f}")
                print(f"Task Split: {stats['task_distribution']}")
                if 'avg_completion_time' in stats:
                    print(f"Avg completion time: {stats['avg_completion_time']:.2f} s")

    # ============================================================================
    # PLOTTING FUNCTIONS
    # ============================================================================

    def plot_spatial_heatmap(self, save_path: Optional[str] = None) -> None:
        """Generates a 2D spatial heatmap of task execution locations."""
        if self.data_selective is None or self.data_greedy is None: return
        logger.info("Generating Spatial Heatmap")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        bounds = [[-2.5, 2.5], [-2.5, 2.5]]

        for ax, data, title in zip([ax1, ax2], [self.data_selective, self.data_greedy], ["Selective Strategy", "Greedy Strategy"]):
            if data is not None and not data.empty:
                h = ax.hist2d(data['x'], data['y'], bins=25, range=bounds, cmap='Greys', alpha=0.3)
                blue_tasks = data[data['task'] == 'BLUE']
                red_tasks = data[data['task'] == 'RED']
                ax.scatter(blue_tasks['x'], blue_tasks['y'], color='blue', s=20, alpha=0.6, label='BLUE Tasks', edgecolors='white', linewidth=0.5)
                ax.scatter(red_tasks['x'], red_tasks['y'], color='red', s=20, alpha=0.6, label='RED Tasks', edgecolors='white', linewidth=0.5)
                ax.set_title(title, fontsize=14, fontweight='bold')
                ax.set_xlabel('X Coordinate (m)', fontsize=12)
                ax.set_ylabel('Y Coordinate (m)', fontsize=12)
                ax.set_xlim(-2.5, 2.5)
                ax.set_ylim(-2.5, 2.5)
                ax.grid(True, linestyle='--', alpha=0.4)
                ax.legend(loc='upper right')
                cbar = fig.colorbar(h[3], ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label('Activity Density', rotation=270, labelpad=15)

        fig.suptitle("Spatial Distribution of Task Executions", fontsize=18, y=1.02)
        plt.tight_layout()
        if save_path: plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_comparison_histograms(self, save_dir: Optional[str] = None) -> None:
        """Plot histograms for both strategies side by side for comparison."""
        if self.data_selective is None or self.data_greedy is None: return
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        bins = np.arange(W_MIN_SEC, W_STD_SEC + 6, 6)
        
        max_freq = 0
        if not self.data_selective.empty:
            counts_sel, _ = np.histogram(self.data_selective["w_sec"].values, bins=bins)
            max_freq = max(max_freq, counts_sel.max())
        if not self.data_greedy.empty:
            counts_gre, _ = np.histogram(self.data_greedy["w_sec"].values, bins=bins)
            max_freq = max(max_freq, counts_gre.max())
            
        y_limit_normalized = max_freq * 1.15

        for ax, data, title, color in zip([ax1, ax2], [self.data_selective, self.data_greedy], ["Selective Strategy", "Greedy Strategy"], ['green', 'orange']):
            if not data.empty:
                w_sec = data["w_sec"].values
                ax.hist(w_sec, bins=bins, edgecolor="black", alpha=0.7, color=color)
                ax.set_xlabel("Task completion time $w_x$ (s)")
                ax.set_ylim(0, y_limit_normalized)
                ax.yaxis.set_major_formatter(FuncFormatter(thousands_formatter))
                ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
                ax.set_ylabel("Number of tasks completed ($10^3$)")
                ax.set_title(title)
                ax.axvline(W_MIN_SEC, linestyle="--", color='red', alpha=0.7)
                ax.axvline(W_STD_SEC, linestyle="--", color='blue', alpha=0.7)
                ax.grid(True, alpha=0.3)
                ax.text(0.02, 0.98, f"Tasks: {len(w_sec)}", transform=ax.transAxes, fontsize=10, va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        fig.suptitle(f"Comparison of Task Completion Times", fontsize=16, y=1.02)
        plt.tight_layout()
        if save_dir: plt.savefig(Path(save_dir) / "strategy_comparison_histograms.png", dpi=300, bbox_inches='tight')
        plt.show()

    def plot_performance_boxplot(self, window_sec: int = 1000, max_time_sec: int = 10000, save_path: Optional[str] = None) -> None:
        """Generates a boxplot showing tasks completed per time window for both strategies."""
        if self.raw_data is None or self.raw_data.empty:
            logger.warning("No data available to plot.")
            return

        logger.info("Generating improved performance boxplot")
        df = self.raw_data.copy()
        df = df[df['robot'] == 'q0']
        # 1. Processing and filtering data
        if df['greedy'].dtype == object:
             df['greedy'] = df['greedy'].astype(str).str.lower().map({'true': True, 'false': False})

        df['strategy'] = df['greedy'].map({True: 'greedy', False: 'selective'})
        df['time_sec'] = df['tick'] / self.ticks_per_sec 

        # 2. Filtering and make tasks per time window
        df = df[(df['time_sec'] >= 0) & (df['time_sec'] <= max_time_sec)]
        df['time_window'] = (np.ceil(df['time_sec'] / window_sec) * window_sec).astype(int)
        df = df[(df['time_window'] > 0) & (df['time_window'] <= max_time_sec)]

        # 3. Data groping to count tasks per strategy and time window
        df['experiment_id'] = df.groupby(['strategy', 'seed']).ngroup()
        perf = df.groupby(['experiment_id', 'strategy', 'time_window']).size().reset_index(name='tasks_completed')

        # Origin to initialize boxplots with 0 tasks at time 0 for each experiment
        dummy_data = []
        for exp_id in perf['experiment_id'].unique():
            strat = perf[perf['experiment_id'] == exp_id]['strategy'].iloc[0]
            dummy_data.append({'experiment_id': exp_id, 'strategy': strat, 'time_window': 0, 'tasks_completed': 0})

        perf = pd.concat([perf, pd.DataFrame(dummy_data)], ignore_index=True)
        perf = perf.sort_values(by='time_window')

        # 4. Size of plot
        fig, ax = plt.subplots(figsize=(10, 10))

        # 
        perf['tasks_completed'] = perf['tasks_completed'].astype(int)
        # Color to diferrentiate strategies
        my_palette = {'selective': '#1f77b4', 'greedy': '#ff7f0e'} # Azul y Naranja

        perf['time_window'] = perf['time_window'].astype(int)
        perf['time_window_str'] = perf['time_window'].astype(str)

        sns.boxplot(
            data=perf,
            x='time_window',
            y='tasks_completed',
            hue='strategy',
            palette=my_palette,
            width=0.6,
            fliersize=5, # Outliers size
            flierprops={'marker': 'o'}, # Outliers marker
            ax=ax
        )
        # Get current tick labels (they are your time_window values)
        labels = [int(t.get_text()) for t in ax.get_xticklabels()]
        # Convert to scientific-style labels
        #new_labels = [f"{int(x/1000)}" if x != 0 else "0" for x in labels]
        ticks = ax.get_xticks()

        ax.set_xticks(ticks)
        #ax.set_xticklabels(new_labels)
        ax.set_xticklabels([str(int(x)) for x in labels])

        for i, box in enumerate(ax.patches):
            # 1. Get the color (RGBA) from actually box
            box_color = box.get_facecolor()

            # ---> SOLUCIÓN: Calcular el centro y dibujar un único punto <---
            median_line_idx = (i * 6) + 4
            if median_line_idx < len(ax.lines):
                median_line = ax.lines[median_line_idx]
                
                # Obtenemos las coordenadas X e Y de la línea de la mediana
                x_coords = median_line.get_xdata()
                y_coords = median_line.get_ydata()
                
                # Calculamos el punto central exacto
                if len(x_coords) == 2 and len(y_coords) == 2:
                    x_center = sum(x_coords) / 2.0
                    y_center = y_coords[0]
                    
                    # Dibujamos un solo punto en ese centro
                    ax.plot(x_center, y_center, marker='o', markersize=8, 
                            markerfacecolor=box_color, markeredgecolor='black', 
                            linestyle='None', zorder=10)

            # 2. Outliers (fliers)
            flier_line_idx = (i * 6) + 5 
            if flier_line_idx < len(ax.lines):
                flier_line = ax.lines[flier_line_idx]
                flier_line.set_markerfacecolor(box_color)  
                flier_line.set_markeredgecolor('black')    
                flier_line.set_alpha(0.7)                  # Opacity for visibility

        # 5. Finally customize axes and grid to match the paper's style
        ax.set_xlabel("Time (s) ", fontsize=12)
        ax.set_ylabel("Number of tasks completed", fontsize=12)

        # Axes limits 
        ax.grid(True, axis='y', linestyle=':', alpha=0.7)

        ax.legend(title="Strategy", fontsize=11, loc='upper left')
        # save figure
        plt.tight_layout()
        if save_path: 
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_search_time_distribution(self, max_time_sec: int = 10000, save_path: Optional[str] = None) -> None:
        """Generates a violin plot to show the overall distribution of task search times per strategy, including statistical legends."""
        if self.raw_data is None or self.raw_data.empty:
            logger.warning("No data available to plot.")
            return

        logger.info("Generating search time distribution plot with statistics")
        df = self.raw_data.copy()
        df = df[df['robot'] == 'q0']
        # 1. Initial data preparation
        if df['greedy'].dtype == object:
            df['greedy'] = df['greedy'].astype(str).str.lower().map({'true': True, 'false': False})

        df['strategy'] = df['greedy'].map({True: 'greedy', False: 'selective'})
        df['time_sec'] = df['tick'] / self.ticks_per_sec
        df = df[(df['time_sec'] >= 0) & (df['time_sec'] <= max_time_sec)]

        # Convert search ticks to seconds for better interpretability
        df['search_time_sec'] = df['search_ticks'] / self.ticks_per_sec

        # Calculate statistics for the legend ---
        # We group by strategy and calculate mean, median, standard deviation, and max values
        stats = df.groupby('strategy')['search_time_sec'].agg(['mean', 'median', 'std', 'max']).round(2)

        # 2. VISUALIZATION
        fig, ax = plt.subplots(figsize=(10, 6)) # Slightly wider to accommodate the descriptive legend

        my_palette = {'selective': '#1f77b4', 'greedy': '#ff7f0e'}

        sns.violinplot(
            data=df,
            x='strategy',
            y='search_time_sec',
            order=['selective', 'greedy'], 
            palette=my_palette,
            inner='quartile', 
            cut=0,            
            linewidth=1.2,
            ax=ax,
            hue='strategy', 
            legend=False
        )

        # 3. AESTHETICS & LEGEND
        # Add the professional English title
        ax.set_title("Spent task search time per Strategy", fontsize=14, fontweight='bold', pad=15)
        ax.set_ylabel("Search Time (s)", fontsize=12)
        ax.set_xlabel("Strategy", fontsize=12)

        # Create custom legend handles with statistics ---
        legend_handles = []
        for strat in ['selective', 'greedy']:
            if strat in stats.index:
                s_mean = stats.loc[strat, 'mean']
                s_med = stats.loc[strat, 'median']
                s_std = stats.loc[strat, 'std']
                s_max = stats.loc[strat, 'max']

                # Format the text that will appear next to each color box
                label_text = f"{strat.capitalize()}\nMean: {s_mean}s\nMedian: {s_med}s\nStd: {s_std}s\nMax: {s_max}s"

                # Create a colored patch for the legend
                patch = Patch(facecolor=my_palette[strat], edgecolor='black', label=label_text)
                legend_handles.append(patch)

        # Place the legend in a spot where it doesn't obstruct the violins (adjust loc if needed)
        ax.legend(handles=legend_handles, title="Descriptive Statistics", 
                  loc='upper right', fontsize=10, title_fontsize=11)

        ax.grid(True, axis='y', linestyle=':', alpha=0.7)

        plt.tight_layout()
        if save_path: 
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_f_measure_boxplot(self, window_sec: int = 1000, max_time_sec: int = 10000, save_path: Optional[str] = None) -> None:
        """Generates a boxplot of the F-measure (consistency of task execution) over time."""
        if self.raw_data is None or self.raw_data.empty:
            logger.warning("No data available to plot.")
            return

        logger.info("Generating improved F-measure boxplot")
        df = self.raw_data.copy()
        df = df[df['robot'] == 'q0']
        # 1. Initial data cleaning and preparation
        if df['greedy'].dtype == object:
            df['greedy'] = df['greedy'].astype(str).str.lower().map({'true': True, 'false': False})

        df['strategy'] = df['greedy'].map({True: 'greedy', False: 'selective'})
        df['time_sec'] = df['tick'] / self.ticks_per_sec
        df = df[(df['time_sec'] >= 0) & (df['time_sec'] <= max_time_sec)]

        # Filter for valid tasks
        df = df[df['task'].isin(['BLUE', 'RED'])]

        # Define time windows
        df['time_window'] = (np.ceil(df['time_sec'] / window_sec) * window_sec).astype(int)
        df = df[(df['time_window'] > 0) & (df['time_window'] <= max_time_sec)]

        # Robust experiment detection using the 'seed' column
        df['experiment_id'] = df.groupby(['strategy', 'seed']).ngroup()

        # 2. VECTORIZED PROCESSING (Performance optimization)
        # Sort chronologically to ensure the sequence of tasks is accurate per robot
        df = df.sort_values(by=['experiment_id', 'time_window', 'robot', 'tick'])

        # Create a shifted column to compare the current task with the previous one for the SAME robot
        df['prev_task'] = df.groupby(['experiment_id', 'time_window', 'robot'])['task'].shift(1)

        # A switch occurs if the current task is different from the previous one (ignoring the first NaN)
        df['is_switch'] = (df['task'] != df['prev_task']) & (df['prev_task'].notnull())

        # Group by robot to count total tasks (N) and the sum of switches
        robot_stats = df.groupby(['experiment_id', 'strategy', 'time_window', 'robot']).agg(
            N=('task', 'count'),
            switches=('is_switch', 'sum')
        ).reset_index()

        # Calculate the F-measure per robot using np.where for vectorized conditional logic
        robot_stats['f_measure'] = np.where(
            robot_stats['N'] == 1, 
            1.0, 
            1.0 - (2.0 * robot_stats['switches'] / robot_stats['N'])
        )

        # Average the F-measure of all robots to get the overall value for the experiment in that time window
        f_df = robot_stats.groupby(['experiment_id', 'strategy', 'time_window'])['f_measure'].mean().reset_index()

        # 3. VISUALIZATION
        fig, ax = plt.subplots(figsize=(10, 10))
        my_palette = {'selective': '#1f77b4', 'greedy': '#ff7f0e'}
        # create origin 0 axis to start boxplots from 0 F-measure at time 0 for each experiment
        # Get the unique time, order the time an put the 0 initial point for each strategy
        time_order = [0] + sorted(f_df['time_window'].unique().tolist())
        f_df['time_window'] = f_df['time_window'].astype(int)
        f_df['time_window_str'] = f_df['time_window'].astype(str)
        
        sns.boxplot(
            data=f_df,
            x='time_window',
            y='f_measure',
            hue='strategy',
            hue_order=['selective', 'greedy'], # ENFORCES ORDER: selective first, greedy second
            order=time_order,
            palette=my_palette,
            width=0.6,
            fliersize=5,
            flierprops={'marker': 'o'},
            ax=ax
        )

        # Get current tick labels (they are your time_window values)
        labels = [int(t.get_text()) for t in ax.get_xticklabels()]
        #new_labels = [f"{int(x/1000)}" if x != 0 else "0" for x in labels]
        # Convert to scientific-style labels    
        ticks = ax.get_xticks()

        ax.set_xticks(ticks)
        #ax.set_xticklabels(new_labels)
        ax.set_xticklabels([str(int(x)) for x in labels])

        # Custom logic to color the outliers based on their respective box color
        # Custom logic to color the outliers AND add a single median dot
        for i, box in enumerate(ax.patches):
            box_color = box.get_facecolor()
            
            # ---> SOLUCIÓN: Calcular el centro y dibujar un único punto <---
            median_line_idx = (i * 6) + 4
            if median_line_idx < len(ax.lines):
                median_line = ax.lines[median_line_idx]
                
                x_coords = median_line.get_xdata()
                y_coords = median_line.get_ydata()
                
                if len(x_coords) == 2 and len(y_coords) == 2:
                    x_center = sum(x_coords) / 2.0
                    y_center = y_coords[0]
                    
                    ax.plot(x_center, y_center, marker='o', markersize=8, 
                            markerfacecolor=box_color, markeredgecolor='black', 
                            linestyle='None', zorder=10)
            
            # Outliers
            flier_line_idx = (i * 6) + 5 
            if flier_line_idx < len(ax.lines):
                flier_line = ax.lines[flier_line_idx]
                flier_line.set_markerfacecolor(box_color)
                flier_line.set_markeredgecolor('black')
                flier_line.set_alpha(0.7)

        # 4. AESTHETICS
        ax.set_ylabel("F-measure (Specialization)", fontsize=12)
        ax.set_xlabel("Time (s) [×10³]", fontsize=12)

        # Set y-limits with extra margin at the top (1.15) to make room for the legend
        ax.set_ylim(-0.15, 1.15) 
        ax.grid(True, axis='y', linestyle=':', alpha=0.7)

        # Place legend in the upper left corner to avoid overlapping with data
        ax.legend(title="Strategy", fontsize=11, loc='upper left') 

        plt.tight_layout()
        if save_path: 
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

# ============================================================================
# SPECIALIZATION SCATTER PLOT 
# ============================================================================
class SpecializationScatterPlotter:
    def __init__(self, csv_path: str):
        self.csv_path = Path(csv_path)
        self.raw_data = None
        self.robot_stats = None
        
    def load_data(self) -> pd.DataFrame:
        self.raw_data = pd.read_csv(self.csv_path)
        return self.raw_data

    def preprocess_data(self) -> pd.DataFrame:
        df = self.raw_data.copy()
        
        # ---> NUEVO: Filtrar exclusivamente para el robot q0 <---
        df = df[df['robot'] == 'q0']
        
        if df['greedy'].dtype == object:
            df['greedy'] = df['greedy'].astype(str).str.lower().map({'true': True, 'false': False})
        df['strategy'] = df['greedy'].map({True: 'greedy', False: 'selective'})

        df['experiment_id'] = df.groupby(['strategy', 'seed']).ngroup()

        TASK_TYPES = ['BLUE', 'RED']
        df_tasks = df[df['task'].isin(TASK_TYPES)].copy()
        
        # AJUSTE CLAVE: Añadimos 'seed' al groupby para no perder la información de la semilla
        self.robot_stats = (df_tasks.groupby(['experiment_id', 'strategy', 'seed', 'robot', 'task'])
                            .size().unstack(fill_value=0).reset_index())

        for task in TASK_TYPES:
            if task not in self.robot_stats.columns:
                self.robot_stats[task] = 0
                
        return self.robot_stats

    def _calculate_spec_index(self, df_subset: pd.DataFrame) -> float:
        if df_subset.empty: return 0.0
        blue = df_subset['BLUE']
        red = df_subset['RED']
        total = blue + red
        active = total > 0
        if not active.any(): return 0.0
        diff = (blue[active] - red[active]).abs()
        return (diff / total[active]).mean()

    # --- SCATTER PLOT---
    def plot_figure(self, save_path: str) -> None:
        if self.robot_stats is None: return
        
        selective_data = self.robot_stats[self.robot_stats['strategy'] == 'selective']
        greedy_data = self.robot_stats[self.robot_stats['strategy'] == 'greedy']
        
        fig, axes = plt.subplots(2, 1, figsize=(8, 11), sharex=True, sharey=True)
        fig.patch.set_facecolor('#2b2b2b')


        def _plot(ax, data, title, show_xlabel=False):
            if data.empty:
                ax.text(0.5, 0.5, "No Data", ha='center', va='center')
                return
            
            total = data['BLUE'] + data['RED']
            bias = (data['BLUE'] - data['RED']) / total.replace(0, 1)
            
            ax.set_facecolor('#2b2b2b')

            ax.tick_params(colors='white')
            ax.title.set_color('white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
        
            scatter = ax.scatter(data['BLUE'], data['RED'], 
                                 c=bias, cmap='RdBu', norm=plt.Normalize(vmin=-0.5, vmax=0.5),
                                 s=30, alpha=0.7, edgecolors='none')
            
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_ylabel("Total tasks $\\tau_r$ (Red)", fontsize=12)
            if show_xlabel:
                ax.set_xlabel("Total tasks $\\tau_b$ (Blue)", fontsize=12)
            
            max_val = max(data['BLUE'].max(), data['RED'].max(), 10) * 1.1
            ax.plot([0, max_val], [0, max_val], color='white', linestyle='--', alpha=0.5, label='Equilibrium')
            ax.grid(True, linestyle=':', alpha=0.4, color='white')
            
            spec_idx = self._calculate_spec_index(data)
            ax.text(0.95, 0.95, f"Swarm Spec. Index: {spec_idx:.2f}", 
                    transform=ax.transAxes, ha='right', va='top', 
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="#FFFFFF", alpha=0.8, edgecolor='none'))

        _plot(axes[0], selective_data, "Selective Strategy (SOLE-R)")
        _plot(axes[1], greedy_data, "Greedy Strategy (Baseline)", show_xlabel=True)

        cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.02])
        cbar = fig.colorbar(plt.cm.ScalarMappable(cmap='RdBu', norm=plt.Normalize(vmin=-1, vmax=1)), 
                            cax=cbar_ax, orientation='horizontal')
        cbar.set_label('Robot Specialization: Red Specialist <--- Generalist ---> Blue Specialist',color='white')
        cbar.ax.tick_params(colors='white')
        plt.subplots_adjust(bottom=0.15, hspace=0.3)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    # --- FUNCIÓN PARA LA MATRIZ DE EXPERIMENTOS ---
    def plot_strategy_grid(self, strategy: str, save_path: str) -> dict:
        """
        Plots a grid (e.g., 4x5) of scatter plots for each individual seed of a given strategy.
        Returns a dictionary with the best seed and its specialization index.
        """
        if self.robot_stats is None:
            print("Please run preprocess_data() first.")
            return {}

        # Filtramos por la estrategia seleccionada 'selective' o 'greedy'
        df_strat = self.robot_stats[self.robot_stats['strategy'] == strategy]
        unique_seeds = df_strat['seed'].unique()
        n_experiments = len(unique_seeds)

        if n_experiments == 0:
            print(f"No data found for strategy: {strategy}")
            return {}

        # NORMALIZACIÓN Encontramos el máximo en TODA la data para establecer límites comunes en todas las gráficas. 
        # Esto garantiza que el eje X y Y sean idénticos en todas las gráficas generadas.
        global_max = max(self.robot_stats['BLUE'].max(), self.robot_stats['RED'].max(), 10) * 1.05

        # Configuramos la matriz se calcula filas y columnas dinámicamente, por defecto 5 columnas
        cols = 5
        rows = math.ceil(n_experiments / cols)
        
        # Creamos la figura ajustando el tamaño según las filas
        fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows), sharex=True, sharey=True)
        # cambiamos el color de fondo 
        fig.patch.set_facecolor('#2b2b2b')  # dark gray background
        # Si es un solo renglón o columna, aseguramos que axes sea plano (flatten)
        axes = np.atleast_1d(axes).flatten()

        best_seed = None
        best_spec_idx = -1.0

        for i, seed in enumerate(unique_seeds):
            ax = axes[i]
            data = df_strat[df_strat['seed'] == seed]
            
            # Cálculo del Spec Index para esta semilla en particular
            spec_idx = self._calculate_spec_index(data)
            
            # Rastreamos la mejor semilla
            if spec_idx > best_spec_idx:
                best_spec_idx = spec_idx
                best_seed = seed

            # Lógica de ploteo
            total = data['BLUE'] + data['RED']
            bias = (data['BLUE'] - data['RED']) / total.replace(0, 1)

            ax.set_facecolor('#2b2b2b') # fondo oscuro

            ax.scatter(data['BLUE'], data['RED'], 
                       c=bias, cmap='coolwarm', norm=plt.Normalize(vmin=-0.5, vmax=0.5),
                       s=15, alpha=0.8, edgecolors='none')
            
            # Títulos y métricas por subplot
            ax.set_title(f"Seed: {seed}", fontsize=10)
            ax.text(0.05, 0.95, f"Spec: {spec_idx:.2f}", 
                    transform=ax.transAxes, ha='left', va='top', fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="#FFFFFF", alpha=0.8, edgecolor='none'))
            

            # CAmbio de color de titulos 
            ax.title.set_color('white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.tick_params(colors='white')
            # Línea de equilibrio y límites globales
            ax.plot([0, global_max], [0, global_max], color='white', linestyle='--', alpha=0.3)
            ax.set_xlim(-1, global_max)
            ax.set_ylim(-1, global_max)
            ax.grid(True, linestyle=':', alpha=0.4)

            # Etiquetas de ejes solo en los bordes para limpiar visualmente
            if i % cols == 0:
                ax.set_ylabel("Red Tasks", fontsize=10)
            if i >= (rows - 1) * cols:
                ax.set_xlabel("Blue Tasks", fontsize=10)

        # Ocultamos los subplots vacíos si n_experiments no es múltiplo exacto de cols
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        # Título global de la figura
        fig.suptitle(f"Individual Experiments - {strategy.capitalize()} Strategy", fontsize=16, fontweight='bold', y=0.98, color='white') 

        # Barra de color global (abajo)
        cbar_ax = fig.add_axes([0.15, 0.02, 0.7, 0.02])
        cbar = fig.colorbar(plt.cm.ScalarMappable(cmap='RdBu', norm=plt.Normalize(vmin=-1, vmax=1)), 
                            cax=cbar_ax, orientation='horizontal')
        cbar.set_label('Robot Specialization: Red Specialist <--- Generalist ---> Blue Specialist',color='white')
        cbar.ax.tick_params(colors='white')
        # Ajuste de márgenes para que quepa todo bien
        plt.subplots_adjust(bottom=0.1, hspace=0.3, wspace=0.1)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

        print(f"[{strategy.capitalize()}] Best performing seed: {best_seed} with Spec Index: {best_spec_idx:.3f}")
        return {'strategy': strategy, 'best_seed': best_seed, 'best_spec_index': best_spec_idx}
        
# ------------------------------------------------------------------
# MAIN EXECUTION
# ------------------------------------------------------------------
def main():
    # Update paths to match the new automated setup
    csv_path = r"/home/gmadro/EXP_CASOS/CASO4/experiment_data.csv"
    output_dir = r"/home/gmadro/EXP_CASOS/CASO4/processing_data"
    
    try:
        processor = RobotDataProcessor(csv_path, output_dir)
        raw_data = processor.load_data()
        
        if not processor.validate_data(raw_data):
            return
            
        data_selective, data_greedy = processor.preprocess_data(raw_data)
        processor.data_selective = data_selective
        processor.data_greedy = data_greedy
        
        processor.print_comparison()
        
        print("\n" + "="*50)
        print("GENERATING PLOTS & FIGURES")
        print("="*50)

        # 1. Plot Comparison Histograms
        processor.plot_comparison_histograms(save_dir=output_dir)
            
        # 2. Plot Spatial Heatmap 
        processor.plot_spatial_heatmap(save_path=f"{output_dir}/spatial_heatmap.png")
            
        # 3. Figure 8 - Specialization Scatter Plot
        spec_plotter = SpecializationScatterPlotter(csv_path)
        spec_plotter.load_data()
        spec_plotter.preprocess_data()
        spec_plotter.plot_figure(save_path=f"{output_dir}/specialization_scatter.png")

        # 3.1 Gráficas matriciales y obtención de mejores semillas por experiento
        best_selective = spec_plotter.plot_strategy_grid('selective', save_path=f"{output_dir}/figure_grid_selective.png")
        best_greedy = spec_plotter.plot_strategy_grid('greedy', save_path=f"{output_dir}/figure_grid_greedy.png")

        print("--- Best experiments seeds ---")
        print("BEST SELECTIVE:", best_selective)
        print("BEST GREEDY:", best_greedy)

        # 4. Figure 6 - Performance Boxplot
        processor.plot_performance_boxplot(
            window_sec=100,
            max_time_sec=2000,
            save_path=f"{output_dir}/figure6_performance_boxplot.png"
        )

        # 5. Time search expend for the task distribution - Violin plot
        processor.plot_search_time_distribution(
            max_time_sec=2000,
            save_path=f"{output_dir}/figure_search_time_.png"
        )

        # 6. Figure 10 - F-measure Boxplot
        processor.plot_f_measure_boxplot(
            window_sec=100,
            max_time_sec=2000,
            save_path=f"{output_dir}/figure_f_measure_boxplot.png"
        )

    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()