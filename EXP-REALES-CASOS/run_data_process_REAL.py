"""
This script processes data collected from REAL robot experiments (QUPA).
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

# === Global parameters for Real Robots ===
W_STD_SEC = 60.0         # w_std from paper (for reference axis)
W_MIN_SEC = 7.9          # w_min from paper (for reference axis)
MIN_TIMESTEP = 0         
MAX_TIMESTEP = 600       # Maximum timestep to consider (Real duration)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RobotDataProcessor:
    """Class to process real robot experiment data."""
    
    def __init__(self, csv_path: str, output_dir: str):
        self.csv_path = Path(csv_path)
        self.output_dir = Path(output_dir)
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

        logger.info("Filtrando datos exclusivamente para el robot 'qupa_7E'")
        df_clean = df_clean[df_clean['robot'] == 'qupa_7E']
        
        # Mapeo de nomenclatura física a lógica
        if 'task' in df_clean.columns:
            df_clean['task'] = df_clean['task'].replace({'TYPE_B': 'BLUE', 'TYPE_R': 'RED'})
                
        logger.info(f"Filtering data: timestep {MIN_TIMESTEP} to {MAX_TIMESTEP}")
        df_clean = df_clean[(df_clean['tick'] >= MIN_TIMESTEP) & (df_clean['tick'] <= MAX_TIMESTEP)]
        
        if df_clean['greedy'].dtype == object:
            df_clean['greedy'] = df_clean['greedy'].astype(str).str.lower().map({'true': True, 'false': False})
        
        data_selective = df_clean[df_clean['greedy'] == False].copy()
        data_greedy = df_clean[df_clean['greedy'] == True].copy()
        
        # Tiempos directos en segundos para el hardware real
        for data in [data_selective, data_greedy]:
            if not data.empty:
                data['time_seconds'] = data['tick'] 
                data['w_sec'] = data['planned_wticks'] 
        
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
        print("STRATEGY COMPARISON (REAL ROBOTS)")
        print("="*50)
        for stats in [stats_selective, stats_greedy]:
            if stats:
                print(f"\n--- {stats['strategy']} Strategy ---")
                print(f"Total tasks: {stats['total_entries']}")
                print(f"Unique robots active: {stats['unique_robots']}")
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
                # Formateador estándar sin división por miles
                ax.yaxis.set_major_locator(MaxNLocator(integer=True))
                ax.set_ylabel("Number of tasks completed")
                ax.set_title(title)
                ax.axvline(W_MIN_SEC, linestyle="--", color='red', alpha=0.7)
                ax.axvline(W_STD_SEC, linestyle="--", color='blue', alpha=0.7)
                ax.grid(True, alpha=0.3)
                ax.text(0.02, 0.98, f"Tasks: {len(w_sec)}", transform=ax.transAxes, fontsize=10, va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        fig.suptitle(f"Comparison of Task Completion Times (Real Swarm)", fontsize=16, y=1.02)
        plt.tight_layout()
        if save_dir: plt.savefig(Path(save_dir) / "real_strategy_comparison_histograms.png", dpi=300, bbox_inches='tight')
        plt.show()

    def plot_performance_boxplot(self, window_sec: int = 60, max_time_sec: int = 600, save_path: Optional[str] = None) -> None:
        """Generates a boxplot showing tasks completed per time window for both strategies."""
        if self.raw_data is None or self.raw_data.empty:
            logger.warning("No data available to plot.")
            return

        logger.info("Generating improved performance boxplot")
        df = self.raw_data.copy()
        
        if df['greedy'].dtype == object:
             df['greedy'] = df['greedy'].astype(str).str.lower().map({'true': True, 'false': False})

        df['strategy'] = df['greedy'].map({True: 'greedy', False: 'selective'})
        df['time_sec'] = df['tick']

        df = df[(df['time_sec'] >= 0) & (df['time_sec'] <= max_time_sec)]
        df['time_window'] = (np.ceil(df['time_sec'] / window_sec) * window_sec).astype(int)
        df = df[(df['time_window'] > 0) & (df['time_window'] <= max_time_sec)]

        df['experiment_id'] = df.groupby(['strategy', 'seed']).ngroup()
        perf = df.groupby(['experiment_id', 'strategy', 'time_window']).size().reset_index(name='tasks_completed')

        dummy_data = []
        for exp_id in perf['experiment_id'].unique():
            strat = perf[perf['experiment_id'] == exp_id]['strategy'].iloc[0]
            dummy_data.append({'experiment_id': exp_id, 'strategy': strat, 'time_window': 0, 'tasks_completed': 0})

        perf = pd.concat([perf, pd.DataFrame(dummy_data)], ignore_index=True)
        perf = perf.sort_values(by='time_window')

        fig, ax = plt.subplots(figsize=(10, 10))

        perf['tasks_completed'] = perf['tasks_completed'].astype(int)
        my_palette = {'selective': '#1f77b4', 'greedy': '#ff7f0e'} 

        perf['time_window'] = perf['time_window'].astype(int)

        sns.boxplot(
            data=perf,
            x='time_window',
            y='tasks_completed',
            hue='strategy',
            palette=my_palette,
            width=0.6,
            fliersize=5,
            flierprops={'marker': 'o'},
            ax=ax
        )
        
        labels = [int(t.get_text()) for t in ax.get_xticklabels()]
        # Sin notación x10^3 para hardware real
        new_labels = [f"{int(x)}" if x != 0 else "0" for x in labels]
        ticks = ax.get_xticks()

        ax.set_xticks(ticks)
        ax.set_xticklabels(new_labels)

        for i, box in enumerate(ax.patches):
            box_color = box.get_facecolor()

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

            flier_line_idx = (i * 6) + 5 
            if flier_line_idx < len(ax.lines):
                flier_line = ax.lines[flier_line_idx]
                flier_line.set_markerfacecolor(box_color)  
                flier_line.set_markeredgecolor('black')    
                flier_line.set_alpha(0.7)

        ax.set_xlabel("Time (s)", fontsize=12)
        ax.set_ylabel("Number of tasks completed", fontsize=12)

        ax.grid(True, axis='y', linestyle=':', alpha=0.7)
        ax.legend(title="Strategy", fontsize=11, loc='upper left')
        
        plt.tight_layout()
        if save_path: 
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_search_time_distribution(self, max_time_sec: int = 600, save_path: Optional[str] = None) -> None:
        """Generates a violin plot to show the overall distribution of task search times per strategy, including statistical legends."""
        if self.raw_data is None or self.raw_data.empty:
            logger.warning("No data available to plot.")
            return

        logger.info("Generating search time distribution plot with statistics")
        df = self.raw_data.copy()
        
        if df['greedy'].dtype == object:
            df['greedy'] = df['greedy'].astype(str).str.lower().map({'true': True, 'false': False})

        df['strategy'] = df['greedy'].map({True: 'greedy', False: 'selective'})
        df['time_sec'] = df['tick'] 
        df = df[(df['time_sec'] >= 0) & (df['time_sec'] <= max_time_sec)]

        df['search_time_sec'] = df['search_ticks'] 

        stats = df.groupby('strategy')['search_time_sec'].agg(['mean', 'median', 'std', 'max']).round(2)

        fig, ax = plt.subplots(figsize=(10, 6)) 

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

        ax.set_title("Spent task search time per Strategy", fontsize=14, fontweight='bold', pad=15)
        ax.set_ylabel("Search Time (s)", fontsize=12)
        ax.set_xlabel("Strategy", fontsize=12)

        legend_handles = []
        for strat in ['selective', 'greedy']:
            if strat in stats.index:
                s_mean = stats.loc[strat, 'mean']
                s_med = stats.loc[strat, 'median']
                s_std = stats.loc[strat, 'std']
                s_max = stats.loc[strat, 'max']

                label_text = f"{strat.capitalize()}\nMean: {s_mean}s\nMedian: {s_med}s\nStd: {s_std}s\nMax: {s_max}s"
                patch = Patch(facecolor=my_palette[strat], edgecolor='black', label=label_text)
                legend_handles.append(patch)

        ax.legend(handles=legend_handles, title="Descriptive Statistics", 
                  loc='upper right', fontsize=10, title_fontsize=11)

        ax.grid(True, axis='y', linestyle=':', alpha=0.7)

        plt.tight_layout()
        if save_path: 
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_f_measure_boxplot(self, window_sec: int = 60, max_time_sec: int = 600, save_path: Optional[str] = None) -> None:
        """Generates a boxplot of the F-measure (consistency of task execution) over time."""
        if self.raw_data is None or self.raw_data.empty:
            logger.warning("No data available to plot.")
            return

        logger.info("Generating improved F-measure boxplot")
        df = self.raw_data.copy()
        
        if df['greedy'].dtype == object:
            df['greedy'] = df['greedy'].astype(str).str.lower().map({'true': True, 'false': False})

        df['strategy'] = df['greedy'].map({True: 'greedy', False: 'selective'})
        df['time_sec'] = df['tick']
        df = df[(df['time_sec'] >= 0) & (df['time_sec'] <= max_time_sec)]

        df = df[df['task'].isin(['BLUE', 'RED'])]

        df['time_window'] = (np.ceil(df['time_sec'] / window_sec) * window_sec).astype(int)
        df = df[(df['time_window'] > 0) & (df['time_window'] <= max_time_sec)]

        df['experiment_id'] = df.groupby(['strategy', 'seed']).ngroup()

        df = df.sort_values(by=['experiment_id', 'time_window', 'robot', 'tick'])

        df['prev_task'] = df.groupby(['experiment_id', 'time_window', 'robot'])['task'].shift(1)
        df['is_switch'] = (df['task'] != df['prev_task']) & (df['prev_task'].notnull())

        robot_stats = df.groupby(['experiment_id', 'strategy', 'time_window', 'robot']).agg(
            N=('task', 'count'),
            switches=('is_switch', 'sum')
        ).reset_index()

        robot_stats['f_measure'] = np.where(
            robot_stats['N'] == 1, 
            1.0, 
            1.0 - (2.0 * robot_stats['switches'] / robot_stats['N'])
        )

        f_df = robot_stats.groupby(['experiment_id', 'strategy', 'time_window'])['f_measure'].mean().reset_index()

        fig, ax = plt.subplots(figsize=(10, 10))
        my_palette = {'selective': '#1f77b4', 'greedy': '#ff7f0e'}
        
        time_order = [0] + sorted(f_df['time_window'].unique().tolist())
        f_df['time_window'] = f_df['time_window'].astype(int)
        
        sns.boxplot(
            data=f_df,
            x='time_window',
            y='f_measure',
            hue='strategy',
            hue_order=['selective', 'greedy'], 
            order=time_order,
            palette=my_palette,
            width=0.6,
            fliersize=5,
            flierprops={'marker': 'o'},
            ax=ax
        )

        labels = [int(t.get_text()) for t in ax.get_xticklabels()]
        new_labels = [f"{int(x)}" if x != 0 else "0" for x in labels]   
        ticks = ax.get_xticks()

        ax.set_xticks(ticks)
        ax.set_xticklabels(new_labels)

        for i, box in enumerate(ax.patches):
            box_color = box.get_facecolor()
            
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
            
            flier_line_idx = (i * 6) + 5 
            if flier_line_idx < len(ax.lines):
                flier_line = ax.lines[flier_line_idx]
                flier_line.set_markerfacecolor(box_color)
                flier_line.set_markeredgecolor('black')
                flier_line.set_alpha(0.7)

        ax.set_ylabel("F-measure (Specialization)", fontsize=12)
        ax.set_xlabel("Time (s)", fontsize=12)

        ax.set_ylim(-0.15, 1.15) 
        ax.grid(True, axis='y', linestyle=':', alpha=0.7)

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
        df = df[df['robot'] == 'qupa_7E'] # filtro robot 7E
        
        if 'task' in df.columns:
            df['task'] = df['task'].replace({'TYPE_B': 'BLUE', 'TYPE_R': 'RED'})
        
        if df['greedy'].dtype == object:
            df['greedy'] = df['greedy'].astype(str).str.lower().map({'true': True, 'false': False})
        df['strategy'] = df['greedy'].map({True: 'greedy', False: 'selective'})

        df['experiment_id'] = df.groupby(['strategy', 'seed']).ngroup()

        TASK_TYPES = ['BLUE', 'RED']
        df_tasks = df[df['task'].isin(TASK_TYPES)].copy()
        
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

    def plot_strategy_grid(self, strategy: str, save_path: str) -> dict:
        if self.robot_stats is None:
            print("Please run preprocess_data() first.")
            return {}

        df_strat = self.robot_stats[self.robot_stats['strategy'] == strategy]
        unique_seeds = df_strat['seed'].unique()
        n_experiments = len(unique_seeds)

        if n_experiments == 0:
            print(f"No data found for strategy: {strategy}")
            return {}

        global_max = max(self.robot_stats['BLUE'].max(), self.robot_stats['RED'].max(), 10) * 1.05

        cols = 5
        rows = math.ceil(n_experiments / cols)
        
        fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows), sharex=True, sharey=True)
        fig.patch.set_facecolor('#2b2b2b')  
        axes = np.atleast_1d(axes).flatten()

        best_seed = None
        best_spec_idx = -1.0

        for i, seed in enumerate(unique_seeds):
            ax = axes[i]
            data = df_strat[df_strat['seed'] == seed]
            
            spec_idx = self._calculate_spec_index(data)
            
            if spec_idx > best_spec_idx:
                best_spec_idx = spec_idx
                best_seed = seed

            total = data['BLUE'] + data['RED']
            bias = (data['BLUE'] - data['RED']) / total.replace(0, 1)

            ax.set_facecolor('#2b2b2b')

            ax.scatter(data['BLUE'], data['RED'], 
                       c=bias, cmap='coolwarm', norm=plt.Normalize(vmin=-0.5, vmax=0.5),
                       s=15, alpha=0.8, edgecolors='none')
            
            ax.set_title(f"Seed: {seed}", fontsize=10)
            ax.text(0.05, 0.95, f"Spec: {spec_idx:.2f}", 
                    transform=ax.transAxes, ha='left', va='top', fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="#FFFFFF", alpha=0.8, edgecolor='none'))
            
            ax.title.set_color('white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.tick_params(colors='white')
            ax.plot([0, global_max], [0, global_max], color='white', linestyle='--', alpha=0.3)
            ax.set_xlim(-1, global_max)
            ax.set_ylim(-1, global_max)
            ax.grid(True, linestyle=':', alpha=0.4)

            if i % cols == 0:
                ax.set_ylabel("Red Tasks", fontsize=10)
            if i >= (rows - 1) * cols:
                ax.set_xlabel("Blue Tasks", fontsize=10)

        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        fig.suptitle(f"Individual Experiments - {strategy.capitalize()} Strategy", fontsize=16, fontweight='bold', y=0.98, color='white') 

        cbar_ax = fig.add_axes([0.15, 0.02, 0.7, 0.02])
        cbar = fig.colorbar(plt.cm.ScalarMappable(cmap='RdBu', norm=plt.Normalize(vmin=-1, vmax=1)), 
                            cax=cbar_ax, orientation='horizontal')
        cbar.set_label('Robot Specialization: Red Specialist <--- Generalist ---> Blue Specialist',color='white')
        cbar.ax.tick_params(colors='white')
        plt.subplots_adjust(bottom=0.1, hspace=0.3, wspace=0.1)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

        print(f"[{strategy.capitalize()}] Best performing seed: {best_seed} with Spec Index: {best_spec_idx:.3f}")
        return {'strategy': strategy, 'best_seed': best_seed, 'best_spec_index': best_spec_idx}
        
# ------------------------------------------------------------------
# MAIN EXECUTION
# ------------------------------------------------------------------
def main():
    csv_path = r"/home/gmadro/EXP_CASOS/EXP-REALES-CASOS/CASO1/experiment_data.csv"
    output_dir = r"/home/gmadro/EXP_CASOS/EXP-REALES-CASOS/CASO1/processing_data"
    
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
        print("GENERATING PLOTS & FIGURES (REAL ROBOTS - SINGLE CASE)")
        print("="*50)

        processor.plot_comparison_histograms(save_dir=output_dir)
            
        processor.plot_spatial_heatmap(save_path=f"{output_dir}/real_spatial_heatmap.png")
            
        spec_plotter = SpecializationScatterPlotter(csv_path)
        spec_plotter.load_data()
        spec_plotter.preprocess_data()
        spec_plotter.plot_figure(save_path=f"{output_dir}/real_specialization_scatter.png")

        best_selective = spec_plotter.plot_strategy_grid('selective', save_path=f"{output_dir}/real_figure_grid_selective.png")
        best_greedy = spec_plotter.plot_strategy_grid('greedy', save_path=f"{output_dir}/real_figure_grid_greedy.png")

        print("--- Best experiments seeds ---")
        print("BEST SELECTIVE:", best_selective)
        print("BEST GREEDY:", best_greedy)

        # Ajuste para hardware real
        processor.plot_performance_boxplot(
            window_sec=60,
            max_time_sec=600,
            save_path=f"{output_dir}/real_figure6_performance_boxplot.png"
        )

        processor.plot_search_time_distribution(
            max_time_sec=600,
            save_path=f"{output_dir}/real_figure_search_time.png"
        )

        processor.plot_f_measure_boxplot(
            window_sec=60,
            max_time_sec=600,
            save_path=f"{output_dir}/real_figure_f_measure_boxplot.png"
        )

    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()