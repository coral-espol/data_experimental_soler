"""
This script processes data collected from REAL QUPA robot experiments (SOLE-R).
It contains functions to clean, transform, and prepare data for subsequent analysis.
Time variables are assumed to be natively logged in seconds.
Author: Gabriel Madroñero
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List
import logging
from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import MaxNLocator
import sys
from scipy import stats
import seaborn as sns
import math
from matplotlib.patches import Patch

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# === Global parameters for Real Robots ===
W_STD_SEC = 60.0         # w_std from methodology
W_MIN_SEC = 7.9          # w_min from methodology
MIN_TIMESTEP = 0         
MAX_TIMESTEP = 600       # Real experiment max duration

def thousands_formatter(x, pos):
    return f"{x/1000:.1f}"

def load_all_cases(base_dir: str) -> pd.DataFrame:
    """Loads experiment_data.csv from all CASOx folders in the base directory."""
    base_path = Path(base_dir)
    case_dirs = sorted([d for d in base_path.iterdir() if d.is_dir() and d.name.startswith("CASO")])
    
    all_data = []
    for c_dir in case_dirs:
        csv_file = c_dir / "experiment_data.csv"
        if csv_file.exists():
            logger.info(f"Loading real robot data for {c_dir.name}...")
            df = pd.read_csv(csv_file)
            
            # Standardize greedy column
            if 'greedy' in df.columns and df['greedy'].dtype == object:
                df['greedy'] = df['greedy'].astype(str).str.lower().map({'true': True, 'false': False})
                
            df['caso'] = c_dir.name
            all_data.append(df)
        else:
            logger.warning(f"File not found: {csv_file}")
            
    if not all_data:
        raise ValueError(f"No experiment_data.csv files found in any CASO directories under {base_dir}")
        
    return pd.concat(all_data, ignore_index=True)


class RealRobotDataProcessor:
    """Class to process real robot experiment data across multiple cases."""
    
    def __init__(self, raw_data: pd.DataFrame, output_dir: str):
        self.output_dir = Path(output_dir)
        self.raw_data = raw_data
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cases = sorted(self.raw_data['caso'].unique())
        
        palette_colors = sns.color_palette("tab10", len(self.cases))
        self.palette = dict(zip(self.cases, palette_colors))
        
    def preprocess_data(self) -> pd.DataFrame:
        df_clean = self.raw_data.copy()

        logger.info("Filtrando datos exclusivamente para el robot 'qupa_7E'")
        df_clean = df_clean[df_clean['robot'] == 'qupa_7E']
        
        # Mapeo estandarizado de nomenclatura física a lógica de gráficas
        if 'task' in df_clean.columns:
            df_clean['task'] = df_clean['task'].replace({'TYPE_B': 'BLUE', 'TYPE_R': 'RED'})
                
        logger.info(f"Filtering data: timestep {MIN_TIMESTEP} to {MAX_TIMESTEP} seconds")
        df_clean = df_clean[(df_clean['tick'] >= MIN_TIMESTEP) & (df_clean['tick'] <= MAX_TIMESTEP)]
        
        # Asignación directa (Los logs de los robots ya están en segundos)
        df_clean['time_seconds'] = df_clean['tick']
        df_clean['w_sec'] = df_clean['planned_wticks']
        
        self.raw_data = df_clean
        return df_clean
    
    def print_comparison(self):
        print("\n" + "="*50)
        print("REAL ROBOTS CASE COMPARISON")
        print("="*50)
        for caso in self.cases:
            df_case = self.raw_data[self.raw_data['caso'] == caso]
            print(f"\n--- {caso} ---")
            print(f"Total tasks: {len(df_case)}")
            print(f"Active robots: {df_case['robot'].nunique()}")
            if 'p_x' in df_case.columns:
                print(f"Average p_x: {df_case['p_x'].mean():.3f}")
            if 'task' in df_case.columns:
                print(f"Task Split: {df_case['task'].value_counts().to_dict()}")
            if 'w_sec' in df_case.columns:
                print(f"Avg completion time: {df_case['w_sec'].mean():.2f} s")

    # ============================================================================
    # PLOTTING FUNCTIONS
    # ============================================================================

    def plot_comparison_histograms(self, save_dir: Optional[str] = None) -> None:
        if self.raw_data is None or self.raw_data.empty: return
        fig, ax = plt.subplots(figsize=(10, 6))
        bins = np.arange(W_MIN_SEC, W_STD_SEC + 6, 6)
        
        sns.histplot(data=self.raw_data, x="w_sec", hue="caso", bins=bins, 
                     palette=self.palette, element="step", fill=True, alpha=0.3, ax=ax)
        
        ax.set_xlabel("Task completion time $w_x$ (s)")
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.set_ylabel("Number of tasks completed")
        ax.set_title("Comparison of Task Completion Times (Real Swarm)")
        ax.axvline(W_MIN_SEC, linestyle="--", color='red', alpha=0.7)
        ax.axvline(W_STD_SEC, linestyle="--", color='blue', alpha=0.7)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_dir: plt.savefig(Path(save_dir) / "real_cases_comparison_histograms.png", dpi=300, bbox_inches='tight')
        plt.show()

    def plot_performance_trend(self, window_sec: int = 60, max_time_sec: int = 600, save_path: Optional[str] = None) -> None:
        if self.raw_data is None or self.raw_data.empty: return
        logger.info("Generating performance trend plot for real cases")
        
        df_all = self.raw_data.copy()
        df_all['time_window_full'] = (np.ceil(df_all['time_seconds'] / window_sec) * window_sec).astype(int)
        global_max_tasks = df_all.groupby(['caso', 'seed', 'time_window_full']).size().max()
        if pd.isna(global_max_tasks): global_max_tasks = 100

        df = self.raw_data.copy()
        df = df[(df['time_seconds'] >= 0) & (df['time_seconds'] <= max_time_sec)]
        df['time_window'] = (np.ceil(df['time_seconds'] / window_sec) * window_sec).astype(int)
        df = df[(df['time_window'] > 0) & (df['time_window'] <= max_time_sec)]

        df['experiment_id'] = df.groupby(['caso', 'seed']).ngroup()
        perf = df.groupby(['experiment_id', 'caso', 'time_window']).size().reset_index(name='tasks_completed')

        # Matriz de tiempo completo para rellenar huecos en los datos físicos
        all_windows = np.arange(window_sec, max_time_sec + window_sec, window_sec)
        experiments = df[['experiment_id', 'caso']].drop_duplicates()
        
        grid = pd.MultiIndex.from_product(
            [experiments['experiment_id'], all_windows],
            names=['experiment_id', 'time_window']
        ).to_frame(index=False)
        
        grid = grid.merge(experiments, on='experiment_id')
        perf = grid.merge(perf, on=['experiment_id', 'caso', 'time_window'], how='left')
        perf['tasks_completed'] = perf['tasks_completed'].fillna(0)

        dummy_data = []
        for exp_id in perf['experiment_id'].unique():
            caso = perf[perf['experiment_id'] == exp_id]['caso'].iloc[0]
            dummy_data.append({'experiment_id': exp_id, 'caso': caso, 'time_window': 0, 'tasks_completed': 0})

        perf = pd.concat([perf, pd.DataFrame(dummy_data)], ignore_index=True)
        perf = perf.sort_values(by='time_window')

        fig, ax = plt.subplots(figsize=(12, 8))

        perf['tasks_completed'] = perf['tasks_completed'].astype(int)
        perf['time_window'] = perf['time_window'].astype(int)

        sns.lineplot(
            data=perf,
            x='time_window',
            y='tasks_completed',
            hue='caso',
            palette=self.palette,
            marker='o',
            linewidth=2.5,
            markersize=6,
            errorbar='sd',
            ax=ax
        )

        time_windows = sorted(perf['time_window'].unique())
        step = max(1, len(time_windows) // 12) 
        ax.set_xticks(time_windows[::step])

        ax.set_ylim(0, global_max_tasks * 1.1)
        ax.set_xlabel("Time (s)", fontsize=12) # Escala directa en segundos
        ax.set_ylabel("Number of tasks completed", fontsize=12)
        ax.grid(True, linestyle=':', alpha=0.7)
        ax.legend(title="Experimental Case", fontsize=11, loc='upper left')

        plt.tight_layout()
        if save_path: plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_search_time_distribution(self, max_time_sec: int = 600, save_path: Optional[str] = None) -> None:
        if self.raw_data is None or self.raw_data.empty: return
        logger.info("Generating search time distribution plot per case")
        df = self.raw_data.copy()
        df = df[(df['time_seconds'] >= 0) & (df['time_seconds'] <= max_time_sec)]
        
        # Asignación directa en segundos
        df['search_time_sec'] = df['search_ticks'] 

        stats = df.groupby('caso')['search_time_sec'].agg(['mean', 'median', 'std', 'max']).round(2)

        fig, ax = plt.subplots(figsize=(12, 6))

        sns.violinplot(
            data=df,
            x='caso',
            y='search_time_sec',
            palette=self.palette,
            inner='quartile', 
            cut=0,            
            linewidth=1.2,
            ax=ax,
            hue='caso', 
            legend=False
        )

        ax.set_title("Spent task search time per Case (Real Swarm)", fontsize=14, fontweight='bold', pad=15)
        ax.set_ylabel("Search Time (s)", fontsize=12)
        ax.set_xlabel("Case", fontsize=12)

        legend_handles = []
        for caso in self.cases:
            if caso in stats.index:
                s_mean = stats.loc[caso, 'mean']
                s_med = stats.loc[caso, 'median']
                label_text = f"{caso}\nMean: {s_mean}s\nMedian: {s_med}s"
                patch = Patch(facecolor=self.palette[caso], edgecolor='black', label=label_text)
                legend_handles.append(patch)

        ax.legend(handles=legend_handles, title="Statistics", loc='upper right', fontsize=10)
        ax.grid(True, axis='y', linestyle=':', alpha=0.7)

        plt.tight_layout()
        if save_path: plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_f_measure_boxplot(self, window_sec: int = 60, max_time_sec: int = 600, save_path: Optional[str] = None) -> None:
        if self.raw_data is None or self.raw_data.empty: return
        logger.info("Generating F-measure boxplot for real cases")
        df = self.raw_data.copy()
        df = df[(df['time_seconds'] >= 0) & (df['time_seconds'] <= max_time_sec)]
        df = df[df['task'].isin(['BLUE', 'RED'])]

        df['time_window'] = (np.ceil(df['time_seconds'] / window_sec) * window_sec).astype(int)
        df = df[(df['time_window'] > 0) & (df['time_window'] <= max_time_sec)]

        df['experiment_id'] = df.groupby(['caso', 'seed']).ngroup()
        df = df.sort_values(by=['experiment_id', 'time_window', 'robot', 'tick'])

        # Lógica F-measure calculada individualmente por cada robot en el enjambre
        df['prev_task'] = df.groupby(['experiment_id', 'time_window', 'robot'])['task'].shift(1)
        df['is_switch'] = (df['task'] != df['prev_task']) & (df['prev_task'].notnull())

        robot_stats = df.groupby(['experiment_id', 'caso', 'time_window', 'robot']).agg(
            N=('task', 'count'),
            switches=('is_switch', 'sum')
        ).reset_index()

        robot_stats['f_measure'] = np.where(
            robot_stats['N'] == 1, 
            1.0, 
            1.0 - (2.0 * robot_stats['switches'] / robot_stats['N'])
        )

        f_df = robot_stats.groupby(['experiment_id', 'caso', 'time_window'])['f_measure'].mean().reset_index()

        fig, ax = plt.subplots(figsize=(12, 8))
        time_order = [0] + sorted(f_df['time_window'].unique().tolist())
        
        sns.boxplot(
            data=f_df,
            x='time_window',
            y='f_measure',
            hue='caso',
            order=time_order,
            palette=self.palette,
            width=0.7,
            fliersize=4,
            ax=ax
        )

        medians = f_df.groupby(['caso', 'time_window'])['f_measure'].median().reset_index()
        time_windows = sorted(time_order)
        tw_to_idx = {tw: i for i, tw in enumerate(time_windows)}
        
        for caso in self.cases:
            caso_data = medians[medians['caso'] == caso].sort_values('time_window')
            x_vals = [tw_to_idx[tw] for tw in caso_data['time_window']]
            y_vals = caso_data['f_measure']
            ax.plot(x_vals, y_vals, color=self.palette[caso], marker='o', 
                    linestyle='-', linewidth=2.5, markersize=6, zorder=10, 
                    label='_nolegend_')

        labels = [float(t.get_text()) for t in ax.get_xticklabels()]
        n_ticks = len(labels)
        step = max(1, n_ticks // 15)
        
        new_labels = []
        for i, x in enumerate(labels):
            if i % step == 0:
                new_labels.append(f"{x:g}" if x != 0 else "0") # Sin notación x10^3
            else:
                new_labels.append("")

        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels(new_labels, rotation=45 if n_ticks > 15 else 0)

        ax.set_ylabel("F-measure (Specialization)", fontsize=12)
        ax.set_xlabel("Time (s)", fontsize=12)
        ax.set_ylim(-0.15, 1.15) 
        ax.grid(True, axis='y', linestyle=':', alpha=0.7)
        ax.legend(title="Case", fontsize=11, loc='upper left') 

        plt.tight_layout()
        if save_path: plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


class RealSpecializationScatterPlotter:
    def __init__(self, raw_data: pd.DataFrame):
        self.raw_data = raw_data
        self.robot_stats = None
        self.cases = sorted(self.raw_data['caso'].unique())
        self.palette = dict(zip(self.cases, sns.color_palette("tab10", len(self.cases))))

    def preprocess_data(self) -> pd.DataFrame:
        df = self.raw_data.copy()
        df = df[df['robot'] == 'qupa_7E']        
        if 'task' in df.columns:
            df['task'] = df['task'].replace({'TYPE_B': 'BLUE', 'TYPE_R': 'RED'})
            
        df['experiment_id'] = df.groupby(['caso', 'seed']).ngroup()

        TASK_TYPES = ['BLUE', 'RED']
        df_tasks = df[df['task'].isin(TASK_TYPES)].copy()
        
        # Agrupa la ejecución de tareas de todos los robots por experimento
        self.robot_stats = (df_tasks.groupby(['experiment_id', 'caso', 'seed', 'robot', 'task'])
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
        
        fig, ax = plt.subplots(figsize=(8, 8))
        fig.patch.set_facecolor('#2b2b2b')
        ax.set_facecolor('#2b2b2b')

        ax.tick_params(colors='white')
        ax.title.set_color('white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')

        max_val = max(self.robot_stats['BLUE'].max(), self.robot_stats['RED'].max(), 10) * 1.1

        for caso in self.cases:
            data = self.robot_stats[self.robot_stats['caso'] == caso]
            if not data.empty:
                ax.scatter(data['BLUE'], data['RED'], 
                           color=self.palette[caso],
                           s=40, alpha=0.8, edgecolors='white', linewidth=0.5, label=caso)

        ax.set_title("Specialization Across Real Cases", fontsize=14, fontweight='bold')
        ax.set_ylabel("Total tasks $\\tau_r$ (Red)", fontsize=12)
        ax.set_xlabel("Total tasks $\\tau_b$ (Blue)", fontsize=12)
        
        ax.plot([0, max_val], [0, max_val], color='white', linestyle='--', alpha=0.5, label='Equilibrium')
        ax.grid(True, linestyle=':', alpha=0.4, color='white')
        ax.legend(title="Cases", loc="upper left")

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

# ------------------------------------------------------------------
# MAIN EXECUTION
# ------------------------------------------------------------------
def main():
    base_dir = r"/home/gmadro/EXP_CASOS/EXP-REALES-CASOS" # Directorio sugerido para aislar datos físicos
    TARGET_STRATEGY = 'selective'  # Cambia a 'both' (ambas estrategias) 'greedy' o 'selective' según el análisis deseado
    
    output_dir = f"/home/gmadro/EXP_CASOS/EXP-REALES-CASOS/processing_data_{TARGET_STRATEGY}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        raw_data = load_all_cases(base_dir)
        
        logger.info(f"Applying strategy filter: {TARGET_STRATEGY.upper()}")
        if TARGET_STRATEGY == 'greedy':
            raw_data = raw_data[raw_data['greedy'] == True]
        elif TARGET_STRATEGY == 'selective':
            raw_data = raw_data[raw_data['greedy'] == False]
            
        if raw_data.empty:
            logger.error("No data available after strategy filter.")
            return
        
        processor = RealRobotDataProcessor(raw_data, output_dir)
        processor.preprocess_data()
        processor.print_comparison()
        
        print("\n" + "="*50)
        print(f"GENERATING PLOTS & FIGURES (REAL SWARM - STRATEGY: {TARGET_STRATEGY.upper()})")
        print("="*50)

        processor.plot_comparison_histograms(save_dir=output_dir)
            
        # Ventanas de 60s hasta un máximo de 600s, ideal para el hardware real
        processor.plot_performance_trend(
            window_sec=60,
            max_time_sec=600,
            save_path=f"{output_dir}/real_figure6_performance_trend.png"
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
        
        spec_plotter = RealSpecializationScatterPlotter(raw_data)
        spec_plotter.preprocess_data()
        spec_plotter.plot_figure(save_path=f"{output_dir}/real_specialization_scatter.png")

    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()