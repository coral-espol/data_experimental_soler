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
from typing import Tuple, Optional, Dict, Any, List
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
W_STD_SEC = 60.0         # w_std from paper (for reference axis)
W_MIN_SEC = 7.9          # w_min from paper (for reference axis)
MIN_TIMESTEP = 0         # Adjusted to 0 to capture all new data
MAX_TIMESTEP = 100000   # Maximum timestep to consider

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def thousands_formatter(x, pos):
    """
    Formatter to display Y-axis values in thousands (x10^3).
    Example: 120000 -> 120
    """
    return f"{x/1000:.1f}"

def load_all_cases(base_dir: str) -> pd.DataFrame:
    """Loads experiment_data.csv from all CASOx folders in the base directory."""
    base_path = Path(base_dir)
    case_dirs = sorted([d for d in base_path.iterdir() if d.is_dir() and d.name.startswith("CASO")])
    
    all_data = []
    for c_dir in case_dirs:
        csv_file = c_dir / "experiment_data.csv"
        if csv_file.exists():
            logger.info(f"Loading data for {c_dir.name}...")
            df = pd.read_csv(csv_file)
            
            # Estandarizar la columna greedy inmediatamente al cargar
            if 'greedy' in df.columns and df['greedy'].dtype == object:
                df['greedy'] = df['greedy'].astype(str).str.lower().map({'true': True, 'false': False})
                
            df['caso'] = c_dir.name  # Añadimos la columna del caso
            all_data.append(df)
        else:
            logger.warning(f"File not found: {csv_file}")
            
    if not all_data:
        raise ValueError(f"No experiment_data.csv files found in any CASO directories under {base_dir}")
        
    return pd.concat(all_data, ignore_index=True)


class RobotDataProcessor:
    """Class to process robot simulation experiment data across multiple cases."""
    
    def __init__(self, raw_data: pd.DataFrame, output_dir: str, ticks_per_sec: float = 10.0):
        self.output_dir = Path(output_dir)
        self.ticks_per_sec = ticks_per_sec
        self.raw_data = raw_data
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cases = sorted(self.raw_data['caso'].unique())
        
        # Paleta de colores dinámica dependiendo del número de casos
        palette_colors = sns.color_palette("tab10", len(self.cases))
        self.palette = dict(zip(self.cases, palette_colors))
        
    def preprocess_data(self) -> pd.DataFrame:
        df_clean = self.raw_data.copy()
        
        logger.info("Filtrando datos exclusivamente para el robot 'q0'")
        df_clean = df_clean[df_clean['robot'] == 'q0']
                
        logger.info(f"Filtering data: timestep {MIN_TIMESTEP} to {MAX_TIMESTEP}")
        df_clean = df_clean[(df_clean['tick'] >= MIN_TIMESTEP) & (df_clean['tick'] <= MAX_TIMESTEP)]
        
        df_clean['time_seconds'] = df_clean['tick'] / self.ticks_per_sec
        df_clean['w_sec'] = df_clean['planned_wticks'] / self.ticks_per_sec
        
        self.raw_data = df_clean
        return df_clean
    
    def print_comparison(self):
        print("\n" + "="*50)
        print("CASE COMPARISON")
        print("="*50)
        for caso in self.cases:
            df_case = self.raw_data[self.raw_data['caso'] == caso]
            print(f"\n--- {caso} ---")
            print(f"Total tasks: {len(df_case)}")
            if 'p_x' in df_case.columns:
                print(f"Average p_x: {df_case['p_x'].mean():.3f}")
            if 'm' in df_case.columns:
                print(f"Average m: {df_case['m'].mean():.3f}")
            if 'task' in df_case.columns:
                print(f"Task Split: {df_case['task'].value_counts().to_dict()}")
            if 'w_sec' in df_case.columns:
                print(f"Avg completion time: {df_case['w_sec'].mean():.2f} s")

    # ============================================================================
    # PLOTTING FUNCTIONS
    # ============================================================================

    def plot_comparison_histograms(self, save_dir: Optional[str] = None) -> None:
        if self.raw_data is None or self.raw_data.empty: return
        
        # 12x12 Cuadrado perfecto
        fig, ax = plt.subplots(figsize=(12, 12))
        bins = np.arange(W_MIN_SEC, W_STD_SEC + 6, 6)
        
        sns.histplot(data=self.raw_data, x="w_sec", hue="caso", bins=bins, 
                     palette=self.palette, element="step", fill=True, alpha=0.3, ax=ax, linewidth=2)
        
        # ---> SI NECESITAS LÍMITE Y MANUAL, DESCOMENTA ESTA LÍNEA Y CAMBIA EL VALOR <---
        # ax.set_ylim(0, 1500)
        
        ax.set_xlabel("Task completion time $w_x$ (s)", fontsize=20, fontweight='bold', labelpad=15)
        ax.set_ylabel("Number of tasks completed ($\\times 10^3$)", fontsize=20, fontweight='bold', labelpad=15)
        ax.set_title("Comparison of Task Completion Times Across Cases", fontsize=24, fontweight='bold', pad=20)
        
        ax.yaxis.set_major_formatter(FuncFormatter(thousands_formatter))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        
        ax.tick_params(axis='both', which='major', labelsize=16)

        # Líneas verticales más notorias
        ax.axvline(W_MIN_SEC, linestyle="--", color='red', alpha=0.8, linewidth=2.5, label='$W_{min}$')
        ax.axvline(W_STD_SEC, linestyle="--", color='blue', alpha=0.8, linewidth=2.5, label='$W_{std}$')
        
        ax.grid(True, axis='y', linestyle='--', alpha=0.6)
        
        legend = ax.legend(loc='upper right', fontsize=16, framealpha=0.9, edgecolor='black')

        plt.tight_layout()
        if save_dir: plt.savefig(Path(save_dir) / "cases_comparison_histograms.png", dpi=300, bbox_inches='tight')
        plt.show()

    def plot_performance_trend(self, window_sec: int = 300, max_time_sec: int = 10000, save_path: Optional[str] = None, target_cases: Optional[List[str]] = None) -> None:
        """Generates a continuous line plot showing tasks completed per time window with shaded variance."""
        if self.raw_data is None or self.raw_data.empty: return
        logger.info("Generating performance trend plot for cases")
        
        # ---> FILTRO DE CASOS CONFIGURABLE <---
        base_df = self.raw_data.copy()
        if target_cases is not None:
            base_df = base_df[base_df['caso'].isin(target_cases)]
            
        if base_df.empty:
            logger.warning("No data to plot after filtering by target_cases.")
            return

        df_all = base_df.copy()
        df_all['time_seconds_full'] = df_all['tick'] / self.ticks_per_sec
        df_all['time_window_full'] = (np.ceil(df_all['time_seconds_full'] / window_sec) * window_sec).astype(int)
        global_max_tasks = df_all.groupby(['caso', 'seed', 'time_window_full']).size().max()
        if pd.isna(global_max_tasks): global_max_tasks = 100

        df = base_df.copy()

        df = df[(df['time_seconds'] >= 0) & (df['time_seconds'] <= max_time_sec)]
        df['time_window'] = (np.ceil(df['time_seconds'] / window_sec) * window_sec).astype(int)
        df = df[(df['time_window'] > 0) & (df['time_window'] <= max_time_sec)]

        df['experiment_id'] = df.groupby(['caso', 'seed']).ngroup()
        perf = df.groupby(['experiment_id', 'caso', 'time_window']).size().reset_index(name='tasks_completed')

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

        # 12x12 Cuadrado perfecto
        fig, ax = plt.subplots(figsize=(12, 12))

        perf['tasks_completed'] = perf['tasks_completed'].astype(int)
        perf['time_window'] = perf['time_window'].astype(int)

        sns.lineplot(
            data=perf,
            x='time_window',
            y='tasks_completed',
            hue='caso',
            palette=self.palette, 
            marker='o',
            linewidth=3.5, 
            markersize=10, 
            errorbar='sd',
            ax=ax
        )

        time_windows = sorted(perf['time_window'].unique())
        step = max(1, len(time_windows) // 12) 
        ax.set_xticks(time_windows[::step])
        
        def time_formatter(x, pos):
            return f"{x/1:g}" if x != 0 else "0"
        ax.xaxis.set_major_formatter(FuncFormatter(time_formatter))
        
        ax.set_ylim(0, 8)

        ax.set_title("Tasks completion over time", fontsize=24, fontweight='bold', pad=20)
        ax.set_xlabel("Time (s)", fontsize=25, fontweight='bold', labelpad=15)
        ax.set_ylabel("Number of tasks completed", fontsize=25, fontweight='bold', labelpad=15)
        
        ax.tick_params(axis='both', which='major', labelsize=16)
        
        ax.grid(True, linestyle='--', alpha=0.6)
        
        legend = ax.legend(title="Experimental Case", fontsize=16, loc='upper left', framealpha=0.9, edgecolor='black')
        legend.get_title().set_fontsize(18)
        legend.get_title().set_fontweight('bold')

        # ---> RE-INDEXACIÓN DINÁMICA DE LA LEYENDA <---
        # 1. Obtenemos los casos que realmente se graficaron y los ordenamos (ej: ['CASO1', 'CASO3', 'CASO4'])
        plotted_cases = sorted(perf['caso'].unique())
        
        # 2. Creamos un diccionario de traducción: {'CASO1': 'CASE 1', 'CASO3': 'CASE 2', 'CASO4': 'CASE 3'}
        case_mapping = {caso: f"CASE {i+1}" for i, caso in enumerate(plotted_cases)}

        # 3. Aplicamos la traducción a los textos de la leyenda
        for text in legend.get_texts():
            current_text = text.get_text()
            if current_text in case_mapping:
                text.set_text(case_mapping[current_text])

        plt.tight_layout()
        if save_path: plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_search_time_distribution(self, max_time_sec: int = 10000, save_path: Optional[str] = None) -> None:
        if self.raw_data is None or self.raw_data.empty: return
        logger.info("Generating search time distribution plot per case")
        df = self.raw_data.copy()
        df = df[(df['time_seconds'] >= 0) & (df['time_seconds'] <= max_time_sec)]
        df['search_time_sec'] = df['search_ticks'] / self.ticks_per_sec

        stats = df.groupby('caso')['search_time_sec'].agg(['mean', 'median', 'std', 'max']).round(2)

        # 12x12 Cuadrado perfecto
        fig, ax = plt.subplots(figsize=(12, 12))

        sns.violinplot(
            data=df,
            x='caso',
            y='search_time_sec',
            palette=self.palette,
            inner='quartile', 
            cut=0,            
            linewidth=2.5, # Bordes del violín más notorios
            ax=ax,
            hue='caso', 
            legend=False
        )

        # ---> SI NECESITAS LÍMITE Y MANUAL, DESCOMENTA ESTA LÍNEA Y CAMBIA EL VALOR <---
        # ax.set_ylim(0, 1000)

        ax.set_title("Spent Task Search Time per Case", fontsize=24, fontweight='bold', pad=20)
        ax.set_ylabel("Search Time (s)", fontsize=25, fontweight='bold', labelpad=15)
        ax.set_xlabel("Experimental Case", fontsize=25, fontweight='bold', labelpad=15)
        
        ax.tick_params(axis='both', which='major', labelsize=16)

        legend_handles = []
        for caso in self.cases:
            if caso in stats.index:
                s_mean = stats.loc[caso, 'mean']
                s_med = stats.loc[caso, 'median']
                label_text = f"{caso}\nMean: {s_mean}s\nMedian: {s_med}s"
                patch = Patch(facecolor=self.palette[caso], edgecolor='black', linewidth=1.5, label=label_text)
                legend_handles.append(patch)

        # labelspacing da espacio para las multilíneas
        legend = ax.legend(handles=legend_handles, title="Statistics", loc='upper right', 
                           fontsize=14, labelspacing=1.3, framealpha=0.9, edgecolor='black')
        legend.get_title().set_fontsize(16)
        legend.get_title().set_fontweight('bold')

        ax.grid(True, axis='y', linestyle='--', alpha=0.6)

        plt.tight_layout()
        if save_path: plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_f_measure_boxplot(self, window_sec: int = 300, max_time_sec: int = 10000, save_path: Optional[str] = None) -> None:
        if self.raw_data is None or self.raw_data.empty: return
        logger.info("Generating F-measure boxplot for cases")
        df = self.raw_data.copy()
        df = df[(df['time_seconds'] >= 0) & (df['time_seconds'] <= max_time_sec)]
        df = df[df['task'].isin(['BLUE', 'RED'])]

        df['time_window'] = (np.ceil(df['time_seconds'] / window_sec) * window_sec).astype(int)
        df = df[(df['time_window'] > 0) & (df['time_window'] <= max_time_sec)]

        df['experiment_id'] = df.groupby(['caso', 'seed']).ngroup()
        df = df.sort_values(by=['experiment_id', 'time_window', 'robot', 'tick'])

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

        # 12x12 Cuadrado perfecto
        fig, ax = plt.subplots(figsize=(12, 12))
        time_order = [0] + sorted(f_df['time_window'].unique().tolist())
        
        sns.boxplot(
            data=f_df,
            x='time_window',
            y='f_measure',
            hue='caso',
            order=time_order,
            palette=self.palette,
            width=0.7,
            fliersize=6, # Outliers más grandes
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
                    linestyle='-', linewidth=3.5, markersize=10, zorder=10, # Líneas más gruesas
                    label='_nolegend_')

        # Formatting ticks para que mantengan tamaño 16
        labels = [float(t.get_text()) for t in ax.get_xticklabels()]
        n_ticks = len(labels)
        step = max(1, n_ticks // 15)
        
        new_labels = []
        for i, x in enumerate(labels):
            if i % step == 0:
                new_labels.append(f"{x/1000:g}" if x != 0 else "0")
            else:
                new_labels.append("")

        ax.tick_params(axis='both', which='major', labelsize=16)
        
        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels(new_labels, rotation=45 if n_ticks > 15 else 0, fontsize=16)

        ax.set_title("F-measure (Task Consistency) Over Time", fontsize=24, fontweight='bold', pad=20)
        ax.set_ylabel("F-measure (Specialization)", fontsize=20, fontweight='bold', labelpad=15)
        ax.set_xlabel("Time (s) [×10³]", fontsize=20, fontweight='bold', labelpad=15)
        
        # ---> LÍMITE Y: 1.15 da espacio arriba para la leyenda <---
        ax.set_ylim(-0.15, 1.15) 
        
        ax.grid(True, axis='y', linestyle='--', alpha=0.6)
        
        legend = ax.legend(title="Case", fontsize=16, loc='upper left', framealpha=0.9, edgecolor='black') 
        legend.get_title().set_fontsize(18)
        legend.get_title().set_fontweight('bold')

        plt.tight_layout()
        if save_path: plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

# ============================================================================
# SPECIALIZATION SCATTER PLOT 
# ============================================================================
class SpecializationScatterPlotter:
    def __init__(self, raw_data: pd.DataFrame):
        self.raw_data = raw_data
        self.robot_stats = None
        self.cases = sorted(self.raw_data['caso'].unique())
        self.palette = dict(zip(self.cases, sns.color_palette("tab10", len(self.cases))))

    def preprocess_data(self) -> pd.DataFrame:
        df = self.raw_data.copy()
        df = df[df['robot'] == 'q0']
        
        df['experiment_id'] = df.groupby(['caso', 'seed']).ngroup()

        TASK_TYPES = ['BLUE', 'RED']
        df_tasks = df[df['task'].isin(TASK_TYPES)].copy()
        
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
        
        # 12x12 Cuadrado perfecto
        fig, ax = plt.subplots(figsize=(12, 12))
        
        # DARK MODE
        fig.patch.set_facecolor('#2b2b2b')
        ax.set_facecolor('#2b2b2b')

        ax.tick_params(colors='white', labelsize=16)
        ax.title.set_color('white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')

        max_val = max(self.robot_stats['BLUE'].max(), self.robot_stats['RED'].max(), 10) * 1.1

        for caso in self.cases:
            data = self.robot_stats[self.robot_stats['caso'] == caso]
            if not data.empty:
                ax.scatter(data['BLUE'], data['RED'], 
                           color=self.palette[caso],
                           s=100, alpha=0.8, edgecolors='white', linewidth=0.8, label=caso) # Puntos grandes

        ax.set_title("Specialization Across All Cases", fontsize=24, fontweight='bold', pad=20)
        ax.set_ylabel("Total tasks $\\tau_r$ (Red)", fontsize=20, fontweight='bold', labelpad=15)
        ax.set_xlabel("Total tasks $\\tau_b$ (Blue)", fontsize=20, fontweight='bold', labelpad=15)
        
        ax.plot([0, max_val], [0, max_val], color='white', linestyle='--', alpha=0.6, linewidth=2, label='Equilibrium')
        
        # ---> SI NECESITAS LÍMITES MANUALES, MODIFICA ESTAS DOS LÍNEAS (deben ser iguales) <---
        ax.set_xlim(0, max_val)
        ax.set_ylim(0, max_val)
        
        # ASPECT EQUAL para que la diagonal sea perfecta
        ax.set_aspect('equal', adjustable='box')
        
        ax.grid(True, linestyle=':', alpha=0.3, color='white')
        
        legend = ax.legend(title="Cases", loc="upper left", fontsize=16, facecolor='#2b2b2b', edgecolor='white', labelcolor='white')
        legend.get_title().set_fontsize(18)
        legend.get_title().set_fontweight('bold')
        legend.get_title().set_color('white')

        # IMPORTANTE: facecolor=fig.get_facecolor() para mantener el fondo oscuro al exportar
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.show()

# ------------------------------------------------------------------
# MAIN EXECUTION
# ------------------------------------------------------------------
def main():
    # UPDATE: Directory containing CASO1, CASO2, etc.
    base_dir = r"/home/gmadro/EXP_CASOS"  # Cambia a tu ruta base si es necesario
    
    # PARÁMETRO DE FILTRO DE ESTRATEGIA ('both', 'greedy', 'selective')
    TARGET_STRATEGY = 'both'  # Cambia a 'greedy' o 'both' según lo que quieras analizar
    
    output_dir = f"/home/gmadro/EXP_CASOS/processing_data_{TARGET_STRATEGY}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        raw_data = load_all_cases(base_dir)
        
        logger.info(f"Aplicando filtro de estrategia: {TARGET_STRATEGY.upper()}")
        if TARGET_STRATEGY == 'greedy':
            raw_data = raw_data[raw_data['greedy'] == True]
        elif TARGET_STRATEGY == 'selective':
            raw_data = raw_data[raw_data['greedy'] == False]
        elif TARGET_STRATEGY != 'both':
            logger.warning("Estrategia no reconocida. Usando 'both'.")
            
        if raw_data.empty:
            logger.error("No hay datos disponibles tras aplicar el filtro.")
            return
        
        processor = RobotDataProcessor(raw_data, output_dir)
        processor.preprocess_data()
        processor.print_comparison()
        
        print("\n" + "="*50)
        print(f"GENERATING PLOTS & FIGURES (STRATEGY: {TARGET_STRATEGY.upper()})")
        print("="*50)

        # 1. Histogramas
        processor.plot_comparison_histograms(save_dir=output_dir)
            
        # 2. Curva de Tendencia de Rendimiento (Líneas continuas)
        processor.plot_performance_trend(
            window_sec=60,  
            max_time_sec=600,
            target_cases=['CASO1', 'CASO3', 'CASO4'],  # None para incluir todos los casos
            save_path=f"{output_dir}/figure6_performance_trend_cases.png"
        )

        # 3. Violin plot
        processor.plot_search_time_distribution(
            max_time_sec=600,
            save_path=f"{output_dir}/figure_search_time_cases.png"
        )

        # 4. F-measure Boxplot
        processor.plot_f_measure_boxplot(
            window_sec=60,  
            max_time_sec=600,
            save_path=f"{output_dir}/figure_f_measure_boxplot_cases.png"
        )
        
        # 5. Scatter Plot general unificado
        spec_plotter = SpecializationScatterPlotter(raw_data)
        spec_plotter.preprocess_data()
        spec_plotter.plot_figure(save_path=f"{output_dir}/specialization_scatter_cases.png")

    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()