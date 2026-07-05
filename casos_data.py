"""
Script de Análisis Comparativo Global - SOLER PROJECT
Autor: Gabriel Madroñero

Este script itera sobre múltiples directorios (CASO1 a CASO6), extrae los datos 
exclusivamente del robot 'q0' y genera gráficas de líneas comparativas para 
evaluar el rendimiento y el F-measure de las distintas configuraciones.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
import logging
import matplotlib.ticker as ticker

# Configuración básica de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# === Parámetros Globales ===
T_TICKS_PER_SEC = 10.0
WINDOW_SEC = 1000
MAX_TIME_SEC = 10000

# Ruta base donde se encuentran las carpetas CASO1, CASO2, ..., CASO6
BASE_DIR = Path(r"/home/gmadro/EXP_CASOS")
# Carpeta donde se guardarán las gráficas comparativas
OUTPUT_DIR = BASE_DIR/"COMPARATIVA_GLOBAL"

def load_and_merge_data() -> pd.DataFrame:
    """Recorre las carpetas de los casos, lee los CSV y los combina."""
    all_data = []
    
    for i in range(1, 5):
        caso_name = f"CASO{i}"
        csv_path = BASE_DIR / caso_name / "experiment_data.csv"
        
        if csv_path.exists():
            logger.info(f"Cargando datos de: {caso_name}")
            try:
                df = pd.read_csv(csv_path)
                
                # ---> FILTRO ESTRICTO: Solo robot q0 <---
                df = df[df['robot'] == 'q0']
                
                # Estandarizar la columna strategy
                if df['greedy'].dtype == object:
                    df['greedy'] = df['greedy'].astype(str).str.lower().map({'true': True, 'false': False})
                df['strategy'] = df['greedy'].map({True: 'greedy', False: 'selective'})
                
                # Calcular el tiempo en segundos
                df['time_sec'] = df['tick'] / T_TICKS_PER_SEC
                
                # Etiquetar con el caso correspondiente
                df['caso'] = caso_name
                
                all_data.append(df)
            except Exception as e:
                logger.error(f"Error al leer {csv_path}: {e}")
        else:
            logger.warning(f"No se encontró el archivo: {csv_path}")
            
    if not all_data:
        raise ValueError("No se encontraron datos en ninguna de las carpetas.")
        
    master_df = pd.concat(all_data, ignore_index=True)
    return master_df

def plot_comparative_performance(df: pd.DataFrame, save_path: str, window_sec: int = 1000, max_time_sec: int = 10000):
    logger.info("Generando gráfica comparativa de Performance...")
    
    dff = df[(df['time_sec'] >= 0) & (df['time_sec'] <= max_time_sec)].copy()
    dff['time_window'] = (np.ceil(dff['time_sec'] / window_sec) * window_sec).astype(int)
    dff = dff[(dff['time_window'] > 0) & (dff['time_window'] <= max_time_sec)]
    
    perf = dff.groupby(['caso', 'strategy', 'seed', 'time_window']).size().reset_index(name='tasks_in_window')
    perf['tasks_completed'] = perf.groupby(['caso', 'strategy', 'seed'])['tasks_in_window'].cumsum()
    
    dummy_data = []
    for caso in perf['caso'].unique():
        for strat in perf[perf['caso'] == caso]['strategy'].unique():
            for seed in perf[(perf['caso'] == caso) & (perf['strategy'] == strat)]['seed'].unique():
                dummy_data.append({'caso': caso, 'strategy': strat, 'seed': seed, 'time_window': 0, 'tasks_completed': 0})
                
    perf = pd.concat([perf, pd.DataFrame(dummy_data)], ignore_index=True)
    perf = perf.sort_values(by=['caso', 'strategy', 'seed', 'time_window'])
    
    # 24x12 para mantener la proporción cuadrada 12x12 por subplot
    fig, axes = plt.subplots(1, 2, figsize=(24, 12), sharey=True)
    casos_order = sorted(perf['caso'].unique())
    palette = sns.color_palette("tab10", n_colors=len(casos_order))
    
    for i, strat in enumerate(['selective', 'greedy']):
        ax = axes[i]
        strat_data = perf[perf['strategy'] == strat]
        
        if not strat_data.empty:
            sns.lineplot(
                data=strat_data, x='time_window', y='tasks_completed',
                hue='caso', hue_order=casos_order, palette=palette,
                marker='o', markersize=12, linewidth=3.5, estimator=np.median, # Aumentado grosor y marcadores
                errorbar=None, ax=ax
            )
            ax.set_xlim(0, max_time_sec)
            
            # ---> SI NECESITAS LÍMITE Y MANUAL, DESCOMENTA Y EDITA ESTA LÍNEA <---
            # ax.set_ylim(0, 1500)
            
            # Jerarquía de títulos grandes
            ax.set_title(f"Performance (Acumulado) - {strat.capitalize()} Strategy", fontsize=24, fontweight='bold', pad=20)
            ax.set_xlabel("Time (s) [×10³]", fontsize=20, fontweight='bold', labelpad=15)
            
            # Ejes en notación científica para el eje x
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{x/1000:g}"))
            
            # Aumentar tamaño de números en los ejes
            ax.tick_params(axis='both', which='major', labelsize=16)
            
            ax.grid(True, linestyle='--', alpha=0.6)
            if i == 0:
                ax.set_ylabel("Total Tasks Completed", fontsize=20, fontweight='bold', labelpad=15)
            else:
                ax.set_ylabel("")
                
            # Leyenda robusta
            legend = ax.legend(title="Configuración", loc='upper left', fontsize=16, framealpha=0.9, edgecolor='black')
            legend.get_title().set_fontsize(18)
            legend.get_title().set_fontweight('bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_comparative_f_measure(df: pd.DataFrame, save_path: str, window_sec: int = 1000, max_time_sec: int = 10000):
    logger.info("Generando gráfica comparativa de F-Measure...")
    
    dff = df[(df['time_sec'] >= 0) & (df['time_sec'] <= max_time_sec)].copy()
    dff = dff[dff['task'].isin(['BLUE', 'RED'])]
    dff['time_window'] = (np.ceil(dff['time_sec'] / window_sec) * window_sec).astype(int)
    dff = dff[(dff['time_window'] > 0) & (dff['time_window'] <= max_time_sec)]
    
    dff = dff.sort_values(by=['caso', 'strategy', 'seed', 'time_window', 'tick'])
    dff['prev_task'] = dff.groupby(['caso', 'strategy', 'seed', 'time_window'])['task'].shift(1)
    dff['is_switch'] = (dff['task'] != dff['prev_task']) & (dff['prev_task'].notnull())
    
    stats = dff.groupby(['caso', 'strategy', 'seed', 'time_window']).agg(
        N=('task', 'count'), switches=('is_switch', 'sum')
    ).reset_index()
    
    stats['f_measure'] = np.where(stats['N'] <= 1, 1.0, 1.0 - (2.0 * stats['switches'] / stats['N']))
    
    # 24x12
    fig, axes = plt.subplots(1, 2, figsize=(24, 12), sharey=True)
    casos_order = sorted(stats['caso'].unique())
    palette = sns.color_palette("tab10", n_colors=len(casos_order))
    
    for i, strat in enumerate(['selective', 'greedy']):
        ax = axes[i]
        strat_data = stats[stats['strategy'] == strat]
        
        if not strat_data.empty:
            sns.lineplot(
                data=strat_data, x='time_window', y='f_measure',
                hue='caso', hue_order=casos_order, palette=palette,
                marker='s', markersize=12, linewidth=3.5, estimator=np.median,
                errorbar=None, ax=ax
            )
            ax.set_xlim(0, max_time_sec)
            
            # Límite con espacio adicional (1.15) arriba para que la leyenda grande encaje bien
            ax.set_ylim(-0.1, 1.15)
            
            ax.set_title(f"F-Measure (Specialization) - {strat.capitalize()} Strategy", fontsize=24, fontweight='bold', pad=20)
            ax.set_xlabel("Time (s) [×10³]", fontsize=20, fontweight='bold', labelpad=15)
            
            # Ejes en notación científica
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{x/1000:g}"))
            ax.tick_params(axis='both', which='major', labelsize=16)
            
            ax.grid(True, linestyle='--', alpha=0.6)
            if i == 0:
                ax.set_ylabel("F-Measure (Median)", fontsize=20, fontweight='bold', labelpad=15)
            else:
                ax.set_ylabel("")
                
            # Leyenda abajo a la derecha
            legend = ax.legend(title="Configuración", loc='lower right', fontsize=16, framealpha=0.9, edgecolor='black')
            legend.get_title().set_fontsize(18)
            legend.get_title().set_fontweight('bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_comparative_m_evolution(df: pd.DataFrame, save_path: str, window_sec: int = 1000, max_time_sec: int = 10000):
    logger.info("Generando gráfica comparativa de la variable 'm'...")
    
    dff = df[(df['time_sec'] >= 0) & (df['time_sec'] <= max_time_sec)].copy()
    
    # 24x12
    fig, axes = plt.subplots(1, 2, figsize=(24, 12), sharey=True)
    casos_order = sorted(dff['caso'].unique())
    palette = sns.color_palette("tab10", n_colors=len(casos_order))
    
    for i, strat in enumerate(['selective', 'greedy']):
        ax = axes[i]
        strat_data = dff[dff['strategy'] == strat]
        
        if not strat_data.empty:
            sns.lineplot(
                data=strat_data, x='time_sec', y='m',
                hue='caso', hue_order=casos_order, palette=palette,
                marker='o', markersize=8, linewidth=3.5, estimator=np.mean, # markersize=8 aquí porque m puede tener muchos puntos
                errorbar=None, ax=ax
            )
            ax.set_xlim(0, max_time_sec)
            
            # ---> SI NECESITAS LÍMITE Y MANUAL, DESCOMENTA Y EDITA ESTA LÍNEA <---
            # ax.set_ylim(-10, 10) 
            
            ax.set_title(f"Evolución de Memoria ('m') - {strat.capitalize()} Strategy", fontsize=24, fontweight='bold', pad=20)
            ax.set_xlabel("Time (s) [×10³]", fontsize=20, fontweight='bold', labelpad=15)
            
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{x/1000:g}"))
            ax.tick_params(axis='both', which='major', labelsize=16)
            
            # Línea horizontal en 0 resaltada
            ax.axhline(0, color='gray', linestyle='--', linewidth=2, alpha=0.8)
            ax.grid(True, linestyle='--', alpha=0.6)
            
            if i == 0:
                ax.set_ylabel("Valor de Memoria 'm' (Promedio)", fontsize=20, fontweight='bold', labelpad=15)
            else:
                ax.set_ylabel("")
                
            legend = ax.legend(title="Configuración", loc='best', fontsize=16, framealpha=0.9, edgecolor='black')
            legend.get_title().set_fontsize(18)
            legend.get_title().set_fontweight('bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Definir el tiempo máximo para las gráficas comparativas
    WINDOW_SEC = 60
    MAX_TIME_SEC = 600    

    try:
        # Cargar todos los datos
        master_df = load_and_merge_data()
        logger.info(f"Datos combinados exitosamente. Total de registros (q0): {len(master_df)}")
        
        # Generar Gráficas Comparativas
        plot_comparative_performance(
            master_df, 
            save_path=str(OUTPUT_DIR / "comparativa_performance_lineas.png"),
            window_sec=WINDOW_SEC,
            max_time_sec=MAX_TIME_SEC
        )
        
        plot_comparative_f_measure(
            master_df, 
            save_path=str(OUTPUT_DIR / "comparativa_fmeasure_lineas.png"),
            window_sec=WINDOW_SEC,
            max_time_sec=MAX_TIME_SEC
        )
        
        plot_comparative_m_evolution(
            master_df,
            save_path=str(OUTPUT_DIR / "comparativa_m_evolution_lineas.png"),
            window_sec=WINDOW_SEC,
            max_time_sec=MAX_TIME_SEC
        )
        
        logger.info(f"¡Proceso finalizado! Las gráficas están en: {OUTPUT_DIR}")
        
    except Exception as e:
        logger.error(f"Error en la ejecución principal: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()