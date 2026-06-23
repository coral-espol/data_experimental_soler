"""
This script allows for automated execution of experiments
Please set up all variables before running
SOLER PROJECT.
Autor: Gabriel Madroñero 
"""
import os
import subprocess
import time 
import xml.etree.ElementTree as ET 
import re
import pyfiglet as pf
import random

"""CONFIGURATION VARIABLES FOR EXPERIMENTS"""
# Configure the variables for the experiments before running the script
# ===== Number of experiments to run for each strategy
num_experiments = 10 # 
# ===== Variable to configure if the arena configuration script should be executed before each experiment or not
arena_conf = True # true to execute arena configuration before each experiment, false to skip it
# ===== Types of strategies to test in the experiments, can be "greedy" or "selective"
use_strategy = ["selective","greedy"] 
# ==== Configure shape, pattern and task size for the arena configuration
geo_shape = "circle" # shape of the arena, can be "circle" or "square"
geo_pattern = "midle" # pattern of the arena, can be "checkerboard", "random", "experimental", "midle"-> recommended use experimental
geo_task_size = "0.15" # task size for the arena, can be "0.35", "0.5" or "0.7" recommended to use 0.12 for circle and square
# ===== Directory paths for control script and arena configuration script
lua_control_strategy_path = r"/home/gmadro/soler_experiment/control/control_soft.lua"
# ===== Directory path for the arena configuration script to execute before each experiment
arena_config_script_path = r"/home/gmadro/soler_experiment/arena_exp.argos"
# ===== Filename where the loop functions store the experiment results
output_data_file = "experiment_data.csv"
# ===== Filename for the seeds log
seeds_log_file = "seeds_summary.txt"
# ===== Use previous seeds or generate new ones
use_previous_seeds = True  # True = use the current file, False = generate news seed

"""CONFIGURATION FUNCTIONS FOR EXPERIMENTS"""
# Function to delete previous data file to ensure a fresh start
def delete_previous_data(file_path):
    """
    Check if the data file exists and delete it before starting a new batch of experiments
    """
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
            print(f"OK: Previous data file '{file_path}' deleted successfully")
        except Exception as e:
            print(f"X: Error deleting previous data file: {e}")
    else:
        print(f"INFO: No previous data file found at '{file_path}'. Starting fresh.")

# Function to update the Lua variable in the control script based on the selected strategy
def update_lua_variable(new_value):
    
    # Convert Python bool to Lua bool
    lua_value = "true" if new_value == "greedy" else "false"
    variable_name = "GREEDY_MODE"
    
    try:
        # Read the Lua file
        with open(lua_control_strategy_path, 'r') as file:
            content = file.read()
        
        # Multiple patterns to handle different formatting styles
        patterns = [
            rf'({re.escape(variable_name)}\s*=\s*)(true|false)',  # Standard assignment
            rf'({re.escape(variable_name)}\s*=\s*)(true|false)\s*',  # With trailing space
            rf'({re.escape(variable_name)}\s*=\s*)(true|false)\s*;',  # With semicolon
            rf'({re.escape(variable_name)}\s*=\s*)(true|false)\s*$',  # At end of line
            rf'(\b{re.escape(variable_name)}\s*=\s*)(true|false)\b',  # Word boundaries
        ]
        
        updated = False
        new_content = content
        
        for pattern in patterns:
            if re.search(pattern, new_content, re.IGNORECASE):
                new_content = re.sub(pattern, rf'\1{lua_value}', new_content, flags=re.IGNORECASE)
                updated = True
                break
        
        if not updated:
            print("Warning: Variable not found with standard patterns. Trying manual search...")
            # Manual search and replace as fallback
            lines = content.split('\n')
            new_lines = []
            for line in lines:
                if variable_name in line and ('true' in line.lower() or 'false' in line.lower()):
                    print(f"Found potential match: {line}")
                    if 'true' in line.lower():
                        new_line = line.replace('true', lua_value).replace('TRUE', lua_value.upper())
                    else:
                        new_line = line.replace('false', lua_value).replace('FALSE', lua_value.upper())
                    new_lines.append(new_line)
                    updated = True
                    print(f"Updated line to: {new_line}")
                else:
                    new_lines.append(line)
            
            if updated:
                new_content = '\n'.join(new_lines)
        
        if updated:
            # Write back to the file
            with open(lua_control_strategy_path, 'w') as file:
                file.write(new_content)
            print(f"+++ Successfully updated {variable_name} to {lua_value} +++")
        else:
            print(f"X: Failed to find and update {variable_name}")
            print("Current file content:")
            print(content)
            
    except FileNotFoundError:
        print(f"Error: File {lua_control_strategy_path} not found")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

# Function to configure the arena .argos file with the specified controller script and visualization settings
def arena_configuration(path_control_script, strategy, visualizations, shape, pattern, task_size):
    """
    Configure the arena .argos file with controller script and visualization settings
    
    Args:
        path_control_script (str): Path to the Lua control script
        visualizations (bool): Whether to enable visualization or not
    """
    # variable to set greedy or selective in the loop functions
    greedy_value = "true" if strategy == "greedy" else "false"
    try:
        # Parse the ARGoS configuration file
        tree = ET.parse(arena_config_script_path)
        root = tree.getroot()
        
        # Configure the controller script
        controller = root.find("controllers")
        for params in controller.iter("params"):
            params.set("script", path_control_script)
        # variable to set greedy or selective in the loop functions
        loop_functions = root.find("loop_functions")
        loop_functions.set("greedy", greedy_value)
        # variable to configure shape, pattern and task_size in the arena
        loop_functions.set("shape", shape)
        loop_functions.set("pattern", pattern)
        loop_functions.set("task_size", task_size)

        # Configure visualization
        if visualizations:
            visualization_elem = root.find("visualization")
            
            if visualization_elem is None:
                visualization_elem = ET.SubElement(root, "visualization")
            else:
                visualization_elem.clear()
            
            qt_opengl_elem = ET.SubElement(visualization_elem, "qt-opengl")
            qt_opengl_elem.set("lua_editor", "false")
            
            camera_elem = ET.SubElement(qt_opengl_elem, "camera")
            
            placement_elem = ET.SubElement(camera_elem, "placement")
            placement_elem.set("idx", "0")
            placement_elem.set("position", "0,0,7")
            placement_elem.set("look_at", "0,0,0")
            placement_elem.set("up", "3,0,0")
            placement_elem.set("lens_focal_length", "70")
            
        else:
            visualization_elem = root.find("visualization")
            if visualization_elem is not None:
                root.remove(visualization_elem)
        
        # Save the changes to the .argos file
        tree.write(arena_config_script_path)
        print(f"<<< Arena configuration updated successfully >>>")

    except Exception as e:
        print(f"X: Error configuring arena: {e}")
        import traceback
        traceback.print_exc()

# Function to update the random seed in the .argos file for experiment reproducibility
def framework_label(file, seed_value):
    """
    Modify the 'framework' tag and its contents in the .argos file to set a unique random seed
    """
    try:
        tree = ET.parse(file)
        root = tree.getroot()
        # Find the framework tag
        framework = root.find("framework")
        if framework is not None:
            # Modify attributes of 'experiment'
            experiment = framework.find("experiment")
            if experiment is not None:
                # Inject the pre-generated seed value
                experiment.set("random_seed", str(seed_value))

        tree.write(file)
    except Exception as e:
        print(f"X: Error updating random seed: {e}")

# Function to save the summary of seeds used in the campaign
def save_seeds_summary(file_path, seeds_data):
    """
    Write a summary of all seeds used during the experiment batch to a text file
    """
    try:
        with open(file_path, 'w') as f:
            f.write("=== SUMMARY OF EXPERIMENT SEEDS ===\n")
            f.write(f"Date: {time.ctime()}\n")
            f.write("-" * 35 + "\n")
            f.write(f"{'Strategy':<12} | {'Exp #':<6} | {'Seed':<10}\n")
            f.write("-" * 35 + "\n")
            for entry in seeds_data:
                f.write(f"{entry['strategy']:<12} | {entry['num']:<6} | {entry['seed']:<10}\n")
        print(f"OK: Seeds summary saved to '{file_path}'")
    except Exception as e:
        print(f"X: Error saving seeds summary: {e}")

# Function to load seed previusly keept
def load_seeds_from_file(file_path):
    """
    Load seeds from a previously generated summary file.
    Returns a list of seeds in the correct execution order.
    """
    seeds = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                # Saltar encabezados y separadores
                if "|" in line and "Strategy" not in line:
                    parts = line.split("|")
                    if len(parts) >= 3:
                        seed = parts[2].strip()
                        if seed.isdigit():
                            seeds.append(int(seed))
        print(f"OK: Loaded {len(seeds)} seeds from '{file_path}'")
    except Exception as e:
        print(f"X: Error loading seeds: {e}")
    
    return seeds

# Execute the experiments in a loop based on the configuration variables
if __name__ == "__main__":
    print(pf.figlet_format("QUPA Experiments", font="bubble"))
    print("Starting automated experiments...")
    
    # Step 1: Delete previous data to start from zero
    delete_previous_data(output_data_file)
    
    # Step 2: Generate a list of strictly UNIQUE seeds for all experiments
    total_total_runs = num_experiments * len(use_strategy)
    
    # random.sample extracts unique elements from a population without replacement.
    # We increased the range up to 99999 to guarantee a massive pool of unique integers.
    #experiment_seeds = random.sample(range(1000, 99999), total_total_runs)
    
    if use_previous_seeds and os.path.exists(seeds_log_file):
        experiment_seeds = load_seeds_from_file(seeds_log_file)

        if len(experiment_seeds) < total_total_runs:
            print("WARNING: Not enough seeds in file, generating missing ones...")
            missing = total_total_runs - len(experiment_seeds)
            extra_seeds = random.sample(range(1000, 99999), missing)
            experiment_seeds.extend(extra_seeds)

    else:
        print("INFO: Generating new random seeds...")
        experiment_seeds = random.sample(range(1000, 99999), total_total_runs)

    seeds_summary_data = []
    seed_index = 0
    
    # Loop to execute experiments for each strategy
    for current_strategy in use_strategy: # Loop through the specified strategies (greedy and selective)
        
        # Update strategy settings before starting the batch
        print("="*50)
        update_lua_variable(current_strategy)
        
        # Loop to execute the specified number of experiments for the current strategy
        for j in range(num_experiments): 
            print(f"--- Experiment {j+1}/{num_experiments} with strategy {current_strategy} ---")
            
            # Step 3: Configure arena and unique random seed for this specific run
            current_seed = experiment_seeds[seed_index]
            
            # Record data for the summary
            seeds_summary_data.append({
                'strategy': current_strategy,
                'num': j + 1,
                'seed': current_seed
            })

            arena_configuration(lua_control_strategy_path, current_strategy, visualizations=False, 
                                shape=geo_shape, pattern=geo_pattern, task_size=geo_task_size)
            framework_label(arena_config_script_path, current_seed)
            seed_index += 1
            
            # ARGoS command to execute the experiment
            command = ["argos3", "-c", arena_config_script_path]
            try:
                # Start the experiment process and capture output and errors
                process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                stdout, stderr = process.communicate()
                
                if process.returncode == 0:
                    print(f" OK: Experiment {j+1} completed with seed {current_seed}")
                else:
                    print(f"X: Experiment {j+1} failed with error:")
                    print(stderr.decode())
            except Exception as e:
                print(f"X: Error executing experiment {j+1}: {e}")
                import traceback
                traceback.print_exc()
            print("-"*50)
            
    # Save the text file with all used seeds
    save_seeds_summary(seeds_log_file, seeds_summary_data)        
    print(pf.figlet_format("Experiments Completed", font="digital"))