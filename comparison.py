import subprocess
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from collections import namedtuple
from simglucose.patient.t1dpatient import T1DPatient

# Run the full loop simulation (assuming the original script is named full_loop.py)
print("Running full loop simulation...")
os.system('python full_loop.py')

# Load the results from the full loop
with open('sim.json', 'r') as f:
    full_loop_results = json.load(f)

# Initialize the step-by-step simulation
print("Initializing step-by-step simulation...")
os.system('python init.py --output_file=step_state_0.json')

# Run the step-by-step simulation
print("Running step-by-step simulation...")
bg_values = []
cho_values = []
insulin_values = []
timestamps = []

# We need to extract the initial glucose value from the state file
with open('step_state_0.json', 'r') as f:
    initial_state = json.load(f)
    # Assuming the state file contains a glucose value
    # You may need to adjust this based on the actual structure
    if 'glucose' in initial_state:
        initial_glucose = initial_state['glucose']
    elif 'Gsub' in initial_state:
        initial_glucose = initial_state['Gsub']
    elif 'observation' in initial_state and 'Gsub' in initial_state['observation']:
        initial_glucose = initial_state['observation']['Gsub']
    else:
        # If we can't find it, use a reasonable default for T1D
        print("Warning: Could not find initial glucose in state file, using default value")
        initial_glucose = 100

# bg_values.append(initial_glucose)
# cho_values.append(0)  # No carbs at time 0
# insulin_values.append(0)  # No insulin at time 0
# timestamps.append(0)

current_glucose = initial_glucose

for t in range(1, 51):
    # Determine if there are carbs at this time step
    carbs = 0
    if t == 6:
        carbs = 20
    
    # Run the step
    cmd = f'python step.py --input_state_file=step_state_{t-1}.json --output_state_file=step_state_{t}.json --glucose_reading={current_glucose} --carbs={carbs} --delta_time=1 --controller_algorithm=basal_bolus'
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(f"Error at step {t}:")
        print(result.stderr)
        break
    
    # Collect the results - assuming step.py outputs state with these fields
    # Adjust as needed based on actual output format
    try:
        with open(f'step_state_{t}.json', 'r') as f:
            state = json.load(f)
            
            # Extract glucose value - adjust field names based on actual structure
            if 'glucose' in state:
                current_glucose = state['glucose']
            elif 'Gsub' in state:
                current_glucose = state['Gsub']
            elif 'observation' in state and 'Gsub' in state['observation']:
                current_glucose = state['observation']['Gsub']
            else:
                print(f"Warning: Could not find glucose value in state file at step {t}")
                current_glucose = bg_values[-1]  # Use previous value as fallback
            
            bg_values.append(current_glucose)
            cho_values.append(carbs)
            
            # Extract insulin value - adjust field names based on actual structure
            if 'insulin' in state:
                insulin = state['insulin']
            elif 'action' in state and 'insulin' in state['action']:
                insulin = state['action']['insulin']
            else:
                print(f"Warning: Could not find insulin value in state file at step {t}")
                insulin = 0  # Default
            
            insulin_values.append(insulin)
            timestamps.append(t)
    except Exception as e:
        print(f"Error processing state file at step {t}: {e}")
        break

# Save the step-by-step results
step_by_step_results = {
    'bg': bg_values,
    'cho': cho_values,
    'insulin': insulin_values
}

with open('step_sim.json', 'w') as f:
    json.dump(step_by_step_results, f)

# Compare the results
# Plot both simulations
fig, axs = plt.subplots(3, 1, figsize=(12, 15), sharex=True)

# Ensure both datasets have the same length for comparison
min_length = min(len(full_loop_results['bg']), len(bg_values))

# Blood Glucose
axs[0].plot(full_loop_results['bg'][:min_length], label='Full Loop', color='blue')
axs[0].plot(bg_values[:min_length], label='Step-by-Step', color='red', linestyle='--')
axs[0].set_ylabel('Blood Glucose (mg/dL)')
axs[0].set_title('Blood Glucose Comparison')
axs[0].legend()
axs[0].grid(True)

# Carbohydrates
axs[1].plot(full_loop_results['cho'][:min_length], label='Full Loop', color='blue')
axs[1].plot(cho_values[:min_length], label='Step-by-Step', color='red', linestyle='--')
axs[1].set_ylabel('Carbohydrates (g)')
axs[1].set_title('Carbohydrate Intake Comparison')
axs[1].legend()
axs[1].grid(True)

# Insulin
axs[2].plot(full_loop_results['insulin'][:min_length], label='Full Loop', color='blue')
axs[2].plot(insulin_values[:min_length], label='Step-by-Step', color='red', linestyle='--')
axs[2].set_ylabel('Insulin (U)')
axs[2].set_title('Insulin Dosing Comparison')
axs[2].set_xlabel('Time (minutes)')
axs[2].legend()
axs[2].grid(True)

plt.tight_layout()
plt.savefig('comparison.png')

# Calculate differences between the two simulations
bg_diff = np.array(full_loop_results['bg'][:min_length]) - np.array(bg_values[:min_length])
cho_diff = np.array(full_loop_results['cho'][:min_length]) - np.array(cho_values[:min_length])
insulin_diff = np.array(full_loop_results['insulin'][:min_length]) - np.array(insulin_values[:min_length])

# Statistics on differences
print("\nComparison Statistics:")
print(f"Blood Glucose Mean Difference: {np.mean(bg_diff):.4f} mg/dL")
print(f"Blood Glucose Max Absolute Difference: {np.max(np.abs(bg_diff)):.4f} mg/dL")
print(f"Blood Glucose RMSE: {np.sqrt(np.mean(bg_diff**2)):.4f} mg/dL")

print(f"Carbohydrate Mean Difference: {np.mean(cho_diff):.4f} g")
print(f"Carbohydrate Max Absolute Difference: {np.max(np.abs(cho_diff)):.4f} g")

print(f"Insulin Mean Difference: {np.mean(insulin_diff):.4f} U")
print(f"Insulin Max Absolute Difference: {np.max(np.abs(insulin_diff)):.4f} U")
print(f"Insulin RMSE: {np.sqrt(np.mean(insulin_diff**2)):.4f} U")

# Plot the differences
fig, axs = plt.subplots(3, 1, figsize=(12, 15), sharex=True)

# Blood Glucose differences
axs[0].plot(bg_diff)
axs[0].set_ylabel('BG Difference (mg/dL)')
axs[0].set_title('Blood Glucose Difference (Full Loop - Step-by-Step)')
axs[0].axhline(y=0, color='r', linestyle='-')
axs[0].grid(True)

# Carbohydrate differences
axs[1].plot(cho_diff)
axs[1].set_ylabel('CHO Difference (g)')
axs[1].set_title('Carbohydrate Difference (Full Loop - Step-by-Step)')
axs[1].axhline(y=0, color='r', linestyle='-')
axs[1].grid(True)

# Insulin differences
axs[2].plot(insulin_diff)
axs[2].set_ylabel('Insulin Difference (U)')
axs[2].set_title('Insulin Difference (Full Loop - Step-by-Step)')
axs[2].set_xlabel('Time (minutes)')
axs[2].axhline(y=0, color='r', linestyle='-')
axs[2].grid(True)

plt.tight_layout()
plt.savefig('difference_plot.png')

print("\nResults saved to 'comparison.png' and 'difference_plot.png'")