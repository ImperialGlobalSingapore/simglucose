import json
import numpy as np
from datetime import datetime
import fire

def minutes_to_time_str(minutes):
    """Convert minutes since midnight to time string HH:MM."""
    hours, mins = divmod(minutes, 60)
    return f"{int(hours):02d}:{int(mins):02d}"

def generate_questions_and_answers(patient_data):
    """Generate questions and calculate ground truth answers."""
    
    # Extract relevant data
    patient_id = patient_data["patient_id"]
    bg_values = patient_data["bg_mgdl"]
    meals = patient_data["meals"]
    insulin_events = patient_data["insulin_events"]
    
    # Basic statistics
    avg_glucose = np.mean(bg_values)
    max_glucose = np.max(bg_values)
    min_glucose = np.min(bg_values)
    max_glucose_time = minutes_to_time_str(np.argmax(bg_values))
    std_glucose = np.std(bg_values)
    
    # Target range calculations
    in_range = [70 <= bg <= 180 for bg in bg_values]
    time_in_range_minutes = sum(in_range)
    time_in_range_hours = time_in_range_minutes / 60
    time_in_range_percentage = (time_in_range_minutes / len(bg_values)) * 100
    
    # Hypoglycemic/hyperglycemic events
    hypo_events = 0
    hyper_events = 0
    in_hypo = False
    in_hyper = False
    
    for bg in bg_values:
        if bg < 70 and not in_hypo:
            in_hypo = True
            hypo_events += 1
        elif bg >= 70 and in_hypo:
            in_hypo = False
            
        if bg > 180 and not in_hyper:
            in_hyper = True
            hyper_events += 1
        elif bg <= 180 and in_hyper:
            in_hyper = False
    
    # Glucose fluctuations
    glucose_rates = [bg_values[i] - bg_values[i-1] for i in range(1, len(bg_values))]
    rapid_fluctuations = [abs(rate) > 2 for rate in glucose_rates]
    has_rapid_fluctuations = any(rapid_fluctuations)
    
    # Meal-related calculations
    meal_responses = {}
    for meal_type, meal_info in meals.items():
        meal_time = meal_info["time"]
        
        # Skip if meal time is outside data range
        if meal_time >= len(bg_values):
            continue
            
        baseline = bg_values[meal_time]
        post_meal_window = min(180, len(bg_values) - meal_time - 1)  # 3 hours or until end of data
        
        if post_meal_window <= 0:
            continue
            
        post_meal_bg = bg_values[meal_time:meal_time + post_meal_window + 1]
        
        if len(post_meal_bg) > 1:
            peak_index = np.argmax(post_meal_bg)
            peak = post_meal_bg[peak_index]
            peak_time = meal_time + peak_index
            peak_time_str = minutes_to_time_str(peak_time)
            
            # Time to return to baseline after peak
            time_to_baseline = None
            if peak_index < len(post_meal_bg) - 1:
                for i in range(peak_index + 1, len(post_meal_bg)):
                    if post_meal_bg[i] <= baseline * 1.1:  # Within 10% above baseline
                        time_to_baseline = i - peak_index
                        break
            
            # Initial rise rate (first 30 min)
            initial_window = min(30, len(post_meal_bg) - 1)
            initial_rise = post_meal_bg[initial_window] - baseline
            rise_rate = initial_rise / initial_window if initial_window > 0 else 0
            
            meal_responses[meal_type] = {
                "baseline": baseline,
                "peak": peak,
                "peak_time": peak_time_str,
                "time_to_baseline": time_to_baseline,
                "rise_rate": rise_rate
            }
    
    # Find meal with highest spike
    max_spike_meal = None
    max_spike = 0
    
    for meal_type, response in meal_responses.items():
        spike = response["peak"] - response["baseline"]
        if spike > max_spike:
            max_spike = spike
            max_spike_meal = meal_type
    
    # Insulin-related calculations
    total_insulin = sum([event["insulin"] for event in insulin_events])
    
    largest_bolus = max(insulin_events, key=lambda x: x["insulin"])
    largest_bolus_time = largest_bolus["time_str"]
    largest_bolus_amount = largest_bolus["insulin"]
    
    # Assume small insulin amounts are basal
    basal_threshold = 0.1
    basal_events = [event for event in insulin_events if event["insulin"] < basal_threshold]
    
    # Generate questions and answers
    questions_and_answers = []
    
    # Basic Statistics questions
    questions_and_answers.append({
        "question": "What was the patient's average blood glucose level during the day?",
        "answer": f"{avg_glucose:.1f} mg/dL",
        "explanation": "Calculate the mean of all blood glucose values."
    })
    
    questions_and_answers.append({
        "question": "What was the highest blood glucose level recorded?",
        "answer": f"{max_glucose:.1f} mg/dL",
        "explanation": "Find the maximum value in the blood glucose array."
    })
    
    questions_and_answers.append({
        "question": "What was the lowest blood glucose level recorded?",
        "answer": f"{min_glucose:.1f} mg/dL",
        "explanation": "Find the minimum value in the blood glucose array."
    })
    
    questions_and_answers.append({
        "question": "When did the patient experience their highest blood glucose peak?",
        "answer": f"{max_glucose_time}",
        "explanation": "Find the index of the maximum glucose value and convert to time format."
    })
    
    questions_and_answers.append({
        "question": "How many hours did the patient spend in the target range (70-180 mg/dL)?",
        "answer": f"{time_in_range_hours:.1f} hours",
        "explanation": "Count minutes where glucose values are between 70-180 mg/dL, then convert to hours."
    })
    
    questions_and_answers.append({
        "question": "How many hypoglycemic events (BG < 70 mg/dL) did the patient experience?",
        "answer": f"{hypo_events}",
        "explanation": "Count transitions from normal to hypoglycemic state (BG < 70 mg/dL)."
    })
    
    questions_and_answers.append({
        "question": "How many hyperglycemic events (BG > 180 mg/dL) did the patient experience?",
        "answer": f"{hyper_events}",
        "explanation": "Count transitions from normal to hyperglycemic state (BG > 180 mg/dL)."
    })
    
    questions_and_answers.append({
        "question": "What was the standard deviation of the patient's glucose levels?",
        "answer": f"{std_glucose:.1f} mg/dL",
        "explanation": "Calculate the standard deviation of all blood glucose values."
    })
    
    questions_and_answers.append({
        "question": "What's the time-in-range percentage for this patient?",
        "answer": f"{time_in_range_percentage:.1f}%",
        "explanation": "Divide time in range (70-180 mg/dL) by total time, multiply by 100."
    })
    
    questions_and_answers.append({
        "question": "Did the patient experience any rapid glucose fluctuations (>2 mg/dL/min)?",
        "answer": "Yes" if has_rapid_fluctuations else "No",
        "explanation": "Calculate rate of change between consecutive readings, check if any exceed 2 mg/dL/min."
    })
    
    # Meal-related questions
    if "breakfast" in meal_responses:
        questions_and_answers.append({
            "question": "How did the patient's blood glucose respond to breakfast?",
            "answer": f"Baseline: {meal_responses['breakfast']['baseline']:.1f} mg/dL, Peak: {meal_responses['breakfast']['peak']:.1f} mg/dL at {meal_responses['breakfast']['peak_time']}",
            "explanation": "Compare baseline glucose at meal time to maximum value in post-meal window."
        })
    
    if "lunch" in meal_responses:
        questions_and_answers.append({
            "question": "What was the peak glucose level after lunch?",
            "answer": f"{meal_responses['lunch']['peak']:.1f} mg/dL",
            "explanation": "Find maximum glucose value in the post-lunch window (typically 3 hours)."
        })
    
    if "dinner" in meal_responses and meal_responses["dinner"]["time_to_baseline"] is not None:
        questions_and_answers.append({
            "question": "How long did it take for glucose levels to return to baseline after dinner?",
            "answer": f"{meal_responses['dinner']['time_to_baseline']} minutes",
            "explanation": "Find first time after peak when glucose returns to within 10% of pre-meal baseline."
        })
    
    if max_spike_meal:
        questions_and_answers.append({
            "question": "Which meal caused the highest glucose spike?",
            "answer": f"{max_spike_meal.replace('_', ' ')} with a {max_spike:.1f} mg/dL increase",
            "explanation": "Compare peak minus baseline values for all meals to find largest increase."
        })
    
    if "snack_morning" in meal_responses:
        questions_and_answers.append({
            "question": "What was the glucose rise rate after the morning snack?",
            "answer": f"{meal_responses['snack_morning']['rise_rate']:.2f} mg/dL/min",
            "explanation": "Calculate rate of increase from baseline to 30-minute post-meal glucose."
        })
    
    # Insulin-related questions
    questions_and_answers.append({
        "question": "What was the patient's total daily insulin dose?",
        "answer": f"{total_insulin:.2f} units",
        "explanation": "Sum all insulin amounts from insulin events."
    })
    
    questions_and_answers.append({
        "question": "When did the patient receive their largest insulin bolus?",
        "answer": f"{largest_bolus_time} ({largest_bolus_amount:.2f} units)",
        "explanation": "Find insulin event with maximum insulin amount."
    })
    
    if len(basal_events) > 1:
        questions_and_answers.append({
            "question": "How did basal rates change throughout the day?",
            "answer": f"{len(basal_events)} basal rate adjustments, averaging {np.mean([e['insulin'] for e in basal_events]):.4f} units per adjustment",
            "explanation": "Analyze insulin events below threshold (assumed to be basal) for frequency and amount."
        })
    
    return questions_and_answers

def process_jsonl_file(input_file, output_file, include_patient_data=True):
    """
    Process JSONL file, generate questions and answers, write to output file.
    
    Args:
        input_file: Path to input JSONL file with glucose data
        output_file: Path to output JSONL file for questions and answers
        include_patient_data: If True, include the original patient data in the output
    """
    print(f"Processing {input_file} -> {output_file}")
    
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        line_count = 0
        processed_count = 0
        
        for line in f_in:
            line_count += 1
            try:
                patient_data = json.loads(line.strip())
                
                results = {
                    "patient_id": patient_data["patient_id"],
                    "patient_data": patient_data,
                    "questions_and_answers": generate_questions_and_answers(patient_data)
                }
                
                if include_patient_data:
                    results["patient_data"] = patient_data
                
                f_out.write(json.dumps(results) + '\n')
                processed_count += 1
                
                if processed_count % 10 == 0:
                    print(f"Processed {processed_count} records...")
                
            except Exception as e:
                print(f"Error processing line {line_count}: {e}")
        
        print(f"Processing complete. Successfully processed {processed_count} of {line_count} records.")

def main(input_file="glucose_traces.jsonl", 
         output_file="glucose_questions_answers.jsonl",
         include_patient_data=True):
    """
    Generate glucose-related questions and answers from patient data.
    
    Args:
        input_file: Path to input JSONL file containing glucose trace data
        output_file: Path to output JSONL file for questions and answers
        include_patient_data: Whether to include original patient data in output (default: True)
    """
    process_jsonl_file(input_file, output_file, include_patient_data)

if __name__ == "__main__":
    fire.Fire(main)