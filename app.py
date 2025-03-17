from collections import namedtuple
from uuid import uuid4
import json
from typing import Dict, Optional, Any
import copy

import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from simglucose.controller.basal_bolus_ctrller import BBController
from simglucose.controller.pid_ctrller import PIDController
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.controller.base import Action, Controller

# Define the namedtuples as in the original code
CtrlObservation = namedtuple("CtrlObservation", ["CGM"])
PatientAction = namedtuple("patient_action", ["CHO", "insulin"])
PatientObservation = namedtuple("observation", ["Gsub"])

# Controller mapping
controller_map = {"basal_bolus": BBController, "pid": PIDController}

# Pydantic models for request/response validation
class InitResponse(BaseModel):
    initial_glucose: float
    patient_id: str

class StepRequest(BaseModel):
    glucose_reading: float
    carbs: int = 0
    delta_time: int = 1
    controller_algorithm: str = "basal_bolus"
    pump: str = "DefaultPump"
    attack_glucose: Optional[float] = None  # Added: fake glucose data for attack scenario

class StepResponse(BaseModel):
    glucose: float
    insulin: float
    attack_scenario: bool = False  # Added: flag for attack scenario
    real_glucose: Optional[float] = None  # Added: real glucose value
    attack_glucose: Optional[float] = None  # Added: attack glucose value

# Create FastAPI app
app = FastAPI(title="Glucose Simulation API")

# Store patients in memory (consider a database for production)
patient_map: Dict[str, T1DPatient] = {}

class ModifiedT1DPatient:
    """Modified T1DPatient with enhanced insulin sensitivity"""
    
    def __init__(self, original_patient):
        self.original_patient = original_patient
        self.name = original_patient.name
        self._observation = copy.deepcopy(original_patient.observation)
        
    @property
    def observation(self):
        return self._observation
    
    def step(self, action):
        """Enhanced insulin effect step function"""
        # Get current glucose
        current_glucose = self._observation.Gsub
        
        # Get action parameters
        insulin = action.insulin
        cho = action.CHO
        
        # Simplified glucose calculation model
        # 1. Carbohydrates increase glucose - each gram increases by ~4 mg/dL
        glucose_from_cho = cho * 4
        
        # 2. Insulin decreases glucose - each unit decreases by ~50 mg/dL (enhanced effect)
        glucose_from_insulin = insulin * 50
        
        # 3. Calculate new glucose value
        new_glucose = current_glucose + glucose_from_cho - glucose_from_insulin
        
        # Ensure glucose doesn't go below 30 mg/dL (dangerous hypoglycemia)
        new_glucose = max(30, new_glucose)
        
        # Update observation
        self._observation = PatientObservation(Gsub=new_glucose)
        
        return self._observation

@app.post("/init", response_model=InitResponse)
def init(patient: str = "adolescent#003"):
    """Initialize a new patient simulation"""
    t1d_patient = T1DPatient.withName(patient, seed=42)
    
    # Generate a unique ID for this patient
    patient_id = str(uuid4())
    
    # Store the patient in our map
    patient_map[patient_id] = t1d_patient
    
    return {
        "initial_glucose": t1d_patient.observation.Gsub,
        "patient_id": patient_id
    }

@app.post("/step/{patient_id}", response_model=StepResponse)
def step(patient_id: str, request: StepRequest):
    """Take a simulation step for a specific patient"""
    # Check if patient exists
    if patient_id not in patient_map:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    patient = patient_map[patient_id]
    
    # Get controller based on algorithm
    if request.controller_algorithm not in controller_map:
        raise HTTPException(status_code=400, detail="Unsupported controller algorithm")
    
    ctrl = controller_map[request.controller_algorithm]()
    
    # Determine if this is an attack scenario
    is_attack = request.attack_glucose is not None
    
    # Create controller observation - if attack scenario, use fake glucose data
    glucose_for_controller = request.attack_glucose if is_attack else request.glucose_reading
    ctrl_obs = CtrlObservation(glucose_for_controller)
    
    # Get controller action
    ctrl_action = ctrl.policy(
        observation=ctrl_obs,
        reward=0,
        done=False,
        patient_name=patient.name,
        meal=request.carbs
    )
    
    # Calculate insulin
    insulin = ctrl_action.basal + ctrl_action.bolus
    
    # Create patient action
    patient_action = PatientAction(insulin=insulin, CHO=request.carbs)
    
    # Update patient state
    patient.step(patient_action)
    
    # Get new glucose level
    glucose = patient.observation.Gsub
    
    # Build response
    response = {
        "glucose": glucose,
        "insulin": insulin
    }
    
    # If attack scenario, add additional information
    if is_attack:
        response["attack_scenario"] = True
        response["real_glucose"] = request.glucose_reading
        response["attack_glucose"] = request.attack_glucose
    
    return response

@app.get("/patients")
def list_patients():
    """List all active patient simulations"""
    return {
        "patient_count": len(patient_map),
        "patients": list(patient_map.keys())
    }

@app.delete("/patients/{patient_id}")
def delete_patient(patient_id: str):
    """Remove a patient from the simulation"""
    if patient_id not in patient_map:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    del patient_map[patient_id]
    return {"status": "deleted", "patient_id": patient_id}

@app.post("/attack_demo/{patient_id}", response_model=StepResponse)
def attack_demo(patient_id: str, request: StepRequest):
    """
    Demonstrate attack scenario: use fake glucose data to calculate insulin dose,
    but apply this dose to real glucose data
    """
    # Check if patient exists
    if patient_id not in patient_map:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    # Check if attack glucose value is provided
    if request.attack_glucose is None:
        raise HTTPException(status_code=400, detail="Attack glucose value is required")
    
    patient = patient_map[patient_id]

    # Use PIDController instead of BBController
    ctrl = PIDController()
    
    # Create controller observation with fake glucose data
    ctrl_obs = CtrlObservation(request.attack_glucose)
    
    # Get controller action
    ctrl_action = ctrl.policy(
        observation=ctrl_obs,
        reward=0,
        done=False,
        patient_name=patient.name,
        meal=request.carbs
    )

    # Calculate insulin dose
    insulin = ctrl_action.basal + ctrl_action.bolus
    
    # Create patient action - use calculated insulin dose and real carb intake
    patient_action = PatientAction(insulin=insulin, CHO=request.carbs)
    
    # Update patient state - this uses the real glucose data
    patient.step(patient_action)
    
    # Get new glucose level
    glucose = patient.observation.Gsub
    
    return {
        "glucose": glucose,
        "insulin": insulin,
        "attack_scenario": True,
        "real_glucose": request.glucose_reading,
        "attack_glucose": request.attack_glucose
    }

@app.post("/attack_demo_with_controller/{patient_id}", response_model=StepResponse)
def attack_demo_with_controller(patient_id: str, request: StepRequest):
    """
    Demonstrate attack scenario: use specified controller algorithm to calculate insulin dose
    based on fake glucose data, but apply this dose to real glucose data
    """
    # Check if patient exists
    if patient_id not in patient_map:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    # Check if attack glucose value is provided
    if request.attack_glucose is None:
        raise HTTPException(status_code=400, detail="Attack glucose value is required")
    
    # Get controller based on algorithm
    if request.controller_algorithm not in controller_map:
        raise HTTPException(status_code=400, detail="Unsupported controller algorithm")

    patient = patient_map[patient_id]
    
    # Use specified controller
    ctrl = controller_map[request.controller_algorithm]()
    
    # Create controller observation with fake glucose data
    ctrl_obs = CtrlObservation(request.attack_glucose)
    
    # Get controller action
    ctrl_action = ctrl.policy(
        observation=ctrl_obs,
        reward=0,
        done=False,
        patient_name=patient.name,
        meal=request.carbs
    )

    # Calculate insulin dose
    insulin = ctrl_action.basal + ctrl_action.bolus
    
    # Create patient action - use calculated insulin dose and real carb intake
    patient_action = PatientAction(insulin=insulin, CHO=request.carbs)
    
    # Update patient state - this uses the real glucose data
    patient.step(patient_action)
    
    # Get new glucose level
    glucose = patient.observation.Gsub
    
    return {
        "glucose": glucose,
        "insulin": insulin,
        "attack_scenario": True,
        "real_glucose": request.glucose_reading,
        "attack_glucose": request.attack_glucose,
        "controller_used": request.controller_algorithm
    }

@app.post("/enhanced_attack_demo/{patient_id}")
def enhanced_attack_demo(patient_id: str, request: StepRequest):
    """
    Use enhanced insulin sensitivity patient model for attack demonstration
    """
    if patient_id not in patient_map:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    if request.attack_glucose is None:
        raise HTTPException(status_code=400, detail="Attack glucose value is required")
    
    original_patient = patient_map[patient_id]
    
    # Create modified patient model
    modified_patient = ModifiedT1DPatient(original_patient)
    
    # Use specified controller
    if request.controller_algorithm not in controller_map:
        raise HTTPException(status_code=400, detail="Unsupported controller algorithm")
    
    ctrl = controller_map[request.controller_algorithm]()
    
    # Calculate insulin dose using attack glucose value
    ctrl_obs = CtrlObservation(request.attack_glucose)
    
    # Get controller action
    ctrl_action = ctrl.policy(
        observation=ctrl_obs,
        reward=0,
        done=False,
        patient_name=modified_patient.name,
        meal=request.carbs
    )
    
    # Calculate insulin dose
    insulin = ctrl_action.basal + ctrl_action.bolus
    
    # Create patient action
    patient_action = PatientAction(insulin=insulin, CHO=request.carbs)
    
    # Set initial glucose to real value
    modified_patient._observation = PatientObservation(Gsub=request.glucose_reading)
    initial_glucose = modified_patient.observation.Gsub
    
    # Update patient state
    modified_patient.step(patient_action)
    
    # Get new glucose level
    glucose = modified_patient.observation.Gsub
    
    return {
        "glucose": glucose,
        "insulin": insulin,
        "attack_scenario": True,
        "real_glucose": request.glucose_reading,
        "attack_glucose": request.attack_glucose,
        "controller_used": request.controller_algorithm,
        "glucose_change": glucose - initial_glucose
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)