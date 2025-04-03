from collections import namedtuple
from uuid import uuid4
import json
from typing import Dict, Optional, Any, List
import copy
import logging

import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from simglucose.controller.basal_bolus_ctrller import BBController
from simglucose.controller.simple_pid_ctrller import SimplePIDController as PIDController
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.controller.base import Action, Controller

# Define the namedtuples as in the original code
CtrlObservation = namedtuple("CtrlObservation", ["CGM"])
PatientAction = namedtuple("patient_action", ["CHO", "insulin"])
PatientObservation = namedtuple("observation", ["Gsub"])

# Controller mapping
controller_map = {"basal_bolus": BBController, "pid": PIDController}

# Pydantic models for request/response validation
class InitRequest(BaseModel):
    patient: str = "adolescent#003"
    controller_algorithm: str = "basal_bolus"
    controller_kwargs: Dict[str, Any] = Field(default_factory=dict)

class InitResponse(BaseModel):
    initial_glucose: float
    patient_id: str
    controller_algorithm: str

class StepRequest(BaseModel):
    glucose_reading: float
    carbs: int = 0
    delta_time: int = 1
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

# Store patients and controllers in memory (consider a database for production)
patient_map: Dict[str, Dict[str, Any]] = {}

# Define logger
logger = logging.getLogger(__name__)

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
def init(request: InitRequest):
    """Initialize a new patient simulation with controller"""
    t1d_patient = T1DPatient.withName(request.patient, seed=42)
    
    # Check if controller algorithm is supported
    if request.controller_algorithm not in controller_map:
        raise HTTPException(status_code=400, detail="Unsupported controller algorithm")
    
    # Initialize controller with provided kwargs
    controller = controller_map[request.controller_algorithm](**request.controller_kwargs)
    
    # Generate a unique ID for this patient
    patient_id = str(uuid4())
    
    # Store the patient and controller in our map
    patient_map[patient_id] = {
        "patient": t1d_patient, 
        "controller": controller,
        "controller_algorithm": request.controller_algorithm
    }
    
    return {
        "initial_glucose": t1d_patient.observation.Gsub,
        "patient_id": patient_id,
        "controller_algorithm": request.controller_algorithm
    }

@app.post("/step/{patient_id}", response_model=StepResponse)
def step(patient_id: str, request: StepRequest):
    """Take a simulation step for a specific patient"""
    # Check if patient exists
    if patient_id not in patient_map:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    # Retrieve patient and controller
    patient_data = patient_map[patient_id]
    patient = patient_data["patient"]
    ctrl = patient_data["controller"]
    
    # Determine if this is an attack scenario
    is_attack = request.attack_glucose is not None
    
    # Create controller observation - if attack scenario, use fake glucose data
    glucose_for_controller = request.attack_glucose if is_attack else request.glucose_reading
    ctrl_obs = CtrlObservation(glucose_for_controller)
    print("observation", ctrl_obs)
    # Get controller action
    ctrl_action = ctrl.policy(
        observation=ctrl_obs,
        reward=0,
        done=False,
        patient_name=patient.name,
        meal=request.carbs,
        time=patient.t
    )
    print("action", ctrl_action)
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
    patient_info = {}
    for patient_id, data in patient_map.items():
        patient_info[patient_id] = {
            "controller_algorithm": data["controller_algorithm"],
            "patient_name": data["patient"].name
        }
    
    return {
        "patient_count": len(patient_map),
        "patients": patient_info
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
    
    # Retrieve patient and controller
    patient_data = patient_map[patient_id]
    patient = patient_data["patient"]
    ctrl = patient_data["controller"]
    
    # Create controller observation with fake glucose data
    ctrl_obs = CtrlObservation(request.attack_glucose)
    
    # Get controller action
    ctrl_action = ctrl.policy(
        observation=ctrl_obs,
        reward=0,
        done=False,
        patient_name=patient.name,
        meal=request.carbs,
        time=request.delta_time
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
def attack_demo_with_controller(patient_id: str, request: StepRequest, override_controller: bool = False):
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
    
    # Retrieve patient data
    patient_data = patient_map[patient_id]
    patient = patient_data["patient"]
    
    # Determine controller to use (stored or specific override)
    if override_controller:
        # Check if controller algorithm is supported
        if request.controller_algorithm not in controller_map:
            raise HTTPException(status_code=400, detail="Unsupported controller algorithm")
        
        # Use specified controller for this request only
        ctrl = controller_map[request.controller_algorithm]()
        controller_used = request.controller_algorithm
        logger.info(f"Created override {controller_used} controller for patient {patient_id}")
    else:
        # Use the stored controller
        ctrl = patient_data["controller"]
        controller_used = patient_data["controller_algorithm"]
        logger.info(f"Using stored {controller_used} controller for patient {patient_id}")
        
    logger.info(f"Attack scenario - Real glucose: {request.glucose_reading}, Attack glucose: {request.attack_glucose}, Carbs: {request.carbs}")
    
    # Create controller observation with fake glucose data
    ctrl_obs = CtrlObservation(request.attack_glucose)
    
    # Get controller action
    ctrl_action = ctrl.policy(
        observation=ctrl_obs,
        reward=0,
        done=False,
        patient_name=patient.name,
        meal=request.carbs,
        time=request.delta_time
    )

    # Calculate insulin dose - include both basal and bolus
    insulin = ctrl_action.basal + ctrl_action.bolus
    logger.info(f"Controller recommended insulin: {insulin} U (basal: {ctrl_action.basal} U/h, bolus: {ctrl_action.bolus} U)")
    
    # Create patient action - use calculated insulin dose and real carb intake
    patient_action = PatientAction(insulin=insulin, CHO=request.carbs)
    
    # Update patient state - this uses the real glucose data
    patient.step(patient_action)
    
    # Get new glucose level
    glucose = patient.observation.Gsub
    logger.info(f"New glucose level after insulin delivery: {glucose} mg/dL")
    
    return {
        "glucose": glucose,
        "insulin": insulin,
        "attack_scenario": True,
        "real_glucose": request.glucose_reading,
        "attack_glucose": request.attack_glucose,
        "controller_used": controller_used,
        "basal": ctrl_action.basal,
        "bolus": ctrl_action.bolus
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
    
    # Check if this is actually an attack (attack_glucose != glucose_reading)
    is_attack = abs(request.attack_glucose - request.glucose_reading) > 1.0  # Allow small differences due to floating point
    
    # Retrieve patient data
    patient_data = patient_map[patient_id]
    original_patient = patient_data["patient"]
    ctrl = patient_data["controller"]
    
    # Create modified patient model
    modified_patient = ModifiedT1DPatient(original_patient)
    
    # Calculate insulin dose using attack glucose value or real glucose value
    glucose_for_controller = request.attack_glucose if is_attack else request.glucose_reading
    ctrl_obs = CtrlObservation(glucose_for_controller)
    
    # Get controller action
    ctrl_action = ctrl.policy(
        observation=ctrl_obs,
        reward=0,
        done=False,
        patient_name=modified_patient.name,
        meal=request.carbs,
        time=request.delta_time
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
    
    response = {
        "glucose": glucose,
        "insulin": insulin,
        "real_glucose": request.glucose_reading,
        "controller_used": patient_data["controller_algorithm"],
        "glucose_change": glucose - initial_glucose
    }
    
    # Only add attack-specific fields if it's actually an attack
    if is_attack:
        response["attack_scenario"] = True
        response["attack_glucose"] = request.attack_glucose
    else:
        response["attack_scenario"] = False
    
    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)