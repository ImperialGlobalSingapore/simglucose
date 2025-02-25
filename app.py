from collections import namedtuple
from uuid import uuid4
import json
from typing import Dict, Optional, Any

import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from simglucose.controller.basal_bolus_ctrller import BBController
from simglucose.controller.pid_ctrller import PIDController
from simglucose.patient.t1dpatient import T1DPatient

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

class StepResponse(BaseModel):
    glucose: float
    insulin: float

# Create FastAPI app
app = FastAPI(title="Glucose Simulation API")

# Store patients in memory (consider a database for production)
patient_map: Dict[str, T1DPatient] = {}

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
    
    # Create controller observation
    ctrl_obs = CtrlObservation(request.glucose_reading)
    
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
    
    return {
        "glucose": glucose,
        "insulin": insulin
    }

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)