import os
import json
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, List
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel

app = FastAPI(title="Dynamic ML Classification Backend")

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins during development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
MODEL_PATH = os.path.join(ARTIFACT_DIR, "model.joblib")
METADATA_PATH = os.path.join(ARTIFACT_DIR, "metadata.json")

# Global state for loaded model and metadata
loaded_model = None
loaded_metadata = None

def load_model_and_metadata():
    global loaded_model, loaded_metadata
    if os.path.exists(MODEL_PATH) and os.path.exists(METADATA_PATH):
        try:
            loaded_model = joblib.load(MODEL_PATH)
            with open(METADATA_PATH, "r") as f:
                loaded_metadata = json.load(f)
            print(f"Successfully loaded model: {loaded_metadata.get('model_name', 'Unknown')}")
            return True
        except Exception as e:
            print(f"Error loading model/metadata: {e}")
            loaded_model = None
            loaded_metadata = None
            return False
    else:
        loaded_model = None
        loaded_metadata = None
        print("No model or metadata found in artifacts.")
        return False

# Initialize directory and load on start
os.makedirs(ARTIFACT_DIR, exist_ok=True)
load_model_and_metadata()

class PredictionInput(BaseModel):
    features: Dict[str, Any]

@app.get("/api/model-info")
def get_model_info():
    if loaded_model is None or loaded_metadata is None:
        return {"loaded": False, "message": "No model artifact loaded."}
    
    return {
        "loaded": True,
        "model_name": loaded_metadata.get("model_name", "Unnamed Model"),
        "model_type": loaded_metadata.get("model_type", "Unknown"),
        "features": loaded_metadata.get("features", []),
        "target": loaded_metadata.get("target", "Target"),
        "classes": [str(c) for c in loaded_metadata.get("classes", [])],
        "metrics": loaded_metadata.get("metrics", {})
    }

@app.post("/api/predict")
def predict(payload: PredictionInput):
    global loaded_model, loaded_metadata
    if loaded_model is None or loaded_metadata is None:
        raise HTTPException(status_code=400, detail="No model artifact is currently loaded.")
    
    input_data = payload.features
    features_config = loaded_metadata.get("features", [])
    
    # Preprocess inputs dynamically based on type
    try:
        # Determine if it's text classification or tabular
        is_text_model = len(features_config) == 1 and features_config[0].get("type") == "text"
        
        if is_text_model:
            feature_name = features_config[0]["name"]
            text_value = str(input_data.get(feature_name, ""))
            X = [text_value]
        else:
            feature_names = [f["name"] for f in features_config]
            processed = {}
            for f in features_config:
                name = f["name"]
                val = input_data.get(name)
                
                if f.get("type") == "numerical":
                    if val is None or val == "":
                        processed[name] = np.nan
                    else:
                        processed[name] = float(val)
                else:
                    processed[name] = val
            
            # Create dataframe and align columns
            X = pd.DataFrame([processed])[feature_names]
            
        # Run prediction
        prediction = loaded_model.predict(X)[0]
        prediction_str = str(prediction)
        
        # Try to map class index to class name label
        classes = loaded_metadata.get("classes", [])
        try:
            pred_idx = int(prediction)
            if 0 <= pred_idx < len(classes):
                prediction_str = str(classes[pred_idx])
        except (ValueError, TypeError):
            pass
        
        # Run probabilities if supported
        probabilities = {}
        if hasattr(loaded_model, "predict_proba"):
            try:
                proba = loaded_model.predict_proba(X)[0]
                classes = loaded_metadata.get("classes", [])
                probabilities = {str(c): float(p) for c, p in zip(classes, proba)}
            except Exception as e:
                print(f"Probabilities error: {e}")
                
        return {
            "prediction": prediction_str,
            "probabilities": probabilities,
            "success": True
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/api/predict-batch")
async def predict_batch(file: UploadFile = File(...)):
    global loaded_model, loaded_metadata
    if loaded_model is None or loaded_metadata is None:
        raise HTTPException(status_code=400, detail="No model artifact is loaded.")
    
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Uploaded file must be a CSV.")
        
    try:
        df = pd.read_csv(file.file)
        features_config = loaded_metadata.get("features", [])
        is_text_model = len(features_config) == 1 and features_config[0].get("type") == "text"
        
        if is_text_model:
            feature_name = features_config[0]["name"]
            if feature_name not in df.columns:
                raise HTTPException(status_code=400, detail=f"CSV must contain column: {feature_name}")
            X = df[feature_name].astype(str).tolist()
        else:
            feature_names = [f["name"] for f in features_config]
            # Verify all expected features are present in CSV
            missing = [f for f in feature_names if f not in df.columns]
            if missing:
                raise HTTPException(
                    status_code=400, 
                    detail=f"CSV is missing the following columns: {', '.join(missing)}"
                )
            X = df[feature_names].copy()
            # Convert numerical columns to float to match schema
            for f in features_config:
                if f.get("type") == "numerical":
                    X[f["name"]] = pd.to_numeric(X[f["name"]], errors='coerce')
        
        # Predict
        predictions = loaded_model.predict(X)
        
        # Try to map class index to class name labels
        classes = loaded_metadata.get("classes", [])
        mapped_predictions = []
        for pred in predictions:
            pred_str = str(pred)
            try:
                pred_idx = int(pred)
                if 0 <= pred_idx < len(classes):
                    pred_str = str(classes[pred_idx])
            except (ValueError, TypeError):
                pass
            mapped_predictions.append(pred_str)
            
        df["prediction"] = mapped_predictions
        
        # Add probability columns if supported
        if hasattr(loaded_model, "predict_proba"):
            try:
                proba = loaded_model.predict_proba(X)
                classes = loaded_metadata.get("classes", [])
                for i, c in enumerate(classes):
                    df[f"prob_{c}"] = proba[:, i]
            except Exception as e:
                print(f"Batch probability mapping error: {e}")
                
        # Save temp output CSV
        temp_out = os.path.join(ARTIFACT_DIR, "batch_predictions_output.csv")
        df.to_csv(temp_out, index=False)
        
        return FileResponse(
            temp_out, 
            media_type="text/csv", 
            filename="predictions_output.csv"
        )
        
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

@app.post("/api/upload-model")
async def upload_model(
    model: UploadFile = File(...),
    metadata: UploadFile = File(...)
):
    try:
        # Write model file
        with open(MODEL_PATH, "wb") as f:
            f.write(await model.read())
            
        # Write metadata file
        with open(METADATA_PATH, "wb") as f:
            f.write(await metadata.read())
            
        # Reload
        success = load_model_and_metadata()
        if not success:
            # Clean up corrupted/invalid files
            if os.path.exists(MODEL_PATH):
                os.remove(MODEL_PATH)
            if os.path.exists(METADATA_PATH):
                os.remove(METADATA_PATH)
            raise HTTPException(
                status_code=400, 
                detail="Uploaded model artifacts could not be loaded successfully. Ensure they are scikit-learn and valid metadata."
            )
            
        return {"success": True, "message": f"Successfully uploaded and loaded model: {loaded_metadata.get('model_name')}"}
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.delete("/api/delete-model")
def delete_model():
    global loaded_model, loaded_metadata
    try:
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)
        if os.path.exists(METADATA_PATH):
            os.remove(METADATA_PATH)
            
        loaded_model = None
        loaded_metadata = None
        return {"success": True, "message": "Model artifacts deleted."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete model artifacts: {str(e)}")

@app.get("/api/charts/{chart_name}")
def get_chart(chart_name: str):
    chart_path = os.path.join(ARTIFACT_DIR, chart_name)
    if os.path.exists(chart_path):
        return FileResponse(chart_path)
    raise HTTPException(status_code=404, detail="Chart not found")
