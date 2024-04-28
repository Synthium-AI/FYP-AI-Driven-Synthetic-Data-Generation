# API Dependencies
import uvicorn
from fastapi import FastAPI, status, File, UploadFile, HTTPException, Depends, BackgroundTasks
from fastapi.responses import StreamingResponse, HTMLResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.models import HTTPBase
from concurrent.futures import ThreadPoolExecutor
from pydantic import BaseModel
from typing import Optional
from io import BytesIO
import uuid
import shutil
import json
import os
from sqlalchemy.orm import Session
from database import Base, engine, SessionLocal, Users, Projects, Models, ModelConfigs, DataArtifacts, SyntheticData, SyntheticQualityReports
from helpers import AutoSyntheticConfigurator
from synthetic_quality_report import SyntheticQualityAssurance
from ctgan_model import CTGANER
from dgan_model import DGANER
from google_drive_api import GoogleDriveAPI
import auth
from typing import Annotated
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

CLIENT_BUFFER_FOLDER_NAME = os.getenv("CLIENT_BUFFER_FOLDER_NAME")

app = FastAPI()
app.include_router(auth.router)

origins = [
    "http://localhost",
    "http://localhost:5173"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# models.Base.metadata.create_all(bind=engine)
Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

db_dependency = Annotated[Session, Depends(get_db)]
user_dependency = Annotated[dict, Depends(auth.get_current_user)]

@app.get("/", status_code=status.HTTP_200_OK)
def root():
    return {"health": "ok"}

@app.get("/user", status_code=status.HTTP_200_OK)
def user(user: user_dependency, db: db_dependency, background_tasks: BackgroundTasks):
    if user is None:
        raise HTTPException(status_code=401, detail="Authentication Failed")
    return {"User": user}

@app.post("/create_new_project")
def create_new_project(user: user_dependency, db: db_dependency, key: str, model="ctgan"):

@app.post("/upload_data_artifact")
async def upload_data_artifact(user: user_dependency, db: db_dependency, background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    # Generate a unique ID for this upload
    data_artifact_id = "data_" + str(uuid.uuid4())
    google_drive_api = GoogleDriveAPI()

    # Define the file path
    data_artifact_local_file_name = data_artifact_id + ".csv"
    data_artifact_local_file_path = os.path.join(CLIENT_BUFFER_FOLDER_NAME, data_artifact_local_file_name)
    
    # Save the uploaded file to the Client Buffer
    with open(data_artifact_local_file_path, "wb+") as file_object:
        file_object.write(await file.read())

    # Upload file to Google Drive Glacier Service
    gdrive_response = google_drive_api.upload_file("data_artifacts", data_artifact_local_file_path)

    # Delete the file from the Client Buffer (Background Task)
    background_tasks.add_task(os.remove, data_artifact_local_file_path)

    if gdrive_response:
        data_artifact_db_record = DataArtifacts(
            data_artifact_id = data_artifact_id,
            original_filename = file.filename,
            user_id = user['id']
        )
        try:
            db.add(data_artifact_db_record)
            db.commit()
            print("[Database][SUCCESS] Data Artifact Record Successfully Added: ", data_artifact_id)
        except Exception as e:
            print("[Database][ERROR] Error Creating Data Artifact Record:",str(e))
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Error Creating Data Artifact Record!")
    else:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error Uploading Data Artifact. Please try again!")
        
    return JSONResponse(status_code=200, content={"data_artifact_id": data_artifact_id})

@app.get("/config/{key}")
def get_config(user: user_dependency, key: str, model="ctgan"):
    folder_path = os.path.join("client", key)
    # Verify the folder exists
    if not os.path.exists(folder_path):
        raise HTTPException(status_code=404, detail="Key not found")

    # Find the CSV file in the folder
    csv_file = None
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):
            csv_file = file_name
            break

    if csv_file is None:
        raise HTTPException(status_code=404, detail="CSV file not found under the provided key")

    file_path = os.path.join(folder_path, csv_file)

    configurator = AutoSyntheticConfigurator(file_path)
    if model == "ctgan":
        dgan_config = configurator.get_ctgan_config()
    if model == "dgan":
        dgan_config = configurator.get_dgan_config()

    return JSONResponse(status_code=200, content=dgan_config)

@app.post("/train_model")
def train_model(user: user_dependency, background_tasks: BackgroundTasks, key: str, config: dict, model: str="ctgan"):
    folder_path = os.path.join("client", key)
    if not os.path.exists(folder_path):
        raise HTTPException(status_code=404, detail="Project key not found")

    csv_file = None
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):
            csv_file = file_name
            break

    if csv_file is None:
        raise HTTPException(status_code=404, detail="CSV file not found under the provided key")

    file_path = os.path.join(folder_path, csv_file)

    def train_and_save_model(model, file_path, config, folder_path):
        if model == "ctgan":
            dganer = CTGANER(file_path, config)
            dganer.train()
            dganer.save(folder_path)
        elif model == "dgan":
            dganer = DGANER(file_path, config)
            dganer.train()
            dganer.save(folder_path)

    background_tasks.add_task(train_and_save_model, model, file_path, config, folder_path)

    return {"model":model, "message": "Training started", "key": key}

@app.post("/generate_synthetic_data/{key}")
def generate_synthetic_data(user: user_dependency, key: str, model: str="ctgan", num_examples: int=1, generate_quality_report=False):
    project_path = os.path.join("client", key)
    if model == "ctgan":
        model_path = os.path.join(project_path, "model.pkl")
        main_config_path = os.path.join(project_path, "ctgan_config.json")
        # Check if model and encoding mappings exist
        if not os.path.exists(model_path) or not os.path.exists(main_config_path):
            return JSONResponse(status_code=404, content={"message": "Model is not trained yet or missing files"})
    elif model == "dgan":
        model_path = os.path.join(project_path, "model.pt")
        main_config_path = os.path.join(project_path, "dgan_config.json")
        encoding_mappings_path = os.path.join(project_path, "encoding_mappings.pkl")
        # Check if model and encoding mappings exist
        if not os.path.exists(model_path) or not os.path.exists(main_config_path) or not os.path.exists(encoding_mappings_path):
            return JSONResponse(status_code=404, content={"message": "Model is not trained yet or missing files"})
    else:
        return JSONResponse(status_code=404, content={"message": "Model is not trained yet or missing files"})
    
    # Find the original CSV file to determine the name
    original_csv = None
    for file_name in os.listdir(project_path):
        if file_name.endswith('.csv'):
            original_csv = file_name
            break

    if original_csv is None:
        return JSONResponse(status_code=404, content={"message": "Original CSV file not found"})

    exports_path = os.path.join(project_path, "exports")
    os.makedirs(exports_path, exist_ok=True)

    # Determine the new filename with versioning
    base_name = original_csv.rsplit('.', 1)[0]
    version = 1
    new_filename = f"{base_name}_{version}.csv"
    while os.path.exists(os.path.join(exports_path, new_filename)):
        version += 1
        new_filename = f"{base_name}_{version}.csv"

    # Initialize DGANER with load_mode
    original_csv_path = os.path.join(project_path, original_csv)
    if model == "ctgan":
        model_agent = CTGANER(file_path=original_csv_path, main_config="load_mode", project_directory_path=project_path)
    elif model == "dgan":    
        model_agent = DGANER(file_path=original_csv_path, main_config="load_mode", project_directory_path=project_path)

    # Generate and save the synthetic data
    model_agent.generate_synthetic_data_csv(os.path.join(exports_path, new_filename), num_examples=num_examples)

    if generate_quality_report:
        quality_manager = SyntheticQualityAssurance(original_csv_path,os.path.join(exports_path, new_filename),model=model)
        quality_manager.generate_report(project_path)

    # Assuming you want to return the file for download
    return FileResponse(path=os.path.join(exports_path, new_filename), filename=new_filename)

@app.get("/get_synthetic_quality_report/{key}")
def get_synthetic_quality_report(user: user_dependency, key: str):
    project_path = os.path.join("client", key)
    report_path = os.path.join(project_path, "synthetic_data_quality_report.json")
    # Check if report exists
    if not os.path.exists(project_path):
        return JSONResponse(status_code=404, content={"message": "Invalid Key!"})
    if not os.path.exists(report_path):
        return JSONResponse(status_code=404, content={"message": "Quality Report does not exists!"})
    
    with open(report_path, "r") as json_file:
        quality_report = json.load(json_file)

    return JSONResponse(status_code=200, content=quality_report)

if __name__ == "__main__":
    uvicorn.run(app)