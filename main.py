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
import time
import os
from sqlalchemy.orm import Session
from database import Base, engine, SessionLocal, Users, Projects, Models, ModelConfigs, ModelLogs, DataArtifacts, SyntheticDataArtifacts, SyntheticQualityReports
# from models import CreateNewProjectRequest, CreateNewProjectResponse, UpdateEmptyProjectRequest, UpdateEmptyProjectResponse, UpdatePendingProjectRequest, UpdatePendingProjectResponse, GenerateSyntheticDataRequest, GenerateSyntheticDataResponse, GetAllProjectsResponse
from models import *
from model_helpers import AutoSyntheticConfigurator, synthetic_model_trainer, synthetic_model_data_generator
from api_helpers import get_model_configuration, start_model_training
from synthetic_quality_report import SyntheticQualityAssurance
from ctgan_model import CTGANER
from dgan_model import DGANER
from google_drive_api import GoogleDriveAPI
import auth
from typing import Annotated
from dotenv import load_dotenv, find_dotenv
import pandas as pd
import ast


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

@app.get("/health", status_code=status.HTTP_200_OK)
def root():
    return {"health": "ok"}

@app.get("/user", status_code=status.HTTP_200_OK)
def user(user: user_dependency, db: db_dependency):
    if user is None:
        raise HTTPException(status_code=401, detail="Authentication Failed")
    return {"User": user}

@app.get("/get_all_data_artifacts")
def get_all_data_artifacts(user: user_dependency, db: db_dependency):
    data_artifacts_db_records = db.query(DataArtifacts).filter(DataArtifacts.user_id == user['id']).all()
    if data_artifacts_db_records is None:
        raise HTTPException(status_code=status.HTTP_204_NO_CONTENT, detail="Data Artifact Not Found!")
    
    data_artifacts = [{"data_artifact_id":data_artifact.data_artifact_id, "name":data_artifact.original_filename, "created_on":data_artifact.created_on} for data_artifact in data_artifacts_db_records]
    
    return GetAllDataArtifactsResponse(
        data_artifacts = data_artifacts
    )

@app.get("/get_all_projects")
def get_all_projects(user: user_dependency, db: db_dependency):
    project_db_records = db.query(Projects).filter(Projects.user_id == user['id']).all()
    if project_db_records is None:
        raise HTTPException(status_code=status.HTTP_204_NO_CONTENT, detail="Projects Not Found!")
    
    projects = [{"project_id":project.project_id, "name":project.name, "description":project.description, "model_type": project.model_type, "status":project.status, "created_on":project.created_on, "updated_on":project.updated_on} for project in project_db_records]
    
    return GetAllProjectsResponse(
        projects = projects
    )

@app.get("/get_project/{project_id}")
def get_project(user: user_dependency, db: db_dependency, project_id: str):
    project_db_record = db.query(Projects).filter(Projects.project_id == project_id).first()
    if project_db_record is None or project_db_record.user_id != user["id"]:
        raise HTTPException(status_code=status.HTTP_204_NO_CONTENT, detail="Specified Project Was Not Found!")
    
    return GetProjectResponse(
        project_id = project_db_record.project_id,
        name = project_db_record.name,
        description = project_db_record.description,
        modelType = project_db_record.model_type,
        status = project_db_record.status,
        modelTraining_time = project_db_record.model_training_time,
        synthetic_quality_score = project_db_record.synthetic_quality_score,
        created_on = project_db_record.created_on,
        updated_on = project_db_record.updated_on
    )

@app.get("/get_model_config/{project_id}")
def get_model_config(user: user_dependency, db: db_dependency, project_id: str):
    project_db_record = db.query(Projects).filter(Projects.project_id == project_id).first()
    if project_db_record is None or project_db_record.user_id != user["id"] or project_db_record.model_config_id is None:
        print("Project")
        raise HTTPException(status_code=status.HTTP_204_NO_CONTENT, detail="Specified Project or it's Model Config Was Not Found!")
    
    model_config_db_record = db.query(ModelConfigs).filter(ModelConfigs.id == project_db_record.model_config_id).first()
    if model_config_db_record is None or model_config_db_record.user_id != user["id"]:
        print("Model Config")
        raise HTTPException(status_code=status.HTTP_204_NO_CONTENT, detail="Specified Project or it's Model Config Was Not Found!")
    
    return GetModelConfigResponse(
        ModelConfig_id = model_config_db_record.model_config_id,
        ModelConfig_data = model_config_db_record.model_config_data,
        created_on = model_config_db_record.created_on
    )

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

    # Step 1: Read the CSV file into a DataFrame
    data_artifact_df = pd.read_csv(data_artifact_local_file_path)

    # Step 2: Iterate through the column names and remove single quotes
    data_artifact_df.columns = [col.replace("'", "") for col in data_artifact_df.columns]

    # Step 3: Save the modified DataFrame back to a CSV file
    data_artifact_df.to_csv(data_artifact_local_file_path, index=False)

    # Get number of rows in the data artifact file
    num_rows = len(data_artifact_df)

    # Upload file to Google Drive Glacier Service
    gdrive_response = google_drive_api.upload_file("data_artifacts", data_artifact_local_file_path)

    if not gdrive_response:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error Uploading Data Artifact. Please try again!")
    
    # Delete the file from the Client Buffer (Background Task)
    background_tasks.add_task(os.remove, data_artifact_local_file_path)

    data_artifact_db_record = DataArtifacts(
        data_artifact_id = data_artifact_id,
        original_filename = file.filename,
        num_rows = num_rows,
        user_id = user['id']
    )
    try:
        db.add(data_artifact_db_record)
        db.commit()
        print("[Database][SUCCESS] Data Artifact Record Successfully Added: ", data_artifact_id)
    except Exception as e:
        print("[Database][ERROR] Error Creating Data Artifact Record:",str(e))
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Error Creating Data Artifact Record!")
        
    return JSONResponse(status_code=200, content={"data_artifact_id": data_artifact_id})

@app.post("/create_new_project")
def create_new_project(user: user_dependency, db: db_dependency, project_data: CreateNewProjectRequest):
    project_id = "proj_" + str(uuid.uuid4())
    project_db_record = Projects(
            project_id = project_id,
            name = project_data.name,
            description = project_data.description,
            status = "empty",
            user_id = user["id"]
        )
    try:
        db.add(project_db_record)
        db.commit()
        print("[Database][SUCCESS] New Project Created Successfully:", project_id)
        return CreateNewProjectResponse(
            project_id = project_id,
            project_name = project_data.name
        )
    except Exception as e:
        print("[Database][ERROR] Failed To Create New Project:",str(e))
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Error Creating New Project Record!")
    
@app.post("/update_empty_project")
def update_empty_project(user: user_dependency, db: db_dependency, project_data: UpdateEmptyProjectRequest, background_tasks: BackgroundTasks):
    model_config_id = "model_config_" + str(uuid.uuid4())
    google_drive_api = GoogleDriveAPI()
    gdrive_response = google_drive_api.download_file("data_artifacts", project_data.data_artifact_id + ".csv")
    
    if not gdrive_response:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error Downloading Data Artifact!")
    
    data_artifact_file_path = gdrive_response
    model_config = get_model_configuration(data_artifact_file_path, project_data.modelType)

    if model_config == None:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error While Generating Model Configuration For Data Artifact: "+project_data.data_artifact_id+" Project ID: "+project_data.project_id)

    # json_dumped_model_config_str = json.dumps(model_config)
    # formatted_json_dumped_model_config_str = json_dumped_model_config_str.replace('"', '\"')

    model_config_db_record = ModelConfigs(
            model_config_id = model_config_id,
            model_config_data = str(model_config),
            project_id = db.query(Projects).filter(Projects.project_id == project_data.project_id).first().id,
            user_id = user["id"]
        )
    try:
        db.add(model_config_db_record)
        db.commit()
        print("[Database][SUCCESS] New Model Config Created Successfully:", model_config_id)
    except Exception as e:
        print("[Database][ERROR] Failed To Create New Model Config:",str(e))
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Error Creating New Model Config Record!")

    project_db_record = db.query(Projects).filter(Projects.project_id == project_data.project_id).first()
    project_db_record.model_type = project_data.modelType
    project_db_record.data_artifact_id = db.query(DataArtifacts).filter(DataArtifacts.data_artifact_id == project_data.data_artifact_id).first().id
    project_db_record.model_config_id = db.query(ModelConfigs).filter(ModelConfigs.model_config_id == model_config_id).first().id
    project_db_record.status = "pending"
    try:
        db.commit()
        print("[Database][SUCCESS] Empty Project Updated Successfully:", project_data.project_id)
    except Exception as e:
        print("[Database][ERROR] Failed To Update Empty Project:", str(e))
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Error Updating Empty Project Record!")
    
    # Delete the file from the Client Buffer (Background Task)
    background_tasks.add_task(os.remove, data_artifact_file_path)
    
    return UpdateEmptyProjectResponse(
        project_id =  project_data.project_id,
        modelConfig_id = model_config_id
    )

@app.post("/update_pending_project")
def update_pending_project(user: user_dependency, db: db_dependency, project_data: UpdatePendingProjectRequest, background_tasks: BackgroundTasks):
    model_log_id = "model_log_" + str(uuid.uuid4())
    background_tasks.add_task(start_model_training, model_log_id, user["id"], project_data)

    return UpdatePendingProjectResponse(
        project_id =  project_data.project_id,
        modelLog_id = model_log_id
    )

@app.post("/generate_synthetic_data")
def generate_synthetic_data(user: user_dependency, db: db_dependency, project_data: GenerateSyntheticDataRequest, background_tasks: BackgroundTasks):
    project_db_record = db.query(Projects).filter(Projects.project_id == project_data.project_id).first()
    if project_db_record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project Not Found!")
    if project_db_record.status != "completed":
        raise HTTPException(status_code=status.HTTP_425_TOO_EARLY, detail="Project Status Not Completed Yet!")
    
    model_db_record = db.query(Models).filter(Models.id == project_db_record.model_id).first()
    model_config_db_record = db.query(ModelConfigs).filter(ModelConfigs.id == project_db_record.model_config_id).first()
    model_file_name = model_db_record.model_id + model_db_record.file_extension

    # Download Model and Encoding file from Google Drive
    google_drive_api = GoogleDriveAPI()
    gdrive_response = google_drive_api.download_file("models", model_file_name)
    
    if not gdrive_response:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error Downloading Model File!")
    
    model_file_path = gdrive_response

    if project_db_record.model_type == "dgan":
        model_encoding_mappings_file_name = "encodings_" + model_db_record.model_id + ".pkl"
        gdrive_response = google_drive_api.download_file("model_encoding_mappings", model_encoding_mappings_file_name)
        if not gdrive_response:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error Downloading Model File!")
        model_encoding_mappings_file_path = gdrive_response

    # Generate Synthetic Data
    synthetic_data_artifact_id = "synthiumAI_" + project_db_record.model_type + "_" + str(uuid.uuid4())
    synthetic_data_artifact_local_file_name = synthetic_data_artifact_id + ".csv"
    synthetic_data_artifact_local_file_path = os.path.join(CLIENT_BUFFER_FOLDER_NAME, synthetic_data_artifact_local_file_name)
    
    synthetic_model_data_generator(
        project_data.num_rows,
        synthetic_data_artifact_local_file_path,
        model_file_path,
        ast.literal_eval(model_config_db_record.model_config_data),
        project_db_record.model_type,
        model_encoding_mappings_file_path if project_db_record.model_type == "dgan" else None
    )

    # Get number of rows in the synthetic data artifact file
    num_rows = len(pd.read_csv(synthetic_data_artifact_local_file_path))

    # Create Synthetic Data Artifact DB Record
    synthetic_data_artifact_db_record = SyntheticDataArtifacts(
            synthetic_data_artifact_id = synthetic_data_artifact_id,
            num_rows = num_rows,
            project_id = project_db_record.id,
            user_id = user["id"]
        )
    try:
        db.add(synthetic_data_artifact_db_record)
        db.commit()
        print("[Database][SUCCESS] New Synthetic Data Artifact Created and Updated Project Training Status Successfully:", synthetic_data_artifact_id)
    except Exception as e:
        print("[Database][ERROR] Failed To Create New Synthetic Data Artifact:",str(e))
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Error Creating New Synthetic Data Artifact Record!")

    # Upload Synthetic Data Artifact to Google Drive
    gdrive_response = google_drive_api.upload_file("synthetic_data_artifacts", synthetic_data_artifact_local_file_path)
    if not gdrive_response:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error Uploading Synthetic Data Artifact File!")
    
    # Delete the file from the Client Buffer (Background Task)
    background_tasks.add_task(os.remove, model_file_path)
    if project_db_record.model_type == "dgan":
        background_tasks.add_task(os.remove, model_encoding_mappings_file_path)
    background_tasks.add_task(os.remove, synthetic_data_artifact_local_file_path)

    print("[SyntheticDataGenerator][SUCCESS] Synthetic Data Generated Successfully!: " + synthetic_data_artifact_id)

    return GenerateSyntheticDataResponse(
        project_id =  project_data.project_id,
        synthetic_data_artifact_id = synthetic_data_artifact_id
    )

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

@app.post("/generate_synthetic_data_old/{key}")
def generate_synthetic_data_old(user: user_dependency, key: str, model: str="ctgan", num_examples: int=1, generate_quality_report=False):
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