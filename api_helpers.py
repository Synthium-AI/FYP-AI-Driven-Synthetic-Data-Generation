from fastapi import status, HTTPException
from database import SessionLocal, Projects, Models, ModelConfigs, ModelLogs, DataArtifacts, SyntheticDataArtifacts, SyntheticQualityReports
from model_helpers import AutoSyntheticConfigurator, synthetic_model_trainer, synthetic_model_data_generator
from synthetic_quality_report import SyntheticQualityAssurance
from google_drive_api import GoogleDriveAPI
from contextlib import contextmanager
from dotenv import load_dotenv, find_dotenv
import pandas as pd
import uuid
import time
import sys
import ast
import os



load_dotenv(find_dotenv())

CLIENT_BUFFER_FOLDER_NAME = os.getenv("CLIENT_BUFFER_FOLDER_NAME")

def get_model_configuration(data_artifact_file_path, model_type):
    try:
        configurator = AutoSyntheticConfigurator(data_artifact_file_path)
        if model_type == "ctgan":
            model_config = configurator.get_ctgan_config()
        if model_type == "dgan":
            model_config = configurator.get_dgan_config()
        print("[ModelConfigGenerator][SUCCESS] Successfully Generated Model Config for: {}_model {}".format(model_type,data_artifact_file_path))
        return model_config
    except Exception as e:
        print("[ModelConfigGenerator][ERROR] Error generating model config:",str(e))
        return None

@contextmanager
def log_to_database(db_session, model_log_db_record):
    log_buffer = []
    old_stdout, old_stderr = sys.stdout, sys.stderr

    class LogCapturer:
        def write(self, data):
            log_buffer.append(data)
            model_log_data = ''.join(log_buffer)
            model_log_db_record.model_log_data = model_log_data
            db_session.commit()

        def flush(self):
            pass

    sys.stdout, sys.stderr = LogCapturer(), LogCapturer()
    try:
        yield
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr

def start_model_training(model_log_id, user_id, project_data):
    db = SessionLocal()
    try:
        project_db_record = db.query(Projects).filter(Projects.project_id == project_data.project_id).first()
        project_db_record.status = "training"

        model_id = project_db_record.model_type + "_model_" + str(uuid.uuid4())
        if project_db_record.model_type == "ctgan":
            model_file_path = os.path.join(CLIENT_BUFFER_FOLDER_NAME, model_id + ".pkl")
        elif project_db_record.model_type == "dgan":
            model_file_path = os.path.join(CLIENT_BUFFER_FOLDER_NAME, model_id + ".pt")
            model_encoding_mappings_path = os.path.join(CLIENT_BUFFER_FOLDER_NAME, "encodings_" + model_id + ".pkl")

        data_artifact_db_record = db.query(DataArtifacts).filter(DataArtifacts.id == project_db_record.data_artifact_id).first()
        model_config_db_record = db.query(ModelConfigs).filter(ModelConfigs.id == project_db_record.model_config_id).first()
        model_config_db_record.model_config_data = project_data.modelConfig_data
        google_drive_api = GoogleDriveAPI()
        gdrive_response = google_drive_api.download_file("data_artifacts", data_artifact_db_record.data_artifact_id + data_artifact_db_record.file_extension)
        
        if not gdrive_response:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error Downloading Data Artifact!")
        
        data_artifact_file_path = gdrive_response

        model_db_record = Models(
                model_id = model_id,
                file_extension = ".pkl" if project_db_record.model_type == "ctgan" else ".pt",
                model_type = project_db_record.model_type,
                project_id = project_db_record.id,
                user_id = user_id
            )
        try:
            db.add(model_db_record)
            db.commit()
            print("[Database][SUCCESS] New Model Created and Updated Project Training Status Successfully:", model_id)
        except Exception as e:
            print("[Database][ERROR] Failed To Create New Model:",str(e))
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Error Creating New Model Record!")
        
        model_log_db_record = ModelLogs(
                model_log_id = model_log_id,
                model_log_data = "----- Model Training Started -----\n",
                project_id = project_db_record.id,
                user_id = user_id
            )
        try:
            db.add(model_log_db_record)
            db.commit()
            print("[Database][SUCCESS] New Model Log Created Successfully:", model_log_id)
        except Exception as e:
            print("[Database][ERROR] Failed To Create New Model Log:",str(e))
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Error Creating New Model Log Record!")
        
        project_db_record.model_id = db.query(Models).filter(Models.model_id == model_id).first().id
        project_db_record.model_log_id = db.query(ModelLogs).filter(ModelLogs.model_log_id == model_log_id).first().id
        try:
            db.commit()
            print("[Database][SUCCESS] Pending Project Updated Successfully:", project_data.project_id)
        except Exception as e:
            print("[Database][ERROR] Failed To Update Pending Project:", str(e))
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Error Updating Pending Project Record!")
        
        start_time = time.time()
        # Model Training Process Starts Here and Ends wiht Saving them to Client Buffer
        with log_to_database(db, model_log_db_record):
            if project_db_record.model_type == "ctgan":
                synthetic_model_trainer(
                    data_artifact_file_path,
                    ast.literal_eval(model_config_db_record.model_config_data),
                    project_db_record.model_type,
                    model_file_path
                )
            elif project_db_record.model_type == "dgan":
                synthetic_model_trainer(
                    data_artifact_file_path,
                    ast.literal_eval(model_config_db_record.model_config_data),
                    project_db_record.model_type,
                    model_file_path,
                    model_encoding_mappings_path
                )
        model_training_time = time.time() - start_time
        # Upload Model files and Encoding mappings to Google Drive
        # Upload Model
        gdrive_response = google_drive_api.upload_file("models", model_file_path)
        if not gdrive_response:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error Uploading Model File!")
        # Upload Model Encoding Mappings
        if project_db_record.model_type == "dgan":
            gdrive_response = google_drive_api.upload_file("model_encoding_mappings", model_encoding_mappings_path)
            if not gdrive_response:
                raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error Uploading Model Encoding Mappings!")
        
        # Generate Synthetic Data
        synthetic_data_artifact_id = "synthiumAI_" + project_db_record.model_type + "_" + str(uuid.uuid4())
        synthetic_data_artifact_local_file_name = synthetic_data_artifact_id + ".csv"
        synthetic_data_artifact_local_file_path = os.path.join(CLIENT_BUFFER_FOLDER_NAME, synthetic_data_artifact_local_file_name)
        
        synthetic_model_data_generator(
            data_artifact_db_record.num_rows,
            synthetic_data_artifact_local_file_path,
            model_file_path,
            ast.literal_eval(model_config_db_record.model_config_data),
            project_db_record.model_type,
            model_encoding_mappings_path if project_db_record.model_type == "dgan" else None
        )

        # Get number of rows in the synthetic data artifact file
        num_rows = len(pd.read_csv(synthetic_data_artifact_local_file_path))

        # Create Synthetic Data Artifact DB Record
        synthetic_data_artifact_db_record = SyntheticDataArtifacts(
                synthetic_data_artifact_id = synthetic_data_artifact_id,
                num_rows = num_rows,
                project_id = project_db_record.id,
                user_id = user_id
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
        
        # Generate Synthetic Quality Report
        quality_manager = SyntheticQualityAssurance(data_artifact_file_path, synthetic_data_artifact_local_file_path, project_db_record.model_type)
        synthetic_quality_report_data = quality_manager.generate_report()
        
        # Create Synthetic Quality Report DB Record
        synthetic_quality_report_id = "synthetic_quality_report_" + str(uuid.uuid4())
        synthetic_quality_report_db_record = SyntheticQualityReports(
                synthetic_quality_report_id = synthetic_quality_report_id,
                synthetic_quality_report_data = str(synthetic_quality_report_data),
                project_id = project_db_record.id,
                user_id = user_id
            )
        try:
            db.add(synthetic_quality_report_db_record)
            db.commit()
            print("[Database][SUCCESS] New Synthetic Quality Report Created and Updated Project Training Status Successfully:", synthetic_quality_report_id)
        except Exception as e:
            print("[Database][ERROR] Failed To Create New Synthetic Quality Report:",str(e))
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Error Creating New Synthetic Quality Report Record!")

        # Update Project DB Record with New Information
        project_db_record.status = "completed"
        project_db_record.synthetic_quality_report_id = db.query(SyntheticQualityReports).filter(SyntheticQualityReports.synthetic_quality_report_id == synthetic_quality_report_id).first().id
        project_db_record.synthetic_quality_score = synthetic_quality_report_data["overall_score"]
        project_db_record.model_training_time = model_training_time
        model_db_record.model_training_time = model_training_time
        try:
            db.commit()
            print("[Database][SUCCESS] Pending Project Finally Updated Successfully:", project_data.project_id)
        except Exception as e:
            print("[Database][ERROR] Failed To Update Pending Project Finally:", str(e))
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Error Updating Pending Project Finally Record!")

        # Delete the files from the Client Buffer
        os.remove(data_artifact_file_path)
        os.remove(model_file_path)
        if project_db_record.model_type == "dgan":
            os.remove(model_encoding_mappings_path)
        os.remove(synthetic_data_artifact_local_file_path)
        
        print("[BackgroundTaskModelTrainer][SUCCESS] Project Completed Successfully! Project ID: " + project_data.project_id)

    except Exception as e:
        print("[BackgroundTaskModelTrainer][ERROR] Failed To Train Model:", str(e))
        project_db_record = db.query(Projects).filter(Projects.project_id == project_data.project_id).first()
        project_db_record.status = "training_failed"
        db.commit()