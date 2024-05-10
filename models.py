import datetime
from pydantic import BaseModel

# Create New Project Models
class CreateNewProjectRequest(BaseModel):
    name: str = "Untitled Project"
    description: str | None
    user_id: int

class CreateNewProjectResponse(BaseModel):
    project_id: str
    project_name: str

# Empty Project Updation Models
class UpdateEmptyProjectRequest(BaseModel):
    project_id: str
    modelType: str
    data_artifact_id: str

class UpdateEmptyProjectResponse(BaseModel):
    project_id: str
    modelConfig_id: str

# Pending Project Updation Models
class UpdatePendingProjectRequest(BaseModel):
    project_id: str
    modelConfig_data: str

class UpdatePendingProjectResponse(BaseModel):
    project_id: str
    modelLog_id: str

# Generate Synthetic Data Models
class GenerateSyntheticDataRequest(BaseModel):
    project_id: str
    num_rows: str

class GenerateSyntheticDataResponse(BaseModel):
    project_id: str
    synthetic_data_artifact_id: str

class GetAllProjectsResponse(BaseModel):
    projects: list

class GetProjectResponse(BaseModel):
    project_id: str
    name: str
    description: str
    modelType: str
    status: str
    modelTraining_time: float
    synthetic_quality_score: float
    created_on: datetime.datetime
    updated_on: datetime.datetime