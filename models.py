import datetime
from pydantic import BaseModel

# Create New Project Models
class CreateNewProjectRequest(BaseModel):
    name: str = "Untitled Project"
    description: str | None

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

class GetAllDataArtifactsResponse(BaseModel):
    data_artifacts: list

class SyntheticDataArtifactMetadata(BaseModel):
    project_id: str
    synthetic_data_artifact_id: str
    file_extension: str
    num_rows: int
    created_on: datetime.datetime

class GetProjectDataArtifactsMetadataResponse(BaseModel):
    synthetic_data_artifacts: list[SyntheticDataArtifactMetadata]

class GetAllProjectsResponse(BaseModel):
    projects: list

class GetProjectResponse(BaseModel):
    project_id: str
    name: str
    description: str | None
    modelType: str | None
    status: str
    modelTraining_time: float | None
    synthetic_quality_score: float | None
    created_on: datetime.datetime
    updated_on: datetime.datetime

class GetDataArtifactMetadataResponse(BaseModel):
    project_id: str
    data_artifact_id: str
    original_filename: str
    file_extension: str
    num_rows: int
    created_on: datetime.datetime

class GetModelConfigResponse(BaseModel):
    ModelConfig_id: str
    ModelConfig_data: str
    created_on: datetime.datetime
