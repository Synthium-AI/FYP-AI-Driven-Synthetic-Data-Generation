from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import Column, Integer, Float, String, DateTime, BIGINT, Text, Boolean
from sqlalchemy.sql import func
from dotenv import load_dotenv, find_dotenv
import os

load_dotenv(find_dotenv()) # read local .env file

Base = declarative_base()

SQLALCHEMY_DATABASE_URL = "sqlite:///database.sqlite"
# SQLALCHEMY_DATABASE_URL = "mysql+pymysql://{}:{}@{}:{}/{}?charset=utf8mb4".format(
#     os.getenv("AWS_RDS_DB_MASTER_USERNAME"),
#     os.getenv("AWS_RDS_DB_MASTER_PASS"),
#     os.getenv("AWS_RDS_DB_HOST"),
#     os.getenv("AWS_RDS_DB_PORT"),
#     os.getenv("AWS_RDS_DB_NAME"),
# )

# Models
class Users(Base):
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(length=256), unique=True)
    hashed_password = Column(String(length=256))
    first_name = Column(String(length=256))
    last_name = Column(String(length=256))

class Projects(Base):
    __tablename__ = 'projects'
    
    id = Column(Integer, primary_key=True)
    project_id = Column(String(length=256), unique=True)
    name = Column(String(length=256))
    description = Column(Text(length=10000))
    model_type = Column(String(length=256))
    status = Column(String(length=256))
    model_training_time = Column(Float())
    synthetic_quality_score = Column(Float)
    model_id = Column(Integer, unique=True)
    model_config_id = Column(Integer, unique=True)
    model_log_id = Column(Integer, unique=True)
    user_id = Column(Integer)
    data_artifact_id = Column(Integer)
    synthetic_quality_report_id = Column(Integer)
    created_on = Column(DateTime(timezone=True), server_default=func.current_timestamp())
    updated_on = Column(DateTime(timezone=True), server_default=func.current_timestamp(), onupdate=func.current_timestamp())

class Models(Base):
    __tablename__ = 'models'
    
    id = Column(Integer, primary_key=True)
    model_id = Column(String(length=256), unique=True)
    file_extension = Column(String(length=256), server_default=".pkl") # OR ".pt" if DGAN Model Type
    model_type = Column(String(length=256))
    model_training_time = Column(Float())
    project_id = Column(Integer, unique=True)
    user_id = Column(Integer)
    created_on = Column(DateTime(timezone=True), server_default=func.current_timestamp())

class ModelConfigs(Base):
    __tablename__ = 'model_configs'
    
    id = Column(Integer, primary_key=True)
    model_config_id = Column(String(length=256), unique=True)
    model_config_data = Column(Text(length=10000))
    project_id = Column(Integer, unique=True)
    user_id = Column(Integer)
    created_on = Column(DateTime(timezone=True), server_default=func.current_timestamp())

class ModelLogs(Base):
    __tablename__ = 'model_logs'
    
    id = Column(Integer, primary_key=True)
    model_log_id = Column(String(length=256), unique=True)
    model_log_data = Column(Text(length=10000))
    project_id = Column(Integer, unique=True)
    user_id = Column(Integer)
    created_on = Column(DateTime(timezone=True), server_default=func.current_timestamp())
    updated_on = Column(DateTime(timezone=True), server_default=func.current_timestamp(), onupdate=func.current_timestamp())

class DataArtifacts(Base):
    __tablename__ = 'data_artifacts'
    
    id = Column(Integer, primary_key=True)
    data_artifact_id = Column(String(length=256),unique=True)
    file_extension = Column(String(length=256), server_default=".csv")
    original_filename = Column(String(length=256))
    num_rows = Column(Integer)
    user_id = Column(Integer)
    created_on = Column(DateTime(timezone=True), server_default=func.current_timestamp())

class SyntheticDataArtifacts(Base):
    __tablename__ = 'synthetic_data_artifacts'
    
    id = Column(Integer, primary_key=True)
    synthetic_data_artifact_id = Column(String(length=256),unique=True)
    file_extension = Column(String(length=256), server_default=".csv")
    num_rows = Column(Integer)
    user_id = Column(Integer)
    project_id = Column(Integer)
    created_on = Column(DateTime(timezone=True), server_default=func.current_timestamp())

class SyntheticQualityReports(Base):
    __tablename__ = 'synthetic_quality_reports'
    
    id = Column(Integer, primary_key=True)
    synthetic_quality_report_id = Column(String(length=256),unique=True)
    synthetic_quality_report_data = Column(Text(length=10000))
    user_id = Column(Integer)
    project_id = Column(Integer)
    created_on = Column(DateTime(timezone=True), server_default=func.current_timestamp())

engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
# engine = create_engine(SQLALCHEMY_DATABASE_URL)

Base.metadata.create_all(bind=engine)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
