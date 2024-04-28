from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import Column, Integer, String, DateTime, BIGINT, Text, Boolean
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
    name = Column(String(length=256))
    description = Column(Text(length=1000))
    model_type = Column(String(length=256))
    status = Column(String(length=256))
    training_time = Column(String(length=256))
    synthetic_quality_score = Column(Integer)
    model_id = Column(String(length=256), unique=True)
    model_config_id = Column(String(length=256), unique=True)
    user_id = Column(Integer)
    data_artifact_id = Column(Integer)
    created_on = Column(DateTime(timezone=True), server_default=func.current_timestamp())
    updated_on = Column(DateTime(timezone=True), server_default=func.current_timestamp(), onupdate=func.current_timestamp())

class Models(Base):
    __tablename__ = 'models'
    
    id = Column(Integer, primary_key=True)
    model_id = Column(String(length=256), unique=True)
    file_extension = Column(String(length=256), server_default=".pkl")
    model_type = Column(String(length=256))
    project_id = Column(Integer, unique=True)
    user_id = Column(Integer)
    created_on = Column(DateTime(timezone=True), server_default=func.current_timestamp())

class ModelConfigs(Base):
    __tablename__ = 'model_configs'
    
    id = Column(Integer, primary_key=True)
    model_config_id = Column(String(length=256), unique=True)
    file_extension = Column(String(length=256), server_default=".json")
    project_id = Column(Integer, unique=True)
    user_id = Column(Integer)
    created_on = Column(DateTime(timezone=True), server_default=func.current_timestamp())

class DataArtifacts(Base):
    __tablename__ = 'data_artifacts'
    
    id = Column(Integer, primary_key=True)
    data_artifact_id = Column(String(length=256),unique=True)
    file_extension = Column(String(length=256), server_default=".csv")
    original_filename = Column(String(length=256))
    user_id = Column(Integer)
    created_on = Column(DateTime(timezone=True), server_default=func.current_timestamp())

class SyntheticData(Base):
    __tablename__ = 'synthetic_data'
    
    id = Column(Integer, primary_key=True)
    synthetic_data_id = Column(String(length=256),unique=True)
    file_extension = Column(String(length=256), server_default=".csv")
    user_id = Column(Integer)
    project_id = Column(Integer)
    created_on = Column(DateTime(timezone=True), server_default=func.current_timestamp())

class SyntheticQualityReports(Base):
    __tablename__ = 'synthetic_quality_reports'
    
    id = Column(Integer, primary_key=True)
    synthetic_data_id = Column(String(length=256),unique=True)
    file_extension = Column(String(length=256), server_default=".csv")
    user_id = Column(Integer)
    created_on = Column(DateTime(timezone=True), server_default=func.current_timestamp())

engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
# engine = create_engine(SQLALCHEMY_DATABASE_URL)

Base.metadata.create_all(bind=engine)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
