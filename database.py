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

class ChatbotAgents(Base):
    __tablename__ = 'chatbot_agents'
    
    id = Column(Integer, primary_key=True)
    chatbot_name = Column(String(length=256))
    chatbot_key = Column(String(length=256),unique=True)
    system_prompt = Column(Text(length=1000))
    vb_loaded = Column(Boolean)
    created_by_user_id = Column(Integer)
    created_on = Column(DateTime(timezone=True), server_default=func.current_timestamp())
    updated_on = Column(DateTime(timezone=True), server_default=func.current_timestamp(), onupdate=func.current_timestamp())

class ChatSessions(Base):
    __tablename__ = 'chat_sessions'
    
    id = Column(Integer, primary_key=True)
    chat_session_id = Column(String(length=256),unique=True)
    chatbot_key = Column(String(length=256))
    status = Column(String(length=256))
    chat_data = Column(Text(length=1000))
    contact_info = Column(Text(length=256))
    started_on = Column(DateTime(timezone=True), server_default=func.current_timestamp())
    ended_on = Column(DateTime(timezone=True), server_default=func.current_timestamp())


engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
# engine = create_engine(SQLALCHEMY_DATABASE_URL)

Base.metadata.create_all(bind=engine)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
