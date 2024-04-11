from pydantic import BaseModel

class CreateChatbotRequest(BaseModel):
    chatbot_name: str = "ImmoIQ"
    system_prompt: str | None = 'default'

class CreateChatbotResponse(BaseModel):
    chatbot_key: str

class GetChatbotRequest(BaseModel):
    chatbot_key: str
    session_id: str | None = None
    content: str = "Hi, my name is Mobeen and I am visiting this website to know more about it."

class GetChatbotResponse(BaseModel):
    session_id: str

class CheckChatbotRequest(BaseModel):
    chatbot_key: str

class CheckChatbotResponse(BaseModel):
    exists: bool