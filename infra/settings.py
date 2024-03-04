from pydantic_settings import BaseSettings
from pydantic import (
    Field
)


class Settings(BaseSettings):
    frontend_url: str = Field(env="FRONTEND_URL", default="http://localhost:5173")

class Config:
    env_file = '../.env'

settings: Settings = Settings()
