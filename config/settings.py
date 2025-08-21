import os
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


# Step 1: Define nested models as simple Pydantic BaseModels
class ApiSettings(BaseModel):
    """API configurations"""
    url: str
    key: str


class ModelParameters(BaseModel):
    """Model parameters"""
    name: str
    temperature: float = 0.7


class SystemSettings(BaseModel):
    """System settings"""
    debug: bool = False


# Step 2: Define a single BaseSettings model to load all variables
class Settings(BaseSettings):
    """Main settings"""
    # Define all settings from .env files here
    API_URL: str
    API_KEY: str
    MODEL_NAME: str
    TEMPERATURE: float = 0.7
    DEBUG: bool = False

    # Configure pydantic-settings
    model_config = SettingsConfigDict(
        env_file_encoding='utf-8'
    )

    # Step 3: Use properties to provide the nested structure
    @property
    def api(self) -> ApiSettings:
        return ApiSettings(url=self.API_URL, key=self.API_KEY)

    @property
    def model(self) -> ModelParameters:
        return ModelParameters(name=self.MODEL_NAME, temperature=self.TEMPERATURE)

    @property
    def system(self) -> SystemSettings:
        return SystemSettings(debug=self.DEBUG)


# Step 4: Simplify the loader function
@lru_cache()
def get_settings() -> Settings:
    """Load settings from the correct .env file and return a Settings object."""
    env = os.getenv("APP_ENV", "development")
    base_dir = Path(__file__).resolve().parent
    env_path = base_dir / f"{env}.env"

    if not os.path.exists(env_path):
        raise FileNotFoundError(f"Environment file not found at {env_path}")

    return Settings(_env_file=env_path)


# Step 5: Create a singleton instance for easy access
settings = get_settings()
