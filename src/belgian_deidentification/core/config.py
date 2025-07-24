"""Configuration management for the Belgian deidentification system."""

import os
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, validator
from enum import Enum


class DeidentificationMode(str, Enum):
    """Deidentification modes supported by the system."""
    ANONYMIZATION = "anonymization"
    PSEUDONYMIZATION = "pseudonymization"


class EntityType(str, Enum):
    """Types of entities that can be detected and deidentified."""
    PERSON = "PERSON"
    ADDRESS = "ADDRESS" 
    DATE = "DATE"
    MEDICAL_ID = "MEDICAL_ID"
    PHONE = "PHONE"
    EMAIL = "EMAIL"
    ORGANIZATION = "ORGANIZATION"


class NLPConfig(BaseModel):
    """Configuration for NLP components."""
    model: str = Field(default="robbert-2023-dutch-large", description="Primary language model")
    use_gpu: bool = Field(default=True, description="Whether to use GPU acceleration")
    batch_size: int = Field(default=32, description="Batch size for model inference")
    max_length: int = Field(default=512, description="Maximum sequence length")
    confidence_threshold: float = Field(default=0.85, description="Minimum confidence for entity detection")
    
    @validator('confidence_threshold')
    def validate_confidence(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError('Confidence threshold must be between 0.0 and 1.0')
        return v


class DeidentificationConfig(BaseModel):
    """Configuration for deidentification processing."""
    mode: DeidentificationMode = Field(default=DeidentificationMode.ANONYMIZATION)
    entities: List[EntityType] = Field(default_factory=lambda: [
        EntityType.PERSON, EntityType.ADDRESS, EntityType.DATE, EntityType.MEDICAL_ID
    ])
    preserve_structure: bool = Field(default=True, description="Whether to preserve document structure")
    date_shift_days: Optional[int] = Field(default=None, description="Days to shift dates (for pseudonymization)")
    replacement_strategy: str = Field(default="realistic", description="Strategy for replacing entities")


class QualityAssuranceConfig(BaseModel):
    """Configuration for quality assurance and validation."""
    enable_validation: bool = Field(default=True)
    confidence_threshold: float = Field(default=0.85)
    expert_review_sampling: float = Field(default=0.1, description="Fraction of documents for expert review")
    statistical_monitoring: bool = Field(default=True)
    
    @validator('expert_review_sampling')
    def validate_sampling_rate(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError('Expert review sampling rate must be between 0.0 and 1.0')
        return v


class SecurityConfig(BaseModel):
    """Configuration for security and compliance."""
    encryption_enabled: bool = Field(default=True)
    audit_logging: bool = Field(default=True)
    access_control: bool = Field(default=True)
    data_retention_days: int = Field(default=90, description="Days to retain audit logs")
    
    @validator('data_retention_days')
    def validate_retention(cls, v):
        if v < 1:
            raise ValueError('Data retention must be at least 1 day')
        return v


class DatabaseConfig(BaseModel):
    """Configuration for database connections."""
    url: str = Field(default="postgresql://localhost:5432/belgian_deidentification")
    pool_size: int = Field(default=10)
    max_overflow: int = Field(default=20)
    echo: bool = Field(default=False, description="Whether to echo SQL queries")


class APIConfig(BaseModel):
    """Configuration for API server."""
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    workers: int = Field(default=4)
    reload: bool = Field(default=False)
    log_level: str = Field(default="info")


class Config(BaseModel):
    """Main configuration class for the Belgian deidentification system."""
    
    # Core configuration sections
    nlp: NLPConfig = Field(default_factory=NLPConfig)
    deidentification: DeidentificationConfig = Field(default_factory=DeidentificationConfig)
    quality_assurance: QualityAssuranceConfig = Field(default_factory=QualityAssuranceConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    
    # General settings
    debug: bool = Field(default=False)
    log_level: str = Field(default="INFO")
    data_dir: str = Field(default="data")
    models_dir: str = Field(default="data/models")
    temp_dir: str = Field(default="/tmp/belgian_deidentification")
    
    class Config:
        """Pydantic configuration."""
        env_prefix = "BELGIAN_DEIDENT_"
        case_sensitive = False
        
    @classmethod
    def from_yaml(cls, config_path: str) -> "Config":
        """Load configuration from YAML file."""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
            
        return cls(**config_data)
    
    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables."""
        return cls()
    
    def to_yaml(self, output_path: str) -> None:
        """Save configuration to YAML file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.dict(), f, default_flow_style=False, indent=2)
    
    def get_model_path(self, model_name: str) -> Path:
        """Get the full path to a model file."""
        return Path(self.models_dir) / model_name
    
    def ensure_directories(self) -> None:
        """Ensure all required directories exist."""
        directories = [
            self.data_dir,
            self.models_dir,
            self.temp_dir,
            f"{self.data_dir}/input",
            f"{self.data_dir}/output",
            f"{self.data_dir}/logs",
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)


def load_config(config_path: Optional[str] = None) -> Config:
    """Load configuration from file or environment variables."""
    if config_path:
        return Config.from_yaml(config_path)
    
    # Try to find default config file
    default_paths = [
        "config/default.yaml",
        "config.yaml",
        os.path.expanduser("~/.belgian_deidentification/config.yaml"),
    ]
    
    for path in default_paths:
        if Path(path).exists():
            return Config.from_yaml(path)
    
    # Fall back to environment variables
    return Config.from_env()


# Global configuration instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = load_config()
    return _config


def set_config(config: Config) -> None:
    """Set the global configuration instance."""
    global _config
    _config = config

