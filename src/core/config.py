"""
应用配置管理
"""
import os
from typing import List
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """应用配置"""
    
    # 应用配置
    APP_ENV: str = "development"
    APP_SECRET_KEY: str = "your-secret-key-change-in-production"
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    LOG_LEVEL: str = "INFO"
    
    # 数据库配置
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "openclaw_rag"
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "postgres"
    
    # Qdrant配置
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    
    # 嵌入服务配置
    EMBEDDING_PROVIDER: str = "tei"  # tei, bge, fallback
    EMBEDDING_MODEL: str = "BAAI/bge-small-zh-v1.5"
    EMBEDDING_DIMENSION: int = 512  # BGE-small的维度
    
    # TEI服务配置
    TEI_BASE_URL: str = "http://tei-bge:80"
    TEI_TIMEOUT: int = 30
    TEI_MAX_RETRIES: int = 3
    
    # DeepSeek LLM配置
    DEEPSEEK_API_KEY: str = ""
    DEEPSEEK_BASE_URL: str = "https://api.deepseek.com"
    DEEPSEEK_MODEL: str = "deepseek-chat"
    DEEPSEEK_MAX_TOKENS: int = 2000
    DEEPSEEK_TEMPERATURE: float = 0.7
    
    # Redis配置
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: str = ""
    
    # 文件监控配置
    WATCH_DIRECTORY: str = "./documents"
    POLL_INTERVAL: int = 5  # 秒
    
    # CORS配置
    CORS_ORIGINS: str = "http://localhost:3000,http://localhost:8000"
    
    @property
    def cors_origins_list(self) -> List[str]:
        """将逗号分隔的CORS_ORIGINS转换为列表"""
        if not self.CORS_ORIGINS:
            return []
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",")]
    
    # 文件上传配置
    UPLOAD_DIR: str = "./data/uploads"
    MAX_UPLOAD_SIZE: int = 104857600  # 100MB
    ALLOWED_EXTENSIONS: List[str] = [".pdf", ".docx", ".txt", ".md", ".json"]
    
    # 向量配置
    EMBEDDING_DIMENSION: int = 512
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()