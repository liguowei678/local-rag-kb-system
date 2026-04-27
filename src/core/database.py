"""
数据库连接和会话管理
"""
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from src.core.config import settings

# 构建数据库URL
SQLALCHEMY_DATABASE_URL = (
    f"postgresql://{settings.POSTGRES_USER}:{settings.POSTGRES_PASSWORD}"
    f"@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}"
)

# 创建数据库引擎
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    pool_size=20,  # 连接池大小
    max_overflow=30,  # 最大溢出连接数
    pool_pre_ping=True,  # 连接前ping检查
    pool_recycle=3600,  # 连接回收时间（秒）
)

# 创建会话工厂
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)

# 声明基类
Base = declarative_base()

def get_db():
    """
    获取数据库会话的依赖函数
    用于FastAPI依赖注入
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    """
    初始化数据库，创建所有表
    注意：生产环境应该使用Alembic迁移
    """
    Base.metadata.create_all(bind=engine)