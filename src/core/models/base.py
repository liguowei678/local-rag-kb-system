"""
基础模型类
"""
from datetime import datetime
from sqlalchemy import Column, DateTime, Integer
from sqlalchemy.ext.declarative import declared_attr

from src.core.database import Base as BaseModel

class Base(BaseModel):
    """所有模型的基类"""
    __abstract__ = True
    
    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    @declared_attr
    def __tablename__(cls):
        """自动生成表名：类名小写 + 's'"""
        return cls.__name__.lower() + 's'
    
    def to_dict(self):
        """将模型转换为字典"""
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}
    
    def update(self, **kwargs):
        """更新模型属性"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)