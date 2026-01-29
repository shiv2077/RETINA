from sqlalchemy import Column, Integer, String, Boolean, DateTime
from sqlalchemy.ext.declarative import declarative_base
import datetime

Base = declarative_base()

"""
SQLAlchemy ORM model in app/db/models.py (this talks to the DB)
Pydantic schemas in app/schemas/user.py (this talks to the API)
"""
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)


class AnomalyRecord(Base):
    __tablename__ = "anomaly_records"

    id = Column(String, primary_key=True, index=True)
    user = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    unsupervised_label = Column(Boolean, nullable=False)
    supervised_label = Column(Boolean, nullable=False)
    mismatch = Column(Boolean, default=False)
    reviewed = Column(Boolean, default=False)
    expert_label = Column(String, nullable=True)
    final_classification = Column(String, nullable=True)

    __all__ = ["Base", "User", "AnomalyRecord"]

# optional: quick serializer to replace .to_dict()
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "user": self.user,
            "file_path": self.file_path,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "unsupervised_label": self.unsupervised_label,
            "supervised_label": self.supervised_label,
            "mismatch": self.mismatch,
            "reviewed": self.reviewed,
            "expert_label": self.expert_label,
            "final_classification": self.final_classification,
        }
