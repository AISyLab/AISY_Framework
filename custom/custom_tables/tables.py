import datetime
from sqlalchemy import *
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from aisy_database.db_tables import Analysis

Base = declarative_base()


def base():
    return Base


def start_custom_tables(database_name):
    engine = create_engine('sqlite:///{}'.format(database_name), echo=False)
    base().metadata.create_all(engine)


def start_custom_tables_session(database_name):
    engine = create_engine('sqlite:///{}'.format(database_name), echo=False)
    return sessionmaker(bind=engine)()


class CustomTable(Base):
    __tablename__ = 'custom_table'
    id = Column(Integer, primary_key=True)
    value1 = Column(Integer)
    value2 = Column(Integer)
    value3 = Column(Integer)
    datetime = Column(DateTime, default=datetime.datetime.utcnow)
    analysis_id = Column(Integer, ForeignKey(Analysis.id))
    analysis = relationship(Analysis)

    def __repr__(self):
        return "<CustomTable(id=%d)>" % self.id
