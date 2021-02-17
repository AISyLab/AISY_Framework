import datetime
from sqlalchemy import *
from sqlalchemy.orm import relationship
from commons.sca_tables import Analysis, Base


def base():
    return Base


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
