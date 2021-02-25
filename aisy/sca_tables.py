import datetime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import *
from sqlalchemy.orm import relationship

Base = declarative_base()


def base():
    return Base


class Analysis(Base):
    __tablename__ = 'analysis'
    id = Column(Integer, primary_key=True)
    datetime = Column(DateTime, default=datetime.datetime.utcnow)
    db_filename = Column(String)
    dataset = Column(String)
    settings = Column(JSON)
    elapsed_time = Column(Float)
    deleted = Column(Boolean)

    def __repr__(self):
        return "<Analysis(datetime=%s, script='%s')>" % (self.datetime, self.db_filename)


class HyperParameter(Base):
    __tablename__ = 'hyper_parameter'
    id = Column(Integer, primary_key=True)
    hyper_parameters = Column(JSON)
    analysis_id = Column(Integer, ForeignKey('analysis.id'))
    analysis = relationship("Analysis", cascade="all, delete")

    def __repr__(self):
        return "<HyperParemeters(id=%d)>" % self.id


class NeuralNetwork(Base):
    __tablename__ = 'neural_network'
    id = Column(Integer, primary_key=True)
    model_name = Column(String)
    description = Column(String)
    analysis_id = Column(Integer, ForeignKey('analysis.id'))
    analysis = relationship("Analysis", cascade="all, delete")

    def __repr__(self):
        return "<NeuralNetwork(name=%s, description='%s')>" % (self.model_name, self.description)


class LeakageModel(Base):
    __tablename__ = 'leakage_model'
    id = Column(Integer, primary_key=True)
    leakage_model = Column(JSON)
    analysis_id = Column(Integer, ForeignKey('analysis.id'))
    analysis = relationship("Analysis", cascade="all, delete")

    def __repr__(self):
        return "<LeakageModel(id=%d)>" % self.id


class Metric(Base):
    __tablename__ = 'metric'
    id = Column(Integer, primary_key=True)
    value = Column(Float)
    key_byte = Column(Integer)
    metric = Column(String)
    analysis_id = Column(Integer, ForeignKey('analysis.id'))
    analysis = relationship("Analysis", cascade="all, delete")

    def __repr__(self):
        return "<Metric(value=%f, metric='%s')>" % (self.value, self.metric)


class KeyRank(Base):
    __tablename__ = 'key_rank_json'
    id = Column(Integer, primary_key=True)
    values = Column(JSON)
    name = Column(String)
    key_byte = Column(Integer)
    report_interval = Column(Integer)
    metric = Column(String)
    analysis_id = Column(Integer, ForeignKey('analysis.id'))
    analysis = relationship("Analysis", cascade="all, delete")

    def __repr__(self):
        return "<KeyRank(key_byte=%d)>" % self.key_byte


class SuccessRate(Base):
    __tablename__ = 'success_rate_json'
    id = Column(Integer, primary_key=True)
    values = Column(JSON)
    key_byte = Column(Integer)
    report_interval = Column(Integer)
    metric = Column(String)
    analysis_id = Column(Integer, ForeignKey('analysis.id'))
    analysis = relationship("Analysis", cascade="all, delete")

    def __repr__(self):
        return "<SuccessRate(key_byte=%d)>" % self.key_byte


class Visualization(Base):
    __tablename__ = 'visualization'
    id = Column(Integer, primary_key=True)
    values = Column(JSON)
    epoch = Column(Integer)
    key_byte = Column(Integer)
    report_interval = Column(Integer)
    metric = Column(String)
    analysis_id = Column(Integer, ForeignKey('analysis.id'))
    analysis = relationship("Analysis", cascade="all, delete")

    def __repr__(self):
        return "<Visualization(metric=%d)>" % self.metric


class HyperParameterSearch(Base):
    __tablename__ = 'hyper_parameter_search'
    id = Column(Integer, primary_key=True)
    search_type = Column(String)
    hyper_parameters_settings = Column(JSON)
    best_hyper_parameters = Column(Integer, ForeignKey('hyper_parameter.id'))
    analysis_id = Column(Integer, ForeignKey('analysis.id'))
    analysis = relationship("Analysis", cascade="all, delete")

    def __repr__(self):
        return "<HyperParemetersSearch(id=%d)>" % self.id


class ConfusionMatrix(Base):
    __tablename__ = 'confusion_matrix'
    id = Column(Integer, primary_key=True)
    y_pred = Column(JSON)
    y_true = Column(Integer)
    key_byte = Column(Integer)
    analysis_id = Column(Integer, ForeignKey('analysis.id'))
    analysis = relationship("Analysis", cascade="all, delete")

    def __repr__(self):
        return "<ConfusionMatrix>"


class ProbabilityRank(Base):
    __tablename__ = 'probability_rank'
    id = Column(Integer, primary_key=True)
    ranks = Column(JSON)
    classes = Column(Integer)
    correct_key_byte = Column(Integer)
    key_guess = Column(Integer)
    title = Column(String)
    key_byte = Column(Integer)
    analysis_id = Column(Integer, ForeignKey('analysis.id'))
    analysis = relationship("Analysis", cascade="all, delete")

    def __repr__(self):
        return "<ProbabilityRank>"
