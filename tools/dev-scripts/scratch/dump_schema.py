
import sys
import os
sys.path.insert(0, os.getcwd())

from sqlalchemy import create_mock_engine
from app.db.base import Base

def dump(sql, *multiparams, **params):
    print(sql.compile(dialect=engine.dialect))

engine = create_mock_engine("postgresql://", dump)
Base.metadata.create_all(engine)
