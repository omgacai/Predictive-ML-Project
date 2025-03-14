import pandas as pd
import sqlite3
from sklearn.base import BaseEstimator, TransformerMixin

class SQLiteLoader(BaseEstimator, TransformerMixin):
    def __init__(self, db_path):
        self.db_path = db_path

    def fit(self, X, y=None):
        return self  # No fitting needed

    def transform(self, X=None):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        table = tables[0][0]

        df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
        conn.close()
        return df