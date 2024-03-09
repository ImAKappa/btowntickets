"""utils

Module for utility functions
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Self

import pandas as pd

class Loader(ABC):
    
    def __init__(self, file_path: Path) -> Self:
        self.file_path = file_path

    @abstractmethod
    def load(self) -> pd.DataFrame:
        """Loads data from a file on disk into a Pandas DataFrame"""
        pass

class CSVLoader(Loader):
    """Loads parking ticket dataset from CSV"""

    def load(self) -> pd.DataFrame:
        dtypes = {
            "X": "Float64",
            "Y": "Float64",
            "OBJECTID": "Int64",
            "ADDRESS": str,
            "LICSTATEPROV": "category",
            "VIODESCRIPTION": "category",
            "VIOFINE": "Float64",
            "VOIDSTATUS": "category",
        }
        datetime_cols = ['ISSUEDATE', 'ISSUETIME']
        df = pd.read_csv(
            self.file_path,
            parse_dates=datetime_cols,
            dtype=dtypes,
            usecols=list(dtypes.keys()) + datetime_cols
        )
        return df
    
class ParquetLoader(Loader):
    """Loads parking ticket dataset from a Parquet file"""

    def load(self) -> pd.DataFrame:
        return pd.read_parquet(self.file_path, engine="pyarrow")

class LoadProcess:
    """Loads the dataset based on the given Loader (Factor pattern)"""

    def __init__(self, data_loader: Loader) -> Self: 
        self.data_loader = data_loader

    def run(self) -> pd.DataFrame:
        return self.data_loader.load()
    

def profile_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """Profiles the null values in the DataFrame"""
    null_profile = df.isna().sum(axis=0).to_frame()\
        .reset_index()\
        .rename(columns={"index": "Column", 0: "Null Count"})
    null_profile["Null Pct"] = null_profile["Null Count"] / len(df)
    return null_profile