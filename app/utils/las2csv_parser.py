

import io
from pathlib import Path

import pandas as pd
import petrologix
 
"""
LAS  2  DataFrame conversion for uploaded well logs.

Delegates to `petrologix.read_las`, which applies exactly the null handling the
training data received -- including the undeclared sentinels (-999.9, -999.0) that
lasio does not strip, because LAS headers in this dataset declare only -999.25.

"""

class UnsupportedFormatError(ValueError):
    """Raised for an upload that is neither LAS nor CSV."""


def parse_las(path: str | Path) -> pd.DataFrame:
    """Parse a .las file into a flat DataFrame with DEPT/DEPTH_MD as a column."""
    return petrologix.read_las(path)


def parse_csv(source: str | Path | io.BytesIO) -> pd.DataFrame:
    """Parse an already-converted CSV, applying the same null handling."""
    df = pd.read_csv(source)
    
    df.columns = [str(c).upper() for c in df.columns]
    
    return petrologix.strip_null_sentinels(df)


def parse_upload(path: str | Path, filename: str | None = None) -> pd.DataFrame:
    """Dispatch on file extension. Returns a model-ready DataFrame."""
    name = (filename or str(path)).lower()
    
    if name.endswith(".las"):
        
        return parse_las(path)
    
    if name.endswith(".csv"):
        
        return parse_csv(path)
    
    raise UnsupportedFormatError(
        f"unsupported file type: {name!r}. Upload a .las or .csv well log."
    )
