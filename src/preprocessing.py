import re
import pandas as pd
from tqdm import tqdm

tqdm.pandas()


def ind_preprocessing(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df[col] = df[col].astype(str)
    df[col] = df[col].str.lower()
    df[col] = df[col].apply(lambda x: re.sub(r"[^a-zA-Z\s]", "", x))
    df[col] = df[col].apply(lambda x: re.sub(r"\s+", " ", x).strip())
    return df


def bnj_preprocessing(df: pd.DataFrame, col: str, add_tokens: bool = True) -> pd.DataFrame:
    df[col] = df[col].str.lower()
    df[col] = df[col].apply(lambda x: re.sub(r"\d", "", x))
    df[col] = df[col].apply(lambda x: re.sub(r"\s+", " ", x))
    df[col] = df[col].apply(lambda x: re.sub(
        r"[-()\"#/@;:<>{}`+=~|.!?,।]", "", x))
    df[col] = df[col].str.strip()

    if add_tokens:
        df[col] = "startseq " + df[col] + " endseq"

    return df
