from typing import Tuple, List, Optional
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


def winsorize_series(s: pd.Series, lower_quantile: float = 0.01, upper_quantile: float = 0.99) -> pd.Series:
    if not np.issubdtype(s.dtype, np.number):
        return s
    lower = s.quantile(lower_quantile)
    upper = s.quantile(upper_quantile)
    return s.clip(lower, upper)


def winsorize_df(df: pd.DataFrame, numeric_cols: Optional[List[str]] = None,
                 lower_quantile: float = 0.01, upper_quantile: float = 0.99) -> pd.DataFrame:
    df = df.copy()
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        df[col] = winsorize_series(df[col], lower_quantile, upper_quantile)
    return df


def build_preprocess_pipeline(df: pd.DataFrame,
                              numeric_cols: Optional[List[str]] = None,
                              categorical_cols: Optional[List[str]] = None,
                              scale: bool = True) -> Tuple[Pipeline, List[str]]:
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if categorical_cols is None:
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    num_steps = [
        ("imputer", SimpleImputer(strategy="median")),
    ]
    if scale:
        num_steps.append(("scaler", StandardScaler()))

    cat_steps = [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False)),
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline(num_steps), numeric_cols),
            ("cat", Pipeline(cat_steps), categorical_cols),
        ],
        remainder="drop",
    )

    pipeline = Pipeline(steps=[("preprocessor", preprocessor)])

    feature_names = []
    feature_names.extend(numeric_cols)
    # OneHot feature names will be available after fit; return placeholders for now
    return pipeline, feature_names


def preprocess_fit_transform(df: pd.DataFrame,
                             id_cols: Optional[List[str]] = None,
                             scale: bool = True,
                             winsorize: bool = True,
                             lower_q: float = 0.01,
                             upper_q: float = 0.99) -> Tuple[np.ndarray, Pipeline]:
    df_use = df.copy()
    if id_cols:
        df_use = df_use.drop(columns=id_cols, errors="ignore")

    if winsorize:
        df_use = winsorize_df(df_use, lower_quantile=lower_q, upper_quantile=upper_q)

    pipeline, _ = build_preprocess_pipeline(df_use, scale=scale)
    X = pipeline.fit_transform(df_use)
    return X, pipeline


def preprocess_transform(df: pd.DataFrame, pipeline: Pipeline, id_cols: Optional[List[str]] = None) -> np.ndarray:
    df_use = df.copy()
    if id_cols:
        df_use = df_use.drop(columns=id_cols, errors="ignore")
    return pipeline.transform(df_use)
