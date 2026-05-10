"""
preprocessor.py
---------------
Cleans and transforms raw data for ML modeling.
Handles missing values, outliers, encoding, and normalization.
"""

import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Stateful preprocessor that fits on training data and
    transforms any dataset consistently.
    """

    def __init__(self, target_column: str, drop_threshold: float = 0.5):
        """
        Args:
            target_column: Name of the label/target column.
            drop_threshold: Drop columns with missing rate above this value (0-1).
        """
        self.target_column = target_column
        self.drop_threshold = drop_threshold

        self.numeric_imputer = SimpleImputer(strategy="median")
        self.categorical_imputer = SimpleImputer(strategy="most_frequent")
        self.scaler = StandardScaler()
        self.label_encoders = {}

        self.numeric_cols = []
        self.categorical_cols = []
        self.dropped_cols = []
        self._fitted = False

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit on df and return cleaned DataFrame."""
        df = df.copy()
        df = self._drop_high_missing(df)
        df = self._remove_duplicates(df)
        df = self._identify_column_types(df)
        df = self._impute_missing(df, fit=True)
        df = self._handle_outliers(df)
        df = self._encode_categoricals(df, fit=True)
        df = self._scale_numerics(df, fit=True)
        self._fitted = True
        logger.info(f"Preprocessing complete. Final shape: {df.shape}")
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using already-fitted preprocessor."""
        if not self._fitted:
            raise RuntimeError("Preprocessor not fitted yet. Call fit_transform() first.")
        df = df.copy()
        df = df.drop(columns=self.dropped_cols, errors="ignore")
        df = self._impute_missing(df, fit=False)
        df = self._encode_categoricals(df, fit=False)
        df = self._scale_numerics(df, fit=False)
        return df

    def _drop_high_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        missing_rate = df.isnull().mean()
        cols_to_drop = missing_rate[missing_rate > self.drop_threshold].index.tolist()
        if self.target_column in cols_to_drop:
            cols_to_drop.remove(self.target_column)
        self.dropped_cols = cols_to_drop
        logger.info(f"Dropping {len(cols_to_drop)} high-missing columns: {cols_to_drop}")
        return df.drop(columns=cols_to_drop)

    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        df = df.drop_duplicates()
        logger.info(f"Removed {before - len(df)} duplicate rows.")
        return df

    def _identify_column_types(self, df: pd.DataFrame) -> pd.DataFrame:
        feature_cols = [c for c in df.columns if c != self.target_column]
        self.numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = df[feature_cols].select_dtypes(include=["object", "category"]).columns.tolist()
        logger.info(f"Numeric cols: {len(self.numeric_cols)}, Categorical cols: {len(self.categorical_cols)}")
        return df

    def _impute_missing(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        if self.numeric_cols:
            if fit:
                df[self.numeric_cols] = self.numeric_imputer.fit_transform(df[self.numeric_cols])
            else:
                df[self.numeric_cols] = self.numeric_imputer.transform(df[self.numeric_cols])

        if self.categorical_cols:
            if fit:
                df[self.categorical_cols] = self.categorical_imputer.fit_transform(df[self.categorical_cols])
            else:
                df[self.categorical_cols] = self.categorical_imputer.transform(df[self.categorical_cols])
        return df

    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cap outliers at 1.5*IQR (Winsorization)."""
        for col in self.numeric_cols:
            if col == self.target_column:
                continue
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            outliers = ((df[col] < lower) | (df[col] > upper)).sum()
            if outliers > 0:
                df[col] = df[col].clip(lower, upper)
                logger.debug(f"Capped {outliers} outliers in '{col}'")
        return df

    def _encode_categoricals(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        for col in self.categorical_cols:
            if fit:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
            else:
                le = self.label_encoders.get(col)
                if le:
                    df[col] = df[col].astype(str).map(
                        lambda x: le.transform([x])[0] if x in le.classes_ else -1
                    )
        return df

    def _scale_numerics(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        scale_cols = [c for c in self.numeric_cols if c != self.target_column]
        if not scale_cols:
            return df
        if fit:
            df[scale_cols] = self.scaler.fit_transform(df[scale_cols])
        else:
            df[scale_cols] = self.scaler.transform(df[scale_cols])
        return df
