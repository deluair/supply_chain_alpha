#!/usr/bin/env python3
"""
Data Processing Utilities
Handles data cleaning, transformation, and analysis for supply chain disruption analysis.

Features:
- Data validation and cleaning
- Time series processing
- Statistical analysis
- Feature engineering
- Data normalization and scaling
- Outlier detection and handling
- Missing data imputation

Author: Supply Chain Alpha Team
Version: 1.0.0
"""

import logging
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from dataclasses import dataclass, field
import json

# Data processing libraries
import pandas as pd
import numpy as np
from scipy import stats
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression

# Time series processing
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Suppress warnings
warnings.filterwarnings('ignore')


@dataclass
class DataQualityReport:
    """Data quality assessment report."""
    total_records: int
    missing_values: Dict[str, int]
    duplicate_records: int
    outliers: Dict[str, int]
    data_types: Dict[str, str]
    value_ranges: Dict[str, Dict[str, float]]
    quality_score: float
    recommendations: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ProcessingConfig:
    """Configuration for data processing operations."""
    # Missing data handling
    missing_strategy: str = 'mean'  # 'mean', 'median', 'mode', 'knn', 'forward_fill', 'backward_fill'
    missing_threshold: float = 0.5  # Drop columns with > 50% missing values
    
    # Outlier detection
    outlier_method: str = 'isolation_forest'  # 'isolation_forest', 'zscore', 'iqr', 'modified_zscore'
    outlier_threshold: float = 0.1  # Contamination rate for isolation forest
    zscore_threshold: float = 3.0
    
    # Scaling and normalization
    scaling_method: str = 'standard'  # 'standard', 'minmax', 'robust', 'none'
    
    # Feature engineering
    create_lag_features: bool = True
    lag_periods: List[int] = field(default_factory=lambda: [1, 7, 30])
    create_rolling_features: bool = True
    rolling_windows: List[int] = field(default_factory=lambda: [7, 14, 30])
    
    # Time series processing
    seasonal_decomposition: bool = True
    stationarity_test: bool = True
    differencing_order: int = 1
    
    # Data validation
    validate_ranges: bool = True
    custom_validators: Dict[str, Callable] = field(default_factory=dict)


class DataProcessor:
    """Advanced data processing and analysis utilities."""
    
    def __init__(self, config: Optional[ProcessingConfig] = None):
        """Initialize data processor.
        
        Args:
            config: Processing configuration
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or ProcessingConfig()
        
        # Initialize scalers
        self.scalers = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'robust': RobustScaler()
        }
        
        # Initialize imputers
        self.imputers = {
            'mean': SimpleImputer(strategy='mean'),
            'median': SimpleImputer(strategy='median'),
            'mode': SimpleImputer(strategy='most_frequent'),
            'knn': KNNImputer(n_neighbors=5)
        }
        
        # Processing history
        self.processing_history = []
        
        # Feature importance cache
        self.feature_importance = {}
    
    def assess_data_quality(self, df: pd.DataFrame, target_column: Optional[str] = None) -> DataQualityReport:
        """Assess data quality and generate report.
        
        Args:
            df: Input DataFrame
            target_column: Target column for supervised learning tasks
            
        Returns:
            Data quality report
        """
        try:
            self.logger.info("Assessing data quality")
            
            # Basic statistics
            total_records = len(df)
            missing_values = df.isnull().sum().to_dict()
            duplicate_records = df.duplicated().sum()
            
            # Data types
            data_types = df.dtypes.astype(str).to_dict()
            
            # Value ranges for numeric columns
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            value_ranges = {}
            for col in numeric_columns:
                value_ranges[col] = {
                    'min': float(df[col].min()),
                    'max': float(df[col].max()),
                    'mean': float(df[col].mean()),
                    'std': float(df[col].std())
                }
            
            # Outlier detection
            outliers = {}
            for col in numeric_columns:
                outlier_mask = self._detect_outliers(df[col].dropna(), method='zscore')
                outliers[col] = int(outlier_mask.sum())
            
            # Calculate quality score
            quality_score = self._calculate_quality_score(df, missing_values, duplicate_records, outliers)
            
            # Generate recommendations
            recommendations = self._generate_quality_recommendations(
                df, missing_values, duplicate_records, outliers, target_column
            )
            
            report = DataQualityReport(
                total_records=total_records,
                missing_values=missing_values,
                duplicate_records=duplicate_records,
                outliers=outliers,
                data_types=data_types,
                value_ranges=value_ranges,
                quality_score=quality_score,
                recommendations=recommendations
            )
            
            self.logger.info(f"Data quality assessment completed. Score: {quality_score:.2f}")
            return report
            
        except Exception as e:
            self.logger.error(f"Error assessing data quality: {e}")
            raise
    
    def clean_data(self, df: pd.DataFrame, target_column: Optional[str] = None) -> pd.DataFrame:
        """Clean and preprocess data.
        
        Args:
            df: Input DataFrame
            target_column: Target column to preserve
            
        Returns:
            Cleaned DataFrame
        """
        try:
            self.logger.info("Starting data cleaning process")
            cleaned_df = df.copy()
            
            # Record original shape
            original_shape = cleaned_df.shape
            
            # 1. Remove completely empty rows and columns
            cleaned_df = cleaned_df.dropna(how='all')
            cleaned_df = cleaned_df.dropna(axis=1, how='all')
            
            # 2. Handle duplicate records
            if cleaned_df.duplicated().any():
                duplicate_count = cleaned_df.duplicated().sum()
                cleaned_df = cleaned_df.drop_duplicates()
                self.logger.info(f"Removed {duplicate_count} duplicate records")
            
            # 3. Drop columns with excessive missing values
            missing_threshold = self.config.missing_threshold
            high_missing_cols = []
            for col in cleaned_df.columns:
                if col != target_column:  # Don't drop target column
                    missing_ratio = cleaned_df[col].isnull().sum() / len(cleaned_df)
                    if missing_ratio > missing_threshold:
                        high_missing_cols.append(col)
            
            if high_missing_cols:
                cleaned_df = cleaned_df.drop(columns=high_missing_cols)
                self.logger.info(f"Dropped {len(high_missing_cols)} columns with >50% missing values")
            
            # 4. Handle missing values
            cleaned_df = self._handle_missing_values(cleaned_df, target_column)
            
            # 5. Detect and handle outliers
            cleaned_df = self._handle_outliers(cleaned_df, target_column)
            
            # 6. Validate data types and ranges
            cleaned_df = self._validate_and_fix_data_types(cleaned_df)
            
            # 7. Apply custom validators
            cleaned_df = self._apply_custom_validators(cleaned_df)
            
            # Record processing step
            self._record_processing_step('clean_data', original_shape, cleaned_df.shape)
            
            self.logger.info(f"Data cleaning completed. Shape: {original_shape} -> {cleaned_df.shape}")
            return cleaned_df
            
        except Exception as e:
            self.logger.error(f"Error cleaning data: {e}")
            raise
    
    def engineer_features(self, df: pd.DataFrame, target_column: Optional[str] = None, 
                         datetime_column: Optional[str] = None) -> pd.DataFrame:
        """Engineer features for machine learning.
        
        Args:
            df: Input DataFrame
            target_column: Target column name
            datetime_column: Datetime column for time-based features
            
        Returns:
            DataFrame with engineered features
        """
        try:
            self.logger.info("Starting feature engineering")
            feature_df = df.copy()
            
            # Identify numeric and categorical columns
            numeric_columns = feature_df.select_dtypes(include=[np.number]).columns.tolist()
            if target_column and target_column in numeric_columns:
                numeric_columns.remove(target_column)
            
            categorical_columns = feature_df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            # 1. Create lag features for time series
            if self.config.create_lag_features and datetime_column:
                feature_df = self._create_lag_features(feature_df, numeric_columns, datetime_column)
            
            # 2. Create rolling window features
            if self.config.create_rolling_features and datetime_column:
                feature_df = self._create_rolling_features(feature_df, numeric_columns, datetime_column)
            
            # 3. Create interaction features
            feature_df = self._create_interaction_features(feature_df, numeric_columns)
            
            # 4. Create polynomial features for important variables
            feature_df = self._create_polynomial_features(feature_df, numeric_columns, degree=2)
            
            # 5. Create statistical features
            feature_df = self._create_statistical_features(feature_df, numeric_columns)
            
            # 6. Encode categorical variables
            feature_df = self._encode_categorical_features(feature_df, categorical_columns)
            
            # 7. Create datetime features
            if datetime_column and datetime_column in feature_df.columns:
                feature_df = self._create_datetime_features(feature_df, datetime_column)
            
            # 8. Create technical indicators for financial data
            feature_df = self._create_technical_indicators(feature_df, numeric_columns)
            
            self.logger.info(f"Feature engineering completed. Features: {len(df.columns)} -> {len(feature_df.columns)}")
            return feature_df
            
        except Exception as e:
            self.logger.error(f"Error engineering features: {e}")
            raise
    
    def scale_features(self, df: pd.DataFrame, target_column: Optional[str] = None, 
                      fit_scaler: bool = True) -> pd.DataFrame:
        """Scale numerical features.
        
        Args:
            df: Input DataFrame
            target_column: Target column to exclude from scaling
            fit_scaler: Whether to fit the scaler (True for training, False for inference)
            
        Returns:
            DataFrame with scaled features
        """
        try:
            if self.config.scaling_method == 'none':
                return df
            
            self.logger.info(f"Scaling features using {self.config.scaling_method} method")
            scaled_df = df.copy()
            
            # Identify numeric columns to scale
            numeric_columns = scaled_df.select_dtypes(include=[np.number]).columns.tolist()
            if target_column and target_column in numeric_columns:
                numeric_columns.remove(target_column)
            
            if not numeric_columns:
                return scaled_df
            
            # Get or create scaler
            scaler = self.scalers[self.config.scaling_method]
            
            if fit_scaler:
                # Fit and transform
                scaled_values = scaler.fit_transform(scaled_df[numeric_columns])
            else:
                # Transform only (for inference)
                scaled_values = scaler.transform(scaled_df[numeric_columns])
            
            # Update DataFrame
            scaled_df[numeric_columns] = scaled_values
            
            self.logger.info(f"Scaled {len(numeric_columns)} numeric features")
            return scaled_df
            
        except Exception as e:
            self.logger.error(f"Error scaling features: {e}")
            raise
    
    def select_features(self, df: pd.DataFrame, target_column: str, 
                       method: str = 'correlation', k: int = 20) -> Tuple[pd.DataFrame, List[str]]:
        """Select most important features.
        
        Args:
            df: Input DataFrame
            target_column: Target column name
            method: Feature selection method ('correlation', 'mutual_info', 'f_test')
            k: Number of features to select
            
        Returns:
            Tuple of (selected DataFrame, selected feature names)
        """
        try:
            self.logger.info(f"Selecting top {k} features using {method} method")
            
            # Separate features and target
            feature_columns = [col for col in df.columns if col != target_column]
            X = df[feature_columns]
            y = df[target_column]
            
            # Handle missing values in target
            valid_mask = ~y.isnull()
            X = X[valid_mask]
            y = y[valid_mask]
            
            selected_features = []
            
            if method == 'correlation':
                # Correlation-based selection
                correlations = X.corrwith(y).abs().sort_values(ascending=False)
                selected_features = correlations.head(k).index.tolist()
                
            elif method == 'f_test':
                # F-test based selection
                # Only use numeric features
                numeric_features = X.select_dtypes(include=[np.number]).columns
                if len(numeric_features) > 0:
                    selector = SelectKBest(score_func=f_regression, k=min(k, len(numeric_features)))
                    selector.fit(X[numeric_features], y)
                    selected_features = numeric_features[selector.get_support()].tolist()
                
            elif method == 'mutual_info':
                # Mutual information based selection
                from sklearn.feature_selection import mutual_info_regression
                numeric_features = X.select_dtypes(include=[np.number]).columns
                if len(numeric_features) > 0:
                    mi_scores = mutual_info_regression(X[numeric_features], y)
                    mi_df = pd.DataFrame({
                        'feature': numeric_features,
                        'score': mi_scores
                    }).sort_values('score', ascending=False)
                    selected_features = mi_df.head(k)['feature'].tolist()
            
            # Ensure we have features selected
            if not selected_features:
                # Fallback to top correlated features
                numeric_features = X.select_dtypes(include=[np.number]).columns
                if len(numeric_features) > 0:
                    correlations = X[numeric_features].corrwith(y).abs().sort_values(ascending=False)
                    selected_features = correlations.head(k).index.tolist()
            
            # Create selected DataFrame
            selected_df = df[selected_features + [target_column]].copy()
            
            # Store feature importance
            self.feature_importance[target_column] = selected_features
            
            self.logger.info(f"Selected {len(selected_features)} features")
            return selected_df, selected_features
            
        except Exception as e:
            self.logger.error(f"Error selecting features: {e}")
            # Return original DataFrame if selection fails
            return df, df.columns.tolist()
    
    def process_time_series(self, df: pd.DataFrame, value_column: str, 
                           datetime_column: str) -> Dict[str, Any]:
        """Process and analyze time series data.
        
        Args:
            df: Input DataFrame
            value_column: Column containing time series values
            datetime_column: Column containing datetime values
            
        Returns:
            Dictionary with time series analysis results
        """
        try:
            self.logger.info("Processing time series data")
            
            # Prepare time series
            ts_df = df[[datetime_column, value_column]].copy()
            ts_df = ts_df.dropna()
            ts_df[datetime_column] = pd.to_datetime(ts_df[datetime_column])
            ts_df = ts_df.set_index(datetime_column).sort_index()
            
            # Remove duplicates by taking mean
            ts_df = ts_df.groupby(ts_df.index).mean()
            
            ts = ts_df[value_column]
            
            results = {
                'original_series': ts,
                'length': len(ts),
                'start_date': ts.index.min(),
                'end_date': ts.index.max(),
                'frequency': pd.infer_freq(ts.index)
            }
            
            # Basic statistics
            results['statistics'] = {
                'mean': float(ts.mean()),
                'std': float(ts.std()),
                'min': float(ts.min()),
                'max': float(ts.max()),
                'skewness': float(ts.skew()),
                'kurtosis': float(ts.kurtosis())
            }
            
            # Stationarity tests
            if self.config.stationarity_test:
                results['stationarity'] = self._test_stationarity(ts)
            
            # Seasonal decomposition
            if self.config.seasonal_decomposition and len(ts) >= 24:  # Need enough data points
                try:
                    decomposition = seasonal_decompose(ts, model='additive', period=min(12, len(ts)//2))
                    results['decomposition'] = {
                        'trend': decomposition.trend,
                        'seasonal': decomposition.seasonal,
                        'residual': decomposition.resid
                    }
                except Exception as e:
                    self.logger.warning(f"Seasonal decomposition failed: {e}")
            
            # Differencing for stationarity
            if results.get('stationarity', {}).get('is_stationary', False) is False:
                diff_series = ts.diff(self.config.differencing_order).dropna()
                results['differenced_series'] = diff_series
                results['differenced_stationarity'] = self._test_stationarity(diff_series)
            
            # Detect anomalies
            results['anomalies'] = self._detect_time_series_anomalies(ts)
            
            # Calculate volatility
            results['volatility'] = self._calculate_volatility(ts)
            
            self.logger.info("Time series processing completed")
            return results
            
        except Exception as e:
            self.logger.error(f"Error processing time series: {e}")
            return {}
    
    def _handle_missing_values(self, df: pd.DataFrame, target_column: Optional[str] = None) -> pd.DataFrame:
        """Handle missing values in the dataset."""
        try:
            processed_df = df.copy()
            
            for column in processed_df.columns:
                if column == target_column:
                    continue  # Don't impute target column
                
                missing_count = processed_df[column].isnull().sum()
                if missing_count == 0:
                    continue
                
                self.logger.info(f"Handling {missing_count} missing values in {column}")
                
                if processed_df[column].dtype in ['object', 'category']:
                    # Categorical data - use mode
                    mode_value = processed_df[column].mode()
                    if len(mode_value) > 0:
                        processed_df[column].fillna(mode_value[0], inplace=True)
                    else:
                        processed_df[column].fillna('Unknown', inplace=True)
                
                else:
                    # Numeric data - use configured strategy
                    if self.config.missing_strategy == 'forward_fill':
                        processed_df[column].fillna(method='ffill', inplace=True)
                    elif self.config.missing_strategy == 'backward_fill':
                        processed_df[column].fillna(method='bfill', inplace=True)
                    elif self.config.missing_strategy in ['mean', 'median', 'mode']:
                        imputer = self.imputers[self.config.missing_strategy]
                        processed_df[column] = imputer.fit_transform(processed_df[[column]]).flatten()
                    elif self.config.missing_strategy == 'knn':
                        # Use KNN imputation for numeric columns only
                        numeric_cols = processed_df.select_dtypes(include=[np.number]).columns
                        if len(numeric_cols) > 1:  # Need at least 2 columns for KNN
                            imputer = self.imputers['knn']
                            processed_df[numeric_cols] = imputer.fit_transform(processed_df[numeric_cols])
                        else:
                            # Fallback to mean imputation
                            processed_df[column].fillna(processed_df[column].mean(), inplace=True)
            
            return processed_df
            
        except Exception as e:
            self.logger.error(f"Error handling missing values: {e}")
            return df
    
    def _handle_outliers(self, df: pd.DataFrame, target_column: Optional[str] = None) -> pd.DataFrame:
        """Detect and handle outliers."""
        try:
            processed_df = df.copy()
            numeric_columns = processed_df.select_dtypes(include=[np.number]).columns
            
            for column in numeric_columns:
                if column == target_column:
                    continue  # Don't modify target column
                
                outlier_mask = self._detect_outliers(processed_df[column], self.config.outlier_method)
                outlier_count = outlier_mask.sum()
                
                if outlier_count > 0:
                    self.logger.info(f"Handling {outlier_count} outliers in {column}")
                    
                    # Cap outliers at 95th and 5th percentiles
                    q95 = processed_df[column].quantile(0.95)
                    q05 = processed_df[column].quantile(0.05)
                    
                    processed_df.loc[processed_df[column] > q95, column] = q95
                    processed_df.loc[processed_df[column] < q05, column] = q05
            
            return processed_df
            
        except Exception as e:
            self.logger.error(f"Error handling outliers: {e}")
            return df
    
    def _detect_outliers(self, series: pd.Series, method: str = 'isolation_forest') -> pd.Series:
        """Detect outliers in a series."""
        try:
            series_clean = series.dropna()
            if len(series_clean) < 10:  # Need minimum data points
                return pd.Series(False, index=series.index)
            
            outlier_mask = pd.Series(False, index=series.index)
            
            if method == 'zscore':
                z_scores = np.abs(stats.zscore(series_clean))
                outlier_indices = series_clean[z_scores > self.config.zscore_threshold].index
                outlier_mask[outlier_indices] = True
                
            elif method == 'modified_zscore':
                median = series_clean.median()
                mad = np.median(np.abs(series_clean - median))
                modified_z_scores = 0.6745 * (series_clean - median) / mad
                outlier_indices = series_clean[np.abs(modified_z_scores) > self.config.zscore_threshold].index
                outlier_mask[outlier_indices] = True
                
            elif method == 'iqr':
                Q1 = series_clean.quantile(0.25)
                Q3 = series_clean.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outlier_indices = series_clean[(series_clean < lower_bound) | (series_clean > upper_bound)].index
                outlier_mask[outlier_indices] = True
                
            elif method == 'isolation_forest':
                iso_forest = IsolationForest(contamination=self.config.outlier_threshold, random_state=42)
                outlier_pred = iso_forest.fit_predict(series_clean.values.reshape(-1, 1))
                outlier_indices = series_clean[outlier_pred == -1].index
                outlier_mask[outlier_indices] = True
            
            return outlier_mask
            
        except Exception as e:
            self.logger.error(f"Error detecting outliers: {e}")
            return pd.Series(False, index=series.index)
    
    def _validate_and_fix_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and fix data types."""
        try:
            processed_df = df.copy()
            
            for column in processed_df.columns:
                # Try to convert string numbers to numeric
                if processed_df[column].dtype == 'object':
                    # Check if it's actually numeric
                    try:
                        numeric_series = pd.to_numeric(processed_df[column], errors='coerce')
                        if numeric_series.notna().sum() > len(processed_df) * 0.8:  # 80% numeric
                            processed_df[column] = numeric_series
                            self.logger.info(f"Converted {column} to numeric")
                    except:
                        pass
                
                # Try to convert datetime strings
                if processed_df[column].dtype == 'object':
                    try:
                        datetime_series = pd.to_datetime(processed_df[column], errors='coerce')
                        if datetime_series.notna().sum() > len(processed_df) * 0.8:  # 80% valid dates
                            processed_df[column] = datetime_series
                            self.logger.info(f"Converted {column} to datetime")
                    except:
                        pass
            
            return processed_df
            
        except Exception as e:
            self.logger.error(f"Error validating data types: {e}")
            return df
    
    def _apply_custom_validators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply custom validation functions."""
        try:
            processed_df = df.copy()
            
            for column, validator in self.config.custom_validators.items():
                if column in processed_df.columns:
                    try:
                        processed_df[column] = processed_df[column].apply(validator)
                        self.logger.info(f"Applied custom validator to {column}")
                    except Exception as e:
                        self.logger.warning(f"Custom validator failed for {column}: {e}")
            
            return processed_df
            
        except Exception as e:
            self.logger.error(f"Error applying custom validators: {e}")
            return df
    
    def _create_lag_features(self, df: pd.DataFrame, columns: List[str], 
                            datetime_column: str) -> pd.DataFrame:
        """Create lag features for time series analysis."""
        try:
            feature_df = df.copy()
            feature_df = feature_df.sort_values(datetime_column)
            
            for column in columns:
                for lag in self.config.lag_periods:
                    lag_column = f"{column}_lag_{lag}"
                    feature_df[lag_column] = feature_df[column].shift(lag)
            
            return feature_df
            
        except Exception as e:
            self.logger.error(f"Error creating lag features: {e}")
            return df
    
    def _create_rolling_features(self, df: pd.DataFrame, columns: List[str], 
                                datetime_column: str) -> pd.DataFrame:
        """Create rolling window features."""
        try:
            feature_df = df.copy()
            feature_df = feature_df.sort_values(datetime_column)
            
            for column in columns:
                for window in self.config.rolling_windows:
                    # Rolling mean
                    feature_df[f"{column}_rolling_mean_{window}"] = feature_df[column].rolling(window=window).mean()
                    
                    # Rolling std
                    feature_df[f"{column}_rolling_std_{window}"] = feature_df[column].rolling(window=window).std()
                    
                    # Rolling min/max
                    feature_df[f"{column}_rolling_min_{window}"] = feature_df[column].rolling(window=window).min()
                    feature_df[f"{column}_rolling_max_{window}"] = feature_df[column].rolling(window=window).max()
            
            return feature_df
            
        except Exception as e:
            self.logger.error(f"Error creating rolling features: {e}")
            return df
    
    def _create_interaction_features(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Create interaction features between important variables."""
        try:
            feature_df = df.copy()
            
            # Create interactions between top correlated features
            if len(columns) >= 2:
                # Limit to top 5 features to avoid explosion
                top_columns = columns[:5]
                
                for i, col1 in enumerate(top_columns):
                    for col2 in top_columns[i+1:]:
                        # Multiplication interaction
                        feature_df[f"{col1}_x_{col2}"] = feature_df[col1] * feature_df[col2]
                        
                        # Ratio interaction (avoid division by zero)
                        with np.errstate(divide='ignore', invalid='ignore'):
                            ratio = feature_df[col1] / (feature_df[col2] + 1e-8)
                            feature_df[f"{col1}_div_{col2}"] = np.where(np.isfinite(ratio), ratio, 0)
            
            return feature_df
            
        except Exception as e:
            self.logger.error(f"Error creating interaction features: {e}")
            return df
    
    def _create_polynomial_features(self, df: pd.DataFrame, columns: List[str], degree: int = 2) -> pd.DataFrame:
        """Create polynomial features."""
        try:
            feature_df = df.copy()
            
            # Only create polynomial features for top 3 most important columns
            top_columns = columns[:3]
            
            for column in top_columns:
                for d in range(2, degree + 1):
                    feature_df[f"{column}_pow_{d}"] = feature_df[column] ** d
            
            return feature_df
            
        except Exception as e:
            self.logger.error(f"Error creating polynomial features: {e}")
            return df
    
    def _create_statistical_features(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Create statistical features."""
        try:
            feature_df = df.copy()
            
            # Create statistical aggregations across all numeric columns
            if len(columns) >= 2:
                numeric_data = feature_df[columns]
                
                # Row-wise statistics
                feature_df['row_mean'] = numeric_data.mean(axis=1)
                feature_df['row_std'] = numeric_data.std(axis=1)
                feature_df['row_min'] = numeric_data.min(axis=1)
                feature_df['row_max'] = numeric_data.max(axis=1)
                feature_df['row_median'] = numeric_data.median(axis=1)
                feature_df['row_skew'] = numeric_data.skew(axis=1)
            
            return feature_df
            
        except Exception as e:
            self.logger.error(f"Error creating statistical features: {e}")
            return df
    
    def _encode_categorical_features(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Encode categorical features."""
        try:
            feature_df = df.copy()
            
            for column in columns:
                if column in feature_df.columns:
                    # Use one-hot encoding for low cardinality, label encoding for high cardinality
                    unique_values = feature_df[column].nunique()
                    
                    if unique_values <= 10:  # One-hot encoding
                        dummies = pd.get_dummies(feature_df[column], prefix=column, drop_first=True)
                        feature_df = pd.concat([feature_df, dummies], axis=1)
                        feature_df.drop(column, axis=1, inplace=True)
                    else:  # Label encoding
                        from sklearn.preprocessing import LabelEncoder
                        le = LabelEncoder()
                        feature_df[f"{column}_encoded"] = le.fit_transform(feature_df[column].astype(str))
            
            return feature_df
            
        except Exception as e:
            self.logger.error(f"Error encoding categorical features: {e}")
            return df
    
    def _create_datetime_features(self, df: pd.DataFrame, datetime_column: str) -> pd.DataFrame:
        """Create datetime-based features."""
        try:
            feature_df = df.copy()
            
            if datetime_column in feature_df.columns:
                dt_col = pd.to_datetime(feature_df[datetime_column])
                
                # Basic datetime features
                feature_df[f"{datetime_column}_year"] = dt_col.dt.year
                feature_df[f"{datetime_column}_month"] = dt_col.dt.month
                feature_df[f"{datetime_column}_day"] = dt_col.dt.day
                feature_df[f"{datetime_column}_dayofweek"] = dt_col.dt.dayofweek
                feature_df[f"{datetime_column}_hour"] = dt_col.dt.hour
                feature_df[f"{datetime_column}_quarter"] = dt_col.dt.quarter
                
                # Cyclical features
                feature_df[f"{datetime_column}_month_sin"] = np.sin(2 * np.pi * dt_col.dt.month / 12)
                feature_df[f"{datetime_column}_month_cos"] = np.cos(2 * np.pi * dt_col.dt.month / 12)
                feature_df[f"{datetime_column}_day_sin"] = np.sin(2 * np.pi * dt_col.dt.day / 31)
                feature_df[f"{datetime_column}_day_cos"] = np.cos(2 * np.pi * dt_col.dt.day / 31)
                
                # Business features
                feature_df[f"{datetime_column}_is_weekend"] = (dt_col.dt.dayofweek >= 5).astype(int)
                feature_df[f"{datetime_column}_is_month_end"] = dt_col.dt.is_month_end.astype(int)
                feature_df[f"{datetime_column}_is_quarter_end"] = dt_col.dt.is_quarter_end.astype(int)
            
            return feature_df
            
        except Exception as e:
            self.logger.error(f"Error creating datetime features: {e}")
            return df
    
    def _create_technical_indicators(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Create technical indicators for financial data."""
        try:
            feature_df = df.copy()
            
            for column in columns:
                if column in feature_df.columns:
                    series = feature_df[column]
                    
                    # Simple moving averages
                    for window in [5, 10, 20]:
                        feature_df[f"{column}_sma_{window}"] = series.rolling(window=window).mean()
                    
                    # Exponential moving averages
                    for span in [5, 10, 20]:
                        feature_df[f"{column}_ema_{span}"] = series.ewm(span=span).mean()
                    
                    # Rate of change
                    for period in [1, 5, 10]:
                        feature_df[f"{column}_roc_{period}"] = series.pct_change(periods=period)
                    
                    # Bollinger Bands
                    rolling_mean = series.rolling(window=20).mean()
                    rolling_std = series.rolling(window=20).std()
                    feature_df[f"{column}_bb_upper"] = rolling_mean + (rolling_std * 2)
                    feature_df[f"{column}_bb_lower"] = rolling_mean - (rolling_std * 2)
                    feature_df[f"{column}_bb_width"] = feature_df[f"{column}_bb_upper"] - feature_df[f"{column}_bb_lower"]
                    
                    # RSI (Relative Strength Index)
                    delta = series.diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                    rs = gain / loss
                    feature_df[f"{column}_rsi"] = 100 - (100 / (1 + rs))
            
            return feature_df
            
        except Exception as e:
            self.logger.error(f"Error creating technical indicators: {e}")
            return df
    
    def _test_stationarity(self, series: pd.Series) -> Dict[str, Any]:
        """Test time series stationarity."""
        try:
            results = {}
            
            # Augmented Dickey-Fuller test
            adf_result = adfuller(series.dropna())
            results['adf'] = {
                'statistic': adf_result[0],
                'p_value': adf_result[1],
                'critical_values': adf_result[4],
                'is_stationary': adf_result[1] < 0.05
            }
            
            # KPSS test
            try:
                kpss_result = kpss(series.dropna())
                results['kpss'] = {
                    'statistic': kpss_result[0],
                    'p_value': kpss_result[1],
                    'critical_values': kpss_result[3],
                    'is_stationary': kpss_result[1] > 0.05
                }
            except Exception as e:
                self.logger.warning(f"KPSS test failed: {e}")
            
            # Overall stationarity assessment
            results['is_stationary'] = results['adf']['is_stationary']
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error testing stationarity: {e}")
            return {'is_stationary': False}
    
    def _detect_time_series_anomalies(self, series: pd.Series) -> Dict[str, Any]:
        """Detect anomalies in time series data."""
        try:
            anomalies = {}
            
            # Statistical outliers
            z_scores = np.abs(stats.zscore(series.dropna()))
            statistical_outliers = series[z_scores > 3]
            anomalies['statistical_outliers'] = {
                'count': len(statistical_outliers),
                'indices': statistical_outliers.index.tolist(),
                'values': statistical_outliers.values.tolist()
            }
            
            # Isolation Forest anomalies
            if len(series.dropna()) > 10:
                iso_forest = IsolationForest(contamination=0.1, random_state=42)
                anomaly_pred = iso_forest.fit_predict(series.dropna().values.reshape(-1, 1))
                isolation_anomalies = series.dropna()[anomaly_pred == -1]
                anomalies['isolation_forest'] = {
                    'count': len(isolation_anomalies),
                    'indices': isolation_anomalies.index.tolist(),
                    'values': isolation_anomalies.values.tolist()
                }
            
            return anomalies
            
        except Exception as e:
            self.logger.error(f"Error detecting time series anomalies: {e}")
            return {}
    
    def _calculate_volatility(self, series: pd.Series) -> Dict[str, float]:
        """Calculate various volatility measures."""
        try:
            returns = series.pct_change().dropna()
            
            volatility = {
                'std_volatility': float(returns.std()),
                'realized_volatility': float(np.sqrt(np.sum(returns**2))),
                'garch_volatility': float(returns.rolling(window=30).std().mean()) if len(returns) > 30 else 0.0
            }
            
            return volatility
            
        except Exception as e:
            self.logger.error(f"Error calculating volatility: {e}")
            return {'std_volatility': 0.0, 'realized_volatility': 0.0, 'garch_volatility': 0.0}
    
    def _calculate_quality_score(self, df: pd.DataFrame, missing_values: Dict[str, int], 
                                duplicate_records: int, outliers: Dict[str, int]) -> float:
        """Calculate overall data quality score."""
        try:
            total_cells = df.shape[0] * df.shape[1]
            total_missing = sum(missing_values.values())
            total_outliers = sum(outliers.values())
            
            # Calculate component scores (0-1 scale)
            completeness_score = 1 - (total_missing / total_cells) if total_cells > 0 else 0
            uniqueness_score = 1 - (duplicate_records / len(df)) if len(df) > 0 else 0
            consistency_score = 1 - (total_outliers / total_cells) if total_cells > 0 else 0
            
            # Weighted average (completeness is most important)
            quality_score = (0.5 * completeness_score + 0.3 * uniqueness_score + 0.2 * consistency_score)
            
            return max(0, min(1, quality_score))  # Ensure score is between 0 and 1
            
        except Exception as e:
            self.logger.error(f"Error calculating quality score: {e}")
            return 0.0
    
    def _generate_quality_recommendations(self, df: pd.DataFrame, missing_values: Dict[str, int], 
                                        duplicate_records: int, outliers: Dict[str, int], 
                                        target_column: Optional[str] = None) -> List[str]:
        """Generate data quality improvement recommendations."""
        recommendations = []
        
        try:
            # Missing values recommendations
            high_missing_cols = [col for col, count in missing_values.items() 
                               if count > len(df) * 0.3 and col != target_column]
            if high_missing_cols:
                recommendations.append(f"Consider dropping columns with >30% missing values: {high_missing_cols}")
            
            moderate_missing_cols = [col for col, count in missing_values.items() 
                                   if 0 < count <= len(df) * 0.3]
            if moderate_missing_cols:
                recommendations.append(f"Apply imputation strategies for columns: {moderate_missing_cols}")
            
            # Duplicate records
            if duplicate_records > 0:
                recommendations.append(f"Remove {duplicate_records} duplicate records")
            
            # Outliers
            high_outlier_cols = [col for col, count in outliers.items() if count > len(df) * 0.1]
            if high_outlier_cols:
                recommendations.append(f"Investigate outliers in columns: {high_outlier_cols}")
            
            # Data types
            object_cols = df.select_dtypes(include=['object']).columns.tolist()
            if object_cols:
                recommendations.append(f"Validate data types for columns: {object_cols}")
            
            # Sample size
            if len(df) < 1000:
                recommendations.append("Consider collecting more data for robust analysis")
            
            # Feature correlation
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                corr_matrix = df[numeric_cols].corr().abs()
                high_corr_pairs = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        if corr_matrix.iloc[i, j] > 0.9:
                            high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j]))
                
                if high_corr_pairs:
                    recommendations.append(f"Consider removing highly correlated features: {high_corr_pairs}")
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {e}")
            return ["Unable to generate specific recommendations due to processing error"]
    
    def _record_processing_step(self, step_name: str, input_shape: Tuple[int, int], 
                               output_shape: Tuple[int, int]):
        """Record processing step in history."""
        self.processing_history.append({
            'step': step_name,
            'timestamp': datetime.utcnow(),
            'input_shape': input_shape,
            'output_shape': output_shape,
            'records_changed': input_shape[0] - output_shape[0],
            'features_changed': input_shape[1] - output_shape[1]
        })
    
    def get_processing_summary(self) -> Dict[str, Any]:
        """Get summary of all processing steps.
        
        Returns:
            Dictionary with processing summary
        """
        return {
            'total_steps': len(self.processing_history),
            'processing_history': self.processing_history,
            'feature_importance': self.feature_importance,
            'config': self.config.__dict__
        }