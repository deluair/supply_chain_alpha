"""Feature engineering utilities for supply chain data.

This module provides feature engineering capabilities for transforming
raw supply chain data into features suitable for machine learning models.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression


@dataclass
class FeatureConfig:
    """Configuration for feature engineering."""
    time_windows: List[int] = None  # Rolling window sizes in days
    lag_features: List[int] = None  # Lag periods in days
    technical_indicators: List[str] = None  # Technical indicators to compute
    aggregation_functions: List[str] = None  # Aggregation functions
    scaling_method: str = "standard"  # Scaling method
    feature_selection_k: Optional[int] = None  # Number of features to select
    
    def __post_init__(self):
        if self.time_windows is None:
            self.time_windows = [7, 14, 30, 60]
        if self.lag_features is None:
            self.lag_features = [1, 3, 7, 14]
        if self.technical_indicators is None:
            self.technical_indicators = ['sma', 'ema', 'rsi', 'bollinger']
        if self.aggregation_functions is None:
            self.aggregation_functions = ['mean', 'std', 'min', 'max', 'median']


class FeatureEngineer:
    """Feature engineering for supply chain data."""
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        """Initialize feature engineer.
        
        Args:
            config: Feature engineering configuration
        """
        self.config = config or FeatureConfig()
        self.scalers = {}
        self.feature_names = []
        
    def create_time_features(self, df: pd.DataFrame, date_column: str = 'date') -> pd.DataFrame:
        """Create time-based features.
        
        Args:
            df: Input dataframe
            date_column: Name of date column
            
        Returns:
            Dataframe with time features
        """
        df = df.copy()
        
        # Ensure date column is datetime
        if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
            df[date_column] = pd.to_datetime(df[date_column])
            
        # Extract time components
        df['year'] = df[date_column].dt.year
        df['month'] = df[date_column].dt.month
        df['day'] = df[date_column].dt.day
        df['dayofweek'] = df[date_column].dt.dayofweek
        df['dayofyear'] = df[date_column].dt.dayofyear
        df['quarter'] = df[date_column].dt.quarter
        df['week'] = df[date_column].dt.isocalendar().week
        
        # Cyclical encoding for periodic features
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
        df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
        df['dayofyear_sin'] = np.sin(2 * np.pi * df['dayofyear'] / 365)
        df['dayofyear_cos'] = np.cos(2 * np.pi * df['dayofyear'] / 365)
        
        # Business day indicators
        df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
        df['is_month_start'] = df[date_column].dt.is_month_start.astype(int)
        df['is_month_end'] = df[date_column].dt.is_month_end.astype(int)
        df['is_quarter_start'] = df[date_column].dt.is_quarter_start.astype(int)
        df['is_quarter_end'] = df[date_column].dt.is_quarter_end.astype(int)
        
        return df
        
    def create_lag_features(self, df: pd.DataFrame, value_columns: List[str], 
                           group_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Create lagged features.
        
        Args:
            df: Input dataframe
            value_columns: Columns to create lags for
            group_columns: Columns to group by for lagging
            
        Returns:
            Dataframe with lag features
        """
        df = df.copy()
        
        for col in value_columns:
            for lag in self.config.lag_features:
                if group_columns:
                    df[f'{col}_lag_{lag}'] = df.groupby(group_columns)[col].shift(lag)
                else:
                    df[f'{col}_lag_{lag}'] = df[col].shift(lag)
                    
        return df
        
    def create_rolling_features(self, df: pd.DataFrame, value_columns: List[str],
                               group_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Create rolling window features.
        
        Args:
            df: Input dataframe
            value_columns: Columns to create rolling features for
            group_columns: Columns to group by for rolling windows
            
        Returns:
            Dataframe with rolling features
        """
        df = df.copy()
        
        for col in value_columns:
            for window in self.config.time_windows:
                for agg_func in self.config.aggregation_functions:
                    feature_name = f'{col}_rolling_{window}d_{agg_func}'
                    
                    if group_columns:
                        df[feature_name] = (
                            df.groupby(group_columns)[col]
                            .rolling(window=window, min_periods=1)
                            .agg(agg_func)
                            .reset_index(level=group_columns, drop=True)
                        )
                    else:
                        df[feature_name] = (
                            df[col].rolling(window=window, min_periods=1).agg(agg_func)
                        )
                        
        return df
        
    def create_technical_indicators(self, df: pd.DataFrame, price_column: str = 'price',
                                   volume_column: Optional[str] = None) -> pd.DataFrame:
        """Create technical indicators.
        
        Args:
            df: Input dataframe
            price_column: Price column name
            volume_column: Volume column name (optional)
            
        Returns:
            Dataframe with technical indicators
        """
        df = df.copy()
        
        if price_column not in df.columns:
            return df
            
        # Simple Moving Average
        if 'sma' in self.config.technical_indicators:
            for window in [5, 10, 20, 50]:
                df[f'sma_{window}'] = df[price_column].rolling(window=window).mean()
                
        # Exponential Moving Average
        if 'ema' in self.config.technical_indicators:
            for span in [5, 10, 20, 50]:
                df[f'ema_{span}'] = df[price_column].ewm(span=span).mean()
                
        # Relative Strength Index
        if 'rsi' in self.config.technical_indicators:
            delta = df[price_column].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
        # Bollinger Bands
        if 'bollinger' in self.config.technical_indicators:
            sma_20 = df[price_column].rolling(window=20).mean()
            std_20 = df[price_column].rolling(window=20).std()
            df['bollinger_upper'] = sma_20 + (2 * std_20)
            df['bollinger_lower'] = sma_20 - (2 * std_20)
            df['bollinger_width'] = df['bollinger_upper'] - df['bollinger_lower']
            df['bollinger_position'] = (df[price_column] - df['bollinger_lower']) / df['bollinger_width']
            
        # MACD
        if 'macd' in self.config.technical_indicators:
            ema_12 = df[price_column].ewm(span=12).mean()
            ema_26 = df[price_column].ewm(span=26).mean()
            df['macd'] = ema_12 - ema_26
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
            
        # Volume indicators (if volume data available)
        if volume_column and volume_column in df.columns:
            # Volume Moving Average
            df['volume_sma_20'] = df[volume_column].rolling(window=20).mean()
            df['volume_ratio'] = df[volume_column] / df['volume_sma_20']
            
            # On-Balance Volume
            df['obv'] = (df[volume_column] * np.sign(df[price_column].diff())).cumsum()
            
        return df
        
    def create_interaction_features(self, df: pd.DataFrame, 
                                   feature_pairs: List[Tuple[str, str]]) -> pd.DataFrame:
        """Create interaction features between pairs of columns.
        
        Args:
            df: Input dataframe
            feature_pairs: List of column pairs to create interactions for
            
        Returns:
            Dataframe with interaction features
        """
        df = df.copy()
        
        for col1, col2 in feature_pairs:
            if col1 in df.columns and col2 in df.columns:
                # Multiplication
                df[f'{col1}_x_{col2}'] = df[col1] * df[col2]
                
                # Division (with zero handling)
                df[f'{col1}_div_{col2}'] = df[col1] / (df[col2] + 1e-8)
                
                # Difference
                df[f'{col1}_diff_{col2}'] = df[col1] - df[col2]
                
                # Ratio
                df[f'{col1}_ratio_{col2}'] = df[col1] / (df[col1] + df[col2] + 1e-8)
                
        return df
        
    def create_statistical_features(self, df: pd.DataFrame, value_columns: List[str],
                                   group_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Create statistical features.
        
        Args:
            df: Input dataframe
            value_columns: Columns to create statistical features for
            group_columns: Columns to group by
            
        Returns:
            Dataframe with statistical features
        """
        df = df.copy()
        
        for col in value_columns:
            if col not in df.columns:
                continue
                
            # Z-score (standardized values)
            if group_columns:
                df[f'{col}_zscore'] = df.groupby(group_columns)[col].transform(
                    lambda x: (x - x.mean()) / (x.std() + 1e-8)
                )
            else:
                df[f'{col}_zscore'] = (df[col] - df[col].mean()) / (df[col].std() + 1e-8)
                
            # Percentile rank
            if group_columns:
                df[f'{col}_percentile'] = df.groupby(group_columns)[col].rank(pct=True)
            else:
                df[f'{col}_percentile'] = df[col].rank(pct=True)
                
            # Change from previous period
            if group_columns:
                df[f'{col}_change'] = df.groupby(group_columns)[col].diff()
                df[f'{col}_pct_change'] = df.groupby(group_columns)[col].pct_change()
            else:
                df[f'{col}_change'] = df[col].diff()
                df[f'{col}_pct_change'] = df[col].pct_change()
                
            # Volatility (rolling standard deviation)
            for window in [7, 14, 30]:
                if group_columns:
                    df[f'{col}_volatility_{window}d'] = (
                        df.groupby(group_columns)[col]
                        .rolling(window=window, min_periods=1)
                        .std()
                        .reset_index(level=group_columns, drop=True)
                    )
                else:
                    df[f'{col}_volatility_{window}d'] = (
                        df[col].rolling(window=window, min_periods=1).std()
                    )
                    
        return df
        
    def scale_features(self, df: pd.DataFrame, feature_columns: List[str],
                      fit: bool = True) -> pd.DataFrame:
        """Scale features using specified scaling method.
        
        Args:
            df: Input dataframe
            feature_columns: Columns to scale
            fit: Whether to fit the scaler
            
        Returns:
            Dataframe with scaled features
        """
        df = df.copy()
        
        # Initialize scaler if not exists
        if self.config.scaling_method not in self.scalers:
            if self.config.scaling_method == 'standard':
                self.scalers[self.config.scaling_method] = StandardScaler()
            elif self.config.scaling_method == 'minmax':
                self.scalers[self.config.scaling_method] = MinMaxScaler()
            elif self.config.scaling_method == 'robust':
                self.scalers[self.config.scaling_method] = RobustScaler()
            else:
                raise ValueError(f"Unknown scaling method: {self.config.scaling_method}")
                
        scaler = self.scalers[self.config.scaling_method]
        
        # Select only existing columns
        existing_columns = [col for col in feature_columns if col in df.columns]
        
        if existing_columns:
            if fit:
                df[existing_columns] = scaler.fit_transform(df[existing_columns])
            else:
                df[existing_columns] = scaler.transform(df[existing_columns])
                
        return df
        
    def select_features(self, df: pd.DataFrame, target_column: str,
                       feature_columns: Optional[List[str]] = None) -> Tuple[pd.DataFrame, List[str]]:
        """Select best features using statistical tests.
        
        Args:
            df: Input dataframe
            target_column: Target column name
            feature_columns: Feature columns to select from
            
        Returns:
            Tuple of (dataframe with selected features, selected feature names)
        """
        if self.config.feature_selection_k is None:
            return df, feature_columns or []
            
        if feature_columns is None:
            feature_columns = [col for col in df.columns if col != target_column]
            
        # Remove non-numeric columns and handle missing values
        numeric_features = []
        for col in feature_columns:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                numeric_features.append(col)
                
        if not numeric_features:
            return df, []
            
        # Prepare data
        X = df[numeric_features].fillna(0)
        y = df[target_column].fillna(0)
        
        # Select features
        selector = SelectKBest(score_func=f_regression, k=min(self.config.feature_selection_k, len(numeric_features)))
        X_selected = selector.fit_transform(X, y)
        
        # Get selected feature names
        selected_features = [numeric_features[i] for i in selector.get_support(indices=True)]
        
        # Create result dataframe
        result_df = df.copy()
        result_df = result_df[[target_column] + selected_features]
        
        return result_df, selected_features
        
    def engineer_features(self, df: pd.DataFrame, 
                         value_columns: List[str],
                         date_column: str = 'date',
                         group_columns: Optional[List[str]] = None,
                         target_column: Optional[str] = None) -> pd.DataFrame:
        """Apply full feature engineering pipeline.
        
        Args:
            df: Input dataframe
            value_columns: Columns to engineer features for
            date_column: Date column name
            group_columns: Grouping columns
            target_column: Target column for feature selection
            
        Returns:
            Dataframe with engineered features
        """
        result_df = df.copy()
        
        # Create time features
        if date_column in result_df.columns:
            result_df = self.create_time_features(result_df, date_column)
            
        # Create lag features
        result_df = self.create_lag_features(result_df, value_columns, group_columns)
        
        # Create rolling features
        result_df = self.create_rolling_features(result_df, value_columns, group_columns)
        
        # Create statistical features
        result_df = self.create_statistical_features(result_df, value_columns, group_columns)
        
        # Create technical indicators (if price column exists)
        price_columns = [col for col in value_columns if 'price' in col.lower()]
        if price_columns:
            result_df = self.create_technical_indicators(result_df, price_columns[0])
            
        # Create interaction features for important pairs
        if len(value_columns) >= 2:
            important_pairs = [(value_columns[i], value_columns[j]) 
                             for i in range(len(value_columns)) 
                             for j in range(i+1, min(i+3, len(value_columns)))]
            result_df = self.create_interaction_features(result_df, important_pairs)
            
        # Scale features
        numeric_columns = result_df.select_dtypes(include=[np.number]).columns.tolist()
        if target_column and target_column in numeric_columns:
            numeric_columns.remove(target_column)
            
        if numeric_columns:
            result_df = self.scale_features(result_df, numeric_columns)
            
        # Feature selection
        if target_column and self.config.feature_selection_k:
            result_df, selected_features = self.select_features(result_df, target_column, numeric_columns)
            self.feature_names = selected_features
            
        return result_df
        
    def get_feature_importance(self, df: pd.DataFrame, target_column: str,
                              feature_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Calculate feature importance using mutual information.
        
        Args:
            df: Input dataframe
            target_column: Target column name
            feature_columns: Feature columns to analyze
            
        Returns:
            Dataframe with feature importance scores
        """
        if feature_columns is None:
            feature_columns = [col for col in df.columns if col != target_column]
            
        # Select numeric features
        numeric_features = []
        for col in feature_columns:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                numeric_features.append(col)
                
        if not numeric_features:
            return pd.DataFrame()
            
        # Prepare data
        X = df[numeric_features].fillna(0)
        y = df[target_column].fillna(0)
        
        # Calculate mutual information
        mi_scores = mutual_info_regression(X, y)
        
        # Create importance dataframe
        importance_df = pd.DataFrame({
            'feature': numeric_features,
            'importance': mi_scores
        }).sort_values('importance', ascending=False)
        
        return importance_df