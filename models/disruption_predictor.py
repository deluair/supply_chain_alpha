#!/usr/bin/env python3
"""
Supply Chain Disruption Predictor
Machine learning models to predict supply chain disruptions and their impact.

Features:
- Port congestion prediction
- Shipping route disruption forecasting
- Freight rate volatility prediction
- Company/sector impact assessment
- Risk scoring and alerts

Author: Supply Chain Alpha Team
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
import joblib
from pathlib import Path

# Machine Learning imports
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, classification_report, roc_auc_score
from sklearn.feature_selection import SelectKBest, f_regression

# Time series analysis
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

# Deep learning (optional)
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

from utils.database import DatabaseManager
from utils.feature_engineering import FeatureEngineer


@dataclass
class DisruptionPrediction:
    """Data structure for disruption predictions."""
    prediction_id: str
    disruption_type: str  # 'port_congestion', 'route_blockage', 'rate_spike'
    location: str
    probability: float  # 0-1
    severity_score: float  # 0-10
    predicted_start_date: datetime
    predicted_duration_days: int
    confidence_interval: Tuple[float, float]
    affected_routes: List[str]
    affected_companies: List[str]
    impact_sectors: List[str]
    timestamp: datetime
    model_version: str


@dataclass
class CompanyImpactScore:
    """Data structure for company impact assessment."""
    company_symbol: str
    company_name: str
    sector: str
    supply_chain_exposure: float  # 0-1
    disruption_sensitivity: float  # 0-1
    predicted_impact_score: float  # -10 to +10
    revenue_at_risk_pct: float
    key_risk_factors: List[str]
    mitigation_score: float  # 0-1
    timestamp: datetime


class DisruptionPredictor:
    """Machine learning system for predicting supply chain disruptions."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the disruption predictor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Model configurations
        self.model_config = config.get('models', {}).get('disruption_predictor', {})
        
        # Database manager
        self.db_manager = DatabaseManager()
        
        # Feature engineer
        self.feature_engineer = FeatureEngineer()
        
        # Model storage
        self.models_dir = Path('models/saved_models')
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize models
        self.models = {
            'port_congestion': None,
            'route_disruption': None,
            'rate_volatility': None,
            'company_impact': None
        }
        
        # Scalers for feature normalization
        self.scalers = {}
        
        # Feature importance tracking
        self.feature_importance = {}
        
        # Load pre-trained models if available
        self._load_models()
    
    async def train_models(self, retrain: bool = False) -> Dict[str, Any]:
        """Train all disruption prediction models.
        
        Args:
            retrain: Whether to retrain existing models
            
        Returns:
            Dictionary with training results
        """
        self.logger.info("Starting disruption prediction model training")
        
        try:
            # Prepare training data
            training_data = await self._prepare_training_data()
            
            if not training_data:
                raise ValueError("No training data available")
            
            results = {}
            
            # Train port congestion model
            if retrain or self.models['port_congestion'] is None:
                port_results = await self._train_port_congestion_model(training_data)
                results['port_congestion'] = port_results
            
            # Train route disruption model
            if retrain or self.models['route_disruption'] is None:
                route_results = await self._train_route_disruption_model(training_data)
                results['route_disruption'] = route_results
            
            # Train rate volatility model
            if retrain or self.models['rate_volatility'] is None:
                rate_results = await self._train_rate_volatility_model(training_data)
                results['rate_volatility'] = rate_results
            
            # Train company impact model
            if retrain or self.models['company_impact'] is None:
                company_results = await self._train_company_impact_model(training_data)
                results['company_impact'] = company_results
            
            # Save trained models
            self._save_models()
            
            self.logger.info(f"Model training completed: {results}")
            return results
            
        except Exception as e:
            self.logger.error(f"Error in model training: {e}")
            raise
    
    async def predict_disruptions(self, forecast_days: int = 30) -> List[DisruptionPrediction]:
        """Generate disruption predictions for the specified forecast period.
        
        Args:
            forecast_days: Number of days to forecast
            
        Returns:
            List of DisruptionPrediction objects
        """
        self.logger.info(f"Generating disruption predictions for {forecast_days} days")
        
        try:
            # Get current data for prediction
            current_data = await self._get_current_data()
            
            if current_data.empty:
                self.logger.warning("No current data available for predictions")
                return []
            
            predictions = []
            
            # Port congestion predictions
            port_predictions = await self._predict_port_congestion(current_data, forecast_days)
            predictions.extend(port_predictions)
            
            # Route disruption predictions
            route_predictions = await self._predict_route_disruptions(current_data, forecast_days)
            predictions.extend(route_predictions)
            
            # Rate volatility predictions
            rate_predictions = await self._predict_rate_volatility(current_data, forecast_days)
            predictions.extend(rate_predictions)
            
            # Filter and rank predictions by probability
            high_confidence_predictions = [
                p for p in predictions if p.probability > 0.3
            ]
            
            # Sort by probability * severity
            high_confidence_predictions.sort(
                key=lambda x: x.probability * x.severity_score, reverse=True
            )
            
            self.logger.info(f"Generated {len(high_confidence_predictions)} high-confidence predictions")
            return high_confidence_predictions
            
        except Exception as e:
            self.logger.error(f"Error generating predictions: {e}")
            raise
    
    async def assess_company_impact(self, companies: List[str]) -> List[CompanyImpactScore]:
        """Assess the impact of predicted disruptions on specific companies.
        
        Args:
            companies: List of company symbols to assess
            
        Returns:
            List of CompanyImpactScore objects
        """
        self.logger.info(f"Assessing impact for {len(companies)} companies")
        
        try:
            # Get current disruption predictions
            disruptions = await self.predict_disruptions()
            
            # Get company data
            company_data = await self._get_company_data(companies)
            
            impact_scores = []
            
            for company in companies:
                try:
                    impact_score = await self._calculate_company_impact(
                        company, disruptions, company_data
                    )
                    if impact_score:
                        impact_scores.append(impact_score)
                        
                except Exception as e:
                    self.logger.error(f"Error assessing impact for {company}: {e}")
            
            # Sort by impact score (most negative first)
            impact_scores.sort(key=lambda x: x.predicted_impact_score)
            
            self.logger.info(f"Completed impact assessment for {len(impact_scores)} companies")
            return impact_scores
            
        except Exception as e:
            self.logger.error(f"Error in company impact assessment: {e}")
            raise
    
    async def _prepare_training_data(self) -> Dict[str, pd.DataFrame]:
        """Prepare training data from historical records.
        
        Returns:
            Dictionary containing training datasets
        """
        try:
            # Get historical data from database
            queries = {
                'port_data': """
                    SELECT * FROM port_metrics 
                    WHERE timestamp >= NOW() - INTERVAL '2 years'
                    ORDER BY timestamp
                """,
                'shipping_data': """
                    SELECT * FROM vessel_data 
                    WHERE timestamp >= NOW() - INTERVAL '2 years'
                    ORDER BY timestamp
                """,
                'freight_data': """
                    SELECT * FROM container_rates 
                    WHERE timestamp >= NOW() - INTERVAL '2 years'
                    ORDER BY timestamp
                """,
                'disruption_events': """
                    SELECT * FROM historical_disruptions 
                    WHERE event_date >= NOW() - INTERVAL '2 years'
                    ORDER BY event_date
                """
            }
            
            training_data = {}
            
            for data_type, query in queries.items():
                try:
                    data = await self.db_manager.execute_query(query)
                    if data:
                        training_data[data_type] = pd.DataFrame(data)
                    else:
                        # Generate synthetic data if no historical data
                        training_data[data_type] = self._generate_synthetic_data(data_type)
                        
                except Exception as e:
                    self.logger.warning(f"Error loading {data_type}: {e}")
                    training_data[data_type] = self._generate_synthetic_data(data_type)
            
            return training_data
            
        except Exception as e:
            self.logger.error(f"Error preparing training data: {e}")
            raise
    
    def _generate_synthetic_data(self, data_type: str) -> pd.DataFrame:
        """Generate synthetic training data when historical data is not available.
        
        Args:
            data_type: Type of data to generate
            
        Returns:
            DataFrame with synthetic data
        """
        np.random.seed(42)  # For reproducibility
        
        if data_type == 'port_data':
            dates = pd.date_range(start='2022-01-01', end='2024-01-01', freq='D')
            return pd.DataFrame({
                'port_name': np.random.choice(['Los Angeles', 'Long Beach', 'Shanghai', 'Singapore'], len(dates)),
                'throughput_teu': np.random.normal(10000, 2000, len(dates)),
                'congestion_index': np.random.beta(2, 5, len(dates)),
                'avg_wait_time': np.random.exponential(2, len(dates)),
                'berth_utilization': np.random.beta(8, 2, len(dates)),
                'timestamp': dates
            })
        
        elif data_type == 'shipping_data':
            dates = pd.date_range(start='2022-01-01', end='2024-01-01', freq='H')
            return pd.DataFrame({
                'vessel_id': np.random.randint(1000, 9999, len(dates)),
                'route': np.random.choice(['Asia-US West', 'Asia-Europe', 'Transpacific'], len(dates)),
                'speed_knots': np.random.normal(15, 3, len(dates)),
                'delay_hours': np.random.exponential(1, len(dates)),
                'timestamp': dates
            })
        
        elif data_type == 'freight_data':
            dates = pd.date_range(start='2022-01-01', end='2024-01-01', freq='D')
            return pd.DataFrame({
                'origin': np.random.choice(['Shanghai', 'Ningbo', 'Shenzhen'], len(dates)),
                'destination': np.random.choice(['Los Angeles', 'Long Beach', 'New York'], len(dates)),
                'rate_40ft': np.random.normal(2500, 500, len(dates)),
                'timestamp': dates
            })
        
        elif data_type == 'disruption_events':
            # Generate fewer disruption events (they're rare)
            dates = pd.date_range(start='2022-01-01', end='2024-01-01', freq='30D')
            return pd.DataFrame({
                'event_type': np.random.choice(['port_strike', 'weather', 'congestion', 'accident'], len(dates)),
                'location': np.random.choice(['Los Angeles', 'Suez Canal', 'Shanghai', 'Singapore'], len(dates)),
                'severity': np.random.randint(1, 11, len(dates)),
                'duration_days': np.random.exponential(3, len(dates)),
                'event_date': dates
            })
        
        else:
            return pd.DataFrame()
    
    async def _train_port_congestion_model(self, training_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Train the port congestion prediction model.
        
        Args:
            training_data: Dictionary containing training datasets
            
        Returns:
            Dictionary with training results
        """
        try:
            port_data = training_data.get('port_data', pd.DataFrame())
            
            if port_data.empty:
                raise ValueError("No port data available for training")
            
            # Feature engineering
            features = self.feature_engineer.create_port_features(port_data)
            
            # Target variable: future congestion level
            target = port_data['congestion_index'].shift(-1).fillna(0)
            
            # Remove rows with missing values
            valid_idx = ~(features.isnull().any(axis=1) | target.isnull())
            X = features[valid_idx]
            y = target[valid_idx]
            
            if len(X) < 10:
                raise ValueError("Insufficient data for training")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            
            model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            train_score = model.score(X_train_scaled, y_train)
            test_score = model.score(X_test_scaled, y_test)
            
            y_pred = model.predict(X_test_scaled)
            mse = mean_squared_error(y_test, y_pred)
            
            # Store model and scaler
            self.models['port_congestion'] = model
            self.scalers['port_congestion'] = scaler
            
            # Feature importance
            self.feature_importance['port_congestion'] = dict(
                zip(X.columns, model.feature_importances_)
            )
            
            return {
                'model_type': 'RandomForestRegressor',
                'train_score': train_score,
                'test_score': test_score,
                'mse': mse,
                'n_features': len(X.columns),
                'n_samples': len(X)
            }
            
        except Exception as e:
            self.logger.error(f"Error training port congestion model: {e}")
            raise
    
    async def _train_route_disruption_model(self, training_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Train the route disruption prediction model.
        
        Args:
            training_data: Dictionary containing training datasets
            
        Returns:
            Dictionary with training results
        """
        try:
            shipping_data = training_data.get('shipping_data', pd.DataFrame())
            disruption_events = training_data.get('disruption_events', pd.DataFrame())
            
            if shipping_data.empty:
                raise ValueError("No shipping data available for training")
            
            # Feature engineering
            features = self.feature_engineer.create_shipping_features(shipping_data)
            
            # Create binary target: disruption or no disruption
            # Mark periods around known disruption events as positive
            target = np.zeros(len(features))
            
            if not disruption_events.empty:
                for _, event in disruption_events.iterrows():
                    event_date = pd.to_datetime(event['event_date'])
                    # Mark 7 days before and during event as disruption period
                    disruption_mask = (
                        (features.index >= event_date - timedelta(days=7)) &
                        (features.index <= event_date + timedelta(days=event.get('duration_days', 1)))
                    )
                    target[disruption_mask] = 1
            
            # Remove rows with missing values
            valid_idx = ~features.isnull().any(axis=1)
            X = features[valid_idx]
            y = target[valid_idx]
            
            if len(X) < 10 or y.sum() < 2:
                # Not enough positive samples, create balanced synthetic data
                n_samples = max(100, len(X))
                X = pd.DataFrame(np.random.randn(n_samples, 5), 
                               columns=['feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5'])
                y = np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=6,
                random_state=42
            )
            
            model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            train_score = model.score(X_train_scaled, y_train)
            test_score = model.score(X_test_scaled, y_test)
            
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            
            try:
                auc_score = roc_auc_score(y_test, y_pred_proba)
            except ValueError:
                auc_score = 0.5  # Default for single class
            
            # Store model and scaler
            self.models['route_disruption'] = model
            self.scalers['route_disruption'] = scaler
            
            # Feature importance
            self.feature_importance['route_disruption'] = dict(
                zip(X.columns, model.feature_importances_)
            )
            
            return {
                'model_type': 'GradientBoostingClassifier',
                'train_score': train_score,
                'test_score': test_score,
                'auc_score': auc_score,
                'n_features': len(X.columns),
                'n_samples': len(X),
                'positive_samples': int(y.sum())
            }
            
        except Exception as e:
            self.logger.error(f"Error training route disruption model: {e}")
            raise
    
    async def _train_rate_volatility_model(self, training_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Train the freight rate volatility prediction model.
        
        Args:
            training_data: Dictionary containing training datasets
            
        Returns:
            Dictionary with training results
        """
        try:
            freight_data = training_data.get('freight_data', pd.DataFrame())
            
            if freight_data.empty:
                raise ValueError("No freight data available for training")
            
            # Calculate rolling volatility as target
            freight_data = freight_data.sort_values('timestamp')
            freight_data['rate_change'] = freight_data['rate_40ft'].pct_change()
            freight_data['volatility'] = freight_data['rate_change'].rolling(window=7).std()
            
            # Feature engineering
            features = self.feature_engineer.create_freight_features(freight_data)
            
            # Target: future volatility (next 7 days)
            target = freight_data['volatility'].shift(-7).fillna(0)
            
            # Remove rows with missing values
            valid_idx = ~(features.isnull().any(axis=1) | target.isnull())
            X = features[valid_idx]
            y = target[valid_idx]
            
            if len(X) < 10:
                # Generate synthetic data
                n_samples = 100
                X = pd.DataFrame(np.random.randn(n_samples, 4), 
                               columns=['price_trend', 'volume_trend', 'seasonal_factor', 'market_stress'])
                y = np.random.exponential(0.1, n_samples)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=8,
                random_state=42,
                n_jobs=-1
            )
            
            model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            train_score = model.score(X_train_scaled, y_train)
            test_score = model.score(X_test_scaled, y_test)
            
            y_pred = model.predict(X_test_scaled)
            mse = mean_squared_error(y_test, y_pred)
            
            # Store model and scaler
            self.models['rate_volatility'] = model
            self.scalers['rate_volatility'] = scaler
            
            # Feature importance
            self.feature_importance['rate_volatility'] = dict(
                zip(X.columns, model.feature_importances_)
            )
            
            return {
                'model_type': 'RandomForestRegressor',
                'train_score': train_score,
                'test_score': test_score,
                'mse': mse,
                'n_features': len(X.columns),
                'n_samples': len(X)
            }
            
        except Exception as e:
            self.logger.error(f"Error training rate volatility model: {e}")
            raise
    
    async def _train_company_impact_model(self, training_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Train the company impact assessment model.
        
        Args:
            training_data: Dictionary containing training datasets
            
        Returns:
            Dictionary with training results
        """
        try:
            # This would typically use company financial data and supply chain exposure
            # For now, create a simple model based on sector and size
            
            # Generate synthetic company data
            companies = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'TSLA', 'META', 'NVDA', 'NFLX']
            sectors = ['Technology', 'Consumer Discretionary', 'Communication Services']
            
            n_samples = 200
            X = pd.DataFrame({
                'market_cap': np.random.lognormal(10, 1, n_samples),
                'supply_chain_complexity': np.random.beta(2, 3, n_samples),
                'international_exposure': np.random.beta(3, 2, n_samples),
                'inventory_turnover': np.random.gamma(2, 2, n_samples),
                'sector_encoded': np.random.randint(0, len(sectors), n_samples)
            })
            
            # Target: impact score (-10 to +10)
            y = (
                -2 * X['supply_chain_complexity'] +
                -1.5 * X['international_exposure'] +
                1 * X['inventory_turnover'] +
                np.random.normal(0, 1, n_samples)
            )
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            model = LinearRegression()
            model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            train_score = model.score(X_train_scaled, y_train)
            test_score = model.score(X_test_scaled, y_test)
            
            y_pred = model.predict(X_test_scaled)
            mse = mean_squared_error(y_test, y_pred)
            
            # Store model and scaler
            self.models['company_impact'] = model
            self.scalers['company_impact'] = scaler
            
            return {
                'model_type': 'LinearRegression',
                'train_score': train_score,
                'test_score': test_score,
                'mse': mse,
                'n_features': len(X.columns),
                'n_samples': len(X)
            }
            
        except Exception as e:
            self.logger.error(f"Error training company impact model: {e}")
            raise
    
    async def _get_current_data(self) -> pd.DataFrame:
        """Get current data for making predictions.
        
        Returns:
            DataFrame with current data
        """
        try:
            # Get recent data from all sources
            queries = {
                'port_metrics': "SELECT * FROM port_metrics WHERE timestamp >= NOW() - INTERVAL '7 days'",
                'vessel_data': "SELECT * FROM vessel_data WHERE timestamp >= NOW() - INTERVAL '24 hours'",
                'container_rates': "SELECT * FROM container_rates WHERE timestamp >= NOW() - INTERVAL '7 days'",
                'baltic_indices': "SELECT * FROM baltic_indices WHERE timestamp >= NOW() - INTERVAL '24 hours'"
            }
            
            current_data = {}
            
            for data_type, query in queries.items():
                try:
                    data = await self.db_manager.execute_query(query)
                    if data:
                        current_data[data_type] = pd.DataFrame(data)
                    else:
                        # Generate current synthetic data
                        current_data[data_type] = self._generate_current_synthetic_data(data_type)
                        
                except Exception as e:
                    self.logger.warning(f"Error loading current {data_type}: {e}")
                    current_data[data_type] = self._generate_current_synthetic_data(data_type)
            
            # Combine all data into a single DataFrame for prediction
            combined_data = pd.DataFrame()
            
            for data_type, df in current_data.items():
                if not df.empty:
                    # Add data type prefix to columns
                    df_prefixed = df.add_prefix(f"{data_type}_")
                    if combined_data.empty:
                        combined_data = df_prefixed
                    else:
                        combined_data = pd.concat([combined_data, df_prefixed], axis=1)
            
            return combined_data
            
        except Exception as e:
            self.logger.error(f"Error getting current data: {e}")
            return pd.DataFrame()
    
    def _generate_current_synthetic_data(self, data_type: str) -> pd.DataFrame:
        """Generate current synthetic data for predictions.
        
        Args:
            data_type: Type of data to generate
            
        Returns:
            DataFrame with current synthetic data
        """
        current_time = datetime.utcnow()
        
        if data_type == 'port_metrics':
            return pd.DataFrame({
                'port_name': ['Los Angeles', 'Long Beach', 'Shanghai'],
                'throughput_teu': [12000, 11000, 15000],
                'congestion_index': [0.6, 0.7, 0.4],
                'avg_wait_time': [3.2, 4.1, 2.1],
                'berth_utilization': [0.85, 0.92, 0.78],
                'timestamp': [current_time] * 3
            })
        
        elif data_type == 'vessel_data':
            return pd.DataFrame({
                'vessel_id': [1001, 1002, 1003],
                'route': ['Asia-US West', 'Asia-Europe', 'Transpacific'],
                'speed_knots': [14.5, 16.2, 15.8],
                'delay_hours': [2.1, 0.5, 1.2],
                'timestamp': [current_time] * 3
            })
        
        elif data_type == 'container_rates':
            return pd.DataFrame({
                'origin': ['Shanghai', 'Ningbo', 'Shenzhen'],
                'destination': ['Los Angeles', 'Los Angeles', 'Long Beach'],
                'rate_40ft': [2800, 2750, 2900],
                'timestamp': [current_time] * 3
            })
        
        elif data_type == 'baltic_indices':
            return pd.DataFrame({
                'index_name': ['BDI', 'BCI', 'BPI'],
                'index_value': [1250, 2100, 1580],
                'change_percent': [-2.1, 1.5, -0.8],
                'timestamp': [current_time] * 3
            })
        
        else:
            return pd.DataFrame()
    
    def _save_models(self) -> None:
        """Save trained models to disk."""
        try:
            for model_name, model in self.models.items():
                if model is not None:
                    model_path = self.models_dir / f"{model_name}_model.joblib"
                    joblib.dump(model, model_path)
                    
                    # Save scaler if exists
                    if model_name in self.scalers:
                        scaler_path = self.models_dir / f"{model_name}_scaler.joblib"
                        joblib.dump(self.scalers[model_name], scaler_path)
            
            # Save feature importance
            importance_path = self.models_dir / "feature_importance.joblib"
            joblib.dump(self.feature_importance, importance_path)
            
            self.logger.info("Models saved successfully")
            
        except Exception as e:
            self.logger.error(f"Error saving models: {e}")
    
    def _load_models(self) -> None:
        """Load pre-trained models from disk."""
        try:
            for model_name in self.models.keys():
                model_path = self.models_dir / f"{model_name}_model.joblib"
                scaler_path = self.models_dir / f"{model_name}_scaler.joblib"
                
                if model_path.exists():
                    self.models[model_name] = joblib.load(model_path)
                    
                    if scaler_path.exists():
                        self.scalers[model_name] = joblib.load(scaler_path)
            
            # Load feature importance
            importance_path = self.models_dir / "feature_importance.joblib"
            if importance_path.exists():
                self.feature_importance = joblib.load(importance_path)
            
            loaded_models = [name for name, model in self.models.items() if model is not None]
            if loaded_models:
                self.logger.info(f"Loaded models: {loaded_models}")
            
        except Exception as e:
            self.logger.warning(f"Error loading models: {e}")
    
    async def _predict_port_congestion(self, current_data: pd.DataFrame, 
                                      forecast_days: int) -> List[DisruptionPrediction]:
        """Predict port congestion disruptions.
        
        Args:
            current_data: Current data for prediction
            forecast_days: Number of days to forecast
            
        Returns:
            List of port congestion predictions
        """
        predictions = []
        
        try:
            if self.models['port_congestion'] is None:
                return predictions
            
            # Extract port-related features
            port_features = [col for col in current_data.columns if 'port' in col.lower()]
            
            if not port_features:
                return predictions
            
            # Make predictions for major ports
            major_ports = ['Los Angeles', 'Long Beach', 'Shanghai', 'Singapore', 'Rotterdam']
            
            for port in major_ports:
                try:
                    # Create feature vector for this port
                    features = self._create_port_feature_vector(current_data, port)
                    
                    if features is not None:
                        # Scale features
                        scaler = self.scalers.get('port_congestion')
                        if scaler:
                            features_scaled = scaler.transform([features])
                        else:
                            features_scaled = [features]
                        
                        # Predict congestion level
                        congestion_prob = self.models['port_congestion'].predict(features_scaled)[0]
                        
                        # Convert to disruption probability
                        disruption_prob = max(0, min(1, (congestion_prob - 0.5) * 2))
                        
                        if disruption_prob > 0.3:  # Only include significant predictions
                            prediction = DisruptionPrediction(
                                prediction_id=f"port_{port}_{datetime.utcnow().strftime('%Y%m%d')}",
                                disruption_type='port_congestion',
                                location=port,
                                probability=disruption_prob,
                                severity_score=min(10, congestion_prob * 10),
                                predicted_start_date=datetime.utcnow() + timedelta(days=1),
                                predicted_duration_days=int(3 + congestion_prob * 7),
                                confidence_interval=(max(0, disruption_prob - 0.2), min(1, disruption_prob + 0.2)),
                                affected_routes=[f"{port}-US", f"{port}-Europe"],
                                affected_companies=[],  # To be filled by company impact model
                                impact_sectors=['Transportation', 'Retail', 'Manufacturing'],
                                timestamp=datetime.utcnow(),
                                model_version='1.0'
                            )
                            predictions.append(prediction)
                
                except Exception as e:
                    self.logger.error(f"Error predicting congestion for {port}: {e}")
        
        except Exception as e:
            self.logger.error(f"Error in port congestion prediction: {e}")
        
        return predictions
    
    def _create_port_feature_vector(self, current_data: pd.DataFrame, port: str) -> Optional[List[float]]:
        """Create feature vector for port congestion prediction.
        
        Args:
            current_data: Current data
            port: Port name
            
        Returns:
            Feature vector or None
        """
        try:
            # Default feature vector if no specific data available
            features = [
                0.6,  # throughput_normalized
                0.7,  # congestion_index
                3.0,  # avg_wait_time
                0.85, # berth_utilization
                1.2   # seasonal_factor
            ]
            
            # Try to extract actual values from current_data if available
            port_columns = [col for col in current_data.columns if 'port' in col.lower()]
            
            if port_columns and not current_data.empty:
                # Use first row of data (most recent)
                row = current_data.iloc[0]
                
                # Extract relevant metrics
                for i, col in enumerate(port_columns[:len(features)]):
                    if pd.notna(row[col]):
                        features[i] = float(row[col])
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error creating feature vector for {port}: {e}")
            return None
    
    async def _predict_route_disruptions(self, current_data: pd.DataFrame, 
                                        forecast_days: int) -> List[DisruptionPrediction]:
        """Predict shipping route disruptions.
        
        Args:
            current_data: Current data for prediction
            forecast_days: Number of days to forecast
            
        Returns:
            List of route disruption predictions
        """
        predictions = []
        
        try:
            if self.models['route_disruption'] is None:
                return predictions
            
            # Major shipping routes to monitor
            major_routes = [
                'Asia-US West Coast',
                'Asia-US East Coast', 
                'Asia-Europe',
                'Transpacific',
                'Transatlantic'
            ]
            
            for route in major_routes:
                try:
                    # Create feature vector for this route
                    features = self._create_route_feature_vector(current_data, route)
                    
                    if features is not None:
                        # Scale features
                        scaler = self.scalers.get('route_disruption')
                        if scaler:
                            features_scaled = scaler.transform([features])
                        else:
                            features_scaled = [features]
                        
                        # Predict disruption probability
                        disruption_prob = self.models['route_disruption'].predict_proba(features_scaled)[0][1]
                        
                        if disruption_prob > 0.3:  # Only include significant predictions
                            prediction = DisruptionPrediction(
                                prediction_id=f"route_{route.replace(' ', '_')}_{datetime.utcnow().strftime('%Y%m%d')}",
                                disruption_type='route_blockage',
                                location=route,
                                probability=disruption_prob,
                                severity_score=min(10, disruption_prob * 8 + 2),
                                predicted_start_date=datetime.utcnow() + timedelta(days=2),
                                predicted_duration_days=int(1 + disruption_prob * 5),
                                confidence_interval=(max(0, disruption_prob - 0.15), min(1, disruption_prob + 0.15)),
                                affected_routes=[route],
                                affected_companies=[],
                                impact_sectors=['Transportation', 'Retail', 'Manufacturing', 'Energy'],
                                timestamp=datetime.utcnow(),
                                model_version='1.0'
                            )
                            predictions.append(prediction)
                
                except Exception as e:
                    self.logger.error(f"Error predicting disruption for {route}: {e}")
        
        except Exception as e:
            self.logger.error(f"Error in route disruption prediction: {e}")
        
        return predictions
    
    def _create_route_feature_vector(self, current_data: pd.DataFrame, route: str) -> Optional[List[float]]:
        """Create feature vector for route disruption prediction.
        
        Args:
            current_data: Current data
            route: Route name
            
        Returns:
            Feature vector or None
        """
        try:
            # Default feature vector
            features = [
                15.0,  # avg_speed_knots
                1.5,   # avg_delay_hours
                0.8,   # route_utilization
                0.2,   # weather_risk
                0.1    # geopolitical_risk
            ]
            
            # Try to extract actual values from current_data if available
            vessel_columns = [col for col in current_data.columns if 'vessel' in col.lower()]
            
            if vessel_columns and not current_data.empty:
                row = current_data.iloc[0]
                
                # Extract relevant metrics
                for i, col in enumerate(vessel_columns[:len(features)]):
                    if pd.notna(row[col]):
                        features[i] = float(row[col])
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error creating route feature vector for {route}: {e}")
            return None
    
    async def _predict_rate_volatility(self, current_data: pd.DataFrame, 
                                      forecast_days: int) -> List[DisruptionPrediction]:
        """Predict freight rate volatility spikes.
        
        Args:
            current_data: Current data for prediction
            forecast_days: Number of days to forecast
            
        Returns:
            List of rate volatility predictions
        """
        predictions = []
        
        try:
            if self.models['rate_volatility'] is None:
                return predictions
            
            # Major trade lanes to monitor
            trade_lanes = [
                'Asia-US West Coast',
                'Asia-US East Coast',
                'Asia-Europe',
                'Europe-US'
            ]
            
            for lane in trade_lanes:
                try:
                    # Create feature vector for this trade lane
                    features = self._create_rate_feature_vector(current_data, lane)
                    
                    if features is not None:
                        # Scale features
                        scaler = self.scalers.get('rate_volatility')
                        if scaler:
                            features_scaled = scaler.transform([features])
                        else:
                            features_scaled = [features]
                        
                        # Predict volatility
                        predicted_volatility = self.models['rate_volatility'].predict(features_scaled)[0]
                        
                        # Convert to disruption probability
                        disruption_prob = min(1, max(0, predicted_volatility * 3))  # Scale volatility to probability
                        
                        if disruption_prob > 0.4:  # Only include significant predictions
                            prediction = DisruptionPrediction(
                                prediction_id=f"rate_{lane.replace(' ', '_')}_{datetime.utcnow().strftime('%Y%m%d')}",
                                disruption_type='rate_spike',
                                location=lane,
                                probability=disruption_prob,
                                severity_score=min(10, predicted_volatility * 20),
                                predicted_start_date=datetime.utcnow() + timedelta(days=3),
                                predicted_duration_days=int(5 + predicted_volatility * 10),
                                confidence_interval=(max(0, disruption_prob - 0.2), min(1, disruption_prob + 0.2)),
                                affected_routes=[lane],
                                affected_companies=[],
                                impact_sectors=['Transportation', 'Retail', 'Manufacturing'],
                                timestamp=datetime.utcnow(),
                                model_version='1.0'
                            )
                            predictions.append(prediction)
                
                except Exception as e:
                    self.logger.error(f"Error predicting rate volatility for {lane}: {e}")
        
        except Exception as e:
            self.logger.error(f"Error in rate volatility prediction: {e}")
        
        return predictions
    
    def _create_rate_feature_vector(self, current_data: pd.DataFrame, lane: str) -> Optional[List[float]]:
        """Create feature vector for rate volatility prediction.
        
        Args:
            current_data: Current data
            lane: Trade lane name
            
        Returns:
            Feature vector or None
        """
        try:
            # Default feature vector
            features = [
                0.05,  # price_trend
                1.2,   # volume_trend
                0.8,   # seasonal_factor
                0.3    # market_stress
            ]
            
            # Try to extract actual values from current_data if available
            rate_columns = [col for col in current_data.columns if 'rate' in col.lower() or 'container' in col.lower()]
            
            if rate_columns and not current_data.empty:
                row = current_data.iloc[0]
                
                # Extract relevant metrics
                for i, col in enumerate(rate_columns[:len(features)]):
                    if pd.notna(row[col]):
                        # Normalize the value
                        features[i] = float(row[col]) / 1000 if 'rate' in col.lower() else float(row[col])
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error creating rate feature vector for {lane}: {e}")
            return None
    
    async def _get_company_data(self, companies: List[str]) -> pd.DataFrame:
        """Get company data for impact assessment.
        
        Args:
            companies: List of company symbols
            
        Returns:
            DataFrame with company data
        """
        try:
            # This would typically fetch from financial data APIs
            # For now, create synthetic company data
            
            company_data = []
            
            for symbol in companies:
                company_data.append({
                    'symbol': symbol,
                    'market_cap': np.random.lognormal(10, 1),
                    'supply_chain_complexity': np.random.beta(2, 3),
                    'international_exposure': np.random.beta(3, 2),
                    'inventory_turnover': np.random.gamma(2, 2),
                    'sector': np.random.choice(['Technology', 'Consumer Discretionary', 'Healthcare'])
                })
            
            return pd.DataFrame(company_data)
            
        except Exception as e:
            self.logger.error(f"Error getting company data: {e}")
            return pd.DataFrame()
    
    async def _calculate_company_impact(self, company: str, 
                                       disruptions: List[DisruptionPrediction],
                                       company_data: pd.DataFrame) -> Optional[CompanyImpactScore]:
        """Calculate impact score for a specific company.
        
        Args:
            company: Company symbol
            disruptions: List of predicted disruptions
            company_data: Company data DataFrame
            
        Returns:
            CompanyImpactScore object or None
        """
        try:
            # Get company info
            company_info = company_data[company_data['symbol'] == company]
            
            if company_info.empty:
                return None
            
            company_row = company_info.iloc[0]
            
            # Calculate base impact from disruptions
            total_impact = 0
            risk_factors = []
            
            for disruption in disruptions:
                # Calculate impact based on disruption type and company exposure
                impact_multiplier = 1.0
                
                if disruption.disruption_type == 'port_congestion':
                    impact_multiplier = company_row['supply_chain_complexity'] * 2
                    risk_factors.append(f"Port congestion at {disruption.location}")
                
                elif disruption.disruption_type == 'route_blockage':
                    impact_multiplier = company_row['international_exposure'] * 1.5
                    risk_factors.append(f"Route disruption: {disruption.location}")
                
                elif disruption.disruption_type == 'rate_spike':
                    impact_multiplier = (2 - company_row['inventory_turnover'] / 5) * 1.2
                    risk_factors.append(f"Freight rate spike: {disruption.location}")
                
                # Calculate weighted impact
                disruption_impact = (
                    disruption.probability * 
                    disruption.severity_score * 
                    impact_multiplier
                )
                
                total_impact += disruption_impact
            
            # Normalize impact score to -10 to +10 scale
            impact_score = max(-10, min(10, total_impact - 5))
            
            # Calculate revenue at risk percentage
            revenue_at_risk = min(50, abs(impact_score) * 2)
            
            # Calculate mitigation score (higher is better)
            mitigation_score = (
                (1 - company_row['supply_chain_complexity']) * 0.4 +
                company_row['inventory_turnover'] / 10 * 0.3 +
                (company_row['market_cap'] / 1e12) * 0.3  # Larger companies have more resources
            )
            mitigation_score = max(0, min(1, mitigation_score))
            
            return CompanyImpactScore(
                company_symbol=company,
                company_name=f"{company} Inc.",  # Simplified
                sector=company_row['sector'],
                supply_chain_exposure=company_row['supply_chain_complexity'],
                disruption_sensitivity=company_row['international_exposure'],
                predicted_impact_score=impact_score,
                revenue_at_risk_pct=revenue_at_risk,
                key_risk_factors=risk_factors[:3],  # Top 3 risks
                mitigation_score=mitigation_score,
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating impact for {company}: {e}")
            return None