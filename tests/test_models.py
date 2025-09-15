#!/usr/bin/env python3
"""
Unit Tests for Predictive Models

Tests for disruption prediction and sector impact analysis models.

Test Coverage:
- Disruption prediction model training and inference
- Sector impact analysis
- Model performance validation
- Feature engineering and data preprocessing
- Model persistence and loading

Author: Supply Chain Alpha Team
Version: 1.0.0
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from typing import List, Dict, Any
import tempfile
import pickle
from pathlib import Path

# Import test base classes
from tests.base_test import BaseTestCase, AsyncTestCase, MockDataGenerator, PerformanceTestMixin

# Import modules to test
from models.disruption_predictor import (
    DisruptionPredictor, DisruptionPrediction, CompanyImpactScore
)
from models.sector_impact_analyzer import (
    SectorImpactAnalyzer, SectorImpactScore, CrossSectorImpact, SectorRotationSignal
)


class TestDisruptionPredictor(BaseTestCase, PerformanceTestMixin):
    """Test cases for DisruptionPredictor."""
    
    def setUp(self):
        """Set up test fixtures."""
        super().setUp()
        
        self.config = {
            'models': {
                'port_congestion': {
                    'algorithm': 'random_forest',
                    'n_estimators': 100,
                    'max_depth': 10
                },
                'route_disruption': {
                    'algorithm': 'gradient_boosting',
                    'n_estimators': 50,
                    'learning_rate': 0.1
                },
                'rate_volatility': {
                    'algorithm': 'lstm',
                    'sequence_length': 30,
                    'hidden_units': 64
                }
            },
            'training': {
                'test_size': 0.2,
                'validation_size': 0.1,
                'cross_validation_folds': 5
            },
            'features': {
                'lag_periods': [1, 7, 30],
                'rolling_windows': [7, 14, 30],
                'technical_indicators': True
            }
        }
        
        self.predictor = DisruptionPredictor(self.config, self.db_manager)
    
    def test_initialization(self):
        """Test predictor initialization."""
        self.assertIsInstance(self.predictor, DisruptionPredictor)
        self.assertEqual(self.predictor.config, self.config)
        self.assertIsNotNone(self.predictor.db_manager)
        self.assertIsInstance(self.predictor.models, dict)
    
    def test_prepare_training_data(self):
        """Test training data preparation."""
        # Generate mock historical data
        vessels = self.mock_data.generate_vessel_data(100)
        ports = self.mock_data.generate_port_metrics(20)
        rates = self.mock_data.generate_freight_rates(200)
        
        # Prepare training data
        training_data = self.predictor.prepare_training_data(
            vessels, ports, rates
        )
        
        self.assertIsInstance(training_data, dict)
        self.assertIn('features', training_data)
        self.assertIn('targets', training_data)
        
        # Validate feature matrix
        features = training_data['features']
        self.assertIsInstance(features, pd.DataFrame)
        self.assertGreater(len(features.columns), 0)
        self.assertGreater(len(features), 0)
        
        # Check for no missing values in critical features
        critical_features = ['vessel_count', 'avg_congestion', 'rate_volatility']
        for feature in critical_features:
            if feature in features.columns:
                self.assertFalse(features[feature].isna().all())
    
    def test_feature_engineering(self):
        """Test feature engineering process."""
        # Create sample time series data
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        data = pd.DataFrame({
            'date': dates,
            'vessel_count': np.random.randint(50, 200, len(dates)),
            'congestion_level': np.random.uniform(0, 1, len(dates)),
            'freight_rate': np.random.uniform(1000, 5000, len(dates))
        })
        
        engineered_features = self.predictor.engineer_features(data)
        
        self.assertIsInstance(engineered_features, pd.DataFrame)
        self.assertGreaterEqual(len(engineered_features.columns), len(data.columns))
        
        # Check for lag features
        lag_columns = [col for col in engineered_features.columns if 'lag_' in col]
        self.assertGreater(len(lag_columns), 0)
        
        # Check for rolling window features
        rolling_columns = [col for col in engineered_features.columns if 'rolling_' in col]
        self.assertGreater(len(rolling_columns), 0)
    
    def test_train_port_congestion_model(self):
        """Test port congestion model training."""
        # Generate training data
        training_data = self._generate_mock_training_data()
        
        # Train model
        model_metrics = self.predictor.train_port_congestion_model(training_data)
        
        self.assertIsInstance(model_metrics, dict)
        self.assertIn('accuracy', model_metrics)
        self.assertIn('precision', model_metrics)
        self.assertIn('recall', model_metrics)
        self.assertIn('f1_score', model_metrics)
        
        # Validate metrics are reasonable
        for metric_name, metric_value in model_metrics.items():
            if metric_name in ['accuracy', 'precision', 'recall', 'f1_score']:
                self.assert_financial_metric_valid(
                    metric_value, metric_name, 0.0, 1.0
                )
    
    def test_train_route_disruption_model(self):
        """Test route disruption model training."""
        training_data = self._generate_mock_training_data()
        
        model_metrics = self.predictor.train_route_disruption_model(training_data)
        
        self.assertIsInstance(model_metrics, dict)
        self.assertIn('mse', model_metrics)
        self.assertIn('mae', model_metrics)
        self.assertIn('r2_score', model_metrics)
        
        # Validate regression metrics
        self.assert_financial_metric_valid(model_metrics['mse'], 'mse', 0)
        self.assert_financial_metric_valid(model_metrics['mae'], 'mae', 0)
    
    def test_predict_disruptions(self):
        """Test disruption prediction."""
        # First train a simple model
        training_data = self._generate_mock_training_data()
        self.predictor.train_port_congestion_model(training_data)
        
        # Generate current data for prediction
        current_vessels = self.mock_data.generate_vessel_data(20)
        current_ports = self.mock_data.generate_port_metrics(5)
        current_rates = self.mock_data.generate_freight_rates(30)
        
        predictions = self.predictor.predict_disruptions(
            current_vessels, current_ports, current_rates
        )
        
        self.assertIsInstance(predictions, list)
        
        for prediction in predictions:
            self.assert_prediction_valid(prediction)
    
    def test_assess_company_impact(self):
        """Test company impact assessment."""
        # Generate mock predictions
        predictions = self.mock_data.generate_disruption_predictions(10)
        
        # Test companies
        companies = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
        
        impact_scores = self.predictor.assess_company_impact(predictions, companies)
        
        self.assertIsInstance(impact_scores, list)
        self.assertEqual(len(impact_scores), len(companies))
        
        for impact_score in impact_scores:
            self.assertIsInstance(impact_score, CompanyImpactScore)
            self.assertIn(impact_score.company_symbol, companies)
            self.assert_financial_metric_valid(
                impact_score.impact_score, 'impact_score', 0.0, 1.0
            )
            self.assert_financial_metric_valid(
                impact_score.confidence, 'confidence', 0.0, 1.0
            )
    
    def test_model_persistence(self):
        """Test model saving and loading."""
        # Train a model
        training_data = self._generate_mock_training_data()
        self.predictor.train_port_congestion_model(training_data)
        
        # Save model
        model_path = self.test_data_dir / 'test_model.pkl'
        success = self.predictor.save_model('port_congestion', str(model_path))
        
        self.assertTrue(success)
        self.assertTrue(model_path.exists())
        
        # Load model
        loaded_success = self.predictor.load_model('port_congestion', str(model_path))
        
        self.assertTrue(loaded_success)
    
    def test_model_performance_validation(self):
        """Test model performance validation."""
        training_data = self._generate_mock_training_data()
        
        # Perform cross-validation
        cv_scores = self.predictor.cross_validate_model(
            'port_congestion', training_data
        )
        
        self.assertIsInstance(cv_scores, dict)
        self.assertIn('mean_score', cv_scores)
        self.assertIn('std_score', cv_scores)
        self.assertIn('individual_scores', cv_scores)
        
        # Validate scores
        self.assert_financial_metric_valid(
            cv_scores['mean_score'], 'mean_score', 0.0, 1.0
        )
    
    def test_feature_importance_analysis(self):
        """Test feature importance analysis."""
        training_data = self._generate_mock_training_data()
        self.predictor.train_port_congestion_model(training_data)
        
        feature_importance = self.predictor.get_feature_importance('port_congestion')
        
        self.assertIsInstance(feature_importance, dict)
        
        # Check that importance scores sum to approximately 1
        total_importance = sum(feature_importance.values())
        self.assertAlmostEqual(total_importance, 1.0, places=2)
        
        # All importance scores should be non-negative
        for feature, importance in feature_importance.items():
            self.assert_financial_metric_valid(importance, f'{feature}_importance', 0.0, 1.0)
    
    def test_prediction_performance(self):
        """Test prediction performance under load."""
        # Train model
        training_data = self._generate_mock_training_data()
        self.predictor.train_port_congestion_model(training_data)
        
        # Generate large dataset for performance testing
        vessels = self.mock_data.generate_vessel_data(1000)
        ports = self.mock_data.generate_port_metrics(50)
        rates = self.mock_data.generate_freight_rates(500)
        
        # Test prediction performance
        def predict_large_dataset():
            return self.predictor.predict_disruptions(vessels, ports, rates)
        
        # Should complete within reasonable time
        predictions = self.assert_execution_time(predict_large_dataset, 10.0)
        
        self.assertIsInstance(predictions, list)
        self.assertGreater(len(predictions), 0)
    
    def _generate_mock_training_data(self) -> Dict[str, Any]:
        """Generate mock training data for model testing.
        
        Returns:
            Dictionary with features and targets
        """
        n_samples = 1000
        n_features = 20
        
        # Generate feature matrix
        features = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        
        # Add some realistic feature names
        features.rename(columns={
            'feature_0': 'vessel_count',
            'feature_1': 'avg_congestion',
            'feature_2': 'rate_volatility',
            'feature_3': 'weather_score',
            'feature_4': 'seasonal_factor'
        }, inplace=True)
        
        # Generate binary targets for classification
        targets = np.random.binomial(1, 0.3, n_samples)  # 30% positive class
        
        return {
            'features': features,
            'targets': targets,
            'feature_names': list(features.columns),
            'target_names': ['no_disruption', 'disruption']
        }


class TestSectorImpactAnalyzer(BaseTestCase):
    """Test cases for SectorImpactAnalyzer."""
    
    def setUp(self):
        """Set up test fixtures."""
        super().setUp()
        
        self.config = {
            'sectors': {
                'transportation': ['FDX', 'UPS', 'DAL', 'UAL'],
                'retail': ['WMT', 'TGT', 'COST', 'HD'],
                'technology': ['AAPL', 'MSFT', 'GOOGL', 'AMZN'],
                'manufacturing': ['CAT', 'DE', 'GE', 'MMM'],
                'energy': ['XOM', 'CVX', 'COP', 'EOG']
            },
            'impact_weights': {
                'direct_exposure': 0.4,
                'supply_chain_dependency': 0.3,
                'geographic_exposure': 0.2,
                'operational_flexibility': 0.1
            },
            'rotation_thresholds': {
                'strong_rotation': 0.7,
                'moderate_rotation': 0.5,
                'weak_rotation': 0.3
            }
        }
        
        self.analyzer = SectorImpactAnalyzer(self.config, self.db_manager)
    
    def test_initialization(self):
        """Test analyzer initialization."""
        self.assertIsInstance(self.analyzer, SectorImpactAnalyzer)
        self.assertEqual(self.analyzer.config, self.config)
        self.assertIsInstance(self.analyzer.sector_mappings, dict)
    
    def test_assess_sector_impact(self):
        """Test sector impact assessment."""
        # Generate mock disruption predictions
        predictions = self.mock_data.generate_disruption_predictions(15)
        
        sector_impacts = self.analyzer.assess_sector_impact(predictions)
        
        self.assertIsInstance(sector_impacts, list)
        
        for impact in sector_impacts:
            self.assertIsInstance(impact, SectorImpactScore)
            self.assertIn(impact.sector_name, self.config['sectors'].keys())
            self.assert_financial_metric_valid(
                impact.impact_score, 'impact_score', 0.0, 1.0
            )
            self.assert_financial_metric_valid(
                impact.confidence, 'confidence', 0.0, 1.0
            )
    
    def test_analyze_cross_sector_impacts(self):
        """Test cross-sector impact analysis."""
        # Generate sector impacts
        predictions = self.mock_data.generate_disruption_predictions(10)
        sector_impacts = self.analyzer.assess_sector_impact(predictions)
        
        cross_impacts = self.analyzer.analyze_cross_sector_impacts(sector_impacts)
        
        self.assertIsInstance(cross_impacts, list)
        
        for cross_impact in cross_impacts:
            self.assertIsInstance(cross_impact, CrossSectorImpact)
            self.assertIn(cross_impact.source_sector, self.config['sectors'].keys())
            self.assertIn(cross_impact.target_sector, self.config['sectors'].keys())
            self.assert_financial_metric_valid(
                cross_impact.impact_strength, 'impact_strength', 0.0, 1.0
            )
    
    def test_generate_rotation_signals(self):
        """Test sector rotation signal generation."""
        # Generate sector impacts
        predictions = self.mock_data.generate_disruption_predictions(20)
        sector_impacts = self.analyzer.assess_sector_impact(predictions)
        
        rotation_signals = self.analyzer.generate_rotation_signals(sector_impacts)
        
        self.assertIsInstance(rotation_signals, list)
        
        for signal in rotation_signals:
            self.assertIsInstance(signal, RotationSignal)
            self.assertIn(signal.from_sector, self.config['sectors'].keys())
            self.assertIn(signal.to_sector, self.config['sectors'].keys())
            self.assertIn(signal.signal_strength, ['weak', 'moderate', 'strong'])
            self.assert_financial_metric_valid(
                signal.confidence, 'confidence', 0.0, 1.0
            )
    
    def test_calculate_sector_correlations(self):
        """Test sector correlation calculation."""
        # Generate mock stock price data
        all_symbols = []
        for sector_stocks in self.config['sectors'].values():
            all_symbols.extend(sector_stocks)
        
        stock_data = self.mock_data.generate_stock_data(all_symbols, days=252)
        
        correlations = self.analyzer.calculate_sector_correlations(stock_data)
        
        self.assertIsInstance(correlations, pd.DataFrame)
        
        # Check correlation matrix properties
        self.assertEqual(correlations.shape[0], correlations.shape[1])  # Square matrix
        
        # Diagonal should be 1.0 (perfect self-correlation)
        for i in range(len(correlations)):
            self.assertAlmostEqual(correlations.iloc[i, i], 1.0, places=2)
        
        # Correlations should be between -1 and 1
        for col in correlations.columns:
            for val in correlations[col]:
                self.assert_financial_metric_valid(val, 'correlation', -1.0, 1.0)
    
    def test_generate_impact_report(self):
        """Test impact report generation."""
        predictions = self.mock_data.generate_disruption_predictions(25)
        
        report = self.analyzer.generate_impact_report(predictions)
        
        self.assertIsInstance(report, dict)
        
        # Check required report sections
        required_sections = [
            'executive_summary',
            'sector_impacts',
            'cross_sector_effects',
            'rotation_recommendations',
            'risk_assessment'
        ]
        
        for section in required_sections:
            self.assertIn(section, report)
        
        # Validate executive summary
        exec_summary = report['executive_summary']
        self.assertIn('total_disruptions', exec_summary)
        self.assertIn('most_affected_sector', exec_summary)
        self.assertIn('overall_risk_level', exec_summary)
    
    def test_sector_exposure_calculation(self):
        """Test sector exposure calculation to supply chain disruptions."""
        sector_name = 'transportation'
        
        exposure_score = self.analyzer.calculate_sector_exposure(
            sector_name, 'port_congestion'
        )
        
        self.assert_financial_metric_valid(
            exposure_score, 'exposure_score', 0.0, 1.0
        )
    
    def test_historical_impact_analysis(self):
        """Test historical impact analysis."""
        # Generate historical data
        start_date = datetime.utcnow() - timedelta(days=365)
        end_date = datetime.utcnow()
        
        historical_analysis = self.analyzer.analyze_historical_impacts(
            start_date, end_date
        )
        
        self.assertIsInstance(historical_analysis, dict)
        self.assertIn('impact_trends', historical_analysis)
        self.assertIn('sector_performance', historical_analysis)
        self.assertIn('correlation_changes', historical_analysis)


class TestModelsIntegration(BaseTestCase):
    """Integration tests for predictive models."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        super().setUp()
        
        # Initialize both models
        self.predictor = DisruptionPredictor({}, self.db_manager)
        self.analyzer = SectorImpactAnalyzer({
            'sectors': {
                'transportation': ['FDX', 'UPS'],
                'retail': ['WMT', 'TGT'],
                'technology': ['AAPL', 'MSFT']
            }
        }, self.db_manager)
    
    def test_end_to_end_prediction_pipeline(self):
        """Test complete prediction pipeline from data to signals."""
        # Generate input data
        vessels = self.mock_data.generate_vessel_data(50)
        ports = self.mock_data.generate_port_metrics(10)
        rates = self.mock_data.generate_freight_rates(100)
        
        # Step 1: Generate disruption predictions
        predictions = self.predictor.predict_disruptions(vessels, ports, rates)
        
        # Step 2: Assess sector impacts
        sector_impacts = self.analyzer.assess_sector_impact(predictions)
        
        # Step 3: Generate rotation signals
        rotation_signals = self.analyzer.generate_rotation_signals(sector_impacts)
        
        # Validate pipeline output
        self.assertIsInstance(predictions, list)
        self.assertIsInstance(sector_impacts, list)
        self.assertIsInstance(rotation_signals, list)
        
        # Ensure data flows correctly through pipeline
        if predictions:
            self.assertGreater(len(sector_impacts), 0)
        
        if sector_impacts:
            # Should have rotation signals if there are significant impacts
            significant_impacts = [s for s in sector_impacts if s.impact_score > 0.5]
            if significant_impacts:
                self.assertGreater(len(rotation_signals), 0)
    
    def test_model_consistency(self):
        """Test consistency between model predictions."""
        # Generate same input data
        vessels = self.mock_data.generate_vessel_data(30)
        ports = self.mock_data.generate_port_metrics(5)
        rates = self.mock_data.generate_freight_rates(50)
        
        # Run predictions multiple times
        predictions_1 = self.predictor.predict_disruptions(vessels, ports, rates)
        predictions_2 = self.predictor.predict_disruptions(vessels, ports, rates)
        
        # Results should be consistent (assuming deterministic models)
        self.assertEqual(len(predictions_1), len(predictions_2))
        
        # Compare prediction probabilities (should be very similar)
        if predictions_1 and predictions_2:
            prob_diff = abs(
                predictions_1[0].probability - predictions_2[0].probability
            )
            self.assertLess(prob_diff, 0.01)  # Less than 1% difference
    
    def test_model_performance_under_load(self):
        """Test model performance with large datasets."""
        # Generate large dataset
        vessels = self.mock_data.generate_vessel_data(500)
        ports = self.mock_data.generate_port_metrics(50)
        rates = self.mock_data.generate_freight_rates(1000)
        
        def run_full_pipeline():
            predictions = self.predictor.predict_disruptions(vessels, ports, rates)
            sector_impacts = self.analyzer.assess_sector_impact(predictions)
            rotation_signals = self.analyzer.generate_rotation_signals(sector_impacts)
            return predictions, sector_impacts, rotation_signals
        
        # Should complete within reasonable time
        results = self.assert_execution_time(run_full_pipeline, 30.0)
        
        predictions, sector_impacts, rotation_signals = results
        
        # Validate results
        self.assertIsInstance(predictions, list)
        self.assertIsInstance(sector_impacts, list)
        self.assertIsInstance(rotation_signals, list)


if __name__ == '__main__':
    unittest.main()