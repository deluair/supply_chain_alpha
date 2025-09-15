#!/usr/bin/env python3
"""
Unit Tests for Utility Modules

Tests for database management, data processing, ESG metrics, and logging utilities.

Test Coverage:
- Database operations and connections
- Data quality assessment and cleaning
- ESG metrics collection and integration
- Logging and performance monitoring
- Configuration management

Author: Supply Chain Alpha Team
Version: 1.0.0
"""

import unittest
from unittest.mock import Mock, patch, MagicMock, mock_open
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
import tempfile
import json
import logging
from pathlib import Path
import sqlite3
import os

# Import test base classes
from tests.base_test import BaseTestCase, AsyncTestCase, MockDataGenerator, PerformanceTestMixin

# Import modules to test
from utils.database import DatabaseManager
from utils.data_processing import DataProcessor, DataQualityReport, ProcessingConfig
from utils.esg_metrics import (
    ESGMetricsIntegrator, ESGScore, SupplyChainSustainabilityMetrics, ESGAdjustedSignal
)
from utils.logging_utils import SupplyChainLogger, LogLevel, PerformanceMetrics


class TestDatabaseManager(BaseTestCase):
    """Test cases for DatabaseManager."""
    
    def setUp(self):
        """Set up test fixtures."""
        super().setUp()
        
        # Use SQLite for testing (in-memory)
        db_config = {
            'db_type': 'sqlite',
            'host': 'localhost',
            'port': 5432,
            'database': ':memory:',
            'file_path': ':memory:'
        }
        
        self.db_manager = DatabaseManager(db_config)
    
    def test_initialization(self):
        """Test database manager initialization."""
        self.assertIsInstance(self.db_manager, DatabaseManager)
        self.assertIsNotNone(self.db_manager.config)
    
    def test_sqlite_connection(self):
        """Test SQLite database connection."""
        # Connect to SQLite
        success = self.db_manager.connect()
        self.assertTrue(success)
        
        # Test connection is active
        self.assertTrue(self.db_manager.is_connected())
        
        # Disconnect
        self.db_manager.disconnect()
    
    def test_create_tables(self):
        """Test table creation."""
        self.db_manager.connect()
        
        # Create test table
        create_sql = """
        CREATE TABLE IF NOT EXISTS test_vessels (
            id INTEGER PRIMARY KEY,
            vessel_name TEXT NOT NULL,
            imo_number TEXT UNIQUE,
            vessel_type TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        
        success = self.db_manager.execute_query(create_sql)
        self.assertTrue(success)
        
        # Verify table exists
        tables = self.db_manager.get_table_names()
        self.assertIn('test_vessels', tables)
        
        self.db_manager.disconnect()
    
    def test_data_insertion_and_retrieval(self):
        """Test data insertion and retrieval."""
        self.db_manager.connect()
        
        # Create test table
        self.db_manager.execute_query("""
            CREATE TABLE test_data (
                id INTEGER PRIMARY KEY,
                name TEXT,
                value REAL,
                timestamp TIMESTAMP
            )
        """)
        
        # Insert test data
        test_records = [
            ('Record 1', 10.5, datetime.utcnow()),
            ('Record 2', 20.3, datetime.utcnow()),
            ('Record 3', 15.7, datetime.utcnow())
        ]
        
        for name, value, timestamp in test_records:
            success = self.db_manager.insert_data(
                'test_data',
                {'name': name, 'value': value, 'timestamp': timestamp}
            )
            self.assertTrue(success)
        
        # Retrieve data
        results = self.db_manager.fetch_data(
            "SELECT * FROM test_data ORDER BY id"
        )
        
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0]['name'], 'Record 1')
        self.assertAlmostEqual(results[0]['value'], 10.5, places=2)
        
        self.db_manager.disconnect()
    
    def test_bulk_operations(self):
        """Test bulk data operations."""
        self.db_manager.connect()
        
        # Create test table
        self.db_manager.execute_query("""
            CREATE TABLE bulk_test (
                id INTEGER PRIMARY KEY,
                symbol TEXT,
                price REAL,
                volume INTEGER
            )
        """)
        
        # Generate bulk data
        bulk_data = []
        for i in range(1000):
            bulk_data.append({
                'symbol': f'STOCK{i:03d}',
                'price': 100 + (i * 0.1),
                'volume': 1000 + (i * 10)
            })
        
        # Insert bulk data
        success = self.db_manager.bulk_insert('bulk_test', bulk_data)
        self.assertTrue(success)
        
        # Verify data count
        count_result = self.db_manager.fetch_data(
            "SELECT COUNT(*) as count FROM bulk_test"
        )
        self.assertEqual(count_result[0]['count'], 1000)
        
        self.db_manager.disconnect()
    
    def test_query_optimization(self):
        """Test query optimization features."""
        self.db_manager.connect()
        
        # Create indexed table
        self.db_manager.execute_query("""
            CREATE TABLE indexed_data (
                id INTEGER PRIMARY KEY,
                symbol TEXT,
                date DATE,
                price REAL
            )
        """)
        
        # Create index
        self.db_manager.execute_query(
            "CREATE INDEX idx_symbol_date ON indexed_data(symbol, date)"
        )
        
        # Insert test data
        test_data = []
        for i in range(100):
            test_data.append({
                'symbol': f'STOCK{i % 10}',
                'date': datetime.utcnow().date(),
                'price': 100 + np.random.random() * 50
            })
        
        self.db_manager.bulk_insert('indexed_data', test_data)
        
        # Test optimized query
        results = self.db_manager.fetch_data(
            "SELECT * FROM indexed_data WHERE symbol = ? ORDER BY date",
            ('STOCK1',)
        )
        
        self.assertGreater(len(results), 0)
        
        self.db_manager.disconnect()
    
    def test_transaction_management(self):
        """Test database transaction management."""
        self.db_manager.connect()
        
        # Create test table
        self.db_manager.execute_query("""
            CREATE TABLE transaction_test (
                id INTEGER PRIMARY KEY,
                value INTEGER
            )
        """)
        
        # Test successful transaction
        with self.db_manager.transaction():
            self.db_manager.insert_data('transaction_test', {'value': 1})
            self.db_manager.insert_data('transaction_test', {'value': 2})
        
        # Verify data was committed
        results = self.db_manager.fetch_data("SELECT COUNT(*) as count FROM transaction_test")
        self.assertEqual(results[0]['count'], 2)
        
        # Test failed transaction (should rollback)
        try:
            with self.db_manager.transaction():
                self.db_manager.insert_data('transaction_test', {'value': 3})
                # Simulate error
                raise Exception("Test error")
        except Exception:
            pass
        
        # Count should still be 2 (rollback occurred)
        results = self.db_manager.fetch_data("SELECT COUNT(*) as count FROM transaction_test")
        self.assertEqual(results[0]['count'], 2)
        
        self.db_manager.disconnect()
    
    @patch('redis.Redis')
    def test_redis_operations(self, mock_redis):
        """Test Redis cache operations."""
        # Mock Redis instance
        mock_redis_instance = Mock()
        mock_redis.return_value = mock_redis_instance
        
        # Test cache operations
        self.db_manager.cache_data('test_key', {'data': 'test_value'}, ttl=3600)
        mock_redis_instance.setex.assert_called_once()
        
        # Test cache retrieval
        mock_redis_instance.get.return_value = json.dumps({'data': 'test_value'})
        cached_data = self.db_manager.get_cached_data('test_key')
        
        self.assertEqual(cached_data['data'], 'test_value')
    
    def test_backup_and_restore(self):
        """Test database manager basic functionality."""
        # Test initialization
        self.assertIsInstance(self.db_manager, DatabaseManager)
        self.assertIsNotNone(self.db_manager.config)
        
        # Test config properties
        self.assertEqual(self.db_manager.config.db_type, 'sqlite')
        self.assertEqual(self.db_manager.config.database, ':memory:')
        
        # Test connection status
        self.assertFalse(self.db_manager.is_connected)


class TestDataProcessor(BaseTestCase):
    """Test cases for DataProcessor."""
    
    def setUp(self):
        """Set up test fixtures."""
        super().setUp()
        
        self.config = ProcessingConfig(
            missing_value_threshold=0.1,
            outlier_method='iqr',
            outlier_threshold=3.0,
            scaling_method='standard',
            feature_selection_method='correlation',
            correlation_threshold=0.95
        )
        
        self.processor = DataProcessor(self.config)
    
    def test_initialization(self):
        """Test data processor initialization."""
        self.assertIsInstance(self.processor, DataProcessor)
        self.assertEqual(self.processor.config, self.config)
    
    def test_data_quality_assessment(self):
        """Test data quality assessment."""
        # Create test data with quality issues
        test_data = pd.DataFrame({
            'price': [100, 105, np.nan, 110, 1000, 95],  # Missing value and outlier
            'volume': [1000, 1100, 1200, np.nan, 1050, 1150],  # Missing value
            'symbol': ['AAPL', 'AAPL', 'AAPL', 'MSFT', 'MSFT', 'MSFT'],
            'date': pd.date_range('2023-01-01', periods=6)
        })
        
        quality_report = self.processor.assess_data_quality(test_data)
        
        self.assertIsInstance(quality_report, DataQualityReport)
        
        # Check missing value detection
        self.assertGreater(quality_report.missing_values['price'], 0)
        self.assertGreater(quality_report.missing_values['volume'], 0)
        
        # Check outlier detection
        self.assertGreater(len(quality_report.outliers), 0)
        
        # Check data types
        self.assertIn('price', quality_report.data_types)
        self.assertIn('volume', quality_report.data_types)
    
    def test_missing_value_handling(self):
        """Test missing value handling."""
        # Create data with missing values
        data = pd.DataFrame({
            'price': [100, np.nan, 110, 105, np.nan, 115],
            'volume': [1000, 1100, np.nan, 1200, 1150, 1300],
            'date': pd.date_range('2023-01-01', periods=6)
        })
        
        # Test forward fill
        filled_data = self.processor.handle_missing_values(data, method='forward_fill')
        self.assertEqual(filled_data.isna().sum().sum(), 1)  # Only first NaN in price remains
        
        # Test interpolation
        interpolated_data = self.processor.handle_missing_values(data, method='interpolate')
        self.assertEqual(interpolated_data.isna().sum().sum(), 0)
        
        # Test mean imputation
        mean_filled_data = self.processor.handle_missing_values(data, method='mean')
        self.assertEqual(mean_filled_data.isna().sum().sum(), 0)
    
    def test_outlier_detection_and_treatment(self):
        """Test outlier detection and treatment."""
        # Create data with outliers
        np.random.seed(42)
        normal_data = np.random.normal(100, 10, 95)
        outliers = [200, 300, -50, 400, 500]  # Clear outliers
        
        data = pd.DataFrame({
            'price': np.concatenate([normal_data, outliers]),
            'index': range(100)
        })
        
        # Detect outliers using IQR method
        outlier_indices = self.processor.detect_outliers(data['price'], method='iqr')
        
        self.assertGreater(len(outlier_indices), 0)
        self.assertLessEqual(len(outlier_indices), 10)  # Should detect the 5 outliers
        
        # Treat outliers
        treated_data = self.processor.treat_outliers(
            data, 'price', method='winsorize', percentile=0.05
        )
        
        # Treated data should have less extreme values
        self.assertLess(treated_data['price'].max(), data['price'].max())
        self.assertGreater(treated_data['price'].min(), data['price'].min())
    
    def test_feature_engineering(self):
        """Test feature engineering capabilities."""
        # Create time series data
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        data = pd.DataFrame({
            'date': dates,
            'price': 100 + np.cumsum(np.random.randn(100) * 0.5),
            'volume': 1000 + np.random.randint(-100, 100, 100)
        })
        
        # Engineer features
        engineered_data = self.processor.engineer_features(data)
        
        self.assertGreater(len(engineered_data.columns), len(data.columns))
        
        # Check for common engineered features
        expected_features = [
            'price_lag_1', 'price_lag_7',
            'price_rolling_mean_7', 'price_rolling_std_7',
            'volume_lag_1', 'volume_rolling_mean_7'
        ]
        
        for feature in expected_features:
            if feature in engineered_data.columns:
                # Feature should have reasonable values
                self.assertFalse(engineered_data[feature].isna().all())
    
    def test_data_scaling(self):
        """Test data scaling methods."""
        # Create test data
        data = pd.DataFrame({
            'price': [100, 150, 200, 250, 300],
            'volume': [1000, 2000, 3000, 4000, 5000],
            'ratio': [0.1, 0.2, 0.3, 0.4, 0.5]
        })
        
        # Test standard scaling
        scaled_data = self.processor.scale_features(data, method='standard')
        
        # Standard scaled data should have mean ~0 and std ~1
        for col in scaled_data.columns:
            if scaled_data[col].dtype in ['float64', 'int64']:
                self.assertAlmostEqual(scaled_data[col].mean(), 0, places=1)
                self.assertAlmostEqual(scaled_data[col].std(), 1, places=1)
        
        # Test min-max scaling
        minmax_scaled = self.processor.scale_features(data, method='minmax')
        
        # Min-max scaled data should be between 0 and 1
        for col in minmax_scaled.columns:
            if minmax_scaled[col].dtype in ['float64', 'int64']:
                self.assertGreaterEqual(minmax_scaled[col].min(), 0)
                self.assertLessEqual(minmax_scaled[col].max(), 1)
    
    def test_feature_selection(self):
        """Test feature selection methods."""
        # Create data with correlated features
        np.random.seed(42)
        n_samples = 1000
        
        # Create base features
        feature1 = np.random.randn(n_samples)
        feature2 = feature1 + np.random.randn(n_samples) * 0.1  # Highly correlated
        feature3 = np.random.randn(n_samples)  # Independent
        
        data = pd.DataFrame({
            'feature1': feature1,
            'feature2': feature2,  # Should be removed due to high correlation
            'feature3': feature3,
            'target': feature1 * 2 + feature3 + np.random.randn(n_samples) * 0.1
        })
        
        # Select features based on correlation
        selected_features = self.processor.select_features(
            data.drop('target', axis=1), data['target'], method='correlation'
        )
        
        self.assertIsInstance(selected_features, list)
        self.assertLess(len(selected_features), len(data.columns) - 1)
        
        # Should keep feature1 and feature3, remove feature2
        self.assertIn('feature1', selected_features)
        self.assertIn('feature3', selected_features)
    
    def test_time_series_processing(self):
        """Test time series specific processing."""
        # Create time series data
        dates = pd.date_range('2023-01-01', periods=365, freq='D')
        
        # Create seasonal pattern
        seasonal = np.sin(2 * np.pi * np.arange(365) / 365) * 10
        trend = np.linspace(100, 150, 365)
        noise = np.random.randn(365) * 2
        
        data = pd.DataFrame({
            'date': dates,
            'value': trend + seasonal + noise
        })
        
        # Process time series
        processed_data = self.processor.process_time_series(data, 'value', 'date')
        
        self.assertIn('value_trend', processed_data.columns)
        self.assertIn('value_seasonal', processed_data.columns)
        self.assertIn('value_residual', processed_data.columns)
        
        # Trend should be generally increasing
        trend_values = processed_data['value_trend'].dropna()
        self.assertGreater(trend_values.iloc[-1], trend_values.iloc[0])
    
    def test_data_validation(self):
        """Test data validation rules."""
        # Create test data with validation issues
        data = pd.DataFrame({
            'price': [100, -50, 200, 0, 150],  # Negative price (invalid)
            'volume': [1000, 2000, -500, 1500, 2500],  # Negative volume (invalid)
            'symbol': ['AAPL', '', 'MSFT', 'GOOGL', None],  # Empty/null symbols
            'date': ['2023-01-01', '2023-01-02', 'invalid', '2023-01-04', '2023-01-05']
        })
        
        # Define validation rules
        validation_rules = {
            'price': {'min_value': 0, 'required': True},
            'volume': {'min_value': 0, 'required': True},
            'symbol': {'required': True, 'min_length': 1},
            'date': {'date_format': '%Y-%m-%d'}
        }
        
        validation_result = self.processor.validate_data(data, validation_rules)
        
        self.assertIsInstance(validation_result, dict)
        self.assertIn('valid_rows', validation_result)
        self.assertIn('validation_errors', validation_result)
        
        # Should detect validation errors
        self.assertGreater(len(validation_result['validation_errors']), 0)
        self.assertLess(validation_result['valid_rows'], len(data))
    
    def test_performance_optimization(self):
        """Test processing performance with large datasets."""
        # Create large dataset
        n_rows = 10000
        large_data = pd.DataFrame({
            'price': np.random.uniform(50, 200, n_rows),
            'volume': np.random.randint(1000, 10000, n_rows),
            'feature1': np.random.randn(n_rows),
            'feature2': np.random.randn(n_rows),
            'feature3': np.random.randn(n_rows)
        })
        
        def process_large_dataset():
            # Perform multiple processing steps
            quality_report = self.processor.assess_data_quality(large_data)
            cleaned_data = self.processor.handle_missing_values(large_data)
            scaled_data = self.processor.scale_features(cleaned_data)
            return quality_report, cleaned_data, scaled_data
        
        # Should complete within reasonable time
        results = self.assert_execution_time(process_large_dataset, 10.0)
        
        quality_report, cleaned_data, scaled_data = results
        
        self.assertIsInstance(quality_report, DataQualityReport)
        self.assertEqual(len(cleaned_data), n_rows)
        self.assertEqual(len(scaled_data), n_rows)


class TestESGMetricsIntegrator(BaseTestCase):
    """Test cases for ESGMetricsIntegrator."""
    
    def setUp(self):
        """Set up test fixtures."""
        super().setUp()
        
        self.config = {
            'esg_providers': {
                'msci': {
                    'api_key': 'test_key',
                    'base_url': 'https://api.msci.com',
                    'weight': 0.4
                },
                'sustainalytics': {
                    'api_key': 'test_key',
                    'base_url': 'https://api.sustainalytics.com',
                    'weight': 0.3
                },
                'refinitiv': {
                    'api_key': 'test_key',
                    'base_url': 'https://api.refinitiv.com',
                    'weight': 0.3
                }
            },
            'scoring': {
                'environmental_weight': 0.4,
                'social_weight': 0.3,
                'governance_weight': 0.3
            },
            'thresholds': {
                'high_esg': 0.8,
                'medium_esg': 0.6,
                'low_esg': 0.4
            }
        }
        
        self.esg_integrator = ESGMetricsIntegrator(self.config, self.db_manager)
    
    def test_initialization(self):
        """Test ESG integrator initialization."""
        self.assertIsInstance(self.esg_integrator, ESGMetricsIntegrator)
        self.assertEqual(self.esg_integrator.config, self.config)
    
    @patch('requests.get')
    def test_collect_esg_data(self, mock_get):
        """Test ESG data collection from external APIs."""
        # Mock API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'symbol': 'AAPL',
            'esg_score': 85,
            'environmental': 88,
            'social': 82,
            'governance': 85,
            'risk_rating': 'low'
        }
        mock_get.return_value = mock_response
        
        # Collect ESG data
        esg_scores = self.esg_integrator.collect_esg_data(['AAPL', 'MSFT'])
        
        self.assertIsInstance(esg_scores, list)
        self.assertGreater(len(esg_scores), 0)
        
        for score in esg_scores:
            self.assertIsInstance(score, ESGScore)
            self.assert_financial_metric_valid(
                score.overall_score, 'overall_score', 0.0, 1.0
            )
    
    def test_assess_supply_chain_sustainability(self):
        """Test supply chain sustainability assessment."""
        # Generate mock supply chain data
        supply_chain_data = {
            'suppliers': [
                {'name': 'Supplier A', 'country': 'China', 'esg_score': 0.7},
                {'name': 'Supplier B', 'country': 'Germany', 'esg_score': 0.9},
                {'name': 'Supplier C', 'country': 'India', 'esg_score': 0.6}
            ],
            'transportation': {
                'carbon_intensity': 0.5,
                'renewable_energy_usage': 0.3
            },
            'facilities': [
                {'location': 'US', 'green_certified': True, 'energy_efficiency': 0.8},
                {'location': 'Mexico', 'green_certified': False, 'energy_efficiency': 0.6}
            ]
        }
        
        sustainability_metrics = self.esg_integrator.assess_supply_chain_sustainability(
            supply_chain_data
        )
        
        self.assertIsInstance(sustainability_metrics, SupplyChainSustainabilityMetrics)
        
        # Validate metrics
        self.assert_financial_metric_valid(
            sustainability_metrics.overall_score, 'overall_score', 0.0, 1.0
        )
        self.assert_financial_metric_valid(
            sustainability_metrics.supplier_score, 'supplier_score', 0.0, 1.0
        )
        self.assert_financial_metric_valid(
            sustainability_metrics.carbon_footprint_score, 'carbon_score', 0.0, 1.0
        )
    
    def test_adjust_investment_signals(self):
        """Test ESG adjustment of investment signals."""
        # Create base investment signal
        base_signal = {
            'symbol': 'AAPL',
            'signal_type': 'long',
            'conviction': 0.8,
            'target_weight': 0.05,
            'expected_return': 0.12
        }
        
        # Create ESG score
        esg_score = ESGScore(
            symbol='AAPL',
            environmental_score=0.85,
            social_score=0.80,
            governance_score=0.90,
            overall_score=0.85,
            esg_risk_rating='low',
            sustainability_rank=25,
            timestamp=datetime.utcnow()
        )
        
        # Adjust signal with ESG
        adjusted_signal = self.esg_integrator.adjust_investment_signals(
            [base_signal], [esg_score]
        )[0]
        
        self.assertIsInstance(adjusted_signal, ESGAdjustedSignal)
        
        # High ESG score should improve conviction
        self.assertGreaterEqual(
            adjusted_signal.adjusted_conviction,
            base_signal['conviction']
        )
        
        # Validate adjusted metrics
        self.assert_financial_metric_valid(
            adjusted_signal.adjusted_conviction, 'adjusted_conviction', 0.0, 1.0
        )
        self.assert_financial_metric_valid(
            adjusted_signal.esg_adjustment_factor, 'esg_adjustment', 0.5, 1.5
        )
    
    def test_generate_esg_report(self):
        """Test ESG report generation."""
        # Generate mock ESG scores
        esg_scores = self.mock_data.generate_esg_scores(['AAPL', 'MSFT', 'GOOGL'])
        
        # Generate sustainability metrics
        sustainability_data = {
            'suppliers': [{'name': 'Test Supplier', 'esg_score': 0.7}],
            'transportation': {'carbon_intensity': 0.4},
            'facilities': [{'green_certified': True}]
        }
        
        sustainability_metrics = self.esg_integrator.assess_supply_chain_sustainability(
            sustainability_data
        )
        
        # Generate report
        report = self.esg_integrator.generate_esg_report(
            esg_scores, [sustainability_metrics]
        )
        
        self.assertIsInstance(report, dict)
        
        # Check required report sections
        required_sections = [
            'executive_summary',
            'portfolio_esg_metrics',
            'supply_chain_assessment',
            'esg_risk_analysis',
            'recommendations'
        ]
        
        for section in required_sections:
            self.assertIn(section, report)
        
        # Validate executive summary
        exec_summary = report['executive_summary']
        self.assertIn('average_esg_score', exec_summary)
        self.assertIn('esg_risk_level', exec_summary)
    
    def test_esg_trend_analysis(self):
        """Test ESG trend analysis over time."""
        # Generate historical ESG data
        symbol = 'AAPL'
        dates = pd.date_range('2023-01-01', '2023-12-31', freq='M')
        
        historical_scores = []
        for i, date in enumerate(dates):
            score = ESGScore(
                symbol=symbol,
                environmental_score=0.8 + (i * 0.01),  # Improving trend
                social_score=0.75 + (i * 0.005),
                governance_score=0.85,  # Stable
                overall_score=0.8 + (i * 0.005),
                esg_risk_rating='low',
                sustainability_rank=30 - i,  # Improving rank
                timestamp=date
            )
            historical_scores.append(score)
        
        # Analyze trends
        trend_analysis = self.esg_integrator.analyze_esg_trends(historical_scores)
        
        self.assertIsInstance(trend_analysis, dict)
        self.assertIn('overall_trend', trend_analysis)
        self.assertIn('environmental_trend', trend_analysis)
        self.assertIn('social_trend', trend_analysis)
        self.assertIn('governance_trend', trend_analysis)
        
        # Should detect improving trend
        self.assertEqual(trend_analysis['overall_trend'], 'improving')
        self.assertEqual(trend_analysis['environmental_trend'], 'improving')
    
    def test_esg_screening(self):
        """Test ESG-based investment screening."""
        # Create universe of stocks with different ESG profiles
        universe = ['AAPL', 'TSLA', 'XOM', 'MSFT', 'GOOGL']
        esg_scores = [
            ESGScore('AAPL', 0.85, 0.80, 0.90, 0.85, 'low', 20, datetime.utcnow()),
            ESGScore('TSLA', 0.90, 0.70, 0.75, 0.78, 'medium', 35, datetime.utcnow()),
            ESGScore('XOM', 0.40, 0.60, 0.70, 0.57, 'high', 180, datetime.utcnow()),
            ESGScore('MSFT', 0.88, 0.85, 0.92, 0.88, 'low', 15, datetime.utcnow()),
            ESGScore('GOOGL', 0.82, 0.78, 0.85, 0.82, 'low', 25, datetime.utcnow())
        ]
        
        # Apply ESG screening
        screened_universe = self.esg_integrator.screen_investments(
            universe, esg_scores, min_esg_score=0.7
        )
        
        self.assertIsInstance(screened_universe, list)
        self.assertLess(len(screened_universe), len(universe))
        
        # XOM should be excluded due to low ESG score
        self.assertNotIn('XOM', screened_universe)
        
        # High ESG stocks should be included
        self.assertIn('AAPL', screened_universe)
        self.assertIn('MSFT', screened_universe)


class TestSupplyChainLogger(BaseTestCase):
    """Test cases for SupplyChainLogger."""
    
    def setUp(self):
        """Set up test fixtures."""
        super().setUp()
        
        self.config = {
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'file_path': str(self.test_data_dir / 'test.log'),
                'max_file_size': '10MB',
                'backup_count': 5
            },
            'performance': {
                'track_execution_time': True,
                'track_memory_usage': True,
                'alert_threshold_seconds': 30
            }
        }
        
        self.logger = SupplyChainLogger('test_logger', self.config)
    
    def test_initialization(self):
        """Test logger initialization."""
        self.assertIsInstance(self.logger, SupplyChainLogger)
        self.assertEqual(self.logger.name, 'test_logger')
    
    def test_basic_logging(self):
        """Test basic logging functionality."""
        # Test different log levels
        self.logger.info("Test info message")
        self.logger.warning("Test warning message")
        self.logger.error("Test error message")
        
        # Check if log file was created
        log_file = Path(self.config['logging']['file_path'])
        self.assertTrue(log_file.exists())
        
        # Read log content
        with open(log_file, 'r') as f:
            log_content = f.read()
        
        self.assertIn("Test info message", log_content)
        self.assertIn("Test warning message", log_content)
        self.assertIn("Test error message", log_content)
    
    def test_structured_logging(self):
        """Test structured logging with metadata."""
        metadata = {
            'user_id': 'test_user',
            'session_id': 'session_123',
            'operation': 'data_processing'
        }
        
        self.logger.log_structured(
            LogLevel.INFO,
            "Processing completed",
            metadata
        )
        
        # Verify structured log entry
        log_file = Path(self.config['logging']['file_path'])
        with open(log_file, 'r') as f:
            log_content = f.read()
        
        self.assertIn("Processing completed", log_content)
        self.assertIn("test_user", log_content)
        self.assertIn("session_123", log_content)
    
    def test_performance_tracking(self):
        """Test performance tracking functionality."""
        # Test execution time tracking
        with self.logger.track_performance("test_operation"):
            # Simulate some work
            import time
            time.sleep(0.1)
        
        # Check performance metrics
        metrics = self.logger.get_performance_metrics()
        self.assertIsInstance(metrics, PerformanceMetrics)
        self.assertIn("test_operation", metrics.operation_times)
        self.assertGreater(metrics.operation_times["test_operation"], 0.05)
    
    def test_error_tracking(self):
        """Test error tracking and alerting."""
        # Log an error with exception
        try:
            raise ValueError("Test error")
        except Exception as e:
            self.logger.log_error("Test operation failed", e)
        
        # Check error metrics
        error_count = self.logger.get_error_count()
        self.assertGreater(error_count, 0)
        
        # Check if error was logged with traceback
        log_file = Path(self.config['logging']['file_path'])
        with open(log_file, 'r') as f:
            log_content = f.read()
        
        self.assertIn("Test operation failed", log_content)
        self.assertIn("ValueError", log_content)
        self.assertIn("Test error", log_content)
    
    def test_log_rotation(self):
        """Test log file rotation."""
        # Generate large amount of log data
        large_message = "X" * 1000  # 1KB message
        
        for i in range(1000):  # Generate ~1MB of logs
            self.logger.info(f"Large message {i}: {large_message}")
        
        # Check if log file exists and has reasonable size
        log_file = Path(self.config['logging']['file_path'])
        self.assertTrue(log_file.exists())
        
        # File size should be managed by rotation
        file_size = log_file.stat().st_size
        self.assertLess(file_size, 50 * 1024 * 1024)  # Less than 50MB
    
    def test_custom_formatters(self):
        """Test custom log formatters."""
        # Test JSON formatter
        json_logger = SupplyChainLogger(
            'json_logger',
            {**self.config, 'logging': {**self.config['logging'], 'format': 'json'}}
        )
        
        json_logger.info("JSON formatted message", extra={'custom_field': 'value'})
        
        # Verify JSON format (basic check)
        log_file = Path(self.config['logging']['file_path'])
        with open(log_file, 'r') as f:
            log_content = f.read()
        
        # Should contain JSON-like structure
        self.assertIn('"message":', log_content)
        self.assertIn('"custom_field":', log_content)


class TestUtilsIntegration(BaseTestCase):
    """Integration tests for utility modules."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        super().setUp()
        
        # Initialize all utility components
        self.db_manager = DatabaseManager({'database': {'type': 'sqlite', 'path': ':memory:'}})
        self.data_processor = DataProcessor(ProcessingConfig())
        self.esg_integrator = ESGMetricsIntegrator({}, self.db_manager)
        self.logger = SupplyChainLogger('integration_test', {})
    
    def test_end_to_end_data_pipeline(self):
        """Test complete data processing pipeline."""
        # Step 1: Generate raw data
        raw_data = self.mock_data.generate_stock_data(['AAPL', 'MSFT'], days=100)
        
        # Step 2: Process data
        quality_report = self.data_processor.assess_data_quality(raw_data)
        cleaned_data = self.data_processor.handle_missing_values(raw_data)
        scaled_data = self.data_processor.scale_features(cleaned_data)
        
        # Step 3: Store in database
        self.db_manager.connect()
        self.db_manager.execute_query("""
            CREATE TABLE processed_data (
                id INTEGER PRIMARY KEY,
                symbol TEXT,
                date DATE,
                price REAL,
                volume INTEGER
            )
        """)
        
        # Convert to records and store
        records = scaled_data.to_dict('records')
        success = self.db_manager.bulk_insert('processed_data', records[:10])
        
        # Step 4: Log the process
        self.logger.info(f"Processed {len(scaled_data)} records")
        
        # Validate pipeline
        self.assertIsInstance(quality_report, DataQualityReport)
        self.assertEqual(len(cleaned_data), len(raw_data))
        self.assertTrue(success)
        
        # Verify data in database
        stored_data = self.db_manager.fetch_data("SELECT COUNT(*) as count FROM processed_data")
        self.assertEqual(stored_data[0]['count'], 10)
        
        self.db_manager.disconnect()
    
    def test_performance_monitoring_integration(self):
        """Test performance monitoring across all utilities."""
        def integrated_operation():
            # Database operations
            self.db_manager.connect()
            
            # Data processing
            test_data = pd.DataFrame({
                'price': np.random.uniform(50, 200, 1000),
                'volume': np.random.randint(1000, 10000, 1000)
            })
            
            processed_data = self.data_processor.scale_features(test_data)
            
            # ESG data collection (mocked)
            esg_scores = self.mock_data.generate_esg_scores(['AAPL', 'MSFT'])
            
            self.db_manager.disconnect()
            
            return processed_data, esg_scores
        
        # Track performance of integrated operation
        with self.logger.track_performance("integrated_operation"):
            results = integrated_operation()
        
        # Validate results
        processed_data, esg_scores = results
        self.assertIsInstance(processed_data, pd.DataFrame)
        self.assertIsInstance(esg_scores, list)
        
        # Check performance metrics
        metrics = self.logger.get_performance_metrics()
        self.assertIn("integrated_operation", metrics.operation_times)
        self.assertGreater(metrics.operation_times["integrated_operation"], 0)


if __name__ == '__main__':
    unittest.main()