#!/usr/bin/env python3
"""
Base Test Classes and Utilities

Provides common testing infrastructure, fixtures, and utilities for all test modules.

Features:
- Base test classes with common setup/teardown
- Mock data generators
- Database test fixtures
- API mocking utilities
- Performance testing helpers
- Assertion helpers for financial data

Author: Supply Chain Alpha Team
Version: 1.0.0
"""

import unittest
import asyncio
import tempfile
import shutil
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Generator
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import json
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
import sqlite3
import pytest

# Import project modules
from utils.database import DatabaseManager
from utils.logging_utils import LogConfig, SupplyChainLogger
from data_scrapers.shipping_scraper import VesselData, ShippingRoute
from data_scrapers.port_scraper import PortMetrics, TerminalData
from data_scrapers.freight_scraper import FreightRate, BalticIndex, ContainerRate
from models.disruption_predictor import DisruptionPrediction, CompanyImpactScore
from strategies.long_short_equity import Position, TradeSignal, PortfolioMetrics
from utils.esg_metrics import ESGScore, SupplyChainSustainabilityMetrics


@dataclass
class TestConfig:
    """Test configuration settings."""
    use_mock_data: bool = True
    database_url: str = 'sqlite:///:memory:'
    log_level: str = 'WARNING'
    timeout_seconds: int = 30
    temp_dir: Optional[str] = None
    mock_api_responses: bool = True
    performance_testing: bool = False
    

class BaseTestCase(unittest.TestCase):
    """Base test case with common setup and utilities."""
    
    @classmethod
    def setUpClass(cls):
        """Set up class-level test fixtures."""
        cls.test_config = TestConfig()
        cls.temp_dir = tempfile.mkdtemp(prefix='supply_chain_test_')
        cls.test_data_dir = Path(cls.temp_dir) / 'test_data'
        cls.test_data_dir.mkdir(exist_ok=True)
        
        # Setup test logging
        log_config = LogConfig(
            log_level=cls.test_config.log_level,
            console_logging=False,
            log_file=None,
            performance_logging=False
        )
        cls.logger = SupplyChainLogger(log_config)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up class-level test fixtures."""
        if hasattr(cls, 'temp_dir') and Path(cls.temp_dir).exists():
            shutil.rmtree(cls.temp_dir)
    
    def setUp(self):
        """Set up test fixtures for each test method."""
        self.start_time = datetime.utcnow()
        
        # Setup test database
        self.db_manager = DatabaseManager({
            'db_type': 'sqlite',
            'host': 'localhost',
            'port': 5432,
            'database': ':memory:',
            'file_path': ':memory:'
        })
        
        # Create mock data generators
        self.mock_data = MockDataGenerator()
        
        # Setup API mocks
        if self.test_config.mock_api_responses:
            self._setup_api_mocks()
    
    def tearDown(self):
        """Clean up after each test method."""
        # Close database connections
        if hasattr(self, 'db_manager'):
            asyncio.run(self.db_manager.disconnect())
        
        # Log test duration if performance testing enabled
        if self.test_config.performance_testing:
            duration = (datetime.utcnow() - self.start_time).total_seconds()
            self.logger.log_performance(f"test_{self._testMethodName}", duration)
    
    def _setup_api_mocks(self):
        """Setup API mocks for external services."""
        # Mock HTTP requests
        self.requests_mock = patch('requests.get')
        self.mock_get = self.requests_mock.start()
        self.addCleanup(self.requests_mock.stop)
        
        # Mock async HTTP requests
        self.aiohttp_mock = patch('aiohttp.ClientSession.get')
        self.mock_aiohttp_get = self.aiohttp_mock.start()
        self.addCleanup(self.aiohttp_mock.stop)
    
    def assert_dataframe_equal(self, df1: pd.DataFrame, df2: pd.DataFrame, 
                              check_dtype: bool = True, rtol: float = 1e-5):
        """Assert that two DataFrames are equal with tolerance for floats.
        
        Args:
            df1: First DataFrame
            df2: Second DataFrame
            check_dtype: Whether to check data types
            rtol: Relative tolerance for float comparison
        """
        try:
            pd.testing.assert_frame_equal(df1, df2, check_dtype=check_dtype, rtol=rtol)
        except AssertionError as e:
            self.fail(f"DataFrames are not equal: {e}")
    
    def assert_financial_metric_valid(self, value: float, metric_name: str, 
                                    min_val: Optional[float] = None, 
                                    max_val: Optional[float] = None):
        """Assert that a financial metric is valid.
        
        Args:
            value: Metric value
            metric_name: Name of the metric
            min_val: Minimum expected value
            max_val: Maximum expected value
        """
        self.assertIsInstance(value, (int, float), 
                            f"{metric_name} should be numeric")
        self.assertFalse(np.isnan(value), 
                        f"{metric_name} should not be NaN")
        self.assertFalse(np.isinf(value), 
                        f"{metric_name} should not be infinite")
        
        if min_val is not None:
            self.assertGreaterEqual(value, min_val, 
                                  f"{metric_name} should be >= {min_val}")
        
        if max_val is not None:
            self.assertLessEqual(value, max_val, 
                               f"{metric_name} should be <= {max_val}")
    
    def assert_prediction_valid(self, prediction: DisruptionPrediction):
        """Assert that a disruption prediction is valid.
        
        Args:
            prediction: Disruption prediction to validate
        """
        self.assertIsInstance(prediction, DisruptionPrediction)
        self.assert_financial_metric_valid(prediction.probability, 'probability', 0.0, 1.0)
        self.assert_financial_metric_valid(prediction.confidence, 'confidence', 0.0, 1.0)
        self.assertIsInstance(prediction.prediction_date, datetime)
        self.assertIsInstance(prediction.disruption_type, str)
        self.assertTrue(len(prediction.disruption_type) > 0)
    
    def assert_trade_signal_valid(self, signal: TradeSignal):
        """Assert that a trade signal is valid.
        
        Args:
            signal: Trade signal to validate
        """
        self.assertIsInstance(signal, TradeSignal)
        self.assertIn(signal.action, ['BUY', 'SELL', 'HOLD'])
        self.assert_financial_metric_valid(signal.confidence, 'confidence', 0.0, 1.0)
        self.assertIsInstance(signal.symbol, str)
        self.assertTrue(len(signal.symbol) > 0)
        self.assertIsInstance(signal.timestamp, datetime)
    
    def assert_esg_score_valid(self, esg_score: ESGScore):
        """Assert that an ESG score is valid.
        
        Args:
            esg_score: ESG score to validate
        """
        self.assertIsInstance(esg_score, ESGScore)
        self.assert_financial_metric_valid(esg_score.environmental_score, 'environmental_score', 0.0, 100.0)
        self.assert_financial_metric_valid(esg_score.social_score, 'social_score', 0.0, 100.0)
        self.assert_financial_metric_valid(esg_score.governance_score, 'governance_score', 0.0, 100.0)
        self.assert_financial_metric_valid(esg_score.overall_score, 'overall_score', 0.0, 100.0)
    
    def create_temp_file(self, content: str, suffix: str = '.txt') -> Path:
        """Create a temporary file with content.
        
        Args:
            content: File content
            suffix: File suffix
            
        Returns:
            Path to temporary file
        """
        temp_file = self.test_data_dir / f"temp_{datetime.utcnow().timestamp()}{suffix}"
        temp_file.write_text(content)
        return temp_file
    
    def create_temp_csv(self, data: Dict[str, List]) -> Path:
        """Create a temporary CSV file with data.
        
        Args:
            data: Dictionary with column names as keys and lists as values
            
        Returns:
            Path to temporary CSV file
        """
        df = pd.DataFrame(data)
        temp_file = self.test_data_dir / f"temp_{datetime.utcnow().timestamp()}.csv"
        df.to_csv(temp_file, index=False)
        return temp_file


class AsyncTestCase(BaseTestCase):
    """Base test case for async operations."""
    
    def setUp(self):
        """Set up async test fixtures."""
        super().setUp()
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
    
    def tearDown(self):
        """Clean up async test fixtures."""
        super().tearDown()
        self.loop.close()
    
    def run_async(self, coro):
        """Run an async coroutine in the test loop.
        
        Args:
            coro: Coroutine to run
            
        Returns:
            Result of the coroutine
        """
        return self.loop.run_until_complete(coro)


class MockDataGenerator:
    """Generate mock data for testing."""
    
    def __init__(self):
        self.random_seed = 42
        np.random.seed(self.random_seed)
    
    def generate_vessel_data(self, count: int = 10) -> List[VesselData]:
        """Generate mock vessel data.
        
        Args:
            count: Number of vessels to generate
            
        Returns:
            List of VesselData objects
        """
        vessels = []
        for i in range(count):
            vessel = VesselData(
                vessel_id=f"VESSEL_{i:03d}",
                name=f"Test Vessel {i}",
                imo_number=f"IMO{1000000 + i}",
                vessel_type=np.random.choice(['Container', 'Bulk', 'Tanker', 'General Cargo']),
                latitude=np.random.uniform(-90, 90),
                longitude=np.random.uniform(-180, 180),
                speed=np.random.uniform(0, 25),
                heading=np.random.uniform(0, 360),
                destination=f"Port {np.random.randint(1, 100)}",
                eta=datetime.utcnow() + timedelta(days=np.random.randint(1, 30)),
                cargo_capacity=np.random.randint(1000, 20000),
                current_cargo=np.random.randint(0, 15000),
                last_update=datetime.utcnow() - timedelta(minutes=np.random.randint(1, 60))
            )
            vessels.append(vessel)
        return vessels
    
    def generate_port_metrics(self, count: int = 5) -> List[PortMetrics]:
        """Generate mock port metrics.
        
        Args:
            count: Number of ports to generate
            
        Returns:
            List of PortMetrics objects
        """
        ports = []
        for i in range(count):
            port = PortMetrics(
                port_code=f"PORT{i:02d}",
                port_name=f"Test Port {i}",
                country=f"Country {i}",
                throughput_teu=np.random.randint(100000, 10000000),
                congestion_level=np.random.uniform(0, 1),
                average_wait_time=np.random.uniform(1, 48),
                berth_utilization=np.random.uniform(0.5, 1.0),
                crane_productivity=np.random.uniform(20, 40),
                vessel_count=np.random.randint(10, 100),
                timestamp=datetime.utcnow() - timedelta(hours=np.random.randint(1, 24))
            )
            ports.append(port)
        return ports
    
    def generate_freight_rates(self, count: int = 20) -> List[FreightRate]:
        """Generate mock freight rates.
        
        Args:
            count: Number of freight rates to generate
            
        Returns:
            List of FreightRate objects
        """
        routes = [
            ('Shanghai', 'Los Angeles'),
            ('Rotterdam', 'New York'),
            ('Singapore', 'Hamburg'),
            ('Hong Kong', 'Long Beach'),
            ('Busan', 'Seattle')
        ]
        
        rates = []
        for i in range(count):
            origin, destination = routes[i % len(routes)]
            rate = FreightRate(
                route_id=f"ROUTE_{i:03d}",
                origin_port=origin,
                destination_port=destination,
                rate_per_teu=np.random.uniform(1000, 5000),
                rate_per_feu=np.random.uniform(1500, 7500),
                currency='USD',
                valid_from=datetime.utcnow() - timedelta(days=np.random.randint(1, 30)),
                valid_to=datetime.utcnow() + timedelta(days=np.random.randint(30, 90)),
                carrier=f"Carrier {np.random.randint(1, 10)}",
                service_type=np.random.choice(['Express', 'Standard', 'Economy']),
                transit_time=np.random.randint(10, 45),
                timestamp=datetime.utcnow() - timedelta(hours=np.random.randint(1, 24))
            )
            rates.append(rate)
        return rates
    
    def generate_stock_data(self, symbols: List[str], days: int = 252) -> pd.DataFrame:
        """Generate mock stock price data.
        
        Args:
            symbols: List of stock symbols
            days: Number of trading days
            
        Returns:
            DataFrame with stock price data
        """
        dates = pd.date_range(end=datetime.utcnow(), periods=days, freq='B')
        data = {'date': dates}
        
        for symbol in symbols:
            # Generate realistic stock price movements
            initial_price = np.random.uniform(50, 200)
            returns = np.random.normal(0.0005, 0.02, days)  # Daily returns
            prices = [initial_price]
            
            for ret in returns[1:]:
                new_price = prices[-1] * (1 + ret)
                prices.append(max(new_price, 0.01))  # Prevent negative prices
            
            data[f'{symbol}_price'] = prices
            data[f'{symbol}_volume'] = np.random.randint(100000, 10000000, days)
        
        return pd.DataFrame(data)
    
    def generate_disruption_predictions(self, count: int = 10) -> List[DisruptionPrediction]:
        """Generate mock disruption predictions.
        
        Args:
            count: Number of predictions to generate
            
        Returns:
            List of DisruptionPrediction objects
        """
        disruption_types = [
            'port_congestion', 'route_blockage', 'weather_delay',
            'labor_strike', 'equipment_failure', 'capacity_shortage'
        ]
        
        predictions = []
        for i in range(count):
            prediction = DisruptionPrediction(
                prediction_id=f"PRED_{i:03d}",
                disruption_type=np.random.choice(disruption_types),
                location=f"Location {i}",
                probability=np.random.uniform(0.1, 0.9),
                confidence=np.random.uniform(0.6, 0.95),
                severity=np.random.choice(['low', 'medium', 'high']),
                estimated_duration=np.random.randint(1, 14),
                affected_routes=[f"Route_{j}" for j in range(np.random.randint(1, 5))],
                prediction_date=datetime.utcnow(),
                event_start_date=datetime.utcnow() + timedelta(days=np.random.randint(1, 30)),
                model_version='test_v1.0',
                features_used=['feature_1', 'feature_2', 'feature_3']
            )
            predictions.append(prediction)
        return predictions
    
    def generate_trade_signals(self, symbols: List[str], count: int = 50) -> List[TradeSignal]:
        """Generate mock trade signals.
        
        Args:
            symbols: List of stock symbols
            count: Number of signals to generate
            
        Returns:
            List of TradeSignal objects
        """
        signals = []
        for i in range(count):
            signal = TradeSignal(
                signal_id=f"SIG_{i:03d}",
                symbol=np.random.choice(symbols),
                action=np.random.choice(['BUY', 'SELL', 'HOLD']),
                quantity=np.random.randint(100, 10000),
                price=np.random.uniform(50, 200),
                confidence=np.random.uniform(0.6, 0.95),
                reasoning=f"Test reasoning {i}",
                timestamp=datetime.utcnow() - timedelta(minutes=np.random.randint(1, 1440)),
                expiry=datetime.utcnow() + timedelta(hours=np.random.randint(1, 24)),
                strategy_name='test_strategy',
                risk_score=np.random.uniform(0.1, 0.8)
            )
            signals.append(signal)
        return signals
    
    def generate_esg_scores(self, companies: List[str]) -> List[ESGScore]:
        """Generate mock ESG scores.
        
        Args:
            companies: List of company identifiers
            
        Returns:
            List of ESGScore objects
        """
        scores = []
        for company in companies:
            score = ESGScore(
                company_id=company,
                company_name=f"Company {company}",
                environmental_score=np.random.uniform(20, 95),
                social_score=np.random.uniform(20, 95),
                governance_score=np.random.uniform(20, 95),
                overall_score=0,  # Will be calculated
                score_date=datetime.utcnow() - timedelta(days=np.random.randint(1, 90)),
                data_provider='test_provider',
                methodology_version='v1.0'
            )
            # Calculate overall score
            score.overall_score = (
                score.environmental_score * 0.4 +
                score.social_score * 0.3 +
                score.governance_score * 0.3
            )
            scores.append(score)
        return scores
    
    def generate_time_series_data(self, start_date: datetime, end_date: datetime, 
                                 frequency: str = 'D') -> pd.DataFrame:
        """Generate mock time series data.
        
        Args:
            start_date: Start date
            end_date: End date
            frequency: Pandas frequency string
            
        Returns:
            DataFrame with time series data
        """
        dates = pd.date_range(start=start_date, end=end_date, freq=frequency)
        
        # Generate correlated time series
        n_points = len(dates)
        trend = np.linspace(0, 1, n_points)
        noise = np.random.normal(0, 0.1, n_points)
        seasonal = np.sin(2 * np.pi * np.arange(n_points) / 365.25) * 0.2
        
        data = {
            'date': dates,
            'value': 100 + trend * 50 + seasonal * 10 + noise * 5,
            'volume': np.random.randint(1000, 100000, n_points),
            'volatility': np.abs(np.random.normal(0.15, 0.05, n_points))
        }
        
        return pd.DataFrame(data)


class MockAPIResponse:
    """Mock API response for testing."""
    
    def __init__(self, status_code: int = 200, json_data: Optional[Dict] = None, 
                 text_data: Optional[str] = None):
        self.status_code = status_code
        self._json_data = json_data or {}
        self._text_data = text_data or ''
    
    def json(self):
        """Return JSON data."""
        return self._json_data
    
    @property
    def text(self):
        """Return text data."""
        return self._text_data
    
    def raise_for_status(self):
        """Raise exception for bad status codes."""
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code} Error")


class PerformanceTestMixin:
    """Mixin for performance testing utilities."""
    
    def assert_execution_time(self, func, max_seconds: float, *args, **kwargs):
        """Assert that function executes within time limit.
        
        Args:
            func: Function to test
            max_seconds: Maximum execution time
            *args: Function arguments
            **kwargs: Function keyword arguments
        """
        start_time = datetime.utcnow()
        result = func(*args, **kwargs)
        execution_time = (datetime.utcnow() - start_time).total_seconds()
        
        self.assertLessEqual(
            execution_time, max_seconds,
            f"Function took {execution_time:.2f}s, expected <= {max_seconds}s"
        )
        
        return result
    
    def benchmark_function(self, func, iterations: int = 100, *args, **kwargs) -> Dict[str, float]:
        """Benchmark function performance.
        
        Args:
            func: Function to benchmark
            iterations: Number of iterations
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Dictionary with performance metrics
        """
        execution_times = []
        
        for _ in range(iterations):
            start_time = datetime.utcnow()
            func(*args, **kwargs)
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            execution_times.append(execution_time)
        
        return {
            'mean_time': np.mean(execution_times),
            'median_time': np.median(execution_times),
            'min_time': np.min(execution_times),
            'max_time': np.max(execution_times),
            'std_time': np.std(execution_times),
            'total_time': np.sum(execution_times),
            'iterations': iterations
        }


# Test decorators
def skip_if_no_internet(func):
    """Skip test if no internet connection available."""
    def wrapper(*args, **kwargs):
        try:
            import requests
            requests.get('https://httpbin.org/get', timeout=5)
            return func(*args, **kwargs)
        except:
            pytest.skip("No internet connection available")
    return wrapper


def requires_database(db_type: str):
    """Skip test if required database is not available."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Check if database is available
            if db_type == 'postgresql':
                try:
                    import psycopg2
                except ImportError:
                    pytest.skip(f"PostgreSQL not available")
            elif db_type == 'mongodb':
                try:
                    import pymongo
                except ImportError:
                    pytest.skip(f"MongoDB not available")
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


def slow_test(func):
    """Mark test as slow (can be skipped with pytest -m "not slow")."""
    return pytest.mark.slow(func)


def integration_test(func):
    """Mark test as integration test."""
    return pytest.mark.integration(func)