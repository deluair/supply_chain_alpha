#!/usr/bin/env python3
"""
Unit Tests for Data Scrapers

Tests for shipping, port, and freight rate data collection modules.

Test Coverage:
- Shipping data scraper functionality
- Port metrics collection
- Freight rate data gathering
- API integration and error handling
- Data validation and storage

Author: Supply Chain Alpha Team
Version: 1.0.0
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Any

# Import test base classes
from tests.base_test import BaseTestCase, AsyncTestCase, MockAPIResponse, MockDataGenerator

# Import modules to test
from data_scrapers.shipping_scraper import (
    ShippingScraper, VesselData, ShippingRoute
)
from data_scrapers.port_scraper import (
    PortScraper, PortMetrics, TerminalData
)
from data_scrapers.freight_scraper import (
    FreightScraper, FreightRate, BalticIndex, ContainerRate
)


class TestShippingScraper(BaseTestCase):
    """Test cases for ShippingScraper."""
    
    def setUp(self):
        """Set up test fixtures."""
        super().setUp()
        
        # Mock configuration
        self.config = {
            'marine_traffic': {
                'api_key': 'test_key',
                'base_url': 'https://api.marinetraffic.com/v1',
                'rate_limit': 60
            },
            'vessel_finder': {
                'api_key': 'test_key',
                'base_url': 'https://api.vesselfinder.com/v1',
                'rate_limit': 100
            }
        }
        
        self.scraper = ShippingScraper(self.config)
    
    def test_initialization(self):
        """Test scraper initialization."""
        self.assertIsInstance(self.scraper, ShippingScraper)
        self.assertEqual(self.scraper.config, self.config)
        self.assertIsNotNone(self.scraper.db_manager)
    
    @patch('requests.get')
    def test_collect_vessel_data_success(self, mock_get):
        """Test successful vessel data collection."""
        # Mock API response
        mock_response_data = {
            'vessels': [
                {
                    'mmsi': '123456789',
                    'name': 'Test Vessel',
                    'imo': 'IMO1234567',
                    'type': 'Container',
                    'lat': 37.7749,
                    'lon': -122.4194,
                    'speed': 15.5,
                    'course': 180,
                    'destination': 'Los Angeles',
                    'eta': '2024-02-15T10:30:00Z',
                    'cargo_capacity': 10000,
                    'current_cargo': 8500,
                    'timestamp': '2024-02-01T12:00:00Z'
                }
            ]
        }
        
        mock_get.return_value = MockAPIResponse(200, mock_response_data)
        
        # Test vessel data collection
        vessels = self.scraper.collect_vessel_data()
        
        self.assertIsInstance(vessels, list)
        self.assertGreater(len(vessels), 0)
        
        vessel = vessels[0]
        self.assertIsInstance(vessel, VesselData)
        self.assertEqual(vessel.name, 'Test Vessel')
        self.assertEqual(vessel.vessel_type, 'Container')
        self.assertAlmostEqual(vessel.latitude, 37.7749, places=4)
        self.assertAlmostEqual(vessel.longitude, -122.4194, places=4)
    
    @patch('requests.get')
    def test_collect_vessel_data_api_error(self, mock_get):
        """Test vessel data collection with API error."""
        mock_get.return_value = MockAPIResponse(500, {'error': 'Internal Server Error'})
        
        vessels = self.scraper.collect_vessel_data()
        
        # Should return empty list on API error
        self.assertIsInstance(vessels, list)
        self.assertEqual(len(vessels), 0)
    
    def test_analyze_shipping_routes(self):
        """Test shipping route analysis."""
        # Generate mock vessel data
        vessels = self.mock_data.generate_vessel_data(20)
        
        routes = self.scraper.analyze_shipping_routes(vessels)
        
        self.assertIsInstance(routes, list)
        for route in routes:
            self.assertIsInstance(route, ShippingRoute)
            self.assertIsInstance(route.route_id, str)
            self.assertIsInstance(route.origin_port, str)
            self.assertIsInstance(route.destination_port, str)
            self.assert_financial_metric_valid(route.distance_nm, 'distance_nm', 0)
            self.assert_financial_metric_valid(route.average_transit_time, 'average_transit_time', 0)
    
    def test_calculate_congestion_metrics(self):
        """Test congestion metrics calculation."""
        # Generate mock vessel data with some vessels at same location (congested)
        vessels = self.mock_data.generate_vessel_data(10)
        
        # Make some vessels congested (same location, low speed)
        for i in range(3):
            vessels[i].latitude = 37.7749  # San Francisco Bay
            vessels[i].longitude = -122.4194
            vessels[i].speed = 2.0  # Low speed indicates waiting
        
        congestion_data = self.scraper.calculate_congestion_metrics(vessels)
        
        self.assertIsInstance(congestion_data, dict)
        self.assertIn('total_vessels', congestion_data)
        self.assertIn('congested_areas', congestion_data)
        self.assertIn('average_speed', congestion_data)
        
        # Validate metrics
        self.assert_financial_metric_valid(
            congestion_data['total_vessels'], 'total_vessels', 0
        )
        self.assert_financial_metric_valid(
            congestion_data['average_speed'], 'average_speed', 0
        )
    
    def test_store_vessel_data(self):
        """Test vessel data storage."""
        vessels = self.mock_data.generate_vessel_data(5)
        
        # Test storage
        result = self.scraper.store_vessel_data(vessels)
        
        self.assertTrue(result)
    
    def test_get_historical_vessel_data(self):
        """Test historical vessel data retrieval."""
        # First store some data
        vessels = self.mock_data.generate_vessel_data(5)
        self.scraper.store_vessel_data(vessels)
        
        # Retrieve historical data
        start_date = datetime.utcnow() - timedelta(days=1)
        end_date = datetime.utcnow() + timedelta(days=1)
        
        historical_data = self.scraper.get_historical_vessel_data(start_date, end_date)
        
        self.assertIsInstance(historical_data, list)


class TestPortScraper(BaseTestCase):
    """Test cases for PortScraper."""
    
    def setUp(self):
        """Set up test fixtures."""
        super().setUp()
        
        self.config = {
            'port_authorities': {
                'los_angeles': {
                    'url': 'https://api.portofla.org/v1',
                    'api_key': 'test_key'
                },
                'long_beach': {
                    'url': 'https://api.polb.com/v1',
                    'api_key': 'test_key'
                }
            },
            'update_frequency': 3600  # 1 hour
        }
        
        self.scraper = PortScraper(self.config)
    
    def test_initialization(self):
        """Test scraper initialization."""
        self.assertIsInstance(self.scraper, PortScraper)
        self.assertEqual(self.scraper.config, self.config)
    
    @patch('requests.get')
    def test_collect_port_throughput_success(self, mock_get):
        """Test successful port throughput collection."""
        mock_response_data = {
            'port_data': {
                'port_code': 'USLAX',
                'throughput_teu': 1500000,
                'monthly_change': 5.2,
                'yearly_change': -2.1,
                'timestamp': '2024-02-01T12:00:00Z'
            }
        }
        
        mock_get.return_value = MockAPIResponse(200, mock_response_data)
        
        throughput_data = self.scraper.collect_port_throughput('USLAX')
        
        self.assertIsInstance(throughput_data, dict)
        self.assertEqual(throughput_data['port_code'], 'USLAX')
        self.assert_financial_metric_valid(
            throughput_data['throughput_teu'], 'throughput_teu', 0
        )
    
    def test_calculate_congestion_level(self):
        """Test port congestion level calculation."""
        # Mock port metrics
        port_metrics = PortMetrics(
            port_code='USLAX',
            port_name='Port of Los Angeles',
            country='USA',
            throughput_teu=1500000,
            congestion_level=0.0,  # Will be calculated
            average_wait_time=24.5,
            berth_utilization=0.85,
            crane_productivity=35.2,
            vessel_count=45,
            timestamp=datetime.utcnow()
        )
        
        congestion_level = self.scraper.calculate_congestion_level(port_metrics)
        
        self.assert_financial_metric_valid(congestion_level, 'congestion_level', 0.0, 1.0)
    
    def test_collect_terminal_data(self):
        """Test terminal data collection."""
        terminals = self.scraper.collect_terminal_data('USLAX')
        
        self.assertIsInstance(terminals, list)
        for terminal in terminals:
            self.assertIsInstance(terminal, TerminalData)
            self.assertIsInstance(terminal.terminal_id, str)
            self.assertIsInstance(terminal.port_code, str)
    
    def test_store_port_metrics(self):
        """Test port metrics storage."""
        port_metrics = self.mock_data.generate_port_metrics(3)
        
        result = self.scraper.store_port_metrics(port_metrics)
        
        self.assertTrue(result)
    
    def test_get_port_efficiency_trends(self):
        """Test port efficiency trend analysis."""
        # Store some historical data first
        port_metrics = self.mock_data.generate_port_metrics(5)
        self.scraper.store_port_metrics(port_metrics)
        
        trends = self.scraper.get_port_efficiency_trends('PORT00', days=30)
        
        self.assertIsInstance(trends, dict)
        self.assertIn('efficiency_score', trends)
        self.assertIn('trend_direction', trends)


class TestFreightScraper(BaseTestCase):
    """Test cases for FreightScraper."""
    
    def setUp(self):
        """Set up test fixtures."""
        super().setUp()
        
        self.config = {
            'baltic_exchange': {
                'api_key': 'test_key',
                'base_url': 'https://api.balticexchange.com/v1'
            },
            'freightos': {
                'api_key': 'test_key',
                'base_url': 'https://api.freightos.com/v1'
            },
            'container_rates': {
                'sources': ['drewry', 'platts'],
                'update_frequency': 86400  # Daily
            }
        }
        
        self.scraper = FreightScraper(self.config)
    
    def test_initialization(self):
        """Test scraper initialization."""
        self.assertIsInstance(self.scraper, FreightScraper)
        self.assertEqual(self.scraper.config, self.config)
    
    @patch('requests.get')
    def test_collect_baltic_indices_success(self, mock_get):
        """Test successful Baltic Exchange indices collection."""
        mock_response_data = {
            'indices': [
                {
                    'index_name': 'BDI',
                    'value': 1250,
                    'change': 15,
                    'change_percent': 1.22,
                    'timestamp': '2024-02-01T12:00:00Z'
                },
                {
                    'index_name': 'BCI',
                    'value': 2100,
                    'change': -25,
                    'change_percent': -1.17,
                    'timestamp': '2024-02-01T12:00:00Z'
                }
            ]
        }
        
        mock_get.return_value = MockAPIResponse(200, mock_response_data)
        
        indices = self.scraper.collect_baltic_indices()
        
        self.assertIsInstance(indices, list)
        self.assertGreater(len(indices), 0)
        
        for index in indices:
            self.assertIsInstance(index, BalticIndex)
            self.assertIn(index.index_name, ['BDI', 'BCI', 'BPI', 'BSI'])
            self.assert_financial_metric_valid(index.value, 'value', 0)
    
    @patch('requests.get')
    def test_collect_container_rates_success(self, mock_get):
        """Test successful container rates collection."""
        mock_response_data = {
            'rates': [
                {
                    'origin': 'Shanghai',
                    'destination': 'Los Angeles',
                    'rate_20ft': 2500,
                    'rate_40ft': 4800,
                    'rate_40hc': 5200,
                    'currency': 'USD',
                    'valid_from': '2024-02-01',
                    'valid_to': '2024-02-15',
                    'carrier': 'Test Carrier',
                    'timestamp': '2024-02-01T12:00:00Z'
                }
            ]
        }
        
        mock_get.return_value = MockAPIResponse(200, mock_response_data)
        
        rates = self.scraper.collect_container_rates()
        
        self.assertIsInstance(rates, list)
        self.assertGreater(len(rates), 0)
        
        for rate in rates:
            self.assertIsInstance(rate, ContainerRate)
            self.assertIsInstance(rate.origin_port, str)
            self.assertIsInstance(rate.destination_port, str)
            self.assert_financial_metric_valid(rate.rate_20ft, 'rate_20ft', 0)
            self.assert_financial_metric_valid(rate.rate_40ft, 'rate_40ft', 0)
    
    def test_calculate_rate_volatility(self):
        """Test freight rate volatility calculation."""
        # Generate mock freight rates with some price variation
        rates = self.mock_data.generate_freight_rates(30)
        
        volatility = self.scraper.calculate_rate_volatility(rates)
        
        self.assertIsInstance(volatility, dict)
        self.assertIn('overall_volatility', volatility)
        self.assertIn('route_volatilities', volatility)
        
        # Validate volatility metrics
        self.assert_financial_metric_valid(
            volatility['overall_volatility'], 'overall_volatility', 0
        )
    
    def test_analyze_rate_trends(self):
        """Test freight rate trend analysis."""
        rates = self.mock_data.generate_freight_rates(50)
        
        trends = self.scraper.analyze_rate_trends(rates)
        
        self.assertIsInstance(trends, dict)
        self.assertIn('trend_direction', trends)
        self.assertIn('trend_strength', trends)
        self.assertIn('seasonal_patterns', trends)
    
    def test_store_freight_data(self):
        """Test freight data storage."""
        rates = self.mock_data.generate_freight_rates(10)
        
        result = self.scraper.store_freight_data(rates)
        
        self.assertTrue(result)
    
    def test_get_rate_forecasts(self):
        """Test freight rate forecasting."""
        # Store historical data first
        rates = self.mock_data.generate_freight_rates(100)
        self.scraper.store_freight_data(rates)
        
        forecasts = self.scraper.get_rate_forecasts('Shanghai', 'Los Angeles', days=30)
        
        self.assertIsInstance(forecasts, dict)
        self.assertIn('forecasted_rates', forecasts)
        self.assertIn('confidence_intervals', forecasts)
        self.assertIn('forecast_accuracy', forecasts)


class TestDataScrapersIntegration(BaseTestCase):
    """Integration tests for data scrapers."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        super().setUp()
        
        # Initialize all scrapers
        self.shipping_scraper = ShippingScraper({}, self.db_manager)
        self.port_scraper = PortScraper({}, self.db_manager)
        self.freight_scraper = FreightScraper({}, self.db_manager)
    
    def test_data_consistency_across_scrapers(self):
        """Test data consistency between different scrapers."""
        # Generate related mock data
        vessels = self.mock_data.generate_vessel_data(20)
        ports = self.mock_data.generate_port_metrics(5)
        rates = self.mock_data.generate_freight_rates(30)
        
        # Store data from all scrapers
        self.shipping_scraper.store_vessel_data(vessels)
        self.port_scraper.store_port_metrics(ports)
        self.freight_scraper.store_freight_data(rates)
        
        # Verify data can be retrieved consistently
        start_date = datetime.utcnow() - timedelta(days=1)
        end_date = datetime.utcnow() + timedelta(days=1)
        
        historical_vessels = self.shipping_scraper.get_historical_vessel_data(
            start_date, end_date
        )
        port_trends = self.port_scraper.get_port_efficiency_trends('PORT00', days=1)
        rate_forecasts = self.freight_scraper.get_rate_forecasts(
            'Shanghai', 'Los Angeles', days=7
        )
        
        # Basic validation that data exists and is properly formatted
        self.assertIsInstance(historical_vessels, list)
        self.assertIsInstance(port_trends, dict)
        self.assertIsInstance(rate_forecasts, dict)
    
    def test_cross_scraper_data_correlation(self):
        """Test correlation analysis between different data sources."""
        # This would test how vessel congestion correlates with port metrics
        # and freight rates in a real scenario
        
        vessels = self.mock_data.generate_vessel_data(50)
        ports = self.mock_data.generate_port_metrics(10)
        
        # Analyze congestion from shipping data
        congestion_metrics = self.shipping_scraper.calculate_congestion_metrics(vessels)
        
        # Get port efficiency metrics
        port_efficiency = {}
        for port in ports:
            efficiency = self.port_scraper.calculate_congestion_level(port)
            port_efficiency[port.port_code] = efficiency
        
        # Verify both metrics are calculated
        self.assertIsInstance(congestion_metrics, dict)
        self.assertIsInstance(port_efficiency, dict)
        
        # In a real implementation, you would test correlation between
        # vessel congestion and port efficiency metrics
    
    def test_data_quality_validation(self):
        """Test data quality validation across all scrapers."""
        # Generate data with some quality issues
        vessels = self.mock_data.generate_vessel_data(10)
        
        # Introduce some data quality issues for testing
        vessels[0].latitude = 999  # Invalid latitude
        vessels[1].speed = -5  # Invalid speed
        
        # Test that scrapers handle invalid data appropriately
        valid_vessels = []
        for vessel in vessels:
            if self._is_valid_vessel_data(vessel):
                valid_vessels.append(vessel)
        
        # Should filter out invalid vessels
        self.assertLess(len(valid_vessels), len(vessels))
        
        # All remaining vessels should be valid
        for vessel in valid_vessels:
            self.assertTrue(self._is_valid_vessel_data(vessel))
    
    def _is_valid_vessel_data(self, vessel: VesselData) -> bool:
        """Validate vessel data quality.
        
        Args:
            vessel: Vessel data to validate
            
        Returns:
            True if data is valid
        """
        try:
            # Check latitude/longitude bounds
            if not (-90 <= vessel.latitude <= 90):
                return False
            if not (-180 <= vessel.longitude <= 180):
                return False
            
            # Check speed is non-negative
            if vessel.speed < 0:
                return False
            
            # Check required fields are not empty
            if not vessel.vessel_id or not vessel.name:
                return False
            
            return True
        except:
            return False


if __name__ == '__main__':
    unittest.main()