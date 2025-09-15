#!/usr/bin/env python3
"""
Unit Tests for Trading Strategies

Tests for long/short equity strategies and portfolio management.

Test Coverage:
- Signal generation and validation
- Portfolio construction and optimization
- Risk management and position sizing
- Performance metrics calculation
- Strategy backtesting
- ESG integration

Author: Supply Chain Alpha Team
Version: 1.0.0
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
import tempfile
from pathlib import Path

# Import test base classes
from tests.base_test import BaseTestCase, AsyncTestCase, MockDataGenerator, PerformanceTestMixin

# Import modules to test
from strategies.long_short_equity import (
    LongShortEquityStrategy, Position, TradeSignal, PortfolioMetrics
)
from models.disruption_predictor import DisruptionPrediction
from models.sector_impact_analyzer import SectorImpactScore, SectorRotationSignal
from utils.esg_metrics import ESGScore, ESGAdjustedSignal


class TestLongShortEquityStrategy(BaseTestCase, PerformanceTestMixin):
    """Test cases for LongShortEquityStrategy."""
    
    def setUp(self):
        """Set up test fixtures."""
        super().setUp()
        
        self.config = {
            'portfolio': {
                'max_positions': 100,
                'max_position_size': 0.05,  # 5% max per position
                'target_leverage': 1.3,
                'cash_buffer': 0.1
            },
            'risk_management': {
                'max_sector_exposure': 0.3,
                'max_single_stock_weight': 0.05,
                'stop_loss_threshold': -0.15,
                'take_profit_threshold': 0.25,
                'var_limit': 0.02  # 2% daily VaR limit
            },
            'signal_generation': {
                'min_conviction_score': 0.6,
                'signal_decay_days': 30,
                'rebalance_frequency': 'weekly',
                'esg_weight': 0.2
            },
            'execution': {
                'slippage_model': 'linear',
                'transaction_costs': 0.001,  # 10 bps
                'market_impact_factor': 0.0005
            }
        }
        
        self.strategy = LongShortEquityStrategy(self.config, self.db_manager)
        
        # Mock universe of stocks
        self.universe = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA',
            'FDX', 'UPS', 'WMT', 'TGT', 'COST',
            'CAT', 'DE', 'GE', 'MMM', 'XOM'
        ]
    
    def test_initialization(self):
        """Test strategy initialization."""
        self.assertIsInstance(self.strategy, LongShortEquityStrategy)
        self.assertEqual(self.strategy.config, self.config)
        self.assertIsInstance(self.strategy.positions, dict)
        self.assertIsInstance(self.strategy.portfolio_metrics, dict)
    
    def test_generate_trade_signals(self):
        """Test trade signal generation."""
        # Generate mock predictions and impacts
        predictions = self.mock_data.generate_disruption_predictions(10)
        sector_impacts = self.mock_data.generate_sector_impacts(5)
        esg_scores = self.mock_data.generate_esg_scores(self.universe)
        
        # Generate signals
        signals = self.strategy.generate_trade_signals(
            predictions, sector_impacts, esg_scores
        )
        
        self.assertIsInstance(signals, list)
        
        for signal in signals:
            self.assertIsInstance(signal, TradeSignal)
            self.assertIn(signal.symbol, self.universe)
            self.assertIn(signal.signal_type, ['long', 'short', 'close'])
            self.assert_financial_metric_valid(
                signal.conviction_score, 'conviction_score', 0.0, 1.0
            )
            self.assert_financial_metric_valid(
                signal.target_weight, 'target_weight', -0.1, 0.1
            )
    
    def test_signal_conviction_calculation(self):
        """Test conviction score calculation."""
        # Create test prediction
        prediction = DisruptionPrediction(
            disruption_type='port_congestion',
            location='Shanghai',
            probability=0.8,
            severity=0.7,
            duration_days=14,
            affected_routes=['Asia-US West Coast'],
            confidence=0.85,
            timestamp=datetime.utcnow()
        )
        
        # Create test sector impact
        sector_impact = SectorImpactScore(
            sector_name='transportation',
            impact_score=0.75,
            confidence=0.8,
            affected_companies=['FDX', 'UPS'],
            impact_factors=['shipping_delays', 'increased_costs'],
            timestamp=datetime.utcnow()
        )
        
        # Calculate conviction
        conviction = self.strategy.calculate_conviction_score(
            'FDX', [prediction], [sector_impact]
        )
        
        self.assert_financial_metric_valid(conviction, 'conviction', 0.0, 1.0)
        
        # High impact should result in high conviction
        self.assertGreater(conviction, 0.5)
    
    def test_portfolio_construction(self):
        """Test portfolio construction from signals."""
        # Generate mock signals
        signals = []
        for i, symbol in enumerate(self.universe[:10]):
            signal = TradeSignal(
                symbol=symbol,
                signal_type='long' if i % 2 == 0 else 'short',
                conviction_score=0.7 + (i * 0.02),
                target_weight=0.03 * (1 if i % 2 == 0 else -1),
                expected_return=0.05 * (1 if i % 2 == 0 else -1),
                risk_score=0.4 + (i * 0.01),
                timestamp=datetime.utcnow()
            )
            signals.append(signal)
        
        # Construct portfolio
        portfolio = self.strategy.construct_portfolio(signals)
        
        self.assertIsInstance(portfolio, dict)
        
        # Check portfolio constraints
        total_long_weight = sum(w for w in portfolio.values() if w > 0)
        total_short_weight = sum(abs(w) for w in portfolio.values() if w < 0)
        
        # Long and short sides should be roughly balanced
        self.assertLess(abs(total_long_weight - total_short_weight), 0.2)
        
        # No position should exceed max size
        for weight in portfolio.values():
            self.assertLessEqual(
                abs(weight), self.config['portfolio']['max_position_size']
            )
    
    def test_risk_management(self):
        """Test risk management constraints."""
        # Create portfolio with some risk violations
        test_portfolio = {
            'AAPL': 0.08,  # Exceeds max position size
            'MSFT': 0.04,
            'GOOGL': 0.03,
            'FDX': -0.06,  # Exceeds max short position
            'UPS': -0.03,
            'WMT': 0.02
        }
        
        # Apply risk management
        adjusted_portfolio = self.strategy.apply_risk_management(test_portfolio)
        
        self.assertIsInstance(adjusted_portfolio, dict)
        
        # Check that violations are corrected
        for symbol, weight in adjusted_portfolio.items():
            self.assertLessEqual(
                abs(weight), self.config['portfolio']['max_position_size']
            )
    
    def test_position_sizing(self):
        """Test position sizing algorithm."""
        # Mock current portfolio value
        portfolio_value = 1000000  # $1M
        
        # Test signal
        signal = TradeSignal(
            symbol='AAPL',
            signal_type='long',
            conviction_score=0.8,
            target_weight=0.04,
            expected_return=0.06,
            risk_score=0.3,
            timestamp=datetime.utcnow()
        )
        
        # Calculate position size
        position_size = self.strategy.calculate_position_size(
            signal, portfolio_value
        )
        
        self.assertIsInstance(position_size, dict)
        self.assertIn('shares', position_size)
        self.assertIn('dollar_amount', position_size)
        self.assertIn('weight', position_size)
        
        # Position should respect risk limits
        self.assertLessEqual(
            position_size['weight'],
            self.config['portfolio']['max_position_size']
        )
    
    def test_execute_strategy(self):
        """Test strategy execution."""
        # Generate mock market data
        market_data = self.mock_data.generate_stock_data(self.universe, days=30)
        
        # Generate mock predictions and impacts
        predictions = self.mock_data.generate_disruption_predictions(15)
        sector_impacts = self.mock_data.generate_sector_impacts(8)
        esg_scores = self.mock_data.generate_esg_scores(self.universe)
        
        # Execute strategy
        execution_result = self.strategy.execute_strategy(
            market_data, predictions, sector_impacts, esg_scores
        )
        
        self.assertIsInstance(execution_result, dict)
        self.assertIn('signals_generated', execution_result)
        self.assertIn('positions_updated', execution_result)
        self.assertIn('portfolio_metrics', execution_result)
        
        # Validate execution metrics
        metrics = execution_result['portfolio_metrics']
        self.assertIsInstance(metrics, PortfolioMetrics)
    
    def test_portfolio_optimization(self):
        """Test portfolio optimization."""
        # Generate mock expected returns and covariance matrix
        n_assets = len(self.universe)
        expected_returns = np.random.normal(0.08, 0.15, n_assets)
        
        # Create realistic covariance matrix
        random_matrix = np.random.randn(n_assets, n_assets)
        covariance_matrix = np.dot(random_matrix, random_matrix.T) * 0.01
        
        # Optimize portfolio
        optimal_weights = self.strategy.optimize_portfolio(
            expected_returns, covariance_matrix, self.universe
        )
        
        self.assertIsInstance(optimal_weights, dict)
        self.assertEqual(len(optimal_weights), len(self.universe))
        
        # Weights should sum to approximately zero (market neutral)
        total_weight = sum(optimal_weights.values())
        self.assertAlmostEqual(total_weight, 0.0, places=2)
        
        # No weight should exceed limits
        for weight in optimal_weights.values():
            self.assertLessEqual(
                abs(weight), self.config['portfolio']['max_position_size']
            )
    
    def test_performance_calculation(self):
        """Test performance metrics calculation."""
        # Generate mock portfolio returns
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        returns = pd.Series(
            np.random.normal(0.0008, 0.02, len(dates)),  # ~20% annual vol
            index=dates
        )
        
        # Calculate performance metrics
        performance = self.strategy.calculate_performance_metrics(returns)
        
        self.assertIsInstance(performance, PortfolioMetrics)
        
        # Validate metrics
        self.assert_financial_metric_valid(
            performance.total_return, 'total_return', -1.0, 5.0
        )
        self.assert_financial_metric_valid(
            performance.annual_volatility, 'volatility', 0.0, 1.0
        )
        self.assert_financial_metric_valid(
            performance.sharpe_ratio, 'sharpe_ratio', -5.0, 5.0
        )
        self.assert_financial_metric_valid(
            performance.max_drawdown, 'max_drawdown', -1.0, 0.0
        )
    
    def test_esg_integration(self):
        """Test ESG factor integration."""
        # Generate base signal
        base_signal = TradeSignal(
            symbol='AAPL',
            signal_type='long',
            conviction_score=0.7,
            target_weight=0.04,
            expected_return=0.05,
            risk_score=0.3,
            timestamp=datetime.utcnow()
        )
        
        # Generate ESG score
        esg_score = ESGScore(
            symbol='AAPL',
            environmental_score=0.8,
            social_score=0.7,
            governance_score=0.9,
            overall_score=0.8,
            esg_risk_rating='low',
            sustainability_rank=15,
            timestamp=datetime.utcnow()
        )
        
        # Apply ESG adjustment
        adjusted_signal = self.strategy.apply_esg_adjustment(base_signal, esg_score)
        
        self.assertIsInstance(adjusted_signal, ESGAdjustedSignal)
        
        # High ESG score should improve the signal
        self.assertGreaterEqual(
            adjusted_signal.adjusted_conviction,
            base_signal.conviction_score
        )
    
    def test_sector_rotation_signals(self):
        """Test sector rotation signal generation."""
        # Generate rotation signals
        rotation_signals = [
            RotationSignal(
                from_sector='energy',
                to_sector='technology',
                signal_strength='strong',
                confidence=0.8,
                expected_duration_days=45,
                rationale='Supply chain disruption favors tech over energy',
                timestamp=datetime.utcnow()
            )
        ]
        
        # Generate sector-based trade signals
        sector_signals = self.strategy.generate_sector_rotation_trades(
            rotation_signals, self.universe
        )
        
        self.assertIsInstance(sector_signals, list)
        
        for signal in sector_signals:
            self.assertIsInstance(signal, TradeSignal)
            # Should have both long and short signals for rotation
            self.assertIn(signal.signal_type, ['long', 'short'])
    
    def test_stop_loss_take_profit(self):
        """Test stop loss and take profit logic."""
        # Create test position
        position = Position(
            symbol='AAPL',
            quantity=1000,
            entry_price=150.0,
            current_price=140.0,  # 6.7% loss
            position_type='long',
            entry_date=datetime.utcnow() - timedelta(days=10),
            stop_loss_price=127.5,  # 15% stop loss
            take_profit_price=187.5  # 25% take profit
        )
        
        # Check if stop loss should trigger
        should_close = self.strategy.should_close_position(position)
        
        # Should not trigger stop loss yet (only 6.7% loss)
        self.assertFalse(should_close)
        
        # Update to trigger stop loss
        position.current_price = 125.0  # 16.7% loss
        should_close = self.strategy.should_close_position(position)
        
        # Should trigger stop loss now
        self.assertTrue(should_close)
    
    def test_market_impact_modeling(self):
        """Test market impact estimation."""
        # Test large order
        order_size = 100000  # shares
        avg_daily_volume = 50000000  # shares
        current_price = 150.0
        
        market_impact = self.strategy.estimate_market_impact(
            order_size, avg_daily_volume, current_price
        )
        
        self.assertIsInstance(market_impact, dict)
        self.assertIn('price_impact', market_impact)
        self.assertIn('cost_bps', market_impact)
        
        # Market impact should be positive for large orders
        self.assertGreater(market_impact['cost_bps'], 0)
    
    def test_strategy_backtesting(self):
        """Test strategy backtesting functionality."""
        # Generate historical data
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 12, 31)
        
        historical_data = self.mock_data.generate_historical_market_data(
            self.universe, start_date, end_date
        )
        
        # Run backtest
        backtest_results = self.strategy.backtest(
            historical_data, start_date, end_date
        )
        
        self.assertIsInstance(backtest_results, dict)
        self.assertIn('performance_metrics', backtest_results)
        self.assertIn('trade_history', backtest_results)
        self.assertIn('portfolio_evolution', backtest_results)
        
        # Validate backtest results
        performance = backtest_results['performance_metrics']
        self.assertIsInstance(performance, PortfolioMetrics)
    
    def test_strategy_performance_under_stress(self):
        """Test strategy performance during market stress."""
        # Generate stressed market conditions
        stress_data = self.mock_data.generate_stressed_market_data(
            self.universe, stress_type='market_crash'
        )
        
        predictions = self.mock_data.generate_disruption_predictions(20)
        sector_impacts = self.mock_data.generate_sector_impacts(10)
        esg_scores = self.mock_data.generate_esg_scores(self.universe)
        
        # Execute strategy under stress
        def execute_under_stress():
            return self.strategy.execute_strategy(
                stress_data, predictions, sector_impacts, esg_scores
            )
        
        # Should handle stress conditions gracefully
        stress_results = self.assert_execution_time(execute_under_stress, 15.0)
        
        self.assertIsInstance(stress_results, dict)
        
        # Strategy should maintain risk controls under stress
        portfolio_metrics = stress_results['portfolio_metrics']
        self.assertIsInstance(portfolio_metrics, PortfolioMetrics)
    
    def test_position_management(self):
        """Test position management and updates."""
        # Create initial positions
        initial_positions = {
            'AAPL': Position(
                symbol='AAPL',
                quantity=1000,
                entry_price=150.0,
                current_price=155.0,
                position_type='long',
                entry_date=datetime.utcnow() - timedelta(days=5)
            ),
            'TSLA': Position(
                symbol='TSLA',
                quantity=-500,
                entry_price=200.0,
                current_price=195.0,
                position_type='short',
                entry_date=datetime.utcnow() - timedelta(days=3)
            )
        }
        
        self.strategy.positions = initial_positions
        
        # Generate new signals
        new_signals = [
            TradeSignal(
                symbol='AAPL',
                signal_type='close',  # Close existing long
                conviction_score=0.3,
                target_weight=0.0,
                expected_return=0.0,
                risk_score=0.2,
                timestamp=datetime.utcnow()
            ),
            TradeSignal(
                symbol='MSFT',
                signal_type='long',  # New long position
                conviction_score=0.8,
                target_weight=0.03,
                expected_return=0.07,
                risk_score=0.25,
                timestamp=datetime.utcnow()
            )
        ]
        
        # Update positions
        updated_positions = self.strategy.update_positions(new_signals)
        
        self.assertIsInstance(updated_positions, dict)
        
        # AAPL position should be closed or reduced
        if 'AAPL' in updated_positions:
            self.assertLess(
                abs(updated_positions['AAPL'].quantity),
                abs(initial_positions['AAPL'].quantity)
            )
        
        # MSFT position should be added
        self.assertIn('MSFT', updated_positions)


class TestStrategyIntegration(BaseTestCase):
    """Integration tests for trading strategies."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        super().setUp()
        
        self.strategy = LongShortEquityStrategy({}, self.db_manager)
        self.universe = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    
    def test_full_strategy_workflow(self):
        """Test complete strategy workflow from signals to execution."""
        # Generate comprehensive input data
        market_data = self.mock_data.generate_stock_data(self.universe, days=60)
        predictions = self.mock_data.generate_disruption_predictions(25)
        sector_impacts = self.mock_data.generate_sector_impacts(12)
        esg_scores = self.mock_data.generate_esg_scores(self.universe)
        
        # Execute full workflow
        def full_workflow():
            # Step 1: Generate signals
            signals = self.strategy.generate_trade_signals(
                predictions, sector_impacts, esg_scores
            )
            
            # Step 2: Construct portfolio
            portfolio = self.strategy.construct_portfolio(signals)
            
            # Step 3: Apply risk management
            risk_adjusted_portfolio = self.strategy.apply_risk_management(portfolio)
            
            # Step 4: Execute trades
            execution_result = self.strategy.execute_strategy(
                market_data, predictions, sector_impacts, esg_scores
            )
            
            return signals, portfolio, risk_adjusted_portfolio, execution_result
        
        # Should complete workflow efficiently
        results = self.assert_execution_time(full_workflow, 20.0)
        
        signals, portfolio, risk_adjusted_portfolio, execution_result = results
        
        # Validate workflow outputs
        self.assertIsInstance(signals, list)
        self.assertIsInstance(portfolio, dict)
        self.assertIsInstance(risk_adjusted_portfolio, dict)
        self.assertIsInstance(execution_result, dict)
    
    def test_strategy_consistency_over_time(self):
        """Test strategy consistency across multiple time periods."""
        # Generate data for multiple periods
        periods = 5
        all_results = []
        
        for period in range(periods):
            # Generate period-specific data
            market_data = self.mock_data.generate_stock_data(
                self.universe, days=30, seed=period
            )
            predictions = self.mock_data.generate_disruption_predictions(
                10, seed=period
            )
            sector_impacts = self.mock_data.generate_sector_impacts(
                5, seed=period
            )
            esg_scores = self.mock_data.generate_esg_scores(
                self.universe, seed=period
            )
            
            # Execute strategy
            result = self.strategy.execute_strategy(
                market_data, predictions, sector_impacts, esg_scores
            )
            
            all_results.append(result)
        
        # Validate consistency
        self.assertEqual(len(all_results), periods)
        
        for result in all_results:
            self.assertIsInstance(result, dict)
            self.assertIn('portfolio_metrics', result)
            
            # Metrics should be within reasonable ranges
            metrics = result['portfolio_metrics']
            self.assertIsInstance(metrics, PortfolioMetrics)


if __name__ == '__main__':
    unittest.main()