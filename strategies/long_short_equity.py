#!/usr/bin/env python3
"""
Long/Short Equity Strategy
Systematic equity strategy based on supply chain disruption predictions.

Features:
- Long/short position generation
- Risk management and position sizing
- Portfolio optimization
- Dynamic hedging strategies
- Performance attribution
- ESG integration

Author: Supply Chain Alpha Team
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
from collections import defaultdict

# Optimization and analysis libraries
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import LedoitWolf
import cvxpy as cp

# Financial libraries
import yfinance as yf
import pandas_datareader as pdr

from utils.database import DatabaseManager
from models.disruption_predictor import DisruptionPrediction, CompanyImpactScore
from models.sector_impact_analyzer import SectorImpactScore, SectorRotationSignal


@dataclass
class Position:
    """Data structure for portfolio positions."""
    symbol: str
    company_name: str
    sector: str
    position_type: str  # 'long', 'short'
    target_weight: float  # -1 to 1
    current_weight: float
    shares: int
    entry_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float
    risk_score: float  # 0-10
    esg_score: float  # 0-100
    conviction_level: float  # 0-1
    entry_date: datetime
    expected_holding_period: int  # days
    stop_loss: Optional[float]
    take_profit: Optional[float]
    rationale: str
    timestamp: datetime


@dataclass
class TradeSignal:
    """Data structure for trade signals."""
    signal_id: str
    symbol: str
    company_name: str
    sector: str
    signal_type: str  # 'buy', 'sell', 'short', 'cover'
    strength: float  # 0-1
    target_weight: float  # -1 to 1
    expected_return: float  # percentage
    risk_level: str  # 'low', 'medium', 'high'
    time_horizon: str  # 'short', 'medium', 'long'
    confidence: float  # 0-1
    rationale: str
    supporting_factors: List[str]
    risk_factors: List[str]
    esg_considerations: List[str]
    entry_price_target: float
    stop_loss_target: float
    take_profit_target: float
    max_position_size: float  # percentage of portfolio
    timestamp: datetime


@dataclass
class PortfolioMetrics:
    """Data structure for portfolio performance metrics."""
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    long_exposure: float
    short_exposure: float
    net_exposure: float
    gross_exposure: float
    beta: float
    alpha: float
    information_ratio: float
    var_95: float  # Value at Risk
    expected_shortfall: float
    sector_allocation: Dict[str, float]
    top_positions: List[Dict[str, Any]]
    timestamp: datetime


class LongShortEquityStrategy:
    """Systematic long/short equity strategy based on supply chain disruptions."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the long/short equity strategy.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Database manager
        self.db_manager = DatabaseManager()
        
        # Strategy parameters
        self.max_gross_exposure = config.get('max_gross_exposure', 1.0)  # 100%
        self.max_net_exposure = config.get('max_net_exposure', 0.3)  # 30%
        self.max_single_position = config.get('max_single_position', 0.05)  # 5%
        self.min_conviction_threshold = config.get('min_conviction_threshold', 0.6)
        self.rebalance_frequency = config.get('rebalance_frequency', 'daily')
        self.risk_budget = config.get('risk_budget', 0.15)  # 15% annual volatility target
        
        # ESG integration
        self.esg_weight = config.get('esg_weight', 0.2)  # 20% weight in scoring
        self.min_esg_score = config.get('min_esg_score', 30)  # Minimum ESG score
        
        # Current portfolio
        self.positions: Dict[str, Position] = {}
        self.cash_balance = config.get('initial_capital', 1000000)  # $1M default
        self.portfolio_value = self.cash_balance
        
        # Performance tracking
        self.performance_history = []
        self.trade_history = []
        
        # Risk models
        self.covariance_matrix = None
        self.factor_loadings = None
        
        # Market data cache
        self.price_data = {}
        self.fundamental_data = {}
        self.esg_data = {}
    
    async def generate_trade_signals(self, 
                                   company_impacts: List[CompanyImpactScore],
                                   sector_impacts: List[SectorImpactScore],
                                   sector_signals: List[SectorRotationSignal]) -> List[TradeSignal]:
        """Generate trade signals based on disruption predictions.
        
        Args:
            company_impacts: List of company impact scores
            sector_impacts: List of sector impact scores
            sector_signals: List of sector rotation signals
            
        Returns:
            List of TradeSignal objects
        """
        self.logger.info(f"Generating trade signals for {len(company_impacts)} companies")
        
        try:
            trade_signals = []
            
            # Create sector impact lookup
            sector_lookup = {si.sector_name: si for si in sector_impacts}
            
            # Generate company-specific signals
            for company_impact in company_impacts:
                try:
                    signal = await self._generate_company_signal(
                        company_impact, sector_lookup
                    )
                    if signal and signal.confidence >= self.min_conviction_threshold:
                        trade_signals.append(signal)
                        
                except Exception as e:
                    self.logger.error(f"Error generating signal for {company_impact.company_name}: {e}")
            
            # Generate sector-based signals
            sector_trade_signals = await self._generate_sector_signals(sector_signals)
            trade_signals.extend(sector_trade_signals)
            
            # Apply portfolio constraints and optimization
            optimized_signals = await self._optimize_signal_portfolio(trade_signals)
            
            # Sort by conviction and expected return
            optimized_signals.sort(
                key=lambda x: x.strength * x.confidence * abs(x.expected_return), 
                reverse=True
            )
            
            self.logger.info(f"Generated {len(optimized_signals)} optimized trade signals")
            return optimized_signals
            
        except Exception as e:
            self.logger.error(f"Error generating trade signals: {e}")
            raise
    
    async def execute_strategy(self, trade_signals: List[TradeSignal]) -> Dict[str, Any]:
        """Execute the trading strategy based on generated signals.
        
        Args:
            trade_signals: List of trade signals to execute
            
        Returns:
            Dictionary with execution results
        """
        self.logger.info(f"Executing strategy with {len(trade_signals)} signals")
        
        try:
            execution_results = {
                'trades_executed': 0,
                'trades_rejected': 0,
                'total_turnover': 0,
                'new_positions': [],
                'closed_positions': [],
                'position_adjustments': [],
                'errors': []
            }
            
            # Update market data
            await self._update_market_data([signal.symbol for signal in trade_signals])
            
            # Calculate current portfolio metrics
            current_metrics = await self._calculate_portfolio_metrics()
            
            # Execute signals in order of priority
            for signal in trade_signals:
                try:
                    result = await self._execute_signal(signal, current_metrics)
                    
                    if result['success']:
                        execution_results['trades_executed'] += 1
                        execution_results['total_turnover'] += result.get('turnover', 0)
                        
                        if result['action'] == 'new_position':
                            execution_results['new_positions'].append(result['position'])
                        elif result['action'] == 'close_position':
                            execution_results['closed_positions'].append(result['position'])
                        elif result['action'] == 'adjust_position':
                            execution_results['position_adjustments'].append(result['position'])
                    else:
                        execution_results['trades_rejected'] += 1
                        execution_results['errors'].append({
                            'symbol': signal.symbol,
                            'error': result.get('error', 'Unknown error')
                        })
                        
                except Exception as e:
                    self.logger.error(f"Error executing signal for {signal.symbol}: {e}")
                    execution_results['trades_rejected'] += 1
                    execution_results['errors'].append({
                        'symbol': signal.symbol,
                        'error': str(e)
                    })
            
            # Update portfolio metrics after execution
            updated_metrics = await self._calculate_portfolio_metrics()
            execution_results['portfolio_metrics'] = updated_metrics
            
            # Risk check after execution
            risk_check = await self._perform_risk_check()
            execution_results['risk_check'] = risk_check
            
            self.logger.info(f"Strategy execution completed: {execution_results['trades_executed']} trades executed")
            return execution_results
            
        except Exception as e:
            self.logger.error(f"Error executing strategy: {e}")
            raise
    
    async def _generate_company_signal(self, 
                                     company_impact: CompanyImpactScore,
                                     sector_lookup: Dict[str, SectorImpactScore]) -> Optional[TradeSignal]:
        """Generate trade signal for a specific company.
        
        Args:
            company_impact: Company impact score
            sector_lookup: Dictionary of sector impacts
            
        Returns:
            TradeSignal object or None
        """
        try:
            # Get sector impact
            sector_impact = sector_lookup.get(company_impact.sector)
            
            # Calculate base signal strength
            impact_score = company_impact.impact_score
            probability = company_impact.probability
            
            # Determine signal direction
            if impact_score < -3:  # Significant negative impact
                signal_type = 'short'
                expected_return = impact_score * 0.5  # Convert to expected return
                target_weight = -min(self.max_single_position, abs(impact_score) / 20)
            elif impact_score > 2:  # Positive impact (beneficiary)
                signal_type = 'buy'
                expected_return = impact_score * 0.3
                target_weight = min(self.max_single_position, impact_score / 25)
            else:
                return None  # No significant signal
            
            # Calculate signal strength
            strength = min(1.0, (abs(impact_score) / 10) * probability)
            
            # Get company data
            company_data = await self._get_company_data(company_impact.symbol)
            
            # Calculate confidence based on multiple factors
            confidence_factors = [
                probability,  # Disruption probability
                min(1.0, abs(impact_score) / 10),  # Impact magnitude
                company_data.get('data_quality', 0.7),  # Data quality
                company_data.get('liquidity_score', 0.8),  # Liquidity
            ]
            
            # ESG adjustment
            esg_score = company_data.get('esg_score', 50)
            if esg_score < self.min_esg_score:
                confidence_factors.append(0.5)  # Reduce confidence for poor ESG
            else:
                confidence_factors.append(min(1.0, esg_score / 100))
            
            confidence = np.mean(confidence_factors)
            
            # Risk assessment
            risk_factors = company_impact.risk_factors.copy()
            if company_data.get('beta', 1.0) > 1.5:
                risk_factors.append('High beta stock')
            if company_data.get('debt_to_equity', 0.5) > 1.0:
                risk_factors.append('High leverage')
            
            risk_level = 'high' if len(risk_factors) > 3 else 'medium' if len(risk_factors) > 1 else 'low'
            
            # Price targets
            current_price = company_data.get('current_price', 100)
            entry_price_target = current_price * (1 + np.random.uniform(-0.02, 0.02))  # Small entry buffer
            
            if signal_type == 'short':
                stop_loss_target = current_price * 1.1  # 10% stop loss for shorts
                take_profit_target = current_price * (1 + expected_return / 100)
            else:
                stop_loss_target = current_price * 0.9  # 10% stop loss for longs
                take_profit_target = current_price * (1 + expected_return / 100)
            
            # Supporting factors
            supporting_factors = [
                f"Supply chain impact score: {impact_score:.1f}",
                f"Disruption probability: {probability:.1%}",
                f"Sector: {company_impact.sector}"
            ]
            
            if sector_impact:
                supporting_factors.append(f"Sector vulnerability: {sector_impact.vulnerability_score:.1f}")
            
            # ESG considerations
            esg_considerations = []
            if esg_score < 40:
                esg_considerations.append('Poor ESG rating - proceed with caution')
            elif esg_score > 70:
                esg_considerations.append('Strong ESG profile supports investment thesis')
            
            # Time horizon based on recovery estimate
            recovery_days = company_impact.recovery_time_estimate
            if recovery_days < 30:
                time_horizon = 'short'
            elif recovery_days < 90:
                time_horizon = 'medium'
            else:
                time_horizon = 'long'
            
            signal = TradeSignal(
                signal_id=f"{company_impact.symbol}_{signal_type}_{datetime.utcnow().strftime('%Y%m%d_%H%M')}",
                symbol=company_impact.symbol,
                company_name=company_impact.company_name,
                sector=company_impact.sector,
                signal_type=signal_type,
                strength=strength,
                target_weight=target_weight,
                expected_return=expected_return,
                risk_level=risk_level,
                time_horizon=time_horizon,
                confidence=confidence,
                rationale=f"Supply chain disruption impact: {company_impact.primary_risk_factor}",
                supporting_factors=supporting_factors,
                risk_factors=risk_factors,
                esg_considerations=esg_considerations,
                entry_price_target=entry_price_target,
                stop_loss_target=stop_loss_target,
                take_profit_target=take_profit_target,
                max_position_size=self.max_single_position,
                timestamp=datetime.utcnow()
            )
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Error generating company signal for {company_impact.company_name}: {e}")
            return None
    
    async def _generate_sector_signals(self, sector_signals: List[SectorRotationSignal]) -> List[TradeSignal]:
        """Generate trade signals based on sector rotation recommendations.
        
        Args:
            sector_signals: List of sector rotation signals
            
        Returns:
            List of TradeSignal objects
        """
        trade_signals = []
        
        try:
            for sector_signal in sector_signals:
                # Get sector ETF or representative stocks
                if sector_signal.signal_type == 'overweight':
                    # Generate long signals for sector leaders
                    sector_stocks = await self._get_sector_leaders(sector_signal.target_sector)
                    
                    for stock in sector_stocks[:3]:  # Top 3 stocks
                        signal = TradeSignal(
                            signal_id=f"{stock['symbol']}_sector_long_{datetime.utcnow().strftime('%Y%m%d_%H%M')}",
                            symbol=stock['symbol'],
                            company_name=stock['name'],
                            sector=sector_signal.target_sector,
                            signal_type='buy',
                            strength=sector_signal.strength * 0.8,  # Slightly lower for sector plays
                            target_weight=self.max_single_position * 0.6,  # Smaller positions for sector plays
                            expected_return=sector_signal.expected_return * 0.7,
                            risk_level='medium',
                            time_horizon=sector_signal.time_horizon,
                            confidence=sector_signal.confidence * 0.9,
                            rationale=f"Sector overweight: {sector_signal.rationale}",
                            supporting_factors=[f"Sector rotation signal", f"Strength: {sector_signal.strength:.2f}"],
                            risk_factors=["Sector concentration risk"],
                            esg_considerations=[],
                            entry_price_target=stock.get('price', 100),
                            stop_loss_target=stock.get('price', 100) * 0.92,
                            take_profit_target=stock.get('price', 100) * (1 + sector_signal.expected_return / 200),
                            max_position_size=self.max_single_position * 0.6,
                            timestamp=datetime.utcnow()
                        )
                        trade_signals.append(signal)
                
                elif sector_signal.signal_type == 'underweight':
                    # Generate short signals for sector laggards
                    sector_stocks = await self._get_sector_laggards(sector_signal.source_sector)
                    
                    for stock in sector_stocks[:2]:  # Top 2 shorts
                        signal = TradeSignal(
                            signal_id=f"{stock['symbol']}_sector_short_{datetime.utcnow().strftime('%Y%m%d_%H%M')}",
                            symbol=stock['symbol'],
                            company_name=stock['name'],
                            sector=sector_signal.source_sector,
                            signal_type='short',
                            strength=sector_signal.strength * 0.7,
                            target_weight=-self.max_single_position * 0.5,
                            expected_return=sector_signal.expected_return * 0.6,
                            risk_level='high',  # Shorts are riskier
                            time_horizon=sector_signal.time_horizon,
                            confidence=sector_signal.confidence * 0.8,
                            rationale=f"Sector underweight: {sector_signal.rationale}",
                            supporting_factors=[f"Sector rotation signal", f"Strength: {sector_signal.strength:.2f}"],
                            risk_factors=["Short selling risk", "Sector concentration risk"],
                            esg_considerations=[],
                            entry_price_target=stock.get('price', 100),
                            stop_loss_target=stock.get('price', 100) * 1.08,
                            take_profit_target=stock.get('price', 100) * (1 + sector_signal.expected_return / 200),
                            max_position_size=self.max_single_position * 0.5,
                            timestamp=datetime.utcnow()
                        )
                        trade_signals.append(signal)
        
        except Exception as e:
            self.logger.error(f"Error generating sector signals: {e}")
        
        return trade_signals
    
    async def _optimize_signal_portfolio(self, signals: List[TradeSignal]) -> List[TradeSignal]:
        """Optimize the portfolio of signals using mean-variance optimization.
        
        Args:
            signals: List of trade signals
            
        Returns:
            List of optimized trade signals
        """
        try:
            if len(signals) < 2:
                return signals
            
            # Create signal matrix
            symbols = [s.symbol for s in signals]
            expected_returns = np.array([s.expected_return / 100 for s in signals])  # Convert to decimal
            target_weights = np.array([s.target_weight for s in signals])
            
            # Get or estimate covariance matrix
            cov_matrix = await self._get_covariance_matrix(symbols)
            
            if cov_matrix is None:
                self.logger.warning("Could not obtain covariance matrix, using original signals")
                return signals
            
            # Portfolio optimization using cvxpy
            n_assets = len(signals)
            weights = cp.Variable(n_assets)
            
            # Objective: maximize expected return - risk penalty
            risk_aversion = 2.0
            objective = cp.Maximize(
                expected_returns.T @ weights - 
                0.5 * risk_aversion * cp.quad_form(weights, cov_matrix)
            )
            
            # Constraints
            constraints = [
                cp.sum(weights) <= self.max_net_exposure,  # Net exposure limit
                cp.sum(cp.abs(weights)) <= self.max_gross_exposure,  # Gross exposure limit
                weights >= -self.max_single_position,  # Individual short limit
                weights <= self.max_single_position,   # Individual long limit
            ]
            
            # Solve optimization
            problem = cp.Problem(objective, constraints)
            problem.solve()
            
            if problem.status == cp.OPTIMAL:
                optimal_weights = weights.value
                
                # Update signal weights
                optimized_signals = []
                for i, signal in enumerate(signals):
                    if abs(optimal_weights[i]) > 0.001:  # Only keep significant positions
                        optimized_signal = signal
                        optimized_signal.target_weight = optimal_weights[i]
                        optimized_signal.strength *= abs(optimal_weights[i]) / abs(signal.target_weight)
                        optimized_signals.append(optimized_signal)
                
                self.logger.info(f"Portfolio optimization successful: {len(optimized_signals)} signals optimized")
                return optimized_signals
            else:
                self.logger.warning(f"Portfolio optimization failed: {problem.status}")
                return signals
                
        except Exception as e:
            self.logger.error(f"Error in portfolio optimization: {e}")
            return signals
    
    async def _execute_signal(self, signal: TradeSignal, current_metrics: PortfolioMetrics) -> Dict[str, Any]:
        """Execute a single trade signal.
        
        Args:
            signal: Trade signal to execute
            current_metrics: Current portfolio metrics
            
        Returns:
            Dictionary with execution results
        """
        try:
            result = {
                'success': False,
                'action': None,
                'position': None,
                'turnover': 0,
                'error': None
            }
            
            # Check if we already have a position in this symbol
            existing_position = self.positions.get(signal.symbol)
            
            # Get current market data
            market_data = await self._get_market_data(signal.symbol)
            if not market_data:
                result['error'] = 'Could not obtain market data'
                return result
            
            current_price = market_data['price']
            
            # Risk checks
            risk_check = self._check_signal_risk(signal, current_metrics, current_price)
            if not risk_check['approved']:
                result['error'] = risk_check['reason']
                return result
            
            # Calculate position size
            position_value = self.portfolio_value * abs(signal.target_weight)
            shares = int(position_value / current_price)
            
            if shares == 0:
                result['error'] = 'Position size too small'
                return result
            
            # Execute the trade
            if existing_position:
                # Adjust existing position
                result['action'] = 'adjust_position'
                
                # Calculate new shares needed
                current_shares = existing_position.shares
                target_shares = shares if signal.signal_type in ['buy', 'long'] else -shares
                shares_to_trade = target_shares - current_shares
                
                if abs(shares_to_trade) > 0:
                    # Update position
                    existing_position.shares = target_shares
                    existing_position.current_weight = signal.target_weight
                    existing_position.current_price = current_price
                    existing_position.unrealized_pnl = (
                        (current_price - existing_position.entry_price) * existing_position.shares
                    )
                    existing_position.timestamp = datetime.utcnow()
                    
                    result['position'] = existing_position
                    result['turnover'] = abs(shares_to_trade * current_price)
                    result['success'] = True
            else:
                # Create new position
                result['action'] = 'new_position'
                
                position_type = 'long' if signal.signal_type in ['buy', 'long'] else 'short'
                final_shares = shares if position_type == 'long' else -shares
                
                new_position = Position(
                    symbol=signal.symbol,
                    company_name=signal.company_name,
                    sector=signal.sector,
                    position_type=position_type,
                    target_weight=signal.target_weight,
                    current_weight=signal.target_weight,
                    shares=final_shares,
                    entry_price=current_price,
                    current_price=current_price,
                    unrealized_pnl=0,
                    realized_pnl=0,
                    risk_score=self._calculate_position_risk_score(signal),
                    esg_score=market_data.get('esg_score', 50),
                    conviction_level=signal.confidence,
                    entry_date=datetime.utcnow(),
                    expected_holding_period=self._estimate_holding_period(signal.time_horizon),
                    stop_loss=signal.stop_loss_target,
                    take_profit=signal.take_profit_target,
                    rationale=signal.rationale,
                    timestamp=datetime.utcnow()
                )
                
                self.positions[signal.symbol] = new_position
                result['position'] = new_position
                result['turnover'] = abs(final_shares * current_price)
                result['success'] = True
            
            # Update cash balance
            if result['success']:
                self.cash_balance -= result['turnover']
                
                # Record trade
                trade_record = {
                    'timestamp': datetime.utcnow(),
                    'symbol': signal.symbol,
                    'action': result['action'],
                    'shares': shares,
                    'price': current_price,
                    'value': result['turnover'],
                    'signal_id': signal.signal_id
                }
                self.trade_history.append(trade_record)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error executing signal for {signal.symbol}: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _check_signal_risk(self, signal: TradeSignal, current_metrics: PortfolioMetrics, current_price: float) -> Dict[str, Any]:
        """Check if signal passes risk management rules.
        
        Args:
            signal: Trade signal to check
            current_metrics: Current portfolio metrics
            current_price: Current market price
            
        Returns:
            Dictionary with approval status and reason
        """
        try:
            # Position size check
            if abs(signal.target_weight) > self.max_single_position:
                return {'approved': False, 'reason': 'Position size exceeds limit'}
            
            # Gross exposure check
            new_gross_exposure = current_metrics.gross_exposure + abs(signal.target_weight)
            if new_gross_exposure > self.max_gross_exposure:
                return {'approved': False, 'reason': 'Would exceed gross exposure limit'}
            
            # Net exposure check
            new_net_exposure = current_metrics.net_exposure + signal.target_weight
            if abs(new_net_exposure) > self.max_net_exposure:
                return {'approved': False, 'reason': 'Would exceed net exposure limit'}
            
            # Liquidity check (simplified)
            if signal.risk_level == 'high' and abs(signal.target_weight) > 0.02:
                return {'approved': False, 'reason': 'High risk position too large'}
            
            # ESG check
            if signal.esg_considerations and any('Poor ESG' in consideration for consideration in signal.esg_considerations):
                if abs(signal.target_weight) > 0.01:  # Limit poor ESG positions
                    return {'approved': False, 'reason': 'Poor ESG score limits position size'}
            
            # Confidence check
            if signal.confidence < self.min_conviction_threshold:
                return {'approved': False, 'reason': 'Signal confidence below threshold'}
            
            return {'approved': True, 'reason': 'All risk checks passed'}
            
        except Exception as e:
            self.logger.error(f"Error in risk check: {e}")
            return {'approved': False, 'reason': f'Risk check error: {e}'}
    
    def _calculate_position_risk_score(self, signal: TradeSignal) -> float:
        """Calculate risk score for a position.
        
        Args:
            signal: Trade signal
            
        Returns:
            Risk score (0-10)
        """
        risk_score = 5.0  # Base risk
        
        # Adjust for signal type
        if signal.signal_type in ['short', 'sell']:
            risk_score += 2.0  # Shorts are riskier
        
        # Adjust for risk level
        risk_adjustments = {'low': -1.0, 'medium': 0.0, 'high': 2.0}
        risk_score += risk_adjustments.get(signal.risk_level, 0.0)
        
        # Adjust for position size
        risk_score += abs(signal.target_weight) * 20  # Larger positions are riskier
        
        # Adjust for confidence
        risk_score -= signal.confidence * 2  # Higher confidence reduces risk
        
        return max(0, min(10, risk_score))
    
    def _estimate_holding_period(self, time_horizon: str) -> int:
        """Estimate holding period in days based on time horizon.
        
        Args:
            time_horizon: Time horizon string
            
        Returns:
            Estimated holding period in days
        """
        horizons = {
            'short': 30,
            'medium': 90,
            'long': 180
        }
        return horizons.get(time_horizon, 60)
    
    async def _calculate_portfolio_metrics(self) -> PortfolioMetrics:
        """Calculate current portfolio performance metrics.
        
        Returns:
            PortfolioMetrics object
        """
        try:
            if not self.positions:
                return PortfolioMetrics(
                    total_return=0, annualized_return=0, volatility=0, sharpe_ratio=0,
                    max_drawdown=0, win_rate=0, profit_factor=0, long_exposure=0,
                    short_exposure=0, net_exposure=0, gross_exposure=0, beta=0,
                    alpha=0, information_ratio=0, var_95=0, expected_shortfall=0,
                    sector_allocation={}, top_positions=[], timestamp=datetime.utcnow()
                )
            
            # Update current prices and P&L
            await self._update_position_values()
            
            # Calculate exposures
            long_exposure = sum(pos.current_weight for pos in self.positions.values() if pos.current_weight > 0)
            short_exposure = sum(abs(pos.current_weight) for pos in self.positions.values() if pos.current_weight < 0)
            net_exposure = long_exposure - short_exposure
            gross_exposure = long_exposure + short_exposure
            
            # Calculate returns
            total_pnl = sum(pos.unrealized_pnl + pos.realized_pnl for pos in self.positions.values())
            total_return = total_pnl / self.portfolio_value if self.portfolio_value > 0 else 0
            
            # Sector allocation
            sector_allocation = defaultdict(float)
            for pos in self.positions.values():
                sector_allocation[pos.sector] += abs(pos.current_weight)
            
            # Top positions
            top_positions = sorted(
                [{
                    'symbol': pos.symbol,
                    'weight': pos.current_weight,
                    'pnl': pos.unrealized_pnl + pos.realized_pnl,
                    'sector': pos.sector
                } for pos in self.positions.values()],
                key=lambda x: abs(x['weight']),
                reverse=True
            )[:10]
            
            # Simplified metrics (in practice, these would be calculated from historical data)
            metrics = PortfolioMetrics(
                total_return=total_return,
                annualized_return=total_return * 252 / 30,  # Rough annualization
                volatility=0.15,  # Placeholder
                sharpe_ratio=total_return / 0.15 if total_return != 0 else 0,
                max_drawdown=0.05,  # Placeholder
                win_rate=0.6,  # Placeholder
                profit_factor=1.5,  # Placeholder
                long_exposure=long_exposure,
                short_exposure=short_exposure,
                net_exposure=net_exposure,
                gross_exposure=gross_exposure,
                beta=1.0,  # Placeholder
                alpha=total_return - 0.05,  # Excess return over risk-free rate
                information_ratio=0.5,  # Placeholder
                var_95=self.portfolio_value * 0.02,  # 2% VaR
                expected_shortfall=self.portfolio_value * 0.03,  # 3% ES
                sector_allocation=dict(sector_allocation),
                top_positions=top_positions,
                timestamp=datetime.utcnow()
            )
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio metrics: {e}")
            raise
    
    async def _update_position_values(self) -> None:
        """Update current values for all positions."""
        try:
            symbols = list(self.positions.keys())
            if not symbols:
                return
            
            # Get current prices
            price_data = await self._get_batch_market_data(symbols)
            
            # Update positions
            for symbol, position in self.positions.items():
                if symbol in price_data:
                    current_price = price_data[symbol]['price']
                    position.current_price = current_price
                    position.unrealized_pnl = (
                        (current_price - position.entry_price) * position.shares
                    )
                    position.current_weight = (
                        position.shares * current_price / self.portfolio_value
                    )
            
        except Exception as e:
            self.logger.error(f"Error updating position values: {e}")
    
    async def _perform_risk_check(self) -> Dict[str, Any]:
        """Perform comprehensive risk check on current portfolio.
        
        Returns:
            Dictionary with risk check results
        """
        try:
            risk_check = {
                'status': 'PASS',
                'warnings': [],
                'violations': [],
                'recommendations': []
            }
            
            # Calculate current metrics
            metrics = await self._calculate_portfolio_metrics()
            
            # Check exposure limits
            if metrics.gross_exposure > self.max_gross_exposure:
                risk_check['violations'].append(f"Gross exposure ({metrics.gross_exposure:.1%}) exceeds limit ({self.max_gross_exposure:.1%})")
                risk_check['status'] = 'VIOLATION'
            
            if abs(metrics.net_exposure) > self.max_net_exposure:
                risk_check['violations'].append(f"Net exposure ({metrics.net_exposure:.1%}) exceeds limit ({self.max_net_exposure:.1%})")
                risk_check['status'] = 'VIOLATION'
            
            # Check individual position sizes
            for pos in self.positions.values():
                if abs(pos.current_weight) > self.max_single_position:
                    risk_check['violations'].append(f"{pos.symbol} position ({pos.current_weight:.1%}) exceeds single position limit")
                    risk_check['status'] = 'VIOLATION'
            
            # Check sector concentration
            for sector, allocation in metrics.sector_allocation.items():
                if allocation > 0.3:  # 30% sector limit
                    risk_check['warnings'].append(f"High concentration in {sector}: {allocation:.1%}")
            
            # Check portfolio volatility
            if metrics.volatility > self.risk_budget:
                risk_check['warnings'].append(f"Portfolio volatility ({metrics.volatility:.1%}) above target ({self.risk_budget:.1%})")
            
            # Generate recommendations
            if risk_check['violations']:
                risk_check['recommendations'].append("Reduce position sizes to comply with risk limits")
            
            if risk_check['warnings']:
                risk_check['recommendations'].append("Consider rebalancing to reduce concentration risk")
            
            return risk_check
            
        except Exception as e:
            self.logger.error(f"Error in risk check: {e}")
            return {
                'status': 'ERROR',
                'error': str(e)
            }
    
    async def _get_company_data(self, symbol: str) -> Dict[str, Any]:
        """Get company fundamental and market data.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary with company data
        """
        try:
            # This would typically fetch from financial data APIs
            # For now, return synthetic data
            
            company_data = {
                'current_price': np.random.lognormal(4, 0.5),  # ~$50-150
                'market_cap': np.random.lognormal(9, 1),  # Billions
                'beta': np.random.normal(1, 0.3),
                'pe_ratio': np.random.lognormal(3, 0.3),
                'debt_to_equity': np.random.beta(2, 5),
                'roa': np.random.normal(0.05, 0.03),
                'roe': np.random.normal(0.12, 0.05),
                'esg_score': np.random.normal(60, 20),
                'liquidity_score': np.random.beta(8, 2),
                'data_quality': np.random.beta(9, 2)
            }
            
            # Ensure reasonable bounds
            company_data['esg_score'] = max(0, min(100, company_data['esg_score']))
            company_data['current_price'] = max(1, company_data['current_price'])
            
            return company_data
            
        except Exception as e:
            self.logger.error(f"Error getting company data for {symbol}: {e}")
            return {}
    
    async def _get_sector_leaders(self, sector: str) -> List[Dict[str, Any]]:
        """Get leading stocks in a sector.
        
        Args:
            sector: Sector name
            
        Returns:
            List of stock information
        """
        # This would typically query a financial database
        # For now, return synthetic sector leaders
        
        sector_leaders = {
            'Energy': [
                {'symbol': 'XOM', 'name': 'Exxon Mobil', 'price': 110},
                {'symbol': 'CVX', 'name': 'Chevron', 'price': 160},
                {'symbol': 'COP', 'name': 'ConocoPhillips', 'price': 120}
            ],
            'Materials': [
                {'symbol': 'LIN', 'name': 'Linde', 'price': 400},
                {'symbol': 'SHW', 'name': 'Sherwin-Williams', 'price': 250},
                {'symbol': 'APD', 'name': 'Air Products', 'price': 280}
            ],
            'Industrials': [
                {'symbol': 'BA', 'name': 'Boeing', 'price': 200},
                {'symbol': 'CAT', 'name': 'Caterpillar', 'price': 250},
                {'symbol': 'GE', 'name': 'General Electric', 'price': 100}
            ]
        }
        
        return sector_leaders.get(sector, [
            {'symbol': 'SPY', 'name': 'SPDR S&P 500', 'price': 450}
        ])
    
    async def _get_sector_laggards(self, sector: str) -> List[Dict[str, Any]]:
        """Get lagging stocks in a sector.
        
        Args:
            sector: Sector name
            
        Returns:
            List of stock information
        """
        # This would typically query for underperforming stocks
        # For now, return synthetic laggards
        
        leaders = await self._get_sector_leaders(sector)
        
        # Create laggards with lower prices (simplified)
        laggards = []
        for leader in leaders:
            laggard = leader.copy()
            laggard['symbol'] = laggard['symbol'] + '_LAG'  # Synthetic symbol
            laggard['price'] *= 0.8  # Lower price
            laggards.append(laggard)
        
        return laggards[:2]  # Return top 2 laggards
    
    async def _get_covariance_matrix(self, symbols: List[str]) -> Optional[np.ndarray]:
        """Get or estimate covariance matrix for symbols.
        
        Args:
            symbols: List of stock symbols
            
        Returns:
            Covariance matrix or None
        """
        try:
            # This would typically calculate from historical returns
            # For now, generate a synthetic covariance matrix
            
            n = len(symbols)
            if n < 2:
                return None
            
            # Generate random correlation matrix
            correlations = np.random.uniform(0.1, 0.7, (n, n))
            correlations = (correlations + correlations.T) / 2  # Make symmetric
            np.fill_diagonal(correlations, 1.0)
            
            # Generate volatilities
            volatilities = np.random.uniform(0.15, 0.4, n)
            
            # Convert to covariance matrix
            cov_matrix = np.outer(volatilities, volatilities) * correlations
            
            return cov_matrix
            
        except Exception as e:
            self.logger.error(f"Error getting covariance matrix: {e}")
            return None
    
    async def _update_market_data(self, symbols: List[str]) -> None:
        """Update market data cache for symbols.
        
        Args:
            symbols: List of symbols to update
        """
        try:
            for symbol in symbols:
                if symbol not in self.price_data or self._is_data_stale(symbol):
                    market_data = await self._get_market_data(symbol)
                    if market_data:
                        self.price_data[symbol] = market_data
                        
        except Exception as e:
            self.logger.error(f"Error updating market data: {e}")
    
    async def _get_market_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get current market data for a symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary with market data or None
        """
        try:
            # This would typically fetch from market data APIs
            # For now, return synthetic data
            
            market_data = {
                'symbol': symbol,
                'price': np.random.lognormal(4, 0.3),
                'volume': np.random.lognormal(12, 1),
                'bid': 0,
                'ask': 0,
                'timestamp': datetime.utcnow()
            }
            
            market_data['bid'] = market_data['price'] * 0.999
            market_data['ask'] = market_data['price'] * 1.001
            
            return market_data
            
        except Exception as e:
            self.logger.error(f"Error getting market data for {symbol}: {e}")
            return None
    
    async def _get_batch_market_data(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """Get market data for multiple symbols.
        
        Args:
            symbols: List of stock symbols
            
        Returns:
            Dictionary mapping symbols to market data
        """
        batch_data = {}
        
        for symbol in symbols:
            data = await self._get_market_data(symbol)
            if data:
                batch_data[symbol] = data
        
        return batch_data
    
    def _is_data_stale(self, symbol: str) -> bool:
        """Check if cached data is stale.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            True if data is stale
        """
        if symbol not in self.price_data:
            return True
        
        data_age = datetime.utcnow() - self.price_data[symbol]['timestamp']
        return data_age > timedelta(minutes=5)  # 5-minute cache
    
    async def generate_strategy_report(self) -> Dict[str, Any]:
        """Generate comprehensive strategy performance report.
        
        Returns:
            Dictionary containing the strategy report
        """
        try:
            metrics = await self._calculate_portfolio_metrics()
            risk_check = await self._perform_risk_check()
            
            # Position analysis
            winning_positions = [pos for pos in self.positions.values() if pos.unrealized_pnl > 0]
            losing_positions = [pos for pos in self.positions.values() if pos.unrealized_pnl < 0]
            
            report = {
                'strategy_summary': {
                    'strategy_name': 'Supply Chain Disruption Alpha',
                    'report_date': datetime.utcnow(),
                    'portfolio_value': self.portfolio_value,
                    'cash_balance': self.cash_balance,
                    'total_positions': len(self.positions),
                    'long_positions': len([p for p in self.positions.values() if p.position_type == 'long']),
                    'short_positions': len([p for p in self.positions.values() if p.position_type == 'short'])
                },
                'performance_metrics': {
                    'total_return': metrics.total_return,
                    'annualized_return': metrics.annualized_return,
                    'volatility': metrics.volatility,
                    'sharpe_ratio': metrics.sharpe_ratio,
                    'max_drawdown': metrics.max_drawdown,
                    'win_rate': len(winning_positions) / len(self.positions) if self.positions else 0,
                    'average_win': np.mean([p.unrealized_pnl for p in winning_positions]) if winning_positions else 0,
                    'average_loss': np.mean([p.unrealized_pnl for p in losing_positions]) if losing_positions else 0
                },
                'risk_metrics': {
                    'gross_exposure': metrics.gross_exposure,
                    'net_exposure': metrics.net_exposure,
                    'long_exposure': metrics.long_exposure,
                    'short_exposure': metrics.short_exposure,
                    'var_95': metrics.var_95,
                    'expected_shortfall': metrics.expected_shortfall,
                    'beta': metrics.beta
                },
                'sector_allocation': metrics.sector_allocation,
                'top_positions': metrics.top_positions,
                'risk_check': risk_check,
                'trade_statistics': {
                    'total_trades': len(self.trade_history),
                    'trades_today': len([t for t in self.trade_history if t['timestamp'].date() == datetime.utcnow().date()]),
                    'total_turnover': sum(t['value'] for t in self.trade_history),
                    'average_trade_size': np.mean([t['value'] for t in self.trade_history]) if self.trade_history else 0
                }
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating strategy report: {e}")
            raise