#!/usr/bin/env python3
"""
Sector Impact Analyzer
Analyzes the impact of supply chain disruptions on different industry sectors.

Features:
- Sector vulnerability assessment
- Cross-sector impact propagation
- Industry-specific risk metrics
- Sector rotation recommendations
- Supply chain dependency mapping

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
import networkx as nx

# Analysis libraries
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import pearsonr
import seaborn as sns
import matplotlib.pyplot as plt

from utils.database import DatabaseManager
from models.disruption_predictor import DisruptionPrediction


@dataclass
class SectorImpactScore:
    """Data structure for sector impact assessment."""
    sector_name: str
    sector_code: str  # GICS sector code
    vulnerability_score: float  # 0-10
    current_impact: float  # -10 to +10
    predicted_impact_7d: float
    predicted_impact_30d: float
    supply_chain_dependency: float  # 0-1
    disruption_sensitivity: Dict[str, float]  # by disruption type
    key_risk_factors: List[str]
    affected_companies_count: int
    market_cap_at_risk: float  # USD billions
    recovery_time_estimate: int  # days
    mitigation_strategies: List[str]
    timestamp: datetime


@dataclass
class CrossSectorImpact:
    """Data structure for cross-sector impact analysis."""
    source_sector: str
    target_sector: str
    impact_strength: float  # 0-1
    propagation_delay: int  # days
    impact_type: str  # 'supply', 'demand', 'financial'
    confidence_level: float  # 0-1
    historical_correlation: float
    timestamp: datetime


@dataclass
class SectorRotationSignal:
    """Data structure for sector rotation recommendations."""
    signal_id: str
    signal_type: str  # 'overweight', 'underweight', 'neutral'
    source_sector: str
    target_sector: str
    strength: float  # 0-1
    time_horizon: str  # 'short', 'medium', 'long'
    rationale: str
    expected_return: float  # percentage
    risk_level: str  # 'low', 'medium', 'high'
    confidence: float  # 0-1
    timestamp: datetime


class SectorImpactAnalyzer:
    """Analyzer for assessing supply chain disruption impact on industry sectors."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the sector impact analyzer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Database manager
        self.db_manager = DatabaseManager()
        
        # Sector definitions (GICS sectors)
        self.sectors = {
            '10': 'Energy',
            '15': 'Materials', 
            '20': 'Industrials',
            '25': 'Consumer Discretionary',
            '30': 'Consumer Staples',
            '35': 'Health Care',
            '40': 'Financials',
            '45': 'Information Technology',
            '50': 'Communication Services',
            '55': 'Utilities',
            '60': 'Real Estate'
        }
        
        # Supply chain dependency matrix (sector to sector)
        self.dependency_matrix = self._initialize_dependency_matrix()
        
        # Disruption sensitivity by sector
        self.disruption_sensitivity = self._initialize_disruption_sensitivity()
        
        # Historical sector performance data
        self.sector_performance = {}
        
        # Cross-sector impact network
        self.impact_network = nx.DiGraph()
        self._build_impact_network()
    
    def _initialize_dependency_matrix(self) -> Dict[str, Dict[str, float]]:
        """Initialize supply chain dependency matrix between sectors.
        
        Returns:
            Dictionary mapping sector dependencies
        """
        # Simplified dependency matrix (in practice, this would be data-driven)
        dependencies = {
            'Energy': {
                'Materials': 0.8, 'Industrials': 0.7, 'Consumer Discretionary': 0.6,
                'Consumer Staples': 0.5, 'Utilities': 0.9
            },
            'Materials': {
                'Industrials': 0.9, 'Consumer Discretionary': 0.7, 'Information Technology': 0.6,
                'Real Estate': 0.8, 'Energy': 0.4
            },
            'Industrials': {
                'Consumer Discretionary': 0.8, 'Information Technology': 0.7, 'Energy': 0.6,
                'Materials': 0.5, 'Real Estate': 0.6
            },
            'Consumer Discretionary': {
                'Consumer Staples': 0.4, 'Information Technology': 0.6, 'Materials': 0.5,
                'Industrials': 0.3
            },
            'Consumer Staples': {
                'Materials': 0.6, 'Energy': 0.5, 'Industrials': 0.4
            },
            'Health Care': {
                'Materials': 0.5, 'Information Technology': 0.6, 'Industrials': 0.3
            },
            'Financials': {
                'Real Estate': 0.7, 'Energy': 0.4, 'Information Technology': 0.5
            },
            'Information Technology': {
                'Materials': 0.7, 'Industrials': 0.6, 'Energy': 0.4
            },
            'Communication Services': {
                'Information Technology': 0.8, 'Industrials': 0.4, 'Energy': 0.3
            },
            'Utilities': {
                'Energy': 0.9, 'Materials': 0.5, 'Industrials': 0.6
            },
            'Real Estate': {
                'Materials': 0.8, 'Industrials': 0.7, 'Financials': 0.6, 'Energy': 0.5
            }
        }
        
        return dependencies
    
    def _initialize_disruption_sensitivity(self) -> Dict[str, Dict[str, float]]:
        """Initialize disruption sensitivity by sector and disruption type.
        
        Returns:
            Dictionary mapping sector sensitivity to disruption types
        """
        sensitivity = {
            'Energy': {
                'port_congestion': 0.9, 'route_blockage': 0.8, 'rate_spike': 0.7,
                'weather': 0.8, 'geopolitical': 0.9
            },
            'Materials': {
                'port_congestion': 0.9, 'route_blockage': 0.9, 'rate_spike': 0.8,
                'weather': 0.6, 'geopolitical': 0.7
            },
            'Industrials': {
                'port_congestion': 0.8, 'route_blockage': 0.8, 'rate_spike': 0.7,
                'weather': 0.5, 'geopolitical': 0.6
            },
            'Consumer Discretionary': {
                'port_congestion': 0.7, 'route_blockage': 0.7, 'rate_spike': 0.8,
                'weather': 0.4, 'geopolitical': 0.5
            },
            'Consumer Staples': {
                'port_congestion': 0.6, 'route_blockage': 0.6, 'rate_spike': 0.6,
                'weather': 0.7, 'geopolitical': 0.4
            },
            'Health Care': {
                'port_congestion': 0.5, 'route_blockage': 0.5, 'rate_spike': 0.4,
                'weather': 0.3, 'geopolitical': 0.6
            },
            'Financials': {
                'port_congestion': 0.3, 'route_blockage': 0.3, 'rate_spike': 0.4,
                'weather': 0.2, 'geopolitical': 0.7
            },
            'Information Technology': {
                'port_congestion': 0.6, 'route_blockage': 0.6, 'rate_spike': 0.5,
                'weather': 0.3, 'geopolitical': 0.8
            },
            'Communication Services': {
                'port_congestion': 0.4, 'route_blockage': 0.4, 'rate_spike': 0.3,
                'weather': 0.3, 'geopolitical': 0.6
            },
            'Utilities': {
                'port_congestion': 0.5, 'route_blockage': 0.5, 'rate_spike': 0.6,
                'weather': 0.8, 'geopolitical': 0.5
            },
            'Real Estate': {
                'port_congestion': 0.4, 'route_blockage': 0.4, 'rate_spike': 0.5,
                'weather': 0.6, 'geopolitical': 0.3
            }
        }
        
        return sensitivity
    
    def _build_impact_network(self) -> None:
        """Build network graph of cross-sector impacts."""
        try:
            # Add nodes (sectors)
            for sector_code, sector_name in self.sectors.items():
                self.impact_network.add_node(sector_name, code=sector_code)
            
            # Add edges (dependencies)
            for source_sector, dependencies in self.dependency_matrix.items():
                for target_sector, strength in dependencies.items():
                    if strength > 0.3:  # Only significant dependencies
                        self.impact_network.add_edge(
                            source_sector, target_sector, 
                            weight=strength, 
                            impact_type='supply_chain'
                        )
            
            self.logger.info(f"Built impact network with {self.impact_network.number_of_nodes()} nodes and {self.impact_network.number_of_edges()} edges")
            
        except Exception as e:
            self.logger.error(f"Error building impact network: {e}")
    
    async def analyze_sector_impacts(self, disruptions: List[DisruptionPrediction]) -> List[SectorImpactScore]:
        """Analyze the impact of disruptions on all sectors.
        
        Args:
            disruptions: List of predicted disruptions
            
        Returns:
            List of SectorImpactScore objects
        """
        self.logger.info(f"Analyzing sector impacts for {len(disruptions)} disruptions")
        
        try:
            sector_impacts = []
            
            # Get current sector data
            sector_data = await self._get_sector_data()
            
            for sector_name in self.sectors.values():
                try:
                    impact_score = await self._calculate_sector_impact(
                        sector_name, disruptions, sector_data
                    )
                    if impact_score:
                        sector_impacts.append(impact_score)
                        
                except Exception as e:
                    self.logger.error(f"Error analyzing impact for {sector_name}: {e}")
            
            # Sort by vulnerability score (highest first)
            sector_impacts.sort(key=lambda x: x.vulnerability_score, reverse=True)
            
            self.logger.info(f"Completed sector impact analysis for {len(sector_impacts)} sectors")
            return sector_impacts
            
        except Exception as e:
            self.logger.error(f"Error in sector impact analysis: {e}")
            raise
    
    async def analyze_cross_sector_impacts(self, disruptions: List[DisruptionPrediction]) -> List[CrossSectorImpact]:
        """Analyze cross-sector impact propagation.
        
        Args:
            disruptions: List of predicted disruptions
            
        Returns:
            List of CrossSectorImpact objects
        """
        self.logger.info("Analyzing cross-sector impact propagation")
        
        try:
            cross_impacts = []
            
            # Calculate direct impacts first
            sector_impacts = await self.analyze_sector_impacts(disruptions)
            impact_by_sector = {si.sector_name: si.current_impact for si in sector_impacts}
            
            # Analyze propagation through dependency network
            for source_sector in self.sectors.values():
                source_impact = impact_by_sector.get(source_sector, 0)
                
                if abs(source_impact) > 1:  # Only significant impacts
                    # Find dependent sectors
                    dependencies = self.dependency_matrix.get(source_sector, {})
                    
                    for target_sector, dependency_strength in dependencies.items():
                        if dependency_strength > 0.3:
                            # Calculate propagated impact
                            propagated_impact = source_impact * dependency_strength * 0.7
                            
                            if abs(propagated_impact) > 0.5:
                                # Calculate propagation delay based on sector characteristics
                                delay_days = self._calculate_propagation_delay(
                                    source_sector, target_sector, dependency_strength
                                )
                                
                                # Get historical correlation
                                correlation = await self._get_sector_correlation(
                                    source_sector, target_sector
                                )
                                
                                cross_impact = CrossSectorImpact(
                                    source_sector=source_sector,
                                    target_sector=target_sector,
                                    impact_strength=abs(propagated_impact) / 10,  # Normalize to 0-1
                                    propagation_delay=delay_days,
                                    impact_type='supply',
                                    confidence_level=dependency_strength * 0.8,
                                    historical_correlation=correlation,
                                    timestamp=datetime.utcnow()
                                )
                                cross_impacts.append(cross_impact)
            
            # Sort by impact strength
            cross_impacts.sort(key=lambda x: x.impact_strength, reverse=True)
            
            self.logger.info(f"Identified {len(cross_impacts)} significant cross-sector impacts")
            return cross_impacts
            
        except Exception as e:
            self.logger.error(f"Error in cross-sector impact analysis: {e}")
            raise
    
    async def generate_sector_rotation_signals(self, 
                                             sector_impacts: List[SectorImpactScore],
                                             cross_impacts: List[CrossSectorImpact]) -> List[SectorRotationSignal]:
        """Generate sector rotation trading signals.
        
        Args:
            sector_impacts: List of sector impact scores
            cross_impacts: List of cross-sector impacts
            
        Returns:
            List of SectorRotationSignal objects
        """
        self.logger.info("Generating sector rotation signals")
        
        try:
            rotation_signals = []
            
            # Create impact lookup
            impact_lookup = {si.sector_name: si for si in sector_impacts}
            
            # Generate underweight signals for highly impacted sectors
            for sector_impact in sector_impacts:
                if sector_impact.predicted_impact_30d < -3:  # Significant negative impact
                    signal = SectorRotationSignal(
                        signal_id=f"underweight_{sector_impact.sector_name}_{datetime.utcnow().strftime('%Y%m%d')}",
                        signal_type='underweight',
                        source_sector=sector_impact.sector_name,
                        target_sector='',  # To be filled with alternative
                        strength=min(1.0, abs(sector_impact.predicted_impact_30d) / 10),
                        time_horizon='medium',
                        rationale=f"High vulnerability to supply chain disruptions (score: {sector_impact.vulnerability_score:.1f})",
                        expected_return=sector_impact.predicted_impact_30d,
                        risk_level='high' if sector_impact.vulnerability_score > 7 else 'medium',
                        confidence=0.7,
                        timestamp=datetime.utcnow()
                    )
                    rotation_signals.append(signal)
            
            # Generate overweight signals for resilient sectors
            resilient_sectors = [
                si for si in sector_impacts 
                if si.vulnerability_score < 4 and si.predicted_impact_30d > -1
            ]
            
            for sector_impact in resilient_sectors:
                # Check if this sector benefits from others' disruptions
                beneficiary_score = 0
                
                for cross_impact in cross_impacts:
                    if (cross_impact.target_sector == sector_impact.sector_name and 
                        cross_impact.impact_strength > 0.3):
                        source_impact = impact_lookup.get(cross_impact.source_sector)
                        if source_impact and source_impact.predicted_impact_30d < -2:
                            beneficiary_score += cross_impact.impact_strength
                
                if beneficiary_score > 0.3 or sector_impact.vulnerability_score < 3:
                    signal = SectorRotationSignal(
                        signal_id=f"overweight_{sector_impact.sector_name}_{datetime.utcnow().strftime('%Y%m%d')}",
                        signal_type='overweight',
                        source_sector='',
                        target_sector=sector_impact.sector_name,
                        strength=min(1.0, (beneficiary_score + (5 - sector_impact.vulnerability_score) / 5) / 2),
                        time_horizon='medium',
                        rationale=f"Low supply chain vulnerability and potential beneficiary of disruptions",
                        expected_return=max(2, 5 - sector_impact.vulnerability_score),
                        risk_level='low' if sector_impact.vulnerability_score < 3 else 'medium',
                        confidence=0.6 + beneficiary_score * 0.2,
                        timestamp=datetime.utcnow()
                    )
                    rotation_signals.append(signal)
            
            # Generate pair trade signals
            pair_signals = self._generate_pair_trade_signals(sector_impacts)
            rotation_signals.extend(pair_signals)
            
            # Sort by strength and confidence
            rotation_signals.sort(key=lambda x: x.strength * x.confidence, reverse=True)
            
            self.logger.info(f"Generated {len(rotation_signals)} sector rotation signals")
            return rotation_signals
            
        except Exception as e:
            self.logger.error(f"Error generating rotation signals: {e}")
            raise
    
    def _generate_pair_trade_signals(self, sector_impacts: List[SectorImpactScore]) -> List[SectorRotationSignal]:
        """Generate pair trade signals between sectors.
        
        Args:
            sector_impacts: List of sector impact scores
            
        Returns:
            List of pair trade signals
        """
        pair_signals = []
        
        try:
            # Sort sectors by predicted impact
            sorted_impacts = sorted(sector_impacts, key=lambda x: x.predicted_impact_30d)
            
            # Create pairs: short worst performers, long best performers
            worst_sectors = sorted_impacts[:3]  # Bottom 3
            best_sectors = sorted_impacts[-3:]  # Top 3
            
            for worst in worst_sectors:
                for best in best_sectors:
                    if worst.sector_name != best.sector_name:
                        impact_spread = best.predicted_impact_30d - worst.predicted_impact_30d
                        
                        if impact_spread > 3:  # Significant spread
                            signal = SectorRotationSignal(
                                signal_id=f"pair_{worst.sector_name}_{best.sector_name}_{datetime.utcnow().strftime('%Y%m%d')}",
                                signal_type='pair_trade',
                                source_sector=worst.sector_name,  # Short
                                target_sector=best.sector_name,   # Long
                                strength=min(1.0, impact_spread / 10),
                                time_horizon='medium',
                                rationale=f"Pair trade: Short {worst.sector_name} (impact: {worst.predicted_impact_30d:.1f}), Long {best.sector_name} (impact: {best.predicted_impact_30d:.1f})",
                                expected_return=impact_spread,
                                risk_level='medium',
                                confidence=0.6,
                                timestamp=datetime.utcnow()
                            )
                            pair_signals.append(signal)
        
        except Exception as e:
            self.logger.error(f"Error generating pair trade signals: {e}")
        
        return pair_signals
    
    async def _get_sector_data(self) -> Dict[str, Any]:
        """Get current sector performance and characteristics data.
        
        Returns:
            Dictionary with sector data
        """
        try:
            # This would typically fetch from financial data APIs
            # For now, generate synthetic sector data
            
            sector_data = {}
            
            for sector_name in self.sectors.values():
                sector_data[sector_name] = {
                    'market_cap': np.random.lognormal(12, 0.5),  # Billions
                    'pe_ratio': np.random.normal(20, 5),
                    'dividend_yield': np.random.beta(2, 8) * 0.05,
                    'beta': np.random.normal(1, 0.3),
                    'rsi': np.random.uniform(30, 70),
                    'price_momentum_1m': np.random.normal(0, 0.05),
                    'price_momentum_3m': np.random.normal(0, 0.1),
                    'earnings_growth': np.random.normal(0.05, 0.1),
                    'revenue_growth': np.random.normal(0.03, 0.08)
                }
            
            return sector_data
            
        except Exception as e:
            self.logger.error(f"Error getting sector data: {e}")
            return {}
    
    async def _calculate_sector_impact(self, 
                                     sector_name: str, 
                                     disruptions: List[DisruptionPrediction],
                                     sector_data: Dict[str, Any]) -> Optional[SectorImpactScore]:
        """Calculate impact score for a specific sector.
        
        Args:
            sector_name: Name of the sector
            disruptions: List of predicted disruptions
            sector_data: Sector characteristics data
            
        Returns:
            SectorImpactScore object or None
        """
        try:
            # Get sector sensitivity
            sensitivity = self.disruption_sensitivity.get(sector_name, {})
            
            # Calculate current and predicted impacts
            current_impact = 0
            impact_7d = 0
            impact_30d = 0
            
            disruption_impacts = defaultdict(float)
            risk_factors = []
            
            for disruption in disruptions:
                # Get sensitivity to this disruption type
                disruption_sensitivity = sensitivity.get(disruption.disruption_type, 0.3)
                
                # Calculate impact magnitude
                base_impact = (
                    disruption.probability * 
                    disruption.severity_score * 
                    disruption_sensitivity
                )
                
                # Time decay for future impacts
                days_to_start = (disruption.predicted_start_date - datetime.utcnow()).days
                
                if days_to_start <= 0:  # Current disruption
                    current_impact += base_impact
                elif days_to_start <= 7:
                    impact_7d += base_impact * 0.8  # Some uncertainty
                elif days_to_start <= 30:
                    impact_30d += base_impact * 0.6  # More uncertainty
                
                # Track disruption type impacts
                disruption_impacts[disruption.disruption_type] += base_impact
                
                # Add to risk factors if significant
                if base_impact > 2:
                    risk_factors.append(f"{disruption.disruption_type} at {disruption.location}")
            
            # Calculate vulnerability score (0-10)
            avg_sensitivity = np.mean(list(sensitivity.values())) if sensitivity else 0.5
            supply_chain_dependency = self._calculate_supply_chain_dependency(sector_name)
            
            vulnerability_score = (
                avg_sensitivity * 4 +  # Base sensitivity (0-4)
                supply_chain_dependency * 3 +  # Dependency factor (0-3)
                min(3, len(disruption_impacts))  # Exposure to multiple disruption types (0-3)
            )
            
            # Normalize impacts to -10 to +10 scale
            current_impact = max(-10, min(10, current_impact - 5))
            impact_7d = max(-10, min(10, impact_7d - 3))
            impact_30d = max(-10, min(10, impact_30d - 2))
            
            # Get sector characteristics
            sector_info = sector_data.get(sector_name, {})
            market_cap_at_risk = sector_info.get('market_cap', 100) * abs(impact_30d) / 100
            
            # Estimate recovery time
            recovery_time = self._estimate_recovery_time(sector_name, vulnerability_score)
            
            # Generate mitigation strategies
            mitigation_strategies = self._generate_mitigation_strategies(sector_name, risk_factors)
            
            # Count affected companies (estimated)
            affected_companies = max(10, int(vulnerability_score * 20))
            
            return SectorImpactScore(
                sector_name=sector_name,
                sector_code=self._get_sector_code(sector_name),
                vulnerability_score=vulnerability_score,
                current_impact=current_impact,
                predicted_impact_7d=impact_7d,
                predicted_impact_30d=impact_30d,
                supply_chain_dependency=supply_chain_dependency,
                disruption_sensitivity=dict(disruption_impacts),
                key_risk_factors=risk_factors[:5],  # Top 5 risks
                affected_companies_count=affected_companies,
                market_cap_at_risk=market_cap_at_risk,
                recovery_time_estimate=recovery_time,
                mitigation_strategies=mitigation_strategies,
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating sector impact for {sector_name}: {e}")
            return None
    
    def _calculate_supply_chain_dependency(self, sector_name: str) -> float:
        """Calculate supply chain dependency score for a sector.
        
        Args:
            sector_name: Name of the sector
            
        Returns:
            Dependency score (0-1)
        """
        try:
            # Count incoming dependencies
            incoming_deps = 0
            total_strength = 0
            
            for source_sector, dependencies in self.dependency_matrix.items():
                if sector_name in dependencies:
                    incoming_deps += 1
                    total_strength += dependencies[sector_name]
            
            # Count outgoing dependencies
            outgoing_deps = len(self.dependency_matrix.get(sector_name, {}))
            
            # Calculate overall dependency
            if incoming_deps + outgoing_deps == 0:
                return 0.3  # Default low dependency
            
            dependency_score = (
                (total_strength / max(1, incoming_deps)) * 0.6 +  # Weighted incoming
                (outgoing_deps / len(self.sectors)) * 0.4  # Outgoing ratio
            )
            
            return min(1.0, dependency_score)
            
        except Exception as e:
            self.logger.error(f"Error calculating dependency for {sector_name}: {e}")
            return 0.5
    
    def _calculate_propagation_delay(self, source_sector: str, target_sector: str, 
                                   dependency_strength: float) -> int:
        """Calculate propagation delay between sectors.
        
        Args:
            source_sector: Source sector name
            target_sector: Target sector name
            dependency_strength: Strength of dependency
            
        Returns:
            Delay in days
        """
        # Base delays by sector type
        sector_delays = {
            'Energy': 1,  # Fast propagation
            'Materials': 2,
            'Industrials': 3,
            'Consumer Discretionary': 5,
            'Consumer Staples': 4,
            'Health Care': 7,  # Slower propagation
            'Financials': 1,
            'Information Technology': 2,
            'Communication Services': 3,
            'Utilities': 2,
            'Real Estate': 10
        }
        
        base_delay = sector_delays.get(target_sector, 5)
        
        # Adjust based on dependency strength
        adjusted_delay = base_delay * (2 - dependency_strength)  # Stronger dependency = faster propagation
        
        return max(1, int(adjusted_delay))
    
    async def _get_sector_correlation(self, sector1: str, sector2: str) -> float:
        """Get historical correlation between two sectors.
        
        Args:
            sector1: First sector name
            sector2: Second sector name
            
        Returns:
            Correlation coefficient (-1 to 1)
        """
        try:
            # This would typically calculate from historical price data
            # For now, return estimated correlation based on sector relationships
            
            # High correlation pairs
            high_corr_pairs = [
                ('Energy', 'Materials'),
                ('Industrials', 'Materials'),
                ('Consumer Discretionary', 'Consumer Staples'),
                ('Information Technology', 'Communication Services'),
                ('Financials', 'Real Estate')
            ]
            
            # Check if sectors are highly correlated
            for pair in high_corr_pairs:
                if (sector1 in pair and sector2 in pair):
                    return np.random.uniform(0.6, 0.8)
            
            # Check dependency relationship
            dependencies = self.dependency_matrix.get(sector1, {})
            if sector2 in dependencies:
                return dependencies[sector2] * 0.7  # Convert dependency to correlation
            
            dependencies = self.dependency_matrix.get(sector2, {})
            if sector1 in dependencies:
                return dependencies[sector1] * 0.7
            
            # Default low correlation
            return np.random.uniform(0.1, 0.4)
            
        except Exception as e:
            self.logger.error(f"Error getting correlation for {sector1}-{sector2}: {e}")
            return 0.3
    
    def _estimate_recovery_time(self, sector_name: str, vulnerability_score: float) -> int:
        """Estimate recovery time for a sector after disruption.
        
        Args:
            sector_name: Name of the sector
            vulnerability_score: Vulnerability score (0-10)
            
        Returns:
            Recovery time in days
        """
        # Base recovery times by sector
        base_recovery = {
            'Energy': 30,
            'Materials': 45,
            'Industrials': 60,
            'Consumer Discretionary': 90,
            'Consumer Staples': 30,
            'Health Care': 120,
            'Financials': 14,
            'Information Technology': 21,
            'Communication Services': 21,
            'Utilities': 45,
            'Real Estate': 180
        }
        
        base_days = base_recovery.get(sector_name, 60)
        
        # Adjust based on vulnerability
        vulnerability_multiplier = 1 + (vulnerability_score - 5) * 0.2
        
        recovery_days = base_days * vulnerability_multiplier
        
        return max(7, int(recovery_days))
    
    def _generate_mitigation_strategies(self, sector_name: str, risk_factors: List[str]) -> List[str]:
        """Generate mitigation strategies for a sector.
        
        Args:
            sector_name: Name of the sector
            risk_factors: List of key risk factors
            
        Returns:
            List of mitigation strategies
        """
        strategies = []
        
        # General strategies by sector
        sector_strategies = {
            'Energy': [
                'Diversify supply sources',
                'Increase strategic reserves',
                'Invest in alternative energy'
            ],
            'Materials': [
                'Build inventory buffers',
                'Develop local suppliers',
                'Implement just-in-case inventory'
            ],
            'Industrials': [
                'Strengthen supplier relationships',
                'Invest in automation',
                'Develop contingency plans'
            ],
            'Consumer Discretionary': [
                'Diversify product mix',
                'Focus on essential categories',
                'Improve demand forecasting'
            ],
            'Consumer Staples': [
                'Secure long-term contracts',
                'Invest in local production',
                'Build distribution redundancy'
            ]
        }
        
        # Add sector-specific strategies
        strategies.extend(sector_strategies.get(sector_name, [
            'Monitor supply chain risks',
            'Develop alternative suppliers',
            'Improve operational flexibility'
        ]))
        
        # Add risk-specific strategies
        if any('port_congestion' in rf for rf in risk_factors):
            strategies.append('Diversify shipping routes and ports')
        
        if any('rate_spike' in rf for rf in risk_factors):
            strategies.append('Hedge freight rate exposure')
        
        if any('route_blockage' in rf for rf in risk_factors):
            strategies.append('Develop alternative transportation modes')
        
        return strategies[:5]  # Return top 5 strategies
    
    def _get_sector_code(self, sector_name: str) -> str:
        """Get GICS sector code for a sector name.
        
        Args:
            sector_name: Name of the sector
            
        Returns:
            GICS sector code
        """
        for code, name in self.sectors.items():
            if name == sector_name:
                return code
        return '00'  # Unknown sector
    
    async def generate_sector_report(self, 
                                   sector_impacts: List[SectorImpactScore],
                                   cross_impacts: List[CrossSectorImpact],
                                   rotation_signals: List[SectorRotationSignal]) -> Dict[str, Any]:
        """Generate comprehensive sector analysis report.
        
        Args:
            sector_impacts: List of sector impact scores
            cross_impacts: List of cross-sector impacts
            rotation_signals: List of rotation signals
            
        Returns:
            Dictionary containing the report
        """
        try:
            # Summary statistics
            avg_vulnerability = np.mean([si.vulnerability_score for si in sector_impacts])
            most_vulnerable = max(sector_impacts, key=lambda x: x.vulnerability_score)
            least_vulnerable = min(sector_impacts, key=lambda x: x.vulnerability_score)
            
            # Impact distribution
            negative_impact_sectors = [si for si in sector_impacts if si.predicted_impact_30d < -2]
            positive_impact_sectors = [si for si in sector_impacts if si.predicted_impact_30d > 1]
            
            # Signal summary
            overweight_signals = [rs for rs in rotation_signals if rs.signal_type == 'overweight']
            underweight_signals = [rs for rs in rotation_signals if rs.signal_type == 'underweight']
            pair_signals = [rs for rs in rotation_signals if rs.signal_type == 'pair_trade']
            
            report = {
                'summary': {
                    'analysis_timestamp': datetime.utcnow(),
                    'sectors_analyzed': len(sector_impacts),
                    'average_vulnerability': avg_vulnerability,
                    'most_vulnerable_sector': most_vulnerable.sector_name,
                    'least_vulnerable_sector': least_vulnerable.sector_name,
                    'sectors_with_negative_impact': len(negative_impact_sectors),
                    'sectors_with_positive_impact': len(positive_impact_sectors),
                    'cross_sector_impacts_identified': len(cross_impacts),
                    'trading_signals_generated': len(rotation_signals)
                },
                'sector_rankings': {
                    'by_vulnerability': [
                        {'sector': si.sector_name, 'score': si.vulnerability_score}
                        for si in sorted(sector_impacts, key=lambda x: x.vulnerability_score, reverse=True)
                    ],
                    'by_30d_impact': [
                        {'sector': si.sector_name, 'impact': si.predicted_impact_30d}
                        for si in sorted(sector_impacts, key=lambda x: x.predicted_impact_30d)
                    ]
                },
                'trading_recommendations': {
                    'overweight': [{
                        'sector': rs.target_sector,
                        'strength': rs.strength,
                        'rationale': rs.rationale,
                        'expected_return': rs.expected_return
                    } for rs in overweight_signals[:5]],
                    'underweight': [{
                        'sector': rs.source_sector,
                        'strength': rs.strength,
                        'rationale': rs.rationale,
                        'expected_return': rs.expected_return
                    } for rs in underweight_signals[:5]],
                    'pair_trades': [{
                        'long': rs.target_sector,
                        'short': rs.source_sector,
                        'strength': rs.strength,
                        'expected_return': rs.expected_return
                    } for rs in pair_signals[:3]]
                },
                'risk_assessment': {
                    'high_risk_sectors': [
                        si.sector_name for si in sector_impacts if si.vulnerability_score > 7
                    ],
                    'total_market_cap_at_risk': sum(si.market_cap_at_risk for si in sector_impacts),
                    'average_recovery_time': np.mean([si.recovery_time_estimate for si in sector_impacts]),
                    'key_risk_factors': list(set([
                        rf for si in sector_impacts for rf in si.key_risk_factors
                    ]))[:10]
                },
                'cross_sector_analysis': {
                    'strongest_propagation_paths': [
                        {
                            'from': ci.source_sector,
                            'to': ci.target_sector,
                            'strength': ci.impact_strength,
                            'delay_days': ci.propagation_delay
                        }
                        for ci in sorted(cross_impacts, key=lambda x: x.impact_strength, reverse=True)[:10]
                    ]
                }
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating sector report: {e}")
            raise