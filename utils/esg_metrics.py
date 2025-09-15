#!/usr/bin/env python3
"""
ESG Metrics Integration
Integrates Environmental, Social, and Governance metrics into supply chain analysis.

Features:
- ESG data collection and scoring
- Supply chain sustainability assessment
- Climate risk integration
- Social impact measurement
- Governance quality evaluation
- ESG-adjusted investment signals

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

# Analysis libraries
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from scipy.stats import percentileofscore

# API libraries
import requests
import json

from utils.database import DatabaseManager


@dataclass
class ESGScore:
    """Data structure for ESG scores."""
    company_symbol: str
    company_name: str
    sector: str
    overall_esg_score: float  # 0-100
    environmental_score: float  # 0-100
    social_score: float  # 0-100
    governance_score: float  # 0-100
    
    # Environmental sub-scores
    carbon_emissions_score: float
    water_usage_score: float
    waste_management_score: float
    renewable_energy_score: float
    biodiversity_score: float
    
    # Social sub-scores
    labor_practices_score: float
    human_rights_score: float
    community_impact_score: float
    product_safety_score: float
    supply_chain_labor_score: float
    
    # Governance sub-scores
    board_composition_score: float
    executive_compensation_score: float
    transparency_score: float
    ethics_compliance_score: float
    shareholder_rights_score: float
    
    # Supply chain specific metrics
    supply_chain_transparency: float  # 0-100
    supplier_diversity: float  # 0-100
    conflict_minerals_compliance: float  # 0-100
    forced_labor_risk: float  # 0-100 (higher = more risk)
    
    # Climate risk metrics
    physical_climate_risk: float  # 0-100
    transition_climate_risk: float  # 0-100
    climate_adaptation_score: float  # 0-100
    
    # Ratings and rankings
    msci_esg_rating: str  # AAA, AA, A, BBB, BB, B, CCC
    sustainalytics_risk_score: float  # 0-100 (lower = better)
    cdp_climate_score: str  # A, A-, B, B-, C, C-, D
    
    # Metadata
    data_quality_score: float  # 0-100
    last_updated: datetime
    data_sources: List[str]
    timestamp: datetime


@dataclass
class SupplyChainSustainabilityMetrics:
    """Data structure for supply chain sustainability assessment."""
    company_symbol: str
    company_name: str
    sector: str
    
    # Supplier assessment
    tier1_suppliers_assessed: int
    tier1_suppliers_total: int
    tier2_suppliers_assessed: int
    tier2_suppliers_total: int
    supplier_esg_average_score: float
    
    # Geographic risk
    high_risk_countries_exposure: float  # 0-1
    water_stress_regions_exposure: float  # 0-1
    conflict_regions_exposure: float  # 0-1
    
    # Transportation sustainability
    low_carbon_transport_percentage: float  # 0-100
    transport_emissions_intensity: float  # kg CO2 per ton-km
    modal_shift_score: float  # 0-100
    
    # Circular economy metrics
    recycled_content_percentage: float  # 0-100
    waste_to_landfill_percentage: float  # 0-100
    product_recyclability_score: float  # 0-100
    
    # Social impact
    living_wage_suppliers_percentage: float  # 0-100
    women_owned_suppliers_percentage: float  # 0-100
    local_sourcing_percentage: float  # 0-100
    
    # Compliance and certifications
    iso14001_certified_suppliers: float  # 0-100
    fair_trade_certified_percentage: float  # 0-100
    organic_certified_percentage: float  # 0-100
    
    # Risk indicators
    supply_chain_disruption_risk: float  # 0-100
    reputational_risk_score: float  # 0-100
    regulatory_compliance_risk: float  # 0-100
    
    timestamp: datetime


@dataclass
class ESGAdjustedSignal:
    """Data structure for ESG-adjusted investment signals."""
    original_signal_strength: float
    esg_adjusted_strength: float
    esg_adjustment_factor: float
    
    # ESG considerations
    esg_positive_factors: List[str]
    esg_negative_factors: List[str]
    esg_risk_factors: List[str]
    
    # Sustainability alignment
    sdg_alignment_score: float  # UN Sustainable Development Goals
    paris_agreement_alignment: bool
    taxonomy_eligibility: float  # EU Taxonomy alignment
    
    # Investment rationale adjustments
    esg_enhanced_rationale: str
    sustainability_thesis: str
    
    timestamp: datetime


class ESGMetricsIntegrator:
    """Integrates ESG metrics into supply chain disruption analysis."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the ESG metrics integrator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Database manager
        self.db_manager = DatabaseManager()
        
        # ESG data providers configuration
        self.esg_providers = config.get('esg_providers', {})
        self.api_keys = {
            'msci': config.get('msci_api_key'),
            'sustainalytics': config.get('sustainalytics_api_key'),
            'refinitiv': config.get('refinitiv_api_key'),
            'bloomberg': config.get('bloomberg_api_key'),
            'cdp': config.get('cdp_api_key')
        }
        
        # ESG scoring weights
        self.esg_weights = config.get('esg_weights', {
            'environmental': 0.4,
            'social': 0.3,
            'governance': 0.3
        })
        
        # Supply chain ESG weights
        self.supply_chain_weights = config.get('supply_chain_esg_weights', {
            'transparency': 0.25,
            'labor_practices': 0.25,
            'environmental_impact': 0.25,
            'governance': 0.25
        })
        
        # ESG adjustment parameters
        self.esg_adjustment_strength = config.get('esg_adjustment_strength', 0.3)
        self.min_esg_threshold = config.get('min_esg_threshold', 30)
        self.esg_momentum_weight = config.get('esg_momentum_weight', 0.1)
        
        # Cache for ESG data
        self.esg_cache = {}
        self.cache_expiry = timedelta(hours=24)  # 24-hour cache
        
        # Sector ESG benchmarks
        self.sector_benchmarks = self._initialize_sector_benchmarks()
        
        # SDG mapping
        self.sdg_mapping = self._initialize_sdg_mapping()
    
    def _initialize_sector_benchmarks(self) -> Dict[str, Dict[str, float]]:
        """Initialize sector-specific ESG benchmarks.
        
        Returns:
            Dictionary with sector ESG benchmarks
        """
        # Typical ESG scores by sector (based on industry averages)
        benchmarks = {
            'Energy': {
                'environmental': 35, 'social': 45, 'governance': 55,
                'supply_chain_risk': 70, 'climate_risk': 85
            },
            'Materials': {
                'environmental': 40, 'social': 50, 'governance': 60,
                'supply_chain_risk': 65, 'climate_risk': 75
            },
            'Industrials': {
                'environmental': 50, 'social': 55, 'governance': 65,
                'supply_chain_risk': 60, 'climate_risk': 60
            },
            'Consumer Discretionary': {
                'environmental': 45, 'social': 50, 'governance': 60,
                'supply_chain_risk': 75, 'climate_risk': 50
            },
            'Consumer Staples': {
                'environmental': 55, 'social': 60, 'governance': 65,
                'supply_chain_risk': 70, 'climate_risk': 45
            },
            'Health Care': {
                'environmental': 60, 'social': 70, 'governance': 70,
                'supply_chain_risk': 55, 'climate_risk': 40
            },
            'Financials': {
                'environmental': 65, 'social': 60, 'governance': 75,
                'supply_chain_risk': 30, 'climate_risk': 35
            },
            'Information Technology': {
                'environmental': 70, 'social': 65, 'governance': 75,
                'supply_chain_risk': 65, 'climate_risk': 45
            },
            'Communication Services': {
                'environmental': 65, 'social': 55, 'governance': 70,
                'supply_chain_risk': 50, 'climate_risk': 40
            },
            'Utilities': {
                'environmental': 45, 'social': 60, 'governance': 65,
                'supply_chain_risk': 40, 'climate_risk': 80
            },
            'Real Estate': {
                'environmental': 55, 'social': 55, 'governance': 60,
                'supply_chain_risk': 45, 'climate_risk': 70
            }
        }
        
        return benchmarks
    
    def _initialize_sdg_mapping(self) -> Dict[str, List[int]]:
        """Initialize mapping of business activities to UN SDGs.
        
        Returns:
            Dictionary mapping activities to SDG numbers
        """
        sdg_mapping = {
            'renewable_energy': [7, 13],  # Affordable Clean Energy, Climate Action
            'water_management': [6, 14],  # Clean Water, Life Below Water
            'waste_reduction': [12, 15],  # Responsible Consumption, Life on Land
            'fair_labor': [8, 10],  # Decent Work, Reduced Inequalities
            'supply_chain_transparency': [8, 12, 16],  # Decent Work, Responsible Consumption, Peace & Justice
            'gender_equality': [5, 10],  # Gender Equality, Reduced Inequalities
            'education_training': [4, 8],  # Quality Education, Decent Work
            'healthcare_access': [3],  # Good Health and Well-being
            'sustainable_cities': [11],  # Sustainable Cities
            'climate_action': [13],  # Climate Action
            'biodiversity': [14, 15],  # Life Below Water, Life on Land
            'poverty_reduction': [1, 2],  # No Poverty, Zero Hunger
            'innovation': [9],  # Industry, Innovation and Infrastructure
            'partnerships': [17]  # Partnerships for the Goals
        }
        
        return sdg_mapping
    
    async def collect_esg_data(self, symbols: List[str]) -> Dict[str, ESGScore]:
        """Collect ESG data for a list of companies.
        
        Args:
            symbols: List of stock symbols
            
        Returns:
            Dictionary mapping symbols to ESG scores
        """
        self.logger.info(f"Collecting ESG data for {len(symbols)} companies")
        
        try:
            esg_scores = {}
            
            for symbol in symbols:
                try:
                    # Check cache first
                    if self._is_esg_data_cached(symbol):
                        esg_scores[symbol] = self.esg_cache[symbol]
                        continue
                    
                    # Collect from multiple sources
                    esg_data = await self._collect_company_esg_data(symbol)
                    
                    if esg_data:
                        esg_score = self._calculate_composite_esg_score(symbol, esg_data)
                        esg_scores[symbol] = esg_score
                        
                        # Cache the result
                        self.esg_cache[symbol] = esg_score
                    
                except Exception as e:
                    self.logger.error(f"Error collecting ESG data for {symbol}: {e}")
            
            self.logger.info(f"Successfully collected ESG data for {len(esg_scores)} companies")
            return esg_scores
            
        except Exception as e:
            self.logger.error(f"Error in ESG data collection: {e}")
            raise
    
    async def assess_supply_chain_sustainability(self, symbols: List[str]) -> Dict[str, SupplyChainSustainabilityMetrics]:
        """Assess supply chain sustainability for companies.
        
        Args:
            symbols: List of stock symbols
            
        Returns:
            Dictionary mapping symbols to sustainability metrics
        """
        self.logger.info(f"Assessing supply chain sustainability for {len(symbols)} companies")
        
        try:
            sustainability_metrics = {}
            
            for symbol in symbols:
                try:
                    # Get company information
                    company_info = await self._get_company_info(symbol)
                    
                    # Assess supply chain sustainability
                    metrics = await self._assess_company_supply_chain(symbol, company_info)
                    
                    if metrics:
                        sustainability_metrics[symbol] = metrics
                    
                except Exception as e:
                    self.logger.error(f"Error assessing supply chain sustainability for {symbol}: {e}")
            
            self.logger.info(f"Completed supply chain sustainability assessment for {len(sustainability_metrics)} companies")
            return sustainability_metrics
            
        except Exception as e:
            self.logger.error(f"Error in supply chain sustainability assessment: {e}")
            raise
    
    def adjust_signals_for_esg(self, 
                              signals: List[Dict[str, Any]], 
                              esg_scores: Dict[str, ESGScore],
                              sustainability_metrics: Dict[str, SupplyChainSustainabilityMetrics]) -> List[Dict[str, Any]]:
        """Adjust investment signals based on ESG considerations.
        
        Args:
            signals: List of investment signals
            esg_scores: Dictionary of ESG scores
            sustainability_metrics: Dictionary of sustainability metrics
            
        Returns:
            List of ESG-adjusted signals
        """
        self.logger.info(f"Adjusting {len(signals)} signals for ESG considerations")
        
        try:
            adjusted_signals = []
            
            for signal in signals:
                try:
                    symbol = signal.get('symbol')
                    if not symbol:
                        adjusted_signals.append(signal)
                        continue
                    
                    # Get ESG data
                    esg_score = esg_scores.get(symbol)
                    sustainability = sustainability_metrics.get(symbol)
                    
                    if not esg_score:
                        # If no ESG data, apply conservative adjustment
                        signal['esg_adjusted_strength'] = signal.get('strength', 1.0) * 0.9
                        signal['esg_adjustment_reason'] = 'No ESG data available'
                        adjusted_signals.append(signal)
                        continue
                    
                    # Calculate ESG adjustment
                    adjustment = self._calculate_esg_adjustment(
                        signal, esg_score, sustainability
                    )
                    
                    # Apply adjustment
                    original_strength = signal.get('strength', 1.0)
                    adjusted_strength = original_strength * adjustment['factor']
                    
                    # Update signal
                    signal['original_strength'] = original_strength
                    signal['esg_adjusted_strength'] = adjusted_strength
                    signal['esg_adjustment_factor'] = adjustment['factor']
                    signal['esg_positive_factors'] = adjustment['positive_factors']
                    signal['esg_negative_factors'] = adjustment['negative_factors']
                    signal['esg_risk_factors'] = adjustment['risk_factors']
                    signal['esg_rationale'] = adjustment['rationale']
                    signal['sdg_alignment_score'] = adjustment['sdg_score']
                    
                    # Update overall strength
                    signal['strength'] = adjusted_strength
                    
                    adjusted_signals.append(signal)
                    
                except Exception as e:
                    self.logger.error(f"Error adjusting signal for ESG: {e}")
                    adjusted_signals.append(signal)  # Keep original signal
            
            self.logger.info(f"Completed ESG adjustment for {len(adjusted_signals)} signals")
            return adjusted_signals
            
        except Exception as e:
            self.logger.error(f"Error in ESG signal adjustment: {e}")
            return signals  # Return original signals on error
    
    async def _collect_company_esg_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Collect ESG data for a single company from multiple sources.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary with ESG data or None
        """
        try:
            esg_data = {
                'symbol': symbol,
                'sources': [],
                'environmental': {},
                'social': {},
                'governance': {},
                'supply_chain': {},
                'climate': {},
                'ratings': {}
            }
            
            # Collect from MSCI (if available)
            msci_data = await self._get_msci_esg_data(symbol)
            if msci_data:
                esg_data['sources'].append('MSCI')
                esg_data.update(msci_data)
            
            # Collect from Sustainalytics (if available)
            sustainalytics_data = await self._get_sustainalytics_data(symbol)
            if sustainalytics_data:
                esg_data['sources'].append('Sustainalytics')
                esg_data.update(sustainalytics_data)
            
            # Collect from Refinitiv (if available)
            refinitiv_data = await self._get_refinitiv_esg_data(symbol)
            if refinitiv_data:
                esg_data['sources'].append('Refinitiv')
                esg_data.update(refinitiv_data)
            
            # Collect from CDP (if available)
            cdp_data = await self._get_cdp_data(symbol)
            if cdp_data:
                esg_data['sources'].append('CDP')
                esg_data.update(cdp_data)
            
            # If no data from APIs, generate synthetic data based on sector
            if not esg_data['sources']:
                esg_data = self._generate_synthetic_esg_data(symbol)
            
            return esg_data if esg_data['sources'] else None
            
        except Exception as e:
            self.logger.error(f"Error collecting ESG data for {symbol}: {e}")
            return None
    
    async def _get_msci_esg_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get ESG data from MSCI.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary with MSCI ESG data or None
        """
        try:
            if not self.api_keys.get('msci'):
                return None
            
            # This would make actual API call to MSCI
            # For now, return synthetic data structure
            
            msci_data = {
                'environmental': {
                    'carbon_emissions': np.random.uniform(20, 80),
                    'water_usage': np.random.uniform(30, 90),
                    'waste_management': np.random.uniform(40, 85),
                    'renewable_energy': np.random.uniform(10, 70)
                },
                'social': {
                    'labor_practices': np.random.uniform(40, 90),
                    'human_rights': np.random.uniform(30, 85),
                    'community_impact': np.random.uniform(35, 80),
                    'product_safety': np.random.uniform(50, 95)
                },
                'governance': {
                    'board_composition': np.random.uniform(45, 90),
                    'executive_compensation': np.random.uniform(40, 85),
                    'transparency': np.random.uniform(50, 90),
                    'ethics_compliance': np.random.uniform(60, 95)
                },
                'ratings': {
                    'msci_esg_rating': np.random.choice(['AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC'], 
                                                       p=[0.05, 0.15, 0.25, 0.30, 0.15, 0.08, 0.02])
                }
            }
            
            return msci_data
            
        except Exception as e:
            self.logger.error(f"Error getting MSCI data for {symbol}: {e}")
            return None
    
    async def _get_sustainalytics_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get ESG data from Sustainalytics.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary with Sustainalytics ESG data or None
        """
        try:
            if not self.api_keys.get('sustainalytics'):
                return None
            
            # This would make actual API call to Sustainalytics
            # For now, return synthetic data structure
            
            sustainalytics_data = {
                'ratings': {
                    'sustainalytics_risk_score': np.random.uniform(10, 50)  # Lower is better
                },
                'supply_chain': {
                    'supply_chain_transparency': np.random.uniform(30, 85),
                    'supplier_diversity': np.random.uniform(20, 70),
                    'conflict_minerals_compliance': np.random.uniform(60, 95),
                    'forced_labor_risk': np.random.uniform(5, 40)  # Higher is worse
                }
            }
            
            return sustainalytics_data
            
        except Exception as e:
            self.logger.error(f"Error getting Sustainalytics data for {symbol}: {e}")
            return None
    
    async def _get_refinitiv_esg_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get ESG data from Refinitiv.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary with Refinitiv ESG data or None
        """
        try:
            if not self.api_keys.get('refinitiv'):
                return None
            
            # This would make actual API call to Refinitiv
            # For now, return synthetic data structure
            
            refinitiv_data = {
                'environmental': {
                    'biodiversity_score': np.random.uniform(30, 80)
                },
                'social': {
                    'supply_chain_labor_score': np.random.uniform(40, 85)
                },
                'governance': {
                    'shareholder_rights_score': np.random.uniform(50, 90)
                }
            }
            
            return refinitiv_data
            
        except Exception as e:
            self.logger.error(f"Error getting Refinitiv data for {symbol}: {e}")
            return None
    
    async def _get_cdp_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get climate data from CDP.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary with CDP climate data or None
        """
        try:
            if not self.api_keys.get('cdp'):
                return None
            
            # This would make actual API call to CDP
            # For now, return synthetic data structure
            
            cdp_data = {
                'climate': {
                    'physical_climate_risk': np.random.uniform(20, 80),
                    'transition_climate_risk': np.random.uniform(15, 75),
                    'climate_adaptation_score': np.random.uniform(30, 85)
                },
                'ratings': {
                    'cdp_climate_score': np.random.choice(['A', 'A-', 'B', 'B-', 'C', 'C-', 'D'], 
                                                         p=[0.10, 0.15, 0.20, 0.20, 0.15, 0.15, 0.05])
                }
            }
            
            return cdp_data
            
        except Exception as e:
            self.logger.error(f"Error getting CDP data for {symbol}: {e}")
            return None
    
    def _generate_synthetic_esg_data(self, symbol: str) -> Dict[str, Any]:
        """Generate synthetic ESG data based on sector characteristics.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary with synthetic ESG data
        """
        try:
            # Get company sector (simplified - would normally look this up)
            sector = 'Industrials'  # Default sector
            
            # Get sector benchmarks
            benchmarks = self.sector_benchmarks.get(sector, {
                'environmental': 50, 'social': 50, 'governance': 50,
                'supply_chain_risk': 50, 'climate_risk': 50
            })
            
            # Generate scores around sector benchmarks with some noise
            synthetic_data = {
                'symbol': symbol,
                'sources': ['Synthetic'],
                'environmental': {
                    'carbon_emissions': max(0, min(100, np.random.normal(benchmarks['environmental'], 15))),
                    'water_usage': max(0, min(100, np.random.normal(benchmarks['environmental'], 10))),
                    'waste_management': max(0, min(100, np.random.normal(benchmarks['environmental'], 12))),
                    'renewable_energy': max(0, min(100, np.random.normal(benchmarks['environmental'], 20))),
                    'biodiversity_score': max(0, min(100, np.random.normal(benchmarks['environmental'], 15)))
                },
                'social': {
                    'labor_practices': max(0, min(100, np.random.normal(benchmarks['social'], 12))),
                    'human_rights': max(0, min(100, np.random.normal(benchmarks['social'], 15))),
                    'community_impact': max(0, min(100, np.random.normal(benchmarks['social'], 10))),
                    'product_safety': max(0, min(100, np.random.normal(benchmarks['social'], 8))),
                    'supply_chain_labor_score': max(0, min(100, np.random.normal(benchmarks['social'], 18)))
                },
                'governance': {
                    'board_composition': max(0, min(100, np.random.normal(benchmarks['governance'], 10))),
                    'executive_compensation': max(0, min(100, np.random.normal(benchmarks['governance'], 12))),
                    'transparency': max(0, min(100, np.random.normal(benchmarks['governance'], 8))),
                    'ethics_compliance': max(0, min(100, np.random.normal(benchmarks['governance'], 6))),
                    'shareholder_rights_score': max(0, min(100, np.random.normal(benchmarks['governance'], 10)))
                },
                'supply_chain': {
                    'supply_chain_transparency': max(0, min(100, np.random.normal(100 - benchmarks['supply_chain_risk'], 15))),
                    'supplier_diversity': max(0, min(100, np.random.normal(50, 20))),
                    'conflict_minerals_compliance': max(0, min(100, np.random.normal(80, 15))),
                    'forced_labor_risk': max(0, min(100, np.random.normal(benchmarks['supply_chain_risk'], 20)))
                },
                'climate': {
                    'physical_climate_risk': max(0, min(100, np.random.normal(benchmarks['climate_risk'], 15))),
                    'transition_climate_risk': max(0, min(100, np.random.normal(benchmarks['climate_risk'], 12))),
                    'climate_adaptation_score': max(0, min(100, np.random.normal(100 - benchmarks['climate_risk'], 18)))
                },
                'ratings': {
                    'msci_esg_rating': 'BBB',  # Default rating
                    'sustainalytics_risk_score': np.random.uniform(15, 35),
                    'cdp_climate_score': 'B'
                }
            }
            
            return synthetic_data
            
        except Exception as e:
            self.logger.error(f"Error generating synthetic ESG data for {symbol}: {e}")
            return {'sources': []}
    
    def _calculate_composite_esg_score(self, symbol: str, esg_data: Dict[str, Any]) -> ESGScore:
        """Calculate composite ESG score from multiple data sources.
        
        Args:
            symbol: Stock symbol
            esg_data: Raw ESG data from multiple sources
            
        Returns:
            ESGScore object
        """
        try:
            # Extract environmental scores
            env_data = esg_data.get('environmental', {})
            environmental_score = np.mean([
                env_data.get('carbon_emissions', 50),
                env_data.get('water_usage', 50),
                env_data.get('waste_management', 50),
                env_data.get('renewable_energy', 50),
                env_data.get('biodiversity_score', 50)
            ])
            
            # Extract social scores
            social_data = esg_data.get('social', {})
            social_score = np.mean([
                social_data.get('labor_practices', 50),
                social_data.get('human_rights', 50),
                social_data.get('community_impact', 50),
                social_data.get('product_safety', 50),
                social_data.get('supply_chain_labor_score', 50)
            ])
            
            # Extract governance scores
            gov_data = esg_data.get('governance', {})
            governance_score = np.mean([
                gov_data.get('board_composition', 50),
                gov_data.get('executive_compensation', 50),
                gov_data.get('transparency', 50),
                gov_data.get('ethics_compliance', 50),
                gov_data.get('shareholder_rights_score', 50)
            ])
            
            # Calculate overall ESG score
            overall_esg_score = (
                environmental_score * self.esg_weights['environmental'] +
                social_score * self.esg_weights['social'] +
                governance_score * self.esg_weights['governance']
            )
            
            # Extract supply chain metrics
            supply_chain_data = esg_data.get('supply_chain', {})
            climate_data = esg_data.get('climate', {})
            ratings_data = esg_data.get('ratings', {})
            
            # Calculate data quality score
            data_quality = len(esg_data.get('sources', [])) * 25  # 25 points per source
            data_quality = min(100, data_quality)
            
            esg_score = ESGScore(
                company_symbol=symbol,
                company_name=f"Company {symbol}",  # Would normally look this up
                sector="Unknown",  # Would normally look this up
                overall_esg_score=overall_esg_score,
                environmental_score=environmental_score,
                social_score=social_score,
                governance_score=governance_score,
                
                # Environmental sub-scores
                carbon_emissions_score=env_data.get('carbon_emissions', 50),
                water_usage_score=env_data.get('water_usage', 50),
                waste_management_score=env_data.get('waste_management', 50),
                renewable_energy_score=env_data.get('renewable_energy', 50),
                biodiversity_score=env_data.get('biodiversity_score', 50),
                
                # Social sub-scores
                labor_practices_score=social_data.get('labor_practices', 50),
                human_rights_score=social_data.get('human_rights', 50),
                community_impact_score=social_data.get('community_impact', 50),
                product_safety_score=social_data.get('product_safety', 50),
                supply_chain_labor_score=social_data.get('supply_chain_labor_score', 50),
                
                # Governance sub-scores
                board_composition_score=gov_data.get('board_composition', 50),
                executive_compensation_score=gov_data.get('executive_compensation', 50),
                transparency_score=gov_data.get('transparency', 50),
                ethics_compliance_score=gov_data.get('ethics_compliance', 50),
                shareholder_rights_score=gov_data.get('shareholder_rights_score', 50),
                
                # Supply chain specific metrics
                supply_chain_transparency=supply_chain_data.get('supply_chain_transparency', 50),
                supplier_diversity=supply_chain_data.get('supplier_diversity', 50),
                conflict_minerals_compliance=supply_chain_data.get('conflict_minerals_compliance', 80),
                forced_labor_risk=supply_chain_data.get('forced_labor_risk', 30),
                
                # Climate risk metrics
                physical_climate_risk=climate_data.get('physical_climate_risk', 50),
                transition_climate_risk=climate_data.get('transition_climate_risk', 50),
                climate_adaptation_score=climate_data.get('climate_adaptation_score', 50),
                
                # Ratings and rankings
                msci_esg_rating=ratings_data.get('msci_esg_rating', 'BBB'),
                sustainalytics_risk_score=ratings_data.get('sustainalytics_risk_score', 25),
                cdp_climate_score=ratings_data.get('cdp_climate_score', 'B'),
                
                # Metadata
                data_quality_score=data_quality,
                last_updated=datetime.utcnow(),
                data_sources=esg_data.get('sources', []),
                timestamp=datetime.utcnow()
            )
            
            return esg_score
            
        except Exception as e:
            self.logger.error(f"Error calculating composite ESG score for {symbol}: {e}")
            raise
    
    async def _assess_company_supply_chain(self, symbol: str, company_info: Dict[str, Any]) -> Optional[SupplyChainSustainabilityMetrics]:
        """Assess supply chain sustainability for a company.
        
        Args:
            symbol: Stock symbol
            company_info: Company information
            
        Returns:
            SupplyChainSustainabilityMetrics object or None
        """
        try:
            # This would typically involve detailed supply chain analysis
            # For now, generate synthetic metrics based on sector and size
            
            sector = company_info.get('sector', 'Unknown')
            market_cap = company_info.get('market_cap', 1000)  # Million USD
            
            # Larger companies typically have better supply chain management
            size_factor = min(1.0, np.log(market_cap) / 10)
            
            # Sector-specific adjustments
            sector_multipliers = {
                'Information Technology': 1.2,
                'Health Care': 1.1,
                'Consumer Staples': 1.0,
                'Industrials': 0.9,
                'Materials': 0.8,
                'Energy': 0.7
            }
            
            sector_mult = sector_multipliers.get(sector, 1.0)
            base_score = 50 * size_factor * sector_mult
            
            metrics = SupplyChainSustainabilityMetrics(
                company_symbol=symbol,
                company_name=company_info.get('name', f'Company {symbol}'),
                sector=sector,
                
                # Supplier assessment
                tier1_suppliers_assessed=int(np.random.uniform(50, 200) * size_factor),
                tier1_suppliers_total=int(np.random.uniform(100, 500) * size_factor),
                tier2_suppliers_assessed=int(np.random.uniform(20, 100) * size_factor),
                tier2_suppliers_total=int(np.random.uniform(200, 1000) * size_factor),
                supplier_esg_average_score=max(20, min(80, np.random.normal(base_score, 15))),
                
                # Geographic risk
                high_risk_countries_exposure=max(0, min(1, np.random.beta(2, 5) * (2 - sector_mult))),
                water_stress_regions_exposure=max(0, min(1, np.random.beta(3, 4) * (2 - sector_mult))),
                conflict_regions_exposure=max(0, min(1, np.random.beta(8, 2) * (2 - sector_mult))),
                
                # Transportation sustainability
                low_carbon_transport_percentage=max(0, min(100, np.random.normal(base_score * 0.8, 20))),
                transport_emissions_intensity=np.random.lognormal(2, 0.5),  # kg CO2 per ton-km
                modal_shift_score=max(0, min(100, np.random.normal(base_score, 15))),
                
                # Circular economy metrics
                recycled_content_percentage=max(0, min(100, np.random.normal(base_score * 0.6, 25))),
                waste_to_landfill_percentage=max(0, min(100, np.random.normal(100 - base_score, 20))),
                product_recyclability_score=max(0, min(100, np.random.normal(base_score, 18))),
                
                # Social impact
                living_wage_suppliers_percentage=max(0, min(100, np.random.normal(base_score * 0.7, 20))),
                women_owned_suppliers_percentage=max(0, min(100, np.random.normal(30, 15))),
                local_sourcing_percentage=max(0, min(100, np.random.normal(40, 20))),
                
                # Compliance and certifications
                iso14001_certified_suppliers=max(0, min(100, np.random.normal(base_score * 0.8, 15))),
                fair_trade_certified_percentage=max(0, min(100, np.random.normal(20, 15))),
                organic_certified_percentage=max(0, min(100, np.random.normal(15, 12))),
                
                # Risk indicators
                supply_chain_disruption_risk=max(0, min(100, np.random.normal(100 - base_score, 20))),
                reputational_risk_score=max(0, min(100, np.random.normal(100 - base_score * 0.8, 15))),
                regulatory_compliance_risk=max(0, min(100, np.random.normal(100 - base_score * 0.9, 12))),
                
                timestamp=datetime.utcnow()
            )
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error assessing supply chain sustainability for {symbol}: {e}")
            return None
    
    def _calculate_esg_adjustment(self, 
                                signal: Dict[str, Any], 
                                esg_score: ESGScore,
                                sustainability: Optional[SupplyChainSustainabilityMetrics]) -> Dict[str, Any]:
        """Calculate ESG adjustment factor for an investment signal.
        
        Args:
            signal: Investment signal
            esg_score: ESG score for the company
            sustainability: Supply chain sustainability metrics
            
        Returns:
            Dictionary with adjustment details
        """
        try:
            # Base adjustment factor (1.0 = no adjustment)
            adjustment_factor = 1.0
            positive_factors = []
            negative_factors = []
            risk_factors = []
            
            # ESG score adjustment
            if esg_score.overall_esg_score > 70:
                adjustment_factor *= 1.1  # 10% boost for high ESG
                positive_factors.append(f"High ESG score ({esg_score.overall_esg_score:.0f})")
            elif esg_score.overall_esg_score < self.min_esg_threshold:
                adjustment_factor *= 0.8  # 20% penalty for low ESG
                negative_factors.append(f"Low ESG score ({esg_score.overall_esg_score:.0f})")
            
            # Environmental factors
            if esg_score.carbon_emissions_score > 75:
                adjustment_factor *= 1.05
                positive_factors.append("Strong carbon management")
            elif esg_score.carbon_emissions_score < 30:
                adjustment_factor *= 0.95
                negative_factors.append("Poor carbon management")
            
            if esg_score.renewable_energy_score > 70:
                adjustment_factor *= 1.03
                positive_factors.append("High renewable energy usage")
            
            # Social factors
            if esg_score.labor_practices_score > 75:
                adjustment_factor *= 1.04
                positive_factors.append("Strong labor practices")
            elif esg_score.labor_practices_score < 35:
                adjustment_factor *= 0.92
                negative_factors.append("Poor labor practices")
                risk_factors.append("Labor practice risks")
            
            if esg_score.supply_chain_labor_score < 40:
                adjustment_factor *= 0.90
                negative_factors.append("Supply chain labor concerns")
                risk_factors.append("Supply chain labor risks")
            
            # Governance factors
            if esg_score.governance_score > 75:
                adjustment_factor *= 1.06
                positive_factors.append("Strong governance")
            elif esg_score.governance_score < 40:
                adjustment_factor *= 0.88
                negative_factors.append("Weak governance")
                risk_factors.append("Governance risks")
            
            # Supply chain specific adjustments
            if sustainability:
                if sustainability.supply_chain_transparency > 70:
                    adjustment_factor *= 1.03
                    positive_factors.append("High supply chain transparency")
                elif sustainability.supply_chain_transparency < 30:
                    adjustment_factor *= 0.95
                    negative_factors.append("Low supply chain transparency")
                
                if sustainability.forced_labor_risk > 60:
                    adjustment_factor *= 0.85
                    negative_factors.append("High forced labor risk")
                    risk_factors.append("Forced labor exposure")
                
                if sustainability.supplier_diversity > 60:
                    adjustment_factor *= 1.02
                    positive_factors.append("Good supplier diversity")
            
            # Climate risk adjustments
            if esg_score.physical_climate_risk > 70:
                adjustment_factor *= 0.95
                risk_factors.append("High physical climate risk")
            
            if esg_score.transition_climate_risk > 70:
                adjustment_factor *= 0.93
                risk_factors.append("High transition climate risk")
            
            if esg_score.climate_adaptation_score > 70:
                adjustment_factor *= 1.04
                positive_factors.append("Strong climate adaptation")
            
            # Signal direction consideration
            signal_type = signal.get('signal_type', 'buy')
            if signal_type in ['short', 'sell']:
                # For short positions, invert some ESG considerations
                if len(negative_factors) > len(positive_factors):
                    adjustment_factor = 1.0 + (1.0 - adjustment_factor)  # Invert adjustment
                    positive_factors.append("ESG concerns support short thesis")
            
            # Calculate SDG alignment score
            sdg_score = self._calculate_sdg_alignment(esg_score, sustainability)
            
            # Apply overall ESG adjustment strength
            final_adjustment = 1.0 + (adjustment_factor - 1.0) * self.esg_adjustment_strength
            
            # Generate rationale
            rationale = self._generate_esg_rationale(
                esg_score, positive_factors, negative_factors, risk_factors
            )
            
            return {
                'factor': final_adjustment,
                'positive_factors': positive_factors,
                'negative_factors': negative_factors,
                'risk_factors': risk_factors,
                'rationale': rationale,
                'sdg_score': sdg_score
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating ESG adjustment: {e}")
            return {
                'factor': 1.0,
                'positive_factors': [],
                'negative_factors': [],
                'risk_factors': ['ESG calculation error'],
                'rationale': 'ESG adjustment could not be calculated',
                'sdg_score': 0
            }
    
    def _calculate_sdg_alignment(self, 
                               esg_score: ESGScore, 
                               sustainability: Optional[SupplyChainSustainabilityMetrics]) -> float:
        """Calculate UN Sustainable Development Goals alignment score.
        
        Args:
            esg_score: ESG score for the company
            sustainability: Supply chain sustainability metrics
            
        Returns:
            SDG alignment score (0-100)
        """
        try:
            sdg_contributions = []
            
            # Environmental SDGs
            if esg_score.renewable_energy_score > 60:
                sdg_contributions.append(esg_score.renewable_energy_score * 0.8)  # SDG 7
            
            if esg_score.water_usage_score > 60:
                sdg_contributions.append(esg_score.water_usage_score * 0.7)  # SDG 6
            
            if esg_score.carbon_emissions_score > 60:
                sdg_contributions.append(esg_score.carbon_emissions_score * 0.9)  # SDG 13
            
            # Social SDGs
            if esg_score.labor_practices_score > 60:
                sdg_contributions.append(esg_score.labor_practices_score * 0.8)  # SDG 8
            
            if sustainability and sustainability.living_wage_suppliers_percentage > 50:
                sdg_contributions.append(sustainability.living_wage_suppliers_percentage * 0.6)  # SDG 1
            
            if sustainability and sustainability.women_owned_suppliers_percentage > 30:
                sdg_contributions.append(sustainability.women_owned_suppliers_percentage * 0.7)  # SDG 5
            
            # Governance SDGs
            if esg_score.transparency_score > 70:
                sdg_contributions.append(esg_score.transparency_score * 0.6)  # SDG 16
            
            # Calculate overall SDG alignment
            if sdg_contributions:
                sdg_score = np.mean(sdg_contributions)
            else:
                sdg_score = 30  # Default low alignment
            
            return min(100, max(0, sdg_score))
            
        except Exception as e:
            self.logger.error(f"Error calculating SDG alignment: {e}")
            return 30  # Default score
    
    def _generate_esg_rationale(self, 
                              esg_score: ESGScore,
                              positive_factors: List[str],
                              negative_factors: List[str],
                              risk_factors: List[str]) -> str:
        """Generate ESG rationale text.
        
        Args:
            esg_score: ESG score for the company
            positive_factors: List of positive ESG factors
            negative_factors: List of negative ESG factors
            risk_factors: List of ESG risk factors
            
        Returns:
            ESG rationale string
        """
        try:
            rationale_parts = []
            
            # Overall ESG assessment
            if esg_score.overall_esg_score > 70:
                rationale_parts.append(f"Strong ESG profile (score: {esg_score.overall_esg_score:.0f})")
            elif esg_score.overall_esg_score < 40:
                rationale_parts.append(f"Weak ESG profile (score: {esg_score.overall_esg_score:.0f})")
            else:
                rationale_parts.append(f"Moderate ESG profile (score: {esg_score.overall_esg_score:.0f})")
            
            # Add positive factors
            if positive_factors:
                rationale_parts.append(f"Strengths: {', '.join(positive_factors[:3])}")
            
            # Add concerns
            if negative_factors:
                rationale_parts.append(f"Concerns: {', '.join(negative_factors[:3])}")
            
            # Add risks
            if risk_factors:
                rationale_parts.append(f"Risks: {', '.join(risk_factors[:2])}")
            
            return ". ".join(rationale_parts) + "."
            
        except Exception as e:
            self.logger.error(f"Error generating ESG rationale: {e}")
            return "ESG assessment completed with limited data."
    
    async def _get_company_info(self, symbol: str) -> Dict[str, Any]:
        """Get basic company information.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary with company information
        """
        try:
            # This would typically fetch from financial data APIs
            # For now, return synthetic company info
            
            company_info = {
                'symbol': symbol,
                'name': f'Company {symbol}',
                'sector': np.random.choice([
                    'Information Technology', 'Health Care', 'Financials',
                    'Consumer Discretionary', 'Industrials', 'Consumer Staples',
                    'Energy', 'Materials', 'Real Estate', 'Utilities'
                ]),
                'market_cap': np.random.lognormal(7, 1.5),  # Million USD
                'employees': int(np.random.lognormal(8, 1.2)),
                'revenue': np.random.lognormal(8, 1),  # Million USD
                'headquarters_country': np.random.choice([
                    'United States', 'Germany', 'Japan', 'United Kingdom',
                    'France', 'Canada', 'Switzerland', 'Netherlands'
                ], p=[0.4, 0.1, 0.1, 0.1, 0.08, 0.07, 0.05, 0.1])
            }
            
            return company_info
            
        except Exception as e:
            self.logger.error(f"Error getting company info for {symbol}: {e}")
            return {'symbol': symbol, 'name': f'Company {symbol}', 'sector': 'Unknown'}
    
    def _is_esg_data_cached(self, symbol: str) -> bool:
        """Check if ESG data is cached and still valid.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            True if cached data is valid
        """
        if symbol not in self.esg_cache:
            return False
        
        cached_data = self.esg_cache[symbol]
        data_age = datetime.utcnow() - cached_data.timestamp
        
        return data_age < self.cache_expiry
    
    async def generate_esg_report(self, 
                                esg_scores: Dict[str, ESGScore],
                                sustainability_metrics: Dict[str, SupplyChainSustainabilityMetrics]) -> Dict[str, Any]:
        """Generate comprehensive ESG analysis report.
        
        Args:
            esg_scores: Dictionary of ESG scores
            sustainability_metrics: Dictionary of sustainability metrics
            
        Returns:
            Dictionary containing the ESG report
        """
        try:
            # Calculate summary statistics
            all_scores = list(esg_scores.values())
            
            if not all_scores:
                return {'error': 'No ESG data available for report generation'}
            
            avg_esg = np.mean([score.overall_esg_score for score in all_scores])
            avg_env = np.mean([score.environmental_score for score in all_scores])
            avg_social = np.mean([score.social_score for score in all_scores])
            avg_gov = np.mean([score.governance_score for score in all_scores])
            
            # Identify leaders and laggards
            esg_leaders = sorted(all_scores, key=lambda x: x.overall_esg_score, reverse=True)[:5]
            esg_laggards = sorted(all_scores, key=lambda x: x.overall_esg_score)[:5]
            
            # Risk assessment
            high_risk_companies = [
                score for score in all_scores 
                if (score.forced_labor_risk > 60 or 
                    score.physical_climate_risk > 70 or 
                    score.overall_esg_score < 30)
            ]
            
            # Sector analysis
            sector_scores = defaultdict(list)
            for score in all_scores:
                sector_scores[score.sector].append(score.overall_esg_score)
            
            sector_averages = {
                sector: np.mean(scores) 
                for sector, scores in sector_scores.items()
            }
            
            report = {
                'summary': {
                    'report_date': datetime.utcnow(),
                    'companies_analyzed': len(all_scores),
                    'average_esg_score': avg_esg,
                    'average_environmental_score': avg_env,
                    'average_social_score': avg_social,
                    'average_governance_score': avg_gov,
                    'high_risk_companies_count': len(high_risk_companies)
                },
                'leaders_and_laggards': {
                    'esg_leaders': [{
                        'symbol': score.company_symbol,
                        'name': score.company_name,
                        'esg_score': score.overall_esg_score,
                        'msci_rating': score.msci_esg_rating
                    } for score in esg_leaders],
                    'esg_laggards': [{
                        'symbol': score.company_symbol,
                        'name': score.company_name,
                        'esg_score': score.overall_esg_score,
                        'msci_rating': score.msci_esg_rating
                    } for score in esg_laggards]
                },
                'sector_analysis': {
                    'sector_averages': sector_averages,
                    'best_sector': max(sector_averages.items(), key=lambda x: x[1])[0] if sector_averages else None,
                    'worst_sector': min(sector_averages.items(), key=lambda x: x[1])[0] if sector_averages else None
                },
                'risk_assessment': {
                    'high_risk_companies': [{
                        'symbol': score.company_symbol,
                        'name': score.company_name,
                        'esg_score': score.overall_esg_score,
                        'forced_labor_risk': score.forced_labor_risk,
                        'climate_risk': max(score.physical_climate_risk, score.transition_climate_risk)
                    } for score in high_risk_companies],
                    'supply_chain_risks': {
                        'high_forced_labor_risk': len([s for s in all_scores if s.forced_labor_risk > 60]),
                        'low_transparency': len([s for s in all_scores if s.supply_chain_transparency < 40]),
                        'poor_conflict_minerals': len([s for s in all_scores if s.conflict_minerals_compliance < 60])
                    }
                },
                'sustainability_highlights': {
                    'renewable_energy_leaders': [{
                        'symbol': score.company_symbol,
                        'renewable_score': score.renewable_energy_score
                    } for score in sorted(all_scores, key=lambda x: x.renewable_energy_score, reverse=True)[:3]],
                    'carbon_management_leaders': [{
                        'symbol': score.company_symbol,
                        'carbon_score': score.carbon_emissions_score
                    } for score in sorted(all_scores, key=lambda x: x.carbon_emissions_score, reverse=True)[:3]]
                },
                'data_quality': {
                    'average_data_quality': np.mean([score.data_quality_score for score in all_scores]),
                    'companies_with_high_quality_data': len([s for s in all_scores if s.data_quality_score > 75]),
                    'data_sources_coverage': {
                        source: len([s for s in all_scores if source in s.data_sources])
                        for source in ['MSCI', 'Sustainalytics', 'Refinitiv', 'CDP', 'Synthetic']
                    }
                },
                'recommendations': {
                    'immediate_actions': [
                        'Engage with high-risk companies on ESG improvements',
                        'Increase allocation to ESG leaders',
                        'Monitor supply chain transparency initiatives'
                    ],
                    'long_term_strategy': [
                        'Develop ESG integration framework',
                        'Establish ESG performance targets',
                        'Build ESG data collection capabilities'
                    ]
                }
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating ESG report: {e}")
            return {'error': f'Failed to generate ESG report: {str(e)}'}
    
    async def store_esg_data(self, esg_scores: Dict[str, ESGScore]) -> bool:
        """Store ESG data in database.
        
        Args:
            esg_scores: Dictionary of ESG scores to store
            
        Returns:
            True if successful
        """
        try:
            self.logger.info(f"Storing ESG data for {len(esg_scores)} companies")
            
            for symbol, esg_score in esg_scores.items():
                # Convert to dictionary for storage
                esg_data = {
                    'symbol': esg_score.company_symbol,
                    'company_name': esg_score.company_name,
                    'sector': esg_score.sector,
                    'overall_esg_score': esg_score.overall_esg_score,
                    'environmental_score': esg_score.environmental_score,
                    'social_score': esg_score.social_score,
                    'governance_score': esg_score.governance_score,
                    'supply_chain_transparency': esg_score.supply_chain_transparency,
                    'forced_labor_risk': esg_score.forced_labor_risk,
                    'climate_risk': max(esg_score.physical_climate_risk, esg_score.transition_climate_risk),
                    'msci_rating': esg_score.msci_esg_rating,
                    'sustainalytics_score': esg_score.sustainalytics_risk_score,
                    'cdp_score': esg_score.cdp_climate_score,
                    'data_quality': esg_score.data_quality_score,
                    'data_sources': ','.join(esg_score.data_sources),
                    'timestamp': esg_score.timestamp
                }
                
                # Store in database
                await self.db_manager.store_data('esg_scores', esg_data)
            
            self.logger.info("ESG data storage completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error storing ESG data: {e}")
            return False
    
    async def get_historical_esg_trends(self, symbols: List[str], days: int = 90) -> Dict[str, List[Dict[str, Any]]]:
        """Get historical ESG trends for companies.
        
        Args:
            symbols: List of stock symbols
            days: Number of days of history to retrieve
            
        Returns:
            Dictionary mapping symbols to historical ESG data
        """
        try:
            self.logger.info(f"Retrieving {days} days of ESG trends for {len(symbols)} companies")
            
            trends = {}
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=days)
            
            for symbol in symbols:
                try:
                    # Query historical data from database
                    historical_data = await self.db_manager.query_data(
                        'esg_scores',
                        filters={
                            'symbol': symbol,
                            'timestamp': {'$gte': start_date, '$lte': end_date}
                        },
                        sort=[('timestamp', 1)]
                    )
                    
                    if historical_data:
                        trends[symbol] = historical_data
                    
                except Exception as e:
                    self.logger.error(f"Error retrieving ESG trends for {symbol}: {e}")
            
            self.logger.info(f"Retrieved ESG trends for {len(trends)} companies")
            return trends
            
        except Exception as e:
            self.logger.error(f"Error retrieving ESG trends: {e}")
            return {}
    
    def calculate_esg_momentum(self, historical_trends: Dict[str, List[Dict[str, Any]]]) -> Dict[str, float]:
        """Calculate ESG momentum scores based on historical trends.
        
        Args:
            historical_trends: Historical ESG data
            
        Returns:
            Dictionary mapping symbols to momentum scores
        """
        try:
            momentum_scores = {}
            
            for symbol, trend_data in historical_trends.items():
                if len(trend_data) < 2:
                    momentum_scores[symbol] = 0.0
                    continue
                
                # Calculate momentum based on ESG score changes
                scores = [data['overall_esg_score'] for data in trend_data]
                
                # Simple momentum: recent average vs older average
                if len(scores) >= 4:
                    recent_avg = np.mean(scores[-2:])
                    older_avg = np.mean(scores[:2])
                    momentum = (recent_avg - older_avg) / older_avg if older_avg > 0 else 0
                else:
                    momentum = (scores[-1] - scores[0]) / scores[0] if scores[0] > 0 else 0
                
                momentum_scores[symbol] = momentum
            
            return momentum_scores
            
        except Exception as e:
            self.logger.error(f"Error calculating ESG momentum: {e}")
            return {}
    
    async def cleanup_old_data(self, days_to_keep: int = 365) -> bool:
        """Clean up old ESG data from database.
        
        Args:
            days_to_keep: Number of days of data to keep
            
        Returns:
            True if successful
        """
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)
            
            deleted_count = await self.db_manager.delete_data(
                'esg_scores',
                filters={'timestamp': {'$lt': cutoff_date}}
            )
            
            self.logger.info(f"Cleaned up {deleted_count} old ESG records")
            return True
            
        except Exception as e:
            self.logger.error(f"Error cleaning up ESG data: {e}")
            return False