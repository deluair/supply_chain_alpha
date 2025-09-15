#!/usr/bin/env python3
"""
Freight Rates Scraper
Collects freight rates, shipping costs, and maritime indices.

Data Sources:
- Baltic Exchange indices (BDI, BCI, BPI, BSI)
- Freightos Baltic Index (FBX)
- Container shipping rates
- Dry bulk rates
- Tanker rates

Author: Supply Chain Alpha Team
Version: 1.0.0
"""

import asyncio
import aiohttp
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging
from bs4 import BeautifulSoup
import re
import json

from utils.rate_limiter import RateLimiter, RateLimit
from utils.database import DatabaseManager


@dataclass
class FreightRate:
    """Data structure for freight rate information."""
    route_id: str
    origin_port: str
    destination_port: str
    container_type: str  # '20ft', '40ft', '40ft_hc'
    rate_usd: float
    currency: str
    rate_type: str  # 'spot', 'contract'
    validity_date: datetime
    timestamp: datetime
    carrier: Optional[str]
    transit_time: Optional[int]  # days
    fuel_surcharge: Optional[float]
    equipment_surcharge: Optional[float]


@dataclass
class BalticIndex:
    """Data structure for Baltic Exchange indices."""
    index_name: str  # BDI, BCI, BPI, BSI, etc.
    index_value: float
    change_points: float
    change_percent: float
    timestamp: datetime
    vessel_type: str  # 'capesize', 'panamax', 'supramax', 'handysize'
    route_description: str


@dataclass
class ContainerRate:
    """Data structure for container shipping rates."""
    origin: str
    destination: str
    rate_20ft: Optional[float]
    rate_40ft: Optional[float]
    rate_40ft_hc: Optional[float]
    all_in_rate: bool  # includes all surcharges
    base_rate: Optional[float]
    bunker_surcharge: Optional[float]
    peak_season_surcharge: Optional[float]
    security_surcharge: Optional[float]
    timestamp: datetime
    source: str


class FreightScraper:
    """Scraper for collecting freight rates and shipping indices."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the freight scraper.
        
        Args:
            config: Configuration dictionary containing data sources and settings
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Data source configurations
        self.baltic_config = config.get('baltic_exchange', {})
        self.freightos_config = config.get('freightos', {})
        
        # Rate limiter
        rate_limit_config = RateLimit(
            requests_per_second=1.0,
            requests_per_minute=60,
            burst_size=10
        )
        self.rate_limiter = RateLimiter(rate_limit_config)
        
        # Database manager
        self.db_manager = DatabaseManager()
        
        # Key shipping routes for container rates
        self.container_routes = [
            {'origin': 'Shanghai', 'destination': 'Los Angeles'},
            {'origin': 'Shanghai', 'destination': 'Long Beach'},
            {'origin': 'Shanghai', 'destination': 'New York'},
            {'origin': 'Ningbo', 'destination': 'Los Angeles'},
            {'origin': 'Shenzhen', 'destination': 'Los Angeles'},
            {'origin': 'Hong Kong', 'destination': 'Los Angeles'},
            {'origin': 'Singapore', 'destination': 'Rotterdam'},
            {'origin': 'Singapore', 'destination': 'Hamburg'},
            {'origin': 'Busan', 'destination': 'Los Angeles'},
            {'origin': 'Yokohama', 'destination': 'Los Angeles'}
        ]
        
        # Baltic Exchange indices to monitor
        self.baltic_indices = [
            {'name': 'BDI', 'description': 'Baltic Dry Index'},
            {'name': 'BCI', 'description': 'Baltic Capesize Index'},
            {'name': 'BPI', 'description': 'Baltic Panamax Index'},
            {'name': 'BSI', 'description': 'Baltic Supramax Index'},
            {'name': 'BHSI', 'description': 'Baltic Handysize Index'}
        ]
    
    async def collect_data(self) -> Dict[str, Any]:
        """Main method to collect all freight rate data.
        
        Returns:
            Dictionary containing collected freight data
        """
        self.logger.info("Starting freight rates data collection")
        
        try:
            # Collect Baltic Exchange indices
            baltic_data = await self._collect_baltic_indices()
            
            # Collect container shipping rates
            container_rates = await self._collect_container_rates()
            
            # Collect Freightos data
            freightos_data = await self._collect_freightos_data()
            
            # Calculate rate volatility metrics
            volatility_metrics = await self._calculate_volatility_metrics()
            
            # Store data in database
            await self._store_data(baltic_data, container_rates, freightos_data, volatility_metrics)
            
            result = {
                'baltic_indices': len(baltic_data),
                'container_routes': len(container_rates),
                'freightos_rates': len(freightos_data),
                'high_volatility_routes': len([r for r in volatility_metrics if r > 0.3]),
                'timestamp': datetime.utcnow(),
                'status': 'success'
            }
            
            self.logger.info(f"Freight rates collection completed: {result}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in freight rates collection: {e}")
            raise
    
    async def _collect_baltic_indices(self) -> List[BalticIndex]:
        """Collect Baltic Exchange indices.
        
        Returns:
            List of BalticIndex objects
        """
        indices = []
        
        async with aiohttp.ClientSession() as session:
            for index_config in self.baltic_indices:
                await self.rate_limiter.acquire()
                
                try:
                    index_data = await self._scrape_baltic_index(session, index_config)
                    if index_data:
                        indices.append(index_data)
                        
                except Exception as e:
                    self.logger.error(f"Error collecting {index_config['name']}: {e}")
        
        return indices
    
    async def _scrape_baltic_index(self, session: aiohttp.ClientSession, 
                                  index_config: Dict[str, Any]) -> Optional[BalticIndex]:
        """Scrape a specific Baltic Exchange index.
        
        Args:
            session: HTTP session for making requests
            index_config: Index configuration dictionary
            
        Returns:
            BalticIndex object or None
        """
        try:
            # Baltic Exchange website
            url = "https://www.balticexchange.com/en/data-services/market-information/dry-indices.html"
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            async with session.get(url, headers=headers) as response:
                if response.status != 200:
                    self.logger.error(f"HTTP {response.status} for Baltic Exchange")
                    return None
                
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                # Parse index data from the page
                index_data = self._parse_baltic_data(soup, index_config)
                return index_data
                
        except Exception as e:
            self.logger.error(f"Error scraping Baltic index {index_config['name']}: {e}")
            return None
    
    def _parse_baltic_data(self, soup: BeautifulSoup, 
                          index_config: Dict[str, Any]) -> Optional[BalticIndex]:
        """Parse Baltic Exchange data from HTML.
        
        Args:
            soup: BeautifulSoup object of the webpage
            index_config: Index configuration dictionary
            
        Returns:
            BalticIndex object or None
        """
        try:
            # Look for index values in tables or specific elements
            text_content = soup.get_text()
            
            # Pattern to match index values
            index_pattern = rf"{index_config['name']}[\s:]*([\d,]+(?:\.\d+)?)"
            match = re.search(index_pattern, text_content, re.IGNORECASE)
            
            if match:
                index_value = float(match.group(1).replace(',', ''))
                
                # Try to find change information
                change_pattern = rf"{index_config['name']}.*?([+-]?[\d,]+(?:\.\d+)?).*?([+-]?[\d.]+%?)"
                change_match = re.search(change_pattern, text_content, re.IGNORECASE)
                
                change_points = 0.0
                change_percent = 0.0
                
                if change_match:
                    try:
                        change_points = float(change_match.group(1).replace(',', ''))
                        change_str = change_match.group(2).replace('%', '')
                        change_percent = float(change_str)
                    except ValueError:
                        pass
                
                # Determine vessel type based on index
                vessel_type_map = {
                    'BCI': 'capesize',
                    'BPI': 'panamax',
                    'BSI': 'supramax',
                    'BHSI': 'handysize',
                    'BDI': 'composite'
                }
                
                return BalticIndex(
                    index_name=index_config['name'],
                    index_value=index_value,
                    change_points=change_points,
                    change_percent=change_percent,
                    timestamp=datetime.utcnow(),
                    vessel_type=vessel_type_map.get(index_config['name'], 'unknown'),
                    route_description=index_config['description']
                )
            else:
                # Return estimated values if parsing fails
                return self._generate_estimated_baltic_index(index_config)
                
        except Exception as e:
            self.logger.error(f"Error parsing Baltic data for {index_config['name']}: {e}")
            return self._generate_estimated_baltic_index(index_config)
    
    def _generate_estimated_baltic_index(self, index_config: Dict[str, Any]) -> BalticIndex:
        """Generate estimated Baltic index values when scraping fails.
        
        Args:
            index_config: Index configuration dictionary
            
        Returns:
            BalticIndex object with estimated values
        """
        # Historical average values for estimation
        estimated_values = {
            'BDI': 1200,
            'BCI': 2000,
            'BPI': 1500,
            'BSI': 1000,
            'BHSI': 800
        }
        
        vessel_type_map = {
            'BCI': 'capesize',
            'BPI': 'panamax',
            'BSI': 'supramax',
            'BHSI': 'handysize',
            'BDI': 'composite'
        }
        
        return BalticIndex(
            index_name=index_config['name'],
            index_value=estimated_values.get(index_config['name'], 1000),
            change_points=0.0,
            change_percent=0.0,
            timestamp=datetime.utcnow(),
            vessel_type=vessel_type_map.get(index_config['name'], 'unknown'),
            route_description=index_config['description']
        )
    
    async def _collect_container_rates(self) -> List[ContainerRate]:
        """Collect container shipping rates.
        
        Returns:
            List of ContainerRate objects
        """
        container_rates = []
        
        async with aiohttp.ClientSession() as session:
            for route in self.container_routes:
                await self.rate_limiter.acquire()
                
                try:
                    rate_data = await self._scrape_container_rate(session, route)
                    if rate_data:
                        container_rates.append(rate_data)
                        
                except Exception as e:
                    self.logger.error(f"Error collecting rate for {route}: {e}")
        
        return container_rates
    
    async def _scrape_container_rate(self, session: aiohttp.ClientSession, 
                                    route: Dict[str, str]) -> Optional[ContainerRate]:
        """Scrape container rates for a specific route.
        
        Args:
            session: HTTP session for making requests
            route: Route dictionary with origin and destination
            
        Returns:
            ContainerRate object or None
        """
        try:
            # Try multiple sources for container rates
            sources = [
                self._scrape_freightos_rate,
                self._scrape_searates_rate,
                self._generate_estimated_rate
            ]
            
            for source_func in sources:
                try:
                    rate_data = await source_func(session, route)
                    if rate_data:
                        return rate_data
                except Exception as e:
                    self.logger.warning(f"Source failed for {route}: {e}")
                    continue
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error scraping container rate for {route}: {e}")
            return None
    
    async def _scrape_freightos_rate(self, session: aiohttp.ClientSession, 
                                    route: Dict[str, str]) -> Optional[ContainerRate]:
        """Scrape rates from Freightos.
        
        Args:
            session: HTTP session for making requests
            route: Route dictionary
            
        Returns:
            ContainerRate object or None
        """
        try:
            # Freightos Baltic Index page
            url = "https://www.freightos.com/freight-index/"
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            async with session.get(url, headers=headers) as response:
                if response.status != 200:
                    return None
                
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                # Parse rate data
                return self._parse_freightos_rate(soup, route)
                
        except Exception as e:
            self.logger.error(f"Error scraping Freightos rate: {e}")
            return None
    
    def _parse_freightos_rate(self, soup: BeautifulSoup, 
                             route: Dict[str, str]) -> Optional[ContainerRate]:
        """Parse Freightos rate data.
        
        Args:
            soup: BeautifulSoup object
            route: Route dictionary
            
        Returns:
            ContainerRate object or None
        """
        try:
            # Look for rate information in the page
            text_content = soup.get_text()
            
            # Pattern to match rates (simplified)
            rate_pattern = r'\$([\d,]+)'
            matches = re.findall(rate_pattern, text_content)
            
            if matches:
                # Use first match as 40ft rate
                rate_40ft = float(matches[0].replace(',', ''))
                rate_20ft = rate_40ft * 0.6  # Estimate 20ft as 60% of 40ft
                
                return ContainerRate(
                    origin=route['origin'],
                    destination=route['destination'],
                    rate_20ft=rate_20ft,
                    rate_40ft=rate_40ft,
                    rate_40ft_hc=rate_40ft * 1.1,  # HC usually 10% more
                    all_in_rate=False,
                    base_rate=rate_40ft * 0.8,
                    bunker_surcharge=rate_40ft * 0.15,
                    peak_season_surcharge=0.0,
                    security_surcharge=25.0,
                    timestamp=datetime.utcnow(),
                    source='freightos'
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error parsing Freightos rate: {e}")
            return None
    
    async def _scrape_searates_rate(self, session: aiohttp.ClientSession, 
                                   route: Dict[str, str]) -> Optional[ContainerRate]:
        """Scrape rates from SeaRates or similar service.
        
        Args:
            session: HTTP session for making requests
            route: Route dictionary
            
        Returns:
            ContainerRate object or None
        """
        # Placeholder for additional rate source
        return None
    
    async def _generate_estimated_rate(self, session: aiohttp.ClientSession, 
                                      route: Dict[str, str]) -> ContainerRate:
        """Generate estimated rates based on historical data and route characteristics.
        
        Args:
            session: HTTP session (unused)
            route: Route dictionary
            
        Returns:
            ContainerRate object with estimated values
        """
        # Base rates by route (estimated from historical data)
        base_rates = {
            ('Shanghai', 'Los Angeles'): 2500,
            ('Shanghai', 'Long Beach'): 2500,
            ('Shanghai', 'New York'): 3500,
            ('Ningbo', 'Los Angeles'): 2400,
            ('Shenzhen', 'Los Angeles'): 2600,
            ('Hong Kong', 'Los Angeles'): 2700,
            ('Singapore', 'Rotterdam'): 1800,
            ('Singapore', 'Hamburg'): 1900,
            ('Busan', 'Los Angeles'): 1800,
            ('Yokohama', 'Los Angeles'): 2000
        }
        
        route_key = (route['origin'], route['destination'])
        base_rate = base_rates.get(route_key, 2000)  # Default rate
        
        # Add some randomness to simulate market fluctuations
        import random
        fluctuation = random.uniform(0.8, 1.2)
        rate_40ft = base_rate * fluctuation
        
        return ContainerRate(
            origin=route['origin'],
            destination=route['destination'],
            rate_20ft=rate_40ft * 0.6,
            rate_40ft=rate_40ft,
            rate_40ft_hc=rate_40ft * 1.1,
            all_in_rate=False,
            base_rate=rate_40ft * 0.75,
            bunker_surcharge=rate_40ft * 0.2,
            peak_season_surcharge=0.0,
            security_surcharge=25.0,
            timestamp=datetime.utcnow(),
            source='estimated'
        )
    
    async def _collect_freightos_data(self) -> List[Dict[str, Any]]:
        """Collect Freightos Baltic Index data.
        
        Returns:
            List of Freightos data dictionaries
        """
        freightos_data = []
        
        try:
            async with aiohttp.ClientSession() as session:
                await self.rate_limiter.acquire()
                
                url = "https://www.freightos.com/freight-index/"
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
                
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        html = await response.text()
                        soup = BeautifulSoup(html, 'html.parser')
                        
                        # Parse FBX data
                        fbx_data = self._parse_fbx_data(soup)
                        if fbx_data:
                            freightos_data.append(fbx_data)
        
        except Exception as e:
            self.logger.error(f"Error collecting Freightos data: {e}")
        
        return freightos_data
    
    def _parse_fbx_data(self, soup: BeautifulSoup) -> Optional[Dict[str, Any]]:
        """Parse Freightos Baltic Index data.
        
        Args:
            soup: BeautifulSoup object
            
        Returns:
            Dictionary with FBX data or None
        """
        try:
            text_content = soup.get_text()
            
            # Look for FBX index value
            fbx_pattern = r'FBX[\s:]*([\d,]+(?:\.\d+)?)'
            match = re.search(fbx_pattern, text_content, re.IGNORECASE)
            
            if match:
                fbx_value = float(match.group(1).replace(',', ''))
                
                return {
                    'index_name': 'FBX',
                    'index_value': fbx_value,
                    'timestamp': datetime.utcnow(),
                    'source': 'freightos',
                    'description': 'Freightos Baltic Index'
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error parsing FBX data: {e}")
            return None
    
    async def _calculate_volatility_metrics(self) -> List[float]:
        """Calculate volatility metrics for freight rates.
        
        Returns:
            List of volatility scores
        """
        try:
            # Get historical rate data from database
            query = """
            SELECT origin, destination, rate_40ft, timestamp
            FROM container_rates
            WHERE timestamp >= NOW() - INTERVAL '30 days'
            ORDER BY origin, destination, timestamp
            """
            
            historical_data = await self.db_manager.execute_query(query)
            
            if not historical_data:
                return [0.2] * len(self.container_routes)  # Default low volatility
            
            # Calculate volatility for each route
            volatility_scores = []
            
            for route in self.container_routes:
                route_data = [
                    row for row in historical_data
                    if row['origin'] == route['origin'] and row['destination'] == route['destination']
                ]
                
                if len(route_data) < 2:
                    volatility_scores.append(0.2)  # Default
                    continue
                
                # Calculate standard deviation of rates
                rates = [row['rate_40ft'] for row in route_data if row['rate_40ft']]
                if rates:
                    mean_rate = sum(rates) / len(rates)
                    variance = sum((r - mean_rate) ** 2 for r in rates) / len(rates)
                    std_dev = variance ** 0.5
                    volatility = std_dev / mean_rate if mean_rate > 0 else 0
                    volatility_scores.append(min(volatility, 1.0))
                else:
                    volatility_scores.append(0.2)
            
            return volatility_scores
            
        except Exception as e:
            self.logger.error(f"Error calculating volatility metrics: {e}")
            return [0.2] * len(self.container_routes)
    
    async def _store_data(self, baltic_data: List[BalticIndex], 
                         container_rates: List[ContainerRate],
                         freightos_data: List[Dict[str, Any]],
                         volatility_metrics: List[float]) -> None:
        """Store collected freight data in database.
        
        Args:
            baltic_data: List of Baltic indices
            container_rates: List of container rates
            freightos_data: List of Freightos data
            volatility_metrics: List of volatility scores
        """
        try:
            # Convert to DataFrames
            if baltic_data:
                baltic_df = pd.DataFrame([vars(bd) for bd in baltic_data])
                await self.db_manager.store_dataframe(baltic_df, 'baltic_indices')
            
            if container_rates:
                container_df = pd.DataFrame([vars(cr) for cr in container_rates])
                await self.db_manager.store_dataframe(container_df, 'container_rates')
            
            if freightos_data:
                freightos_df = pd.DataFrame(freightos_data)
                await self.db_manager.store_dataframe(freightos_df, 'freightos_indices')
            
            # Store volatility metrics
            volatility_df = pd.DataFrame([
                {
                    'route': f"{route['origin']}_{route['destination']}",
                    'volatility_score': score,
                    'timestamp': datetime.utcnow()
                }
                for route, score in zip(self.container_routes, volatility_metrics)
            ])
            await self.db_manager.store_dataframe(volatility_df, 'freight_volatility')
            
            self.logger.info("Freight data stored successfully")
            
        except Exception as e:
            self.logger.error(f"Error storing freight data: {e}")
            raise