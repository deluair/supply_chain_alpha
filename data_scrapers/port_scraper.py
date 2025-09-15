#!/usr/bin/env python3
"""
Port Congestion Scraper
Collects port throughput, congestion metrics, and operational data.

Data Sources:
- Port authority websites and APIs
- Container terminal data
- Berth availability and utilization
- Cargo handling statistics

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

from utils.rate_limiter import RateLimiter, RateLimit
from utils.database import DatabaseManager


@dataclass
class PortMetrics:
    """Data structure for port operational metrics."""
    port_name: str
    port_code: str
    country: str
    timestamp: datetime
    
    # Throughput metrics
    container_throughput_teu: Optional[float]  # Twenty-foot Equivalent Units
    cargo_volume_tons: Optional[float]
    vessel_arrivals: Optional[int]
    vessel_departures: Optional[int]
    
    # Congestion metrics
    berth_utilization: Optional[float]  # 0-1 scale
    avg_waiting_time: Optional[float]  # hours
    queue_length: Optional[int]  # number of vessels waiting
    congestion_index: Optional[float]  # 0-1 scale
    
    # Operational metrics
    crane_productivity: Optional[float]  # moves per hour
    truck_turnaround_time: Optional[float]  # minutes
    rail_connectivity: Optional[bool]
    storage_utilization: Optional[float]  # 0-1 scale
    
    # Economic indicators
    port_charges: Optional[float]  # USD per TEU
    fuel_availability: Optional[bool]
    labor_availability: Optional[float]  # 0-1 scale


@dataclass
class TerminalData:
    """Data structure for individual terminal information."""
    terminal_name: str
    port_name: str
    operator: str
    capacity_teu: Optional[float]
    current_utilization: Optional[float]
    berth_count: int
    crane_count: int
    max_vessel_size: Optional[int]  # TEU capacity
    specialization: List[str]  # e.g., ['container', 'bulk', 'ro-ro']


class PortScraper:
    """Scraper for collecting port congestion and operational data."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the port scraper.
        
        Args:
            config: Configuration dictionary containing port authorities and settings
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Port authorities configuration
        self.port_authorities = config.get('port_authorities', [])
        
        # Rate limiter for web scraping
        rate_limit_config = RateLimit(
            requests_per_second=0.5,
            requests_per_minute=30,
            burst_size=5
        )
        self.rate_limiter = RateLimiter(rate_limit_config)
        
        # Database manager
        self.db_manager = DatabaseManager()
        
        # Major ports to monitor
        self.major_ports = [
            {
                'name': 'Port of Los Angeles',
                'code': 'USLAX',
                'country': 'USA',
                'url': 'https://www.portoflosangeles.org/business/statistics',
                'api_endpoint': None,
                'scrape_method': 'web'
            },
            {
                'name': 'Port of Long Beach',
                'code': 'USLGB',
                'country': 'USA',
                'url': 'https://www.polb.com/business/port-statistics',
                'api_endpoint': None,
                'scrape_method': 'web'
            },
            {
                'name': 'Port of Shanghai',
                'code': 'CNSHA',
                'country': 'China',
                'url': 'https://www.portshanghai.com.cn/en',
                'api_endpoint': None,
                'scrape_method': 'web'
            },
            {
                'name': 'Port of Singapore',
                'code': 'SGSIN',
                'country': 'Singapore',
                'url': 'https://www.mpa.gov.sg/web/portal/home/port-of-singapore/operations',
                'api_endpoint': None,
                'scrape_method': 'web'
            },
            {
                'name': 'Port of Rotterdam',
                'code': 'NLRTM',
                'country': 'Netherlands',
                'url': 'https://www.portofrotterdam.com/en/our-port/facts-figures',
                'api_endpoint': None,
                'scrape_method': 'web'
            },
            {
                'name': 'Port of Hamburg',
                'code': 'DEHAM',
                'country': 'Germany',
                'url': 'https://www.hafen-hamburg.de/en/statistics',
                'api_endpoint': None,
                'scrape_method': 'web'
            }
        ]
    
    async def collect_data(self) -> Dict[str, Any]:
        """Main method to collect all port data.
        
        Returns:
            Dictionary containing collected port data
        """
        self.logger.info("Starting port data collection")
        
        try:
            # Collect port metrics
            port_metrics = await self._collect_port_metrics()
            
            # Collect terminal data
            terminal_data = await self._collect_terminal_data()
            
            # Calculate congestion indices
            congestion_indices = await self._calculate_congestion_indices(port_metrics)
            
            # Store data in database
            await self._store_data(port_metrics, terminal_data, congestion_indices)
            
            result = {
                'ports_monitored': len(port_metrics),
                'terminals_tracked': len(terminal_data),
                'congestion_alerts': sum(1 for ci in congestion_indices.values() if ci > 0.7),
                'timestamp': datetime.utcnow(),
                'status': 'success'
            }
            
            self.logger.info(f"Port data collection completed: {result}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in port data collection: {e}")
            raise
    
    async def _collect_port_metrics(self) -> List[PortMetrics]:
        """Collect metrics for all monitored ports.
        
        Returns:
            List of PortMetrics objects
        """
        port_metrics = []
        
        async with aiohttp.ClientSession() as session:
            tasks = []
            for port in self.major_ports:
                task = self._scrape_port_data(session, port)
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    self.logger.error(f"Error scraping {self.major_ports[i]['name']}: {result}")
                elif result:
                    port_metrics.append(result)
        
        return port_metrics
    
    async def _scrape_port_data(self, session: aiohttp.ClientSession, 
                               port_config: Dict[str, Any]) -> Optional[PortMetrics]:
        """Scrape data for a specific port.
        
        Args:
            session: HTTP session for making requests
            port_config: Port configuration dictionary
            
        Returns:
            PortMetrics object or None
        """
        await self.rate_limiter.acquire()
        
        try:
            if port_config['scrape_method'] == 'web':
                return await self._scrape_web_data(session, port_config)
            elif port_config['scrape_method'] == 'api':
                return await self._scrape_api_data(session, port_config)
            else:
                self.logger.warning(f"Unknown scrape method for {port_config['name']}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error scraping {port_config['name']}: {e}")
            return None
    
    async def _scrape_web_data(self, session: aiohttp.ClientSession, 
                              port_config: Dict[str, Any]) -> Optional[PortMetrics]:
        """Scrape port data from web pages.
        
        Args:
            session: HTTP session for making requests
            port_config: Port configuration dictionary
            
        Returns:
            PortMetrics object or None
        """
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            async with session.get(port_config['url'], headers=headers) as response:
                if response.status != 200:
                    self.logger.error(f"HTTP {response.status} for {port_config['name']}")
                    return None
                
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                # Parse data based on port-specific patterns
                if 'Los Angeles' in port_config['name']:
                    return self._parse_la_port_data(soup, port_config)
                elif 'Long Beach' in port_config['name']:
                    return self._parse_lb_port_data(soup, port_config)
                elif 'Shanghai' in port_config['name']:
                    return self._parse_shanghai_port_data(soup, port_config)
                elif 'Singapore' in port_config['name']:
                    return self._parse_singapore_port_data(soup, port_config)
                elif 'Rotterdam' in port_config['name']:
                    return self._parse_rotterdam_port_data(soup, port_config)
                elif 'Hamburg' in port_config['name']:
                    return self._parse_hamburg_port_data(soup, port_config)
                else:
                    return self._parse_generic_port_data(soup, port_config)
                    
        except Exception as e:
            self.logger.error(f"Error parsing web data for {port_config['name']}: {e}")
            return None
    
    def _parse_la_port_data(self, soup: BeautifulSoup, 
                           port_config: Dict[str, Any]) -> Optional[PortMetrics]:
        """Parse Port of Los Angeles specific data.
        
        Args:
            soup: BeautifulSoup object of the webpage
            port_config: Port configuration dictionary
            
        Returns:
            PortMetrics object or None
        """
        try:
            # Look for TEU statistics
            teu_pattern = r'([\d,]+)\s*TEU'
            container_throughput = None
            
            # Search for container statistics in text
            text_content = soup.get_text()
            teu_matches = re.findall(teu_pattern, text_content)
            if teu_matches:
                # Take the largest number (likely annual throughput)
                teu_values = [int(match.replace(',', '')) for match in teu_matches]
                container_throughput = max(teu_values) / 12  # Convert to monthly
            
            # Estimate congestion based on available information
            congestion_index = self._estimate_congestion_from_text(text_content)
            
            return PortMetrics(
                port_name=port_config['name'],
                port_code=port_config['code'],
                country=port_config['country'],
                timestamp=datetime.utcnow(),
                container_throughput_teu=container_throughput,
                cargo_volume_tons=None,
                vessel_arrivals=None,
                vessel_departures=None,
                berth_utilization=None,
                avg_waiting_time=None,
                queue_length=None,
                congestion_index=congestion_index,
                crane_productivity=None,
                truck_turnaround_time=None,
                rail_connectivity=True,  # LA port has rail connectivity
                storage_utilization=None,
                port_charges=None,
                fuel_availability=True,
                labor_availability=None
            )
            
        except Exception as e:
            self.logger.error(f"Error parsing LA port data: {e}")
            return None
    
    def _parse_lb_port_data(self, soup: BeautifulSoup, 
                           port_config: Dict[str, Any]) -> Optional[PortMetrics]:
        """Parse Port of Long Beach specific data."""
        # Similar implementation to LA port
        return self._parse_generic_port_data(soup, port_config)
    
    def _parse_shanghai_port_data(self, soup: BeautifulSoup, 
                                 port_config: Dict[str, Any]) -> Optional[PortMetrics]:
        """Parse Port of Shanghai specific data."""
        return self._parse_generic_port_data(soup, port_config)
    
    def _parse_singapore_port_data(self, soup: BeautifulSoup, 
                                  port_config: Dict[str, Any]) -> Optional[PortMetrics]:
        """Parse Port of Singapore specific data."""
        return self._parse_generic_port_data(soup, port_config)
    
    def _parse_rotterdam_port_data(self, soup: BeautifulSoup, 
                                  port_config: Dict[str, Any]) -> Optional[PortMetrics]:
        """Parse Port of Rotterdam specific data."""
        return self._parse_generic_port_data(soup, port_config)
    
    def _parse_hamburg_port_data(self, soup: BeautifulSoup, 
                                port_config: Dict[str, Any]) -> Optional[PortMetrics]:
        """Parse Port of Hamburg specific data."""
        return self._parse_generic_port_data(soup, port_config)
    
    def _parse_generic_port_data(self, soup: BeautifulSoup, 
                                port_config: Dict[str, Any]) -> Optional[PortMetrics]:
        """Parse generic port data when specific parser is not available.
        
        Args:
            soup: BeautifulSoup object of the webpage
            port_config: Port configuration dictionary
            
        Returns:
            PortMetrics object with estimated values
        """
        try:
            text_content = soup.get_text()
            
            # Extract numerical data using regex patterns
            teu_pattern = r'([\d,]+)\s*TEU'
            tons_pattern = r'([\d,]+)\s*(?:tons?|tonnes?)'
            vessel_pattern = r'([\d,]+)\s*(?:vessels?|ships?)'
            
            teu_matches = re.findall(teu_pattern, text_content, re.IGNORECASE)
            tons_matches = re.findall(tons_pattern, text_content, re.IGNORECASE)
            vessel_matches = re.findall(vessel_pattern, text_content, re.IGNORECASE)
            
            # Process matches
            container_throughput = None
            if teu_matches:
                teu_values = [int(match.replace(',', '')) for match in teu_matches]
                container_throughput = max(teu_values) / 12  # Estimate monthly from annual
            
            cargo_volume = None
            if tons_matches:
                ton_values = [int(match.replace(',', '')) for match in tons_matches]
                cargo_volume = max(ton_values) / 12  # Estimate monthly from annual
            
            vessel_count = None
            if vessel_matches:
                vessel_values = [int(match.replace(',', '')) for match in vessel_matches]
                vessel_count = max(vessel_values) / 365  # Estimate daily from annual
            
            # Estimate congestion
            congestion_index = self._estimate_congestion_from_text(text_content)
            
            return PortMetrics(
                port_name=port_config['name'],
                port_code=port_config['code'],
                country=port_config['country'],
                timestamp=datetime.utcnow(),
                container_throughput_teu=container_throughput,
                cargo_volume_tons=cargo_volume,
                vessel_arrivals=vessel_count,
                vessel_departures=vessel_count,
                berth_utilization=0.75,  # Estimated
                avg_waiting_time=24.0,  # Estimated hours
                queue_length=5,  # Estimated
                congestion_index=congestion_index,
                crane_productivity=30.0,  # Estimated moves per hour
                truck_turnaround_time=45.0,  # Estimated minutes
                rail_connectivity=True,
                storage_utilization=0.8,  # Estimated
                port_charges=500.0,  # Estimated USD per TEU
                fuel_availability=True,
                labor_availability=0.9  # Estimated
            )
            
        except Exception as e:
            self.logger.error(f"Error in generic port data parsing: {e}")
            return None
    
    def _estimate_congestion_from_text(self, text: str) -> float:
        """Estimate congestion level from text content.
        
        Args:
            text: Text content to analyze
            
        Returns:
            Congestion index (0-1 scale)
        """
        congestion_keywords = {
            'high': ['congestion', 'delay', 'waiting', 'queue', 'bottleneck', 'backlog'],
            'medium': ['busy', 'active', 'operations', 'traffic'],
            'low': ['smooth', 'efficient', 'fast', 'quick', 'normal']
        }
        
        text_lower = text.lower()
        
        high_score = sum(text_lower.count(word) for word in congestion_keywords['high'])
        medium_score = sum(text_lower.count(word) for word in congestion_keywords['medium'])
        low_score = sum(text_lower.count(word) for word in congestion_keywords['low'])
        
        total_score = high_score + medium_score + low_score
        
        if total_score == 0:
            return 0.5  # Default moderate congestion
        
        # Calculate weighted congestion index
        congestion_index = (high_score * 1.0 + medium_score * 0.5 + low_score * 0.1) / total_score
        return min(congestion_index, 1.0)
    
    async def _scrape_api_data(self, session: aiohttp.ClientSession, 
                              port_config: Dict[str, Any]) -> Optional[PortMetrics]:
        """Scrape port data from API endpoints.
        
        Args:
            session: HTTP session for making requests
            port_config: Port configuration dictionary
            
        Returns:
            PortMetrics object or None
        """
        # Implementation for API-based data collection
        # This would be used when ports provide API access
        pass
    
    async def _collect_terminal_data(self) -> List[TerminalData]:
        """Collect terminal-specific data.
        
        Returns:
            List of TerminalData objects
        """
        terminals = []
        
        # Major container terminals
        terminal_configs = [
            {
                'name': 'APM Terminals Los Angeles',
                'port': 'Port of Los Angeles',
                'operator': 'APM Terminals',
                'capacity': 2400000,  # TEU
                'berths': 8,
                'cranes': 14
            },
            {
                'name': 'Long Beach Container Terminal',
                'port': 'Port of Long Beach',
                'operator': 'LBCT',
                'capacity': 3300000,  # TEU
                'berths': 10,
                'cranes': 18
            },
            {
                'name': 'Shanghai International Port',
                'port': 'Port of Shanghai',
                'operator': 'SIPG',
                'capacity': 47000000,  # TEU
                'berths': 50,
                'cranes': 120
            }
        ]
        
        for config in terminal_configs:
            terminal = TerminalData(
                terminal_name=config['name'],
                port_name=config['port'],
                operator=config['operator'],
                capacity_teu=config['capacity'],
                current_utilization=0.8,  # Estimated
                berth_count=config['berths'],
                crane_count=config['cranes'],
                max_vessel_size=24000,  # TEU
                specialization=['container']
            )
            terminals.append(terminal)
        
        return terminals
    
    async def _calculate_congestion_indices(self, port_metrics: List[PortMetrics]) -> Dict[str, float]:
        """Calculate congestion indices for all ports.
        
        Args:
            port_metrics: List of port metrics
            
        Returns:
            Dictionary mapping port names to congestion indices
        """
        congestion_indices = {}
        
        for metrics in port_metrics:
            # Calculate composite congestion index
            factors = []
            
            if metrics.berth_utilization is not None:
                factors.append(metrics.berth_utilization)
            
            if metrics.avg_waiting_time is not None:
                # Normalize waiting time (assume 48 hours is maximum)
                normalized_wait = min(metrics.avg_waiting_time / 48.0, 1.0)
                factors.append(normalized_wait)
            
            if metrics.queue_length is not None:
                # Normalize queue length (assume 20 vessels is maximum)
                normalized_queue = min(metrics.queue_length / 20.0, 1.0)
                factors.append(normalized_queue)
            
            if metrics.storage_utilization is not None:
                factors.append(metrics.storage_utilization)
            
            if metrics.congestion_index is not None:
                factors.append(metrics.congestion_index)
            
            # Calculate weighted average
            if factors:
                congestion_index = sum(factors) / len(factors)
            else:
                congestion_index = 0.5  # Default moderate congestion
            
            congestion_indices[metrics.port_name] = congestion_index
        
        return congestion_indices
    
    async def _store_data(self, port_metrics: List[PortMetrics], 
                         terminal_data: List[TerminalData],
                         congestion_indices: Dict[str, float]) -> None:
        """Store collected data in database.
        
        Args:
            port_metrics: List of port metrics
            terminal_data: List of terminal data
            congestion_indices: Congestion indices by port
        """
        try:
            # Convert to DataFrames
            ports_df = pd.DataFrame([vars(pm) for pm in port_metrics])
            terminals_df = pd.DataFrame([vars(td) for td in terminal_data])
            congestion_df = pd.DataFrame([
                {'port_name': port, 'congestion_index': index, 'timestamp': datetime.utcnow()}
                for port, index in congestion_indices.items()
            ])
            
            # Store in database
            await self.db_manager.store_dataframe(ports_df, 'port_metrics')
            await self.db_manager.store_dataframe(terminals_df, 'terminal_data')
            await self.db_manager.store_dataframe(congestion_df, 'port_congestion_indices')
            
            self.logger.info("Port data stored successfully")
            
        except Exception as e:
            self.logger.error(f"Error storing port data: {e}")
            raise