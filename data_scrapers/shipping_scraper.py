#!/usr/bin/env python3
"""
Shipping Data Scraper
Collects vessel tracking, shipping routes, and maritime traffic data.

Data Sources:
- Marine Traffic API for vessel positions and details
- Vessel Finder for additional shipping information
- AIS (Automatic Identification System) data

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
from pathlib import Path

from utils.rate_limiter import RateLimiter, RateLimit
from utils.database import DatabaseManager


@dataclass
class VesselData:
    """Data structure for vessel information."""
    mmsi: str  # Maritime Mobile Service Identity
    imo: str   # International Maritime Organization number
    vessel_name: str
    vessel_type: str
    flag: str
    latitude: float
    longitude: float
    speed: float
    course: float
    destination: str
    eta: Optional[datetime]
    draught: float
    length: int
    width: int
    timestamp: datetime
    cargo_capacity: Optional[float]
    current_port: Optional[str]
    next_port: Optional[str]


@dataclass
class ShippingRoute:
    """Data structure for shipping route information."""
    route_id: str
    origin_port: str
    destination_port: str
    distance_nm: float  # nautical miles
    avg_transit_time: float  # hours
    vessel_count: int
    congestion_level: float  # 0-1 scale
    freight_rate: Optional[float]
    route_risk_score: float


class ShippingScraper:
    """Scraper for collecting shipping and vessel data."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the shipping scraper.
        
        Args:
            config: Configuration dictionary containing API keys and settings
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # API configurations
        self.marine_traffic_config = config.get('marine_traffic', {})
        self.vessel_finder_config = config.get('vessel_finder', {})
        
        # Rate limiters
        marine_traffic_rate = RateLimit(
            requests_per_second=self.marine_traffic_config.get('rate_limit', 100) / 60,
            requests_per_minute=self.marine_traffic_config.get('rate_limit', 100),
            burst_size=10
        )
        vessel_finder_rate = RateLimit(
            requests_per_second=self.vessel_finder_config.get('rate_limit', 60) / 60,
            requests_per_minute=self.vessel_finder_config.get('rate_limit', 60),
            burst_size=10
        )
        self.marine_traffic_limiter = RateLimiter(marine_traffic_rate)
        self.vessel_finder_limiter = RateLimiter(vessel_finder_rate)
        
        # Database manager
        self.db_manager = DatabaseManager()
        
        # Key shipping routes to monitor
        self.key_routes = [
            {'origin': 'Shanghai', 'destination': 'Los Angeles'},
            {'origin': 'Shanghai', 'destination': 'Long Beach'},
            {'origin': 'Shenzhen', 'destination': 'Los Angeles'},
            {'origin': 'Singapore', 'destination': 'Rotterdam'},
            {'origin': 'Dubai', 'destination': 'Hamburg'},
            {'origin': 'Hong Kong', 'destination': 'New York'},
            {'origin': 'Busan', 'destination': 'Tacoma'},
            {'origin': 'Ningbo', 'destination': 'Savannah'}
        ]
    
    async def collect_data(self) -> Dict[str, Any]:
        """Main method to collect all shipping data.
        
        Returns:
            Dictionary containing collected shipping data
        """
        self.logger.info("Starting shipping data collection")
        
        try:
            # Collect vessel positions and details
            vessel_data = await self._collect_vessel_data()
            
            # Collect shipping route information
            route_data = await self._collect_route_data()
            
            # Calculate congestion metrics
            congestion_data = await self._calculate_congestion_metrics(vessel_data)
            
            # Store data in database
            await self._store_data(vessel_data, route_data, congestion_data)
            
            result = {
                'vessels': len(vessel_data),
                'routes': len(route_data),
                'congestion_points': len(congestion_data),
                'timestamp': datetime.utcnow(),
                'status': 'success'
            }
            
            self.logger.info(f"Shipping data collection completed: {result}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in shipping data collection: {e}")
            raise
    
    async def _collect_vessel_data(self) -> List[VesselData]:
        """Collect vessel position and details data.
        
        Returns:
            List of VesselData objects
        """
        vessels = []
        
        async with aiohttp.ClientSession() as session:
            # Get vessel data from Marine Traffic
            mt_vessels = await self._get_marine_traffic_vessels(session)
            vessels.extend(mt_vessels)
            
            # Get additional vessel data from Vessel Finder
            vf_vessels = await self._get_vessel_finder_vessels(session)
            vessels.extend(vf_vessels)
        
        # Remove duplicates based on MMSI
        unique_vessels = {}
        for vessel in vessels:
            if vessel.mmsi not in unique_vessels:
                unique_vessels[vessel.mmsi] = vessel
        
        return list(unique_vessels.values())
    
    async def _get_marine_traffic_vessels(self, session: aiohttp.ClientSession) -> List[VesselData]:
        """Get vessel data from Marine Traffic API.
        
        Args:
            session: HTTP session for making requests
            
        Returns:
            List of VesselData objects
        """
        vessels = []
        base_url = self.marine_traffic_config.get('base_url')
        api_key = self.marine_traffic_config.get('api_key')
        
        if not base_url or not api_key:
            self.logger.warning("Marine Traffic API not configured")
            return vessels
        
        # Define areas of interest (major shipping lanes)
        areas = [
            {'name': 'Suez Canal', 'minlat': 29.5, 'maxlat': 31.5, 'minlon': 32.0, 'maxlon': 33.0},
            {'name': 'Panama Canal', 'minlat': 8.5, 'maxlat': 9.5, 'minlon': -80.0, 'maxlon': -79.0},
            {'name': 'Strait of Malacca', 'minlat': 1.0, 'maxlat': 6.0, 'minlon': 100.0, 'maxlon': 105.0},
            {'name': 'English Channel', 'minlat': 49.5, 'maxlat': 51.5, 'minlon': -2.0, 'maxlon': 2.0}
        ]
        
        for area in areas:
            await self.marine_traffic_limiter.acquire()
            
            params = {
                'v': '2',
                'key': api_key,
                'minlat': area['minlat'],
                'maxlat': area['maxlat'],
                'minlon': area['minlon'],
                'maxlon': area['maxlon'],
                'msg_type': 1  # Position reports
            }
            
            try:
                url = f"{base_url}/exportvessels"
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        area_vessels = self._parse_marine_traffic_data(data, area['name'])
                        vessels.extend(area_vessels)
                        self.logger.info(f"Collected {len(area_vessels)} vessels from {area['name']}")
                    else:
                        self.logger.error(f"Marine Traffic API error for {area['name']}: {response.status}")
            
            except Exception as e:
                self.logger.error(f"Error fetching Marine Traffic data for {area['name']}: {e}")
        
        return vessels
    
    async def _get_vessel_finder_vessels(self, session: aiohttp.ClientSession) -> List[VesselData]:
        """Get vessel data from Vessel Finder API.
        
        Args:
            session: HTTP session for making requests
            
        Returns:
            List of VesselData objects
        """
        vessels = []
        base_url = self.vessel_finder_config.get('base_url')
        api_key = self.vessel_finder_config.get('api_key')
        
        if not base_url or not api_key:
            self.logger.warning("Vessel Finder API not configured")
            return vessels
        
        # Get vessels in major ports
        major_ports = [
            {'name': 'Shanghai', 'lat': 31.2304, 'lon': 121.4737},
            {'name': 'Singapore', 'lat': 1.3521, 'lon': 103.8198},
            {'name': 'Rotterdam', 'lat': 51.9244, 'lon': 4.4777},
            {'name': 'Los Angeles', 'lat': 33.7701, 'lon': -118.1937}
        ]
        
        for port in major_ports:
            await self.vessel_finder_limiter.acquire()
            
            params = {
                'userkey': api_key,
                'lat': port['lat'],
                'lon': port['lon'],
                'radius': 50,  # 50 km radius
                'format': 'json'
            }
            
            try:
                url = f"{base_url}/vessels"
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        port_vessels = self._parse_vessel_finder_data(data, port['name'])
                        vessels.extend(port_vessels)
                        self.logger.info(f"Collected {len(port_vessels)} vessels from {port['name']}")
                    else:
                        self.logger.error(f"Vessel Finder API error for {port['name']}: {response.status}")
            
            except Exception as e:
                self.logger.error(f"Error fetching Vessel Finder data for {port['name']}: {e}")
        
        return vessels
    
    def _parse_marine_traffic_data(self, data: Dict, area_name: str) -> List[VesselData]:
        """Parse Marine Traffic API response data.
        
        Args:
            data: API response data
            area_name: Name of the area being processed
            
        Returns:
            List of VesselData objects
        """
        vessels = []
        
        try:
            for vessel_info in data.get('data', []):
                vessel = VesselData(
                    mmsi=str(vessel_info.get('MMSI', '')),
                    imo=str(vessel_info.get('IMO', '')),
                    vessel_name=vessel_info.get('SHIPNAME', ''),
                    vessel_type=vessel_info.get('TYPE_NAME', ''),
                    flag=vessel_info.get('FLAG', ''),
                    latitude=float(vessel_info.get('LAT', 0)),
                    longitude=float(vessel_info.get('LON', 0)),
                    speed=float(vessel_info.get('SPEED', 0)),
                    course=float(vessel_info.get('COURSE', 0)),
                    destination=vessel_info.get('DESTINATION', ''),
                    eta=self._parse_eta(vessel_info.get('ETA')),
                    draught=float(vessel_info.get('DRAUGHT', 0)),
                    length=int(vessel_info.get('LENGTH', 0)),
                    width=int(vessel_info.get('WIDTH', 0)),
                    timestamp=datetime.utcnow(),
                    cargo_capacity=vessel_info.get('DWT'),
                    current_port=area_name,
                    next_port=vessel_info.get('DESTINATION', '')
                )
                vessels.append(vessel)
        
        except Exception as e:
            self.logger.error(f"Error parsing Marine Traffic data: {e}")
        
        return vessels
    
    def _parse_vessel_finder_data(self, data: Dict, port_name: str) -> List[VesselData]:
        """Parse Vessel Finder API response data.
        
        Args:
            data: API response data
            port_name: Name of the port being processed
            
        Returns:
            List of VesselData objects
        """
        vessels = []
        
        try:
            for vessel_info in data.get('vessels', []):
                vessel = VesselData(
                    mmsi=str(vessel_info.get('mmsi', '')),
                    imo=str(vessel_info.get('imo', '')),
                    vessel_name=vessel_info.get('name', ''),
                    vessel_type=vessel_info.get('type', ''),
                    flag=vessel_info.get('flag', ''),
                    latitude=float(vessel_info.get('lat', 0)),
                    longitude=float(vessel_info.get('lon', 0)),
                    speed=float(vessel_info.get('speed', 0)),
                    course=float(vessel_info.get('course', 0)),
                    destination=vessel_info.get('destination', ''),
                    eta=self._parse_eta(vessel_info.get('eta')),
                    draught=float(vessel_info.get('draught', 0)),
                    length=int(vessel_info.get('length', 0)),
                    width=int(vessel_info.get('width', 0)),
                    timestamp=datetime.utcnow(),
                    cargo_capacity=vessel_info.get('dwt'),
                    current_port=port_name,
                    next_port=vessel_info.get('destination', '')
                )
                vessels.append(vessel)
        
        except Exception as e:
            self.logger.error(f"Error parsing Vessel Finder data: {e}")
        
        return vessels
    
    def _parse_eta(self, eta_str: Optional[str]) -> Optional[datetime]:
        """Parse ETA string to datetime object.
        
        Args:
            eta_str: ETA string from API
            
        Returns:
            Parsed datetime or None
        """
        if not eta_str:
            return None
        
        try:
            # Handle different ETA formats
            if isinstance(eta_str, str):
                # Try common formats
                formats = ['%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%S', '%m/%d/%Y %H:%M']
                for fmt in formats:
                    try:
                        return datetime.strptime(eta_str, fmt)
                    except ValueError:
                        continue
        except Exception as e:
            self.logger.warning(f"Could not parse ETA: {eta_str}, error: {e}")
        
        return None
    
    async def _collect_route_data(self) -> List[ShippingRoute]:
        """Collect shipping route information.
        
        Returns:
            List of ShippingRoute objects
        """
        routes = []
        
        for route_config in self.key_routes:
            route = await self._analyze_route(
                route_config['origin'],
                route_config['destination']
            )
            if route:
                routes.append(route)
        
        return routes
    
    async def _analyze_route(self, origin: str, destination: str) -> Optional[ShippingRoute]:
        """Analyze a specific shipping route.
        
        Args:
            origin: Origin port name
            destination: Destination port name
            
        Returns:
            ShippingRoute object or None
        """
        try:
            # This would typically involve:
            # 1. Querying historical vessel movements
            # 2. Calculating average transit times
            # 3. Assessing current congestion levels
            # 4. Gathering freight rate data
            
            # For now, return a placeholder route
            route = ShippingRoute(
                route_id=f"{origin}_{destination}",
                origin_port=origin,
                destination_port=destination,
                distance_nm=self._calculate_route_distance(origin, destination),
                avg_transit_time=self._estimate_transit_time(origin, destination),
                vessel_count=0,  # Will be calculated from vessel data
                congestion_level=0.5,  # Placeholder
                freight_rate=None,  # Will be filled by freight scraper
                route_risk_score=0.3  # Placeholder
            )
            
            return route
            
        except Exception as e:
            self.logger.error(f"Error analyzing route {origin} -> {destination}: {e}")
            return None
    
    def _calculate_route_distance(self, origin: str, destination: str) -> float:
        """Calculate approximate distance between ports.
        
        Args:
            origin: Origin port name
            destination: Destination port name
            
        Returns:
            Distance in nautical miles
        """
        # Placeholder distances (in practice, use great circle distance)
        distances = {
            ('Shanghai', 'Los Angeles'): 6434,
            ('Shanghai', 'Long Beach'): 6434,
            ('Singapore', 'Rotterdam'): 8288,
            ('Dubai', 'Hamburg'): 6200,
            ('Hong Kong', 'New York'): 8439,
            ('Busan', 'Tacoma'): 4500,
            ('Ningbo', 'Savannah'): 7800
        }
        
        return distances.get((origin, destination), 5000)  # Default distance
    
    def _estimate_transit_time(self, origin: str, destination: str) -> float:
        """Estimate transit time between ports.
        
        Args:
            origin: Origin port name
            destination: Destination port name
            
        Returns:
            Transit time in hours
        """
        distance = self._calculate_route_distance(origin, destination)
        avg_speed = 15  # knots
        return distance / avg_speed
    
    async def _calculate_congestion_metrics(self, vessels: List[VesselData]) -> Dict[str, float]:
        """Calculate congestion metrics for key areas.
        
        Args:
            vessels: List of vessel data
            
        Returns:
            Dictionary of congestion metrics by area
        """
        congestion_data = {}
        
        # Define key congestion areas
        areas = {
            'Suez Canal': {'lat_range': (29.5, 31.5), 'lon_range': (32.0, 33.0)},
            'Panama Canal': {'lat_range': (8.5, 9.5), 'lon_range': (-80.0, -79.0)},
            'Strait of Malacca': {'lat_range': (1.0, 6.0), 'lon_range': (100.0, 105.0)},
            'Los Angeles Port': {'lat_range': (33.5, 34.0), 'lon_range': (-118.5, -118.0)}
        }
        
        for area_name, bounds in areas.items():
            vessels_in_area = [
                v for v in vessels
                if (bounds['lat_range'][0] <= v.latitude <= bounds['lat_range'][1] and
                    bounds['lon_range'][0] <= v.longitude <= bounds['lon_range'][1])
            ]
            
            # Calculate congestion score based on vessel density and speed
            vessel_count = len(vessels_in_area)
            avg_speed = sum(v.speed for v in vessels_in_area) / max(vessel_count, 1)
            
            # Congestion score: higher vessel count and lower speeds indicate congestion
            congestion_score = min(vessel_count / 50.0, 1.0)  # Normalize to 0-1
            if avg_speed < 5:  # Very slow movement indicates congestion
                congestion_score = min(congestion_score * 1.5, 1.0)
            
            congestion_data[area_name] = congestion_score
        
        return congestion_data
    
    async def _store_data(self, vessels: List[VesselData], routes: List[ShippingRoute], 
                         congestion: Dict[str, float]) -> None:
        """Store collected data in database.
        
        Args:
            vessels: List of vessel data
            routes: List of route data
            congestion: Congestion metrics
        """
        try:
            # Convert to DataFrames for easier database insertion
            vessels_df = pd.DataFrame([vars(v) for v in vessels])
            routes_df = pd.DataFrame([vars(r) for r in routes])
            congestion_df = pd.DataFrame([
                {'area': area, 'congestion_score': score, 'timestamp': datetime.utcnow()}
                for area, score in congestion.items()
            ])
            
            # Store in database
            await self.db_manager.store_dataframe(vessels_df, 'vessel_positions')
            await self.db_manager.store_dataframe(routes_df, 'shipping_routes')
            await self.db_manager.store_dataframe(congestion_df, 'congestion_metrics')
            
            self.logger.info("Shipping data stored successfully")
            
        except Exception as e:
            self.logger.error(f"Error storing shipping data: {e}")
            raise