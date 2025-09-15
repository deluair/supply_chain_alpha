# API Reference Guide

## Overview

This document provides comprehensive API reference for the Supply Chain Alpha system. The APIs are organized into several modules:

- **Data Scrapers**: Real-time data collection from external sources
- **Models**: Predictive analytics and machine learning models
- **Strategies**: Trading strategy implementation
- **Utils**: Utility functions and helpers

## Data Scrapers API

### ShippingScraper

Collects maritime traffic and vessel data from multiple sources.

#### Class: `ShippingScraper`

```python
from data_scrapers import ShippingScraper

scraper = ShippingScraper(api_config)
```

##### Methods

###### `collect_vessel_data(region, vessel_types, time_range)`

Collects vessel data for specified region and vessel types.

**Parameters:**
- `region` (str): Geographic region (e.g., "North Atlantic", "Mediterranean")
- `vessel_types` (List[str]): List of vessel types to collect
- `time_range` (Tuple[datetime, datetime], optional): Time range for historical data

**Returns:**
- `List[VesselData]`: List of vessel data objects

**Example:**
```python
vessels = await scraper.collect_vessel_data(
    region="North Atlantic",
    vessel_types=["Container Ship", "Bulk Carrier"],
    time_range=(start_date, end_date)
)
```

###### `analyze_shipping_routes(origin_ports, destination_ports, date_range)`

Analyzes shipping routes between specified ports.

**Parameters:**
- `origin_ports` (List[str]): List of origin port codes
- `destination_ports` (List[str]): List of destination port codes
- `date_range` (Tuple[datetime, datetime], optional): Analysis date range

**Returns:**
- `List[ShippingRoute]`: List of shipping route objects

**Example:**
```python
routes = await scraper.analyze_shipping_routes(
    origin_ports=["USNYC", "USLAX"],
    destination_ports=["NLRTM", "DEHAM"]
)
```

###### `calculate_congestion_metrics(routes, time_window)`

Calculates congestion metrics for shipping routes.

**Parameters:**
- `routes` (List[ShippingRoute]): List of routes to analyze
- `time_window` (int): Time window in days

**Returns:**
- `Dict[str, float]`: Congestion metrics by route

### PortScraper

Collects port operational data and performance metrics.

#### Class: `PortScraper`

```python
from data_scrapers import PortScraper

scraper = PortScraper(api_config)
```

##### Methods

###### `collect_port_metrics(ports, metrics, time_range)`

Collects operational metrics for specified ports.

**Parameters:**
- `ports` (List[str]): List of port codes
- `metrics` (List[str]): Metrics to collect ("throughput", "congestion", "waiting_time")
- `time_range` (Tuple[datetime, datetime], optional): Time range for data

**Returns:**
- `List[PortMetrics]`: List of port metrics objects

**Example:**
```python
metrics = await scraper.collect_port_metrics(
    ports=["USNYC", "NLRTM", "SGSIN"],
    metrics=["throughput", "congestion", "waiting_time"]
)
```

###### `calculate_congestion_metrics(port_code, time_window)`

Calculates detailed congestion metrics for a specific port.

**Parameters:**
- `port_code` (str): Port code (e.g., "USNYC")
- `time_window` (int): Analysis window in days

**Returns:**
- `Dict[str, Any]`: Detailed congestion analysis

### FreightScraper

Collects freight rates and shipping cost data.

#### Class: `FreightScraper`

```python
from data_scrapers import FreightScraper

scraper = FreightScraper(api_config)
```

##### Methods

###### `collect_baltic_indices(indices, date_range)`

Collects Baltic Exchange shipping indices.

**Parameters:**
- `indices` (List[str], optional): Specific indices to collect
- `date_range` (Tuple[datetime, datetime], optional): Date range

**Returns:**
- `List[BalticIndex]`: List of Baltic index data

###### `collect_container_rates(routes, container_types)`

Collects container shipping rates for specified routes.

**Parameters:**
- `routes` (List[str]): Shipping routes
- `container_types` (List[str]): Container types ("20ft", "40ft", "40ft_hc")

**Returns:**
- `List[ContainerRate]`: List of container rate data

## Models API

### DisruptionPredictor

Predictive models for supply chain disruptions.

#### Class: `DisruptionPredictor`

```python
from models import DisruptionPredictor

predictor = DisruptionPredictor(model_config)
```

##### Methods

###### `train_port_congestion_model(training_data, validation_data)`

Trains port congestion prediction model.

**Parameters:**
- `training_data` (pd.DataFrame): Training dataset
- `validation_data` (pd.DataFrame, optional): Validation dataset

**Returns:**
- `Dict[str, Any]`: Training results and metrics

###### `predict_disruptions(prediction_horizon, confidence_threshold)`

Generates disruption predictions.

**Parameters:**
- `prediction_horizon` (int): Prediction horizon in days
- `confidence_threshold` (float): Minimum confidence threshold (0-1)

**Returns:**
- `List[DisruptionPrediction]`: List of disruption predictions

**Example:**
```python
predictions = predictor.predict_disruptions(
    prediction_horizon=30,
    confidence_threshold=0.7
)
```

###### `assess_company_impact(disruptions, companies)`

Assesses impact of disruptions on specific companies.

**Parameters:**
- `disruptions` (List[DisruptionPrediction]): Disruption predictions
- `companies` (List[str]): Company symbols to analyze

**Returns:**
- `List[CompanyImpactScore]`: Company impact assessments

### SectorImpactAnalyzer

Analyzes cross-sector impacts of supply chain disruptions.

#### Class: `SectorImpactAnalyzer`

```python
from models import SectorImpactAnalyzer

analyzer = SectorImpactAnalyzer(model_config)
```

##### Methods

###### `assess_sector_impacts(disruptions, sectors)`

Assesses impact on different economic sectors.

**Parameters:**
- `disruptions` (List[DisruptionPrediction]): Disruption predictions
- `sectors` (List[str]): Sectors to analyze

**Returns:**
- `Dict[str, SectorImpactScore]`: Sector impact scores

###### `analyze_cross_sector_propagation(initial_impacts)`

Analyzes how impacts propagate across sectors.

**Parameters:**
- `initial_impacts` (Dict[str, float]): Initial sector impacts

**Returns:**
- `Dict[str, CrossSectorImpact]`: Cross-sector propagation analysis

## Strategies API

### LongShortEquityStrategy

Implements long/short equity trading strategy.

#### Class: `LongShortEquityStrategy`

```python
from strategies import LongShortEquityStrategy

strategy = LongShortEquityStrategy(trading_config)
```

##### Methods

###### `generate_signals(supply_chain_data, market_data, esg_scores)`

Generates trading signals based on supply chain intelligence.

**Parameters:**
- `supply_chain_data` (Dict[str, Any]): Supply chain disruption data
- `market_data` (Dict[str, Any]): Market price and volume data
- `esg_scores` (Dict[str, ESGScore], optional): ESG scores for stocks

**Returns:**
- `List[TradeSignal]`: List of trading signals

**Example:**
```python
signals = strategy.generate_signals(
    supply_chain_data=disruption_data,
    market_data=market_data,
    esg_scores=esg_scores
)
```

###### `execute_strategy(signals, current_positions, market_data)`

Executes trading strategy based on generated signals.

**Parameters:**
- `signals` (List[TradeSignal]): Trading signals
- `current_positions` (Dict[str, Position]): Current portfolio positions
- `market_data` (Dict[str, Any]): Current market data

**Returns:**
- `Dict[str, Position]`: Updated portfolio positions

###### `calculate_performance_metrics(portfolio, benchmark_returns)`

Calculates strategy performance metrics.

**Parameters:**
- `portfolio` (Dict[str, Position]): Portfolio positions
- `benchmark_returns` (pd.Series): Benchmark return series

**Returns:**
- `PortfolioMetrics`: Performance metrics object

## Utils API

### DatabaseManager

Manages database connections and operations.

#### Class: `DatabaseManager`

```python
from utils import DatabaseManager

db_manager = DatabaseManager(database_config)
```

##### Methods

###### `connect(database_type)`

Establishes database connection.

**Parameters:**
- `database_type` (str): Database type ("postgresql", "sqlite", "mongodb")

**Returns:**
- `Any`: Database connection object

###### `store_data(table_name, data, batch_size)`

Stores data in specified table.

**Parameters:**
- `table_name` (str): Target table name
- `data` (List[Dict] or pd.DataFrame): Data to store
- `batch_size` (int, optional): Batch size for bulk operations

**Returns:**
- `bool`: Success status

###### `query_data(query, parameters)`

Executes database query.

**Parameters:**
- `query` (str): SQL query string
- `parameters` (Dict, optional): Query parameters

**Returns:**
- `pd.DataFrame`: Query results

### DataProcessor

Data preprocessing and feature engineering utilities.

#### Class: `DataProcessor`

```python
from utils import DataProcessor

processor = DataProcessor()
```

##### Methods

###### `assess_data_quality(data)`

Assesses data quality and identifies issues.

**Parameters:**
- `data` (pd.DataFrame): Input dataset

**Returns:**
- `DataQualityReport`: Data quality assessment report

###### `clean_data(data, cleaning_config)`

Cleans and preprocesses data.

**Parameters:**
- `data` (pd.DataFrame): Input dataset
- `cleaning_config` (Dict[str, Any]): Cleaning configuration

**Returns:**
- `pd.DataFrame`: Cleaned dataset

###### `engineer_features(data, feature_config)`

Performs feature engineering.

**Parameters:**
- `data` (pd.DataFrame): Input dataset
- `feature_config` (Dict[str, Any]): Feature engineering configuration

**Returns:**
- `pd.DataFrame`: Dataset with engineered features

### ESGMetricsIntegrator

Integrates ESG metrics into investment decisions.

#### Class: `ESGMetricsIntegrator`

```python
from utils import ESGMetricsIntegrator

esg_integrator = ESGMetricsIntegrator(api_config)
```

##### Methods

###### `collect_esg_data(companies, data_sources)`

Collects ESG data for specified companies.

**Parameters:**
- `companies` (List[str]): Company symbols
- `data_sources` (List[str]): ESG data sources to use

**Returns:**
- `Dict[str, ESGScore]`: ESG scores by company

###### `adjust_signals_for_esg(signals, esg_scores, esg_weight)`

Adjusts trading signals based on ESG factors.

**Parameters:**
- `signals` (List[TradeSignal]): Original trading signals
- `esg_scores` (Dict[str, ESGScore]): ESG scores
- `esg_weight` (float): Weight for ESG adjustment (0-1)

**Returns:**
- `List[ESGAdjustedSignal]`: ESG-adjusted signals

## Data Models

### VesselData

```python
@dataclass
class VesselData:
    vessel_id: str
    vessel_name: str
    vessel_type: str
    latitude: float
    longitude: float
    speed: float
    course: float
    destination: str
    eta: datetime
    timestamp: datetime
    source: str
```

### PortMetrics

```python
@dataclass
class PortMetrics:
    port_code: str
    port_name: str
    throughput_teu: int
    congestion_level: float
    avg_waiting_time: float
    berth_utilization: float
    timestamp: datetime
```

### DisruptionPrediction

```python
@dataclass
class DisruptionPrediction:
    prediction_type: str
    target_entity: str
    prediction_value: float
    confidence_score: float
    prediction_date: date
    impact_severity: str
    affected_routes: List[str]
    estimated_duration: int
    mitigation_suggestions: List[str]
```

### TradeSignal

```python
@dataclass
class TradeSignal:
    symbol: str
    signal_type: str  # "BUY", "SELL", "HOLD"
    signal_strength: float
    conviction_score: float
    target_price: Optional[float]
    stop_loss: Optional[float]
    position_size: float
    rationale: str
    generated_at: datetime
    expires_at: datetime
```

### ESGScore

```python
@dataclass
class ESGScore:
    company_symbol: str
    environmental_score: float
    social_score: float
    governance_score: float
    overall_score: float
    score_date: date
    data_source: str
    sector_percentile: float
```

## Error Handling

### Common Exceptions

#### `DataCollectionError`

Raised when data collection fails.

```python
try:
    data = await scraper.collect_vessel_data(region="North Atlantic")
except DataCollectionError as e:
    logger.error(f"Data collection failed: {e}")
```

#### `ModelTrainingError`

Raised when model training fails.

```python
try:
    predictor.train_port_congestion_model(training_data)
except ModelTrainingError as e:
    logger.error(f"Model training failed: {e}")
```

#### `StrategyExecutionError`

Raised when strategy execution fails.

```python
try:
    portfolio = strategy.execute_strategy(signals, positions, market_data)
except StrategyExecutionError as e:
    logger.error(f"Strategy execution failed: {e}")
```

## Rate Limiting

All API calls are subject to rate limiting to ensure fair usage and prevent overloading external services.

### Rate Limit Headers

- `X-RateLimit-Limit`: Maximum requests per time window
- `X-RateLimit-Remaining`: Remaining requests in current window
- `X-RateLimit-Reset`: Time when rate limit resets

### Handling Rate Limits

```python
import asyncio
from data_scrapers import RateLimitExceeded

try:
    data = await scraper.collect_vessel_data(region="North Atlantic")
except RateLimitExceeded as e:
    # Wait for rate limit reset
    await asyncio.sleep(e.retry_after)
    data = await scraper.collect_vessel_data(region="North Atlantic")
```

## Authentication

### API Key Authentication

Most external APIs require API key authentication:

```python
api_config = APIConfig(
    marine_traffic_api_key="your_api_key",
    vessel_finder_api_key="your_api_key"
)
```

### OAuth Authentication

Some APIs use OAuth 2.0:

```python
oauth_config = {
    "client_id": "your_client_id",
    "client_secret": "your_client_secret",
    "redirect_uri": "your_redirect_uri"
}
```

## Pagination

Large datasets are paginated to manage memory usage:

```python
# Paginated data collection
all_data = []
page = 1
page_size = 1000

while True:
    page_data = await scraper.collect_vessel_data(
        region="North Atlantic",
        page=page,
        page_size=page_size
    )
    
    if not page_data:
        break
    
    all_data.extend(page_data)
    page += 1
```

## Caching

Frequently accessed data is cached to improve performance:

```python
# Enable caching
scraper = ShippingScraper(
    api_config,
    cache_enabled=True,
    cache_ttl=3600  # 1 hour
)

# Force cache refresh
data = await scraper.collect_vessel_data(
    region="North Atlantic",
    force_refresh=True
)
```

## Monitoring

### Health Checks

```python
# Check API health
health_status = await scraper.check_health()
print(f"API Status: {health_status['status']}")
print(f"Response Time: {health_status['response_time']}ms")
```

### Performance Metrics

```python
# Get performance metrics
metrics = scraper.get_performance_metrics()
print(f"Total Requests: {metrics['total_requests']}")
print(f"Average Response Time: {metrics['avg_response_time']}ms")
print(f"Error Rate: {metrics['error_rate']}%")
```

## Best Practices

### Error Handling

1. Always use try-catch blocks for API calls
2. Implement exponential backoff for retries
3. Log errors with sufficient context
4. Gracefully handle rate limiting

### Performance

1. Use async/await for concurrent operations
2. Implement proper caching strategies
3. Batch API requests when possible
4. Monitor and optimize database queries

### Security

1. Store API keys securely
2. Use environment variables for sensitive data
3. Implement proper access controls
4. Regularly rotate API keys

### Data Quality

1. Validate input data before processing
2. Implement data quality checks
3. Handle missing or invalid data gracefully
4. Monitor data freshness and accuracy