#!/usr/bin/env python3
"""
Configuration Management System

Centralized configuration management for the Supply Chain Alpha system.
Handles environment variables, API keys, database settings, and application parameters.

Features:
- Environment-specific configurations (dev, staging, prod)
- Secure API key management
- Database connection settings
- Model parameters and hyperparameters
- Logging and monitoring configuration
- Data source configurations

Author: Supply Chain Alpha Team
Version: 1.0.0
"""

import os
import json
import yaml
from typing import Dict, Any, Optional, Union
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from cryptography.fernet import Fernet
import base64


class Environment(Enum):
    """Environment types."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


@dataclass
class DatabaseConfig:
    """Database configuration."""
    type: str = "postgresql"
    host: str = "localhost"
    port: int = 5432
    database: str = "supply_chain_alpha"
    username: str = ""
    password: str = ""
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30
    ssl_mode: str = "prefer"
    
    # SQLite specific
    sqlite_path: Optional[str] = None
    
    # MongoDB specific
    mongodb_uri: Optional[str] = None
    
    # Redis configuration
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None


@dataclass
class APIConfig:
    """API configuration for external data sources."""
    # Marine Traffic API
    marine_traffic_api_key: str = ""
    marine_traffic_base_url: str = "https://api.marinetraffic.com"
    
    # Vessel Finder API
    vessel_finder_api_key: str = ""
    vessel_finder_base_url: str = "https://api.vesselfinder.com"
    
    # Baltic Exchange API
    baltic_exchange_api_key: str = ""
    baltic_exchange_base_url: str = "https://api.balticexchange.com"
    
    # Freightos API
    freightos_api_key: str = ""
    freightos_base_url: str = "https://api.freightos.com"
    
    # Financial data APIs
    alpha_vantage_api_key: str = ""
    quandl_api_key: str = ""
    bloomberg_api_key: str = ""
    
    # ESG data APIs
    msci_esg_api_key: str = ""
    sustainalytics_api_key: str = ""
    refinitiv_esg_api_key: str = ""
    
    # Rate limiting
    rate_limit_requests_per_minute: int = 60
    rate_limit_requests_per_hour: int = 1000
    
    # Timeout settings
    request_timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0


@dataclass
class ModelConfig:
    """Machine learning model configuration."""
    # Disruption prediction models
    port_congestion_model: Dict[str, Any] = None
    route_disruption_model: Dict[str, Any] = None
    rate_volatility_model: Dict[str, Any] = None
    
    # Training parameters
    train_test_split: float = 0.8
    validation_split: float = 0.1
    cross_validation_folds: int = 5
    random_state: int = 42
    
    # Feature engineering
    feature_lag_periods: list = None
    rolling_window_sizes: list = None
    technical_indicators: bool = True
    
    # Model persistence
    model_save_path: str = "models/saved"
    model_versioning: bool = True
    
    def __post_init__(self):
        if self.port_congestion_model is None:
            self.port_congestion_model = {
                'algorithm': 'random_forest',
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2
            }
        
        if self.route_disruption_model is None:
            self.route_disruption_model = {
                'algorithm': 'gradient_boosting',
                'n_estimators': 50,
                'learning_rate': 0.1,
                'max_depth': 6
            }
        
        if self.rate_volatility_model is None:
            self.rate_volatility_model = {
                'algorithm': 'lstm',
                'sequence_length': 30,
                'hidden_units': 64,
                'dropout_rate': 0.2,
                'epochs': 100
            }
        
        if self.feature_lag_periods is None:
            self.feature_lag_periods = [1, 7, 14, 30]
        
        if self.rolling_window_sizes is None:
            self.rolling_window_sizes = [7, 14, 30, 60]


@dataclass
class TradingConfig:
    """Trading strategy configuration."""
    # Portfolio parameters
    max_positions: int = 100
    max_position_size: float = 0.05  # 5% max per position
    target_leverage: float = 1.3
    cash_buffer: float = 0.1
    
    # Risk management
    max_sector_exposure: float = 0.3
    max_single_stock_weight: float = 0.05
    stop_loss_threshold: float = -0.15
    take_profit_threshold: float = 0.25
    var_limit: float = 0.02  # 2% daily VaR limit
    
    # Signal generation
    min_conviction_score: float = 0.6
    signal_decay_days: int = 30
    rebalance_frequency: str = "weekly"
    esg_weight: float = 0.2
    
    # Execution
    slippage_model: str = "linear"
    transaction_costs: float = 0.001  # 10 bps
    market_impact_factor: float = 0.0005
    
    # Backtesting
    backtest_start_date: str = "2020-01-01"
    benchmark_symbol: str = "SPY"
    
    # Universe definition
    stock_universe: list = None
    
    def __post_init__(self):
        if self.stock_universe is None:
            self.stock_universe = [
                # Transportation
                'FDX', 'UPS', 'DAL', 'UAL', 'AAL', 'LUV',
                # Retail
                'WMT', 'TGT', 'COST', 'HD', 'LOW', 'AMZN',
                # Technology
                'AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA', 'TSLA',
                # Manufacturing
                'CAT', 'DE', 'GE', 'MMM', 'HON', 'UTX',
                # Energy
                'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'HAL',
                # Materials
                'LIN', 'APD', 'ECL', 'DD', 'DOW', 'PPG'
            ]


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: str = "logs/supply_chain_alpha.log"
    max_file_size: str = "100MB"
    backup_count: int = 10
    
    # Performance tracking
    track_execution_time: bool = True
    track_memory_usage: bool = True
    alert_threshold_seconds: float = 30.0
    
    # Structured logging
    use_json_format: bool = False
    include_metadata: bool = True
    
    # External logging services
    elasticsearch_host: Optional[str] = None
    elasticsearch_port: Optional[int] = None
    elasticsearch_index: str = "supply-chain-alpha"


@dataclass
class MonitoringConfig:
    """System monitoring configuration."""
    # Health checks
    health_check_interval: int = 60  # seconds
    database_health_check: bool = True
    api_health_check: bool = True
    
    # Metrics collection
    collect_system_metrics: bool = True
    collect_business_metrics: bool = True
    metrics_retention_days: int = 90
    
    # Alerting
    email_alerts: bool = False
    slack_webhook_url: Optional[str] = None
    alert_thresholds: Dict[str, float] = None
    
    def __post_init__(self):
        if self.alert_thresholds is None:
            self.alert_thresholds = {
                'cpu_usage': 80.0,
                'memory_usage': 85.0,
                'disk_usage': 90.0,
                'error_rate': 5.0,
                'response_time': 5.0
            }


class ConfigManager:
    """Configuration manager for the Supply Chain Alpha system."""
    
    def __init__(self, config_dir: Optional[str] = None, environment: Optional[Environment] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_dir: Directory containing configuration files
            environment: Target environment (dev, staging, prod)
        """
        self.config_dir = Path(config_dir) if config_dir else Path(__file__).parent
        self.environment = environment or self._detect_environment()
        self.encryption_key = self._get_encryption_key()
        
        # Configuration components
        self.database: DatabaseConfig = DatabaseConfig()
        self.api: APIConfig = APIConfig()
        self.models: ModelConfig = ModelConfig()
        self.trading: TradingConfig = TradingConfig()
        self.logging: LoggingConfig = LoggingConfig()
        self.monitoring: MonitoringConfig = MonitoringConfig()
        
        # Load configurations
        self._load_configurations()
    
    def _detect_environment(self) -> Environment:
        """Detect current environment from environment variables."""
        env_name = os.getenv('SUPPLY_CHAIN_ENV', 'development').lower()
        
        env_mapping = {
            'dev': Environment.DEVELOPMENT,
            'development': Environment.DEVELOPMENT,
            'staging': Environment.STAGING,
            'stage': Environment.STAGING,
            'prod': Environment.PRODUCTION,
            'production': Environment.PRODUCTION,
            'test': Environment.TESTING,
            'testing': Environment.TESTING
        }
        
        return env_mapping.get(env_name, Environment.DEVELOPMENT)
    
    def _get_encryption_key(self) -> Optional[Fernet]:
        """Get encryption key for sensitive data."""
        key_env = os.getenv('SUPPLY_CHAIN_ENCRYPTION_KEY')
        if key_env:
            try:
                key = base64.urlsafe_b64decode(key_env)
                return Fernet(key)
            except Exception as e:
                logging.warning(f"Invalid encryption key: {e}")
        
        return None
    
    def _load_configurations(self):
        """Load configurations from files and environment variables."""
        # Load base configuration
        self._load_config_file('base.yaml')
        
        # Load environment-specific configuration
        env_file = f'{self.environment.value}.yaml'
        self._load_config_file(env_file)
        
        # Override with environment variables
        self._load_environment_variables()
        
        # Load secrets
        self._load_secrets()
    
    def _load_config_file(self, filename: str):
        """Load configuration from YAML file."""
        config_file = self.config_dir / filename
        
        if not config_file.exists():
            logging.info(f"Configuration file {filename} not found, using defaults")
            return
        
        try:
            with open(config_file, 'r') as f:
                config_data = yaml.safe_load(f)
            
            if config_data:
                self._update_configurations(config_data)
                
        except Exception as e:
            logging.error(f"Error loading configuration file {filename}: {e}")
    
    def _update_configurations(self, config_data: Dict[str, Any]):
        """Update configuration objects with loaded data."""
        if 'database' in config_data:
            self._update_dataclass(self.database, config_data['database'])
        
        if 'api' in config_data:
            self._update_dataclass(self.api, config_data['api'])
        
        if 'models' in config_data:
            self._update_dataclass(self.models, config_data['models'])
        
        if 'trading' in config_data:
            self._update_dataclass(self.trading, config_data['trading'])
        
        if 'logging' in config_data:
            self._update_dataclass(self.logging, config_data['logging'])
        
        if 'monitoring' in config_data:
            self._update_dataclass(self.monitoring, config_data['monitoring'])
    
    def _update_dataclass(self, obj: Any, data: Dict[str, Any]):
        """Update dataclass object with dictionary data."""
        for key, value in data.items():
            if hasattr(obj, key):
                setattr(obj, key, value)
    
    def _load_environment_variables(self):
        """Load configuration from environment variables."""
        # Database configuration
        if os.getenv('DATABASE_URL'):
            self.database.mongodb_uri = os.getenv('DATABASE_URL')
        
        if os.getenv('DATABASE_HOST'):
            self.database.host = os.getenv('DATABASE_HOST')
        
        if os.getenv('DATABASE_PORT'):
            self.database.port = int(os.getenv('DATABASE_PORT'))
        
        if os.getenv('DATABASE_NAME'):
            self.database.database = os.getenv('DATABASE_NAME')
        
        if os.getenv('DATABASE_USER'):
            self.database.username = os.getenv('DATABASE_USER')
        
        if os.getenv('DATABASE_PASSWORD'):
            self.database.password = os.getenv('DATABASE_PASSWORD')
        
        # API keys
        api_key_mapping = {
            'MARINE_TRAFFIC_API_KEY': 'marine_traffic_api_key',
            'VESSEL_FINDER_API_KEY': 'vessel_finder_api_key',
            'BALTIC_EXCHANGE_API_KEY': 'baltic_exchange_api_key',
            'FREIGHTOS_API_KEY': 'freightos_api_key',
            'ALPHA_VANTAGE_API_KEY': 'alpha_vantage_api_key',
            'QUANDL_API_KEY': 'quandl_api_key',
            'BLOOMBERG_API_KEY': 'bloomberg_api_key',
            'MSCI_ESG_API_KEY': 'msci_esg_api_key',
            'SUSTAINALYTICS_API_KEY': 'sustainalytics_api_key',
            'REFINITIV_ESG_API_KEY': 'refinitiv_esg_api_key'
        }
        
        for env_var, attr_name in api_key_mapping.items():
            if os.getenv(env_var):
                setattr(self.api, attr_name, os.getenv(env_var))
        
        # Redis configuration
        if os.getenv('REDIS_URL'):
            # Parse Redis URL
            import urllib.parse
            parsed = urllib.parse.urlparse(os.getenv('REDIS_URL'))
            self.database.redis_host = parsed.hostname or 'localhost'
            self.database.redis_port = parsed.port or 6379
            self.database.redis_password = parsed.password
        
        # Logging configuration
        if os.getenv('LOG_LEVEL'):
            self.logging.level = os.getenv('LOG_LEVEL')
        
        if os.getenv('LOG_FILE_PATH'):
            self.logging.file_path = os.getenv('LOG_FILE_PATH')
    
    def _load_secrets(self):
        """Load encrypted secrets."""
        secrets_file = self.config_dir / 'secrets.json'
        
        if not secrets_file.exists():
            return
        
        try:
            with open(secrets_file, 'r') as f:
                encrypted_secrets = json.load(f)
            
            if self.encryption_key:
                for key, encrypted_value in encrypted_secrets.items():
                    try:
                        decrypted_value = self.encryption_key.decrypt(
                            encrypted_value.encode()
                        ).decode()
                        
                        # Map to appropriate configuration
                        self._set_secret_value(key, decrypted_value)
                        
                    except Exception as e:
                        logging.warning(f"Failed to decrypt secret {key}: {e}")
            
        except Exception as e:
            logging.error(f"Error loading secrets: {e}")
    
    def _set_secret_value(self, key: str, value: str):
        """Set decrypted secret value to appropriate configuration."""
        # Map secret keys to configuration attributes
        secret_mapping = {
            'database_password': ('database', 'password'),
            'marine_traffic_api_key': ('api', 'marine_traffic_api_key'),
            'vessel_finder_api_key': ('api', 'vessel_finder_api_key'),
            # Add more mappings as needed
        }
        
        if key in secret_mapping:
            config_obj, attr_name = secret_mapping[key]
            config_instance = getattr(self, config_obj)
            setattr(config_instance, attr_name, value)
    
    def get_database_url(self) -> str:
        """Get database connection URL."""
        if self.database.type == 'postgresql':
            return (
                f"postgresql://{self.database.username}:{self.database.password}@"
                f"{self.database.host}:{self.database.port}/{self.database.database}"
            )
        elif self.database.type == 'sqlite':
            return f"sqlite:///{self.database.sqlite_path or 'supply_chain_alpha.db'}"
        elif self.database.type == 'mongodb':
            return self.database.mongodb_uri or "mongodb://localhost:27017/supply_chain_alpha"
        else:
            raise ValueError(f"Unsupported database type: {self.database.type}")
    
    def get_redis_url(self) -> str:
        """Get Redis connection URL."""
        auth = f":{self.database.redis_password}@" if self.database.redis_password else ""
        return f"redis://{auth}{self.database.redis_host}:{self.database.redis_port}/{self.database.redis_db}"
    
    def validate_configuration(self) -> Dict[str, list]:
        """Validate configuration and return any issues."""
        issues = {
            'errors': [],
            'warnings': []
        }
        
        # Validate database configuration
        if not self.database.username and self.database.type == 'postgresql':
            issues['errors'].append("Database username is required for PostgreSQL")
        
        if not self.database.password and self.database.type == 'postgresql':
            issues['warnings'].append("Database password is not set")
        
        # Validate API keys
        required_api_keys = [
            'marine_traffic_api_key',
            'vessel_finder_api_key',
            'alpha_vantage_api_key'
        ]
        
        for key in required_api_keys:
            if not getattr(self.api, key):
                issues['warnings'].append(f"API key {key} is not configured")
        
        # Validate trading configuration
        if self.trading.max_position_size > 0.1:
            issues['warnings'].append("Max position size is greater than 10%")
        
        if self.trading.target_leverage > 2.0:
            issues['warnings'].append("Target leverage is greater than 2.0")
        
        return issues
    
    def export_configuration(self, include_secrets: bool = False) -> Dict[str, Any]:
        """Export current configuration to dictionary."""
        config_dict = {
            'environment': self.environment.value,
            'database': asdict(self.database),
            'api': asdict(self.api),
            'models': asdict(self.models),
            'trading': asdict(self.trading),
            'logging': asdict(self.logging),
            'monitoring': asdict(self.monitoring)
        }
        
        if not include_secrets:
            # Remove sensitive information
            config_dict['database']['password'] = '***'
            
            api_keys = [
                'marine_traffic_api_key', 'vessel_finder_api_key',
                'baltic_exchange_api_key', 'freightos_api_key',
                'alpha_vantage_api_key', 'quandl_api_key',
                'bloomberg_api_key', 'msci_esg_api_key',
                'sustainalytics_api_key', 'refinitiv_esg_api_key'
            ]
            
            for key in api_keys:
                if key in config_dict['api']:
                    config_dict['api'][key] = '***'
        
        return config_dict
    
    def save_configuration(self, filename: str, include_secrets: bool = False):
        """Save current configuration to file."""
        config_dict = self.export_configuration(include_secrets)
        
        output_file = self.config_dir / filename
        
        try:
            with open(output_file, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            
            logging.info(f"Configuration saved to {output_file}")
            
        except Exception as e:
            logging.error(f"Error saving configuration: {e}")
    
    def encrypt_secret(self, value: str) -> Optional[str]:
        """Encrypt a secret value."""
        if not self.encryption_key:
            logging.warning("No encryption key available")
            return None
        
        try:
            encrypted_value = self.encryption_key.encrypt(value.encode())
            return encrypted_value.decode()
        except Exception as e:
            logging.error(f"Error encrypting secret: {e}")
            return None
    
    @staticmethod
    def generate_encryption_key() -> str:
        """Generate a new encryption key."""
        key = Fernet.generate_key()
        return base64.urlsafe_b64encode(key).decode()


# Global configuration instance
_config_manager: Optional[ConfigManager] = None


def get_config() -> ConfigManager:
    """Get global configuration manager instance."""
    global _config_manager
    
    if _config_manager is None:
        _config_manager = ConfigManager()
    
    return _config_manager


def initialize_config(config_dir: Optional[str] = None, 
                     environment: Optional[Environment] = None) -> ConfigManager:
    """Initialize global configuration manager."""
    global _config_manager
    
    _config_manager = ConfigManager(config_dir, environment)
    return _config_manager


if __name__ == '__main__':
    # Example usage and testing
    config = ConfigManager()
    
    print(f"Environment: {config.environment.value}")
    print(f"Database URL: {config.get_database_url()}")
    print(f"Redis URL: {config.get_redis_url()}")
    
    # Validate configuration
    issues = config.validate_configuration()
    if issues['errors']:
        print("Configuration errors:")
        for error in issues['errors']:
            print(f"  - {error}")
    
    if issues['warnings']:
        print("Configuration warnings:")
        for warning in issues['warnings']:
            print(f"  - {warning}")
    
    # Export configuration (without secrets)
    config_dict = config.export_configuration(include_secrets=False)
    print("\nConfiguration exported successfully")