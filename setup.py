#!/usr/bin/env python3
"""
Supply Chain Alpha - Setup Script

This script sets up the development environment for the Supply Chain Alpha system.
It handles dependency installation, database setup, configuration, and initial data preparation.

Usage:
    python setup.py [options]

Options:
    --env {dev,staging,prod}    Target environment (default: dev)
    --install-deps             Install Python dependencies
    --setup-db                 Set up database and tables
    --create-config            Create configuration files
    --download-data            Download sample data
    --run-tests                Run test suite
    --all                      Run all setup steps

Author: Supply Chain Alpha Team
Version: 1.0.0
"""

import os
import sys
import argparse
import subprocess
import sqlite3
import json
import yaml
from pathlib import Path
from typing import List, Dict, Any
import logging
from datetime import datetime

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


class SupplyChainAlphaSetup:
    """Setup manager for Supply Chain Alpha system."""
    
    def __init__(self, environment: str = "dev"):
        """
        Initialize setup manager.
        
        Args:
            environment: Target environment (dev, staging, prod)
        """
        self.environment = environment
        self.project_root = Path(__file__).parent
        self.setup_logging()
        
        # Directory structure
        self.directories = [
            "data",
            "data/raw",
            "data/processed",
            "data/models",
            "logs",
            "models/saved",
            "models/dev",
            "models/prod",
            "config",
            "tests/data",
            "tests/fixtures",
            "docs",
            "scripts"
        ]
        
        # Required Python packages
        self.python_dependencies = [
            "pandas>=1.5.0",
            "numpy>=1.21.0",
            "scikit-learn>=1.1.0",
            "tensorflow>=2.10.0",
            "torch>=1.12.0",
            "xgboost>=1.6.0",
            "lightgbm>=3.3.0",
            "requests>=2.28.0",
            "aiohttp>=3.8.0",
            "sqlalchemy>=1.4.0",
            "psycopg2-binary>=2.9.0",
            "pymongo>=4.2.0",
            "redis>=4.3.0",
            "pyyaml>=6.0",
            "cryptography>=37.0.0",
            "python-dotenv>=0.20.0",
            "click>=8.1.0",
            "rich>=12.5.0",
            "plotly>=5.10.0",
            "dash>=2.6.0",
            "streamlit>=1.12.0",
            "fastapi>=0.85.0",
            "uvicorn>=0.18.0",
            "celery>=5.2.0",
            "pytest>=7.1.0",
            "pytest-asyncio>=0.19.0",
            "pytest-cov>=3.0.0",
            "black>=22.6.0",
            "flake8>=5.0.0",
            "mypy>=0.971",
            "pre-commit>=2.20.0"
        ]
        
        # System dependencies (for reference)
        self.system_dependencies = {
            "ubuntu": [
                "python3-dev",
                "postgresql-client",
                "redis-tools",
                "git",
                "curl",
                "wget"
            ],
            "macos": [
                "postgresql",
                "redis",
                "git"
            ]
        }
    
    def setup_logging(self):
        """Set up logging for setup process."""
        log_dir = self.project_root / "logs"
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / "setup.log"),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger(__name__)
    
    def create_directories(self):
        """Create required directory structure."""
        self.logger.info("Creating directory structure...")
        
        for directory in self.directories:
            dir_path = self.project_root / directory
            dir_path.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Created directory: {directory}")
        
        # Create __init__.py files for Python packages
        python_packages = [
            "data_scrapers",
            "models",
            "strategies",
            "utils",
            "tests",
            "config"
        ]
        
        for package in python_packages:
            init_file = self.project_root / package / "__init__.py"
            if not init_file.exists():
                init_file.touch()
                self.logger.info(f"Created __init__.py for {package}")
    
    def install_python_dependencies(self):
        """Install Python dependencies using pip."""
        self.logger.info("Installing Python dependencies...")
        
        try:
            # Upgrade pip first
            subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], 
                         check=True, capture_output=True, text=True)
            
            # Install dependencies
            for dependency in self.python_dependencies:
                self.logger.info(f"Installing {dependency}...")
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", dependency],
                    check=True, capture_output=True, text=True
                )
                
            self.logger.info("Python dependencies installed successfully")
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error installing dependencies: {e}")
            self.logger.error(f"Error output: {e.stderr}")
            raise
    
    def create_requirements_file(self):
        """Create requirements.txt file."""
        self.logger.info("Creating requirements.txt...")
        
        requirements_content = "\n".join(self.python_dependencies)
        
        requirements_file = self.project_root / "requirements.txt"
        with open(requirements_file, 'w') as f:
            f.write(requirements_content)
        
        self.logger.info("requirements.txt created")
    
    def setup_database(self):
        """Set up database for the specified environment."""
        self.logger.info(f"Setting up database for {self.environment} environment...")
        
        if self.environment == "dev":
            self._setup_sqlite_database()
        else:
            self._setup_postgresql_database()
    
    def _setup_sqlite_database(self):
        """Set up SQLite database for development."""
        db_path = self.project_root / "data" / "dev_supply_chain_alpha.db"
        
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Create tables
            self._create_database_tables(cursor)
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"SQLite database created at {db_path}")
            
        except Exception as e:
            self.logger.error(f"Error setting up SQLite database: {e}")
            raise
    
    def _setup_postgresql_database(self):
        """Set up PostgreSQL database for staging/production."""
        self.logger.info("PostgreSQL setup requires manual configuration")
        self.logger.info("Please ensure PostgreSQL is installed and configured")
        
        # Create SQL script for database setup
        sql_script = self._generate_postgresql_setup_script()
        
        script_path = self.project_root / "scripts" / "setup_postgresql.sql"
        with open(script_path, 'w') as f:
            f.write(sql_script)
        
        self.logger.info(f"PostgreSQL setup script created at {script_path}")
    
    def _create_database_tables(self, cursor):
        """Create database tables."""
        tables = {
            "vessel_data": """
                CREATE TABLE IF NOT EXISTS vessel_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    vessel_id TEXT NOT NULL,
                    vessel_name TEXT,
                    vessel_type TEXT,
                    latitude REAL,
                    longitude REAL,
                    speed REAL,
                    course REAL,
                    destination TEXT,
                    eta TIMESTAMP,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    source TEXT
                )
            """,
            "port_metrics": """
                CREATE TABLE IF NOT EXISTS port_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    port_code TEXT NOT NULL,
                    port_name TEXT,
                    throughput_teu INTEGER,
                    congestion_level REAL,
                    avg_waiting_time REAL,
                    berth_utilization REAL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """,
            "freight_rates": """
                CREATE TABLE IF NOT EXISTS freight_rates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    route TEXT NOT NULL,
                    rate_type TEXT,
                    rate_value REAL,
                    currency TEXT DEFAULT 'USD',
                    container_type TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    source TEXT
                )
            """,
            "predictions": """
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    prediction_type TEXT NOT NULL,
                    target_entity TEXT,
                    prediction_value REAL,
                    confidence_score REAL,
                    prediction_date DATE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    model_version TEXT
                )
            """,
            "trading_signals": """
                CREATE TABLE IF NOT EXISTS trading_signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    signal_type TEXT,
                    signal_strength REAL,
                    conviction_score REAL,
                    target_price REAL,
                    stop_loss REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP
                )
            """,
            "esg_scores": """
                CREATE TABLE IF NOT EXISTS esg_scores (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    company_symbol TEXT NOT NULL,
                    environmental_score REAL,
                    social_score REAL,
                    governance_score REAL,
                    overall_score REAL,
                    score_date DATE,
                    data_source TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
        }
        
        for table_name, create_sql in tables.items():
            cursor.execute(create_sql)
            self.logger.info(f"Created table: {table_name}")
    
    def _generate_postgresql_setup_script(self) -> str:
        """Generate PostgreSQL setup script."""
        return """
-- PostgreSQL Setup Script for Supply Chain Alpha
-- Run this script as a PostgreSQL superuser

-- Create database
CREATE DATABASE supply_chain_alpha_prod;

-- Create user
CREATE USER supply_chain_user WITH PASSWORD 'your_secure_password';

-- Grant privileges
GRANT ALL PRIVILEGES ON DATABASE supply_chain_alpha_prod TO supply_chain_user;

-- Connect to the database
\c supply_chain_alpha_prod;

-- Create tables (same structure as SQLite but with PostgreSQL syntax)
CREATE TABLE IF NOT EXISTS vessel_data (
    id SERIAL PRIMARY KEY,
    vessel_id VARCHAR(50) NOT NULL,
    vessel_name VARCHAR(255),
    vessel_type VARCHAR(100),
    latitude DECIMAL(10, 8),
    longitude DECIMAL(11, 8),
    speed DECIMAL(5, 2),
    course DECIMAL(5, 2),
    destination VARCHAR(255),
    eta TIMESTAMP,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    source VARCHAR(100)
);

-- Add indexes for performance
CREATE INDEX idx_vessel_data_vessel_id ON vessel_data(vessel_id);
CREATE INDEX idx_vessel_data_timestamp ON vessel_data(timestamp);

-- Additional tables would be created here...
-- (Similar to SQLite tables but with PostgreSQL-specific syntax)
"""
    
    def create_environment_file(self):
        """Create .env file for environment variables."""
        self.logger.info("Creating .env file...")
        
        env_content = f"""
# Supply Chain Alpha Environment Configuration
# Generated on {datetime.now().isoformat()}

# Environment
SUPPLY_CHAIN_ENV={self.environment}

# Database Configuration
DATABASE_HOST=localhost
DATABASE_PORT=5432
DATABASE_NAME=supply_chain_alpha_{self.environment}
DATABASE_USER=supply_chain_user
DATABASE_PASSWORD=your_secure_password

# Redis Configuration
REDIS_URL=redis://localhost:6379/0

# API Keys (Replace with actual keys)
MARINE_TRAFFIC_API_KEY=your_marine_traffic_api_key
VESSEL_FINDER_API_KEY=your_vessel_finder_api_key
BALTIC_EXCHANGE_API_KEY=your_baltic_exchange_api_key
FREIGHTOS_API_KEY=your_freightos_api_key
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_api_key
QUANDL_API_KEY=your_quandl_api_key
BLOOMBERG_API_KEY=your_bloomberg_api_key
MSCI_ESG_API_KEY=your_msci_esg_api_key
SUSTAINALYTICS_API_KEY=your_sustainalytics_api_key
REFINITIV_ESG_API_KEY=your_refinitiv_esg_api_key

# Encryption Key (Generate a new one for production)
SUPPLY_CHAIN_ENCRYPTION_KEY=generate_new_key_for_production

# Logging Configuration
LOG_LEVEL=INFO
LOG_FILE_PATH=logs/supply_chain_alpha.log

# Monitoring and Alerting
SLACK_WEBHOOK_URL=your_slack_webhook_url
EMAIL_SMTP_SERVER=smtp.gmail.com
EMAIL_SMTP_PORT=587
EMAIL_USERNAME=your_email@gmail.com
EMAIL_PASSWORD=your_email_password

# External Services
ELASTICSEARCH_HOST=localhost
ELASTICSEARCH_PORT=9200
"""
        
        env_file = self.project_root / ".env"
        with open(env_file, 'w') as f:
            f.write(env_content)
        
        self.logger.info(".env file created")
        self.logger.warning("Please update .env file with actual API keys and passwords")
    
    def download_sample_data(self):
        """Download sample data for development and testing."""
        self.logger.info("Creating sample data files...")
        
        # Create sample vessel data
        sample_vessel_data = [
            {
                "vessel_id": "IMO9234567",
                "vessel_name": "EVER GIVEN",
                "vessel_type": "Container Ship",
                "latitude": 31.1065,
                "longitude": 32.5498,
                "speed": 12.5,
                "course": 45.0,
                "destination": "ROTTERDAM",
                "eta": "2024-01-15T10:00:00Z"
            },
            {
                "vessel_id": "IMO9876543",
                "vessel_name": "MSC OSCAR",
                "vessel_type": "Container Ship",
                "latitude": 51.9225,
                "longitude": 4.4792,
                "speed": 8.2,
                "course": 180.0,
                "destination": "SINGAPORE",
                "eta": "2024-01-20T14:30:00Z"
            }
        ]
        
        # Save sample data
        sample_data_file = self.project_root / "data" / "raw" / "sample_vessel_data.json"
        with open(sample_data_file, 'w') as f:
            json.dump(sample_vessel_data, f, indent=2)
        
        self.logger.info("Sample data files created")
    
    def run_tests(self):
        """Run the test suite."""
        self.logger.info("Running test suite...")
        
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", "tests/", "-v", "--cov=."],
                cwd=self.project_root,
                check=True,
                capture_output=True,
                text=True
            )
            
            self.logger.info("Tests completed successfully")
            self.logger.info(result.stdout)
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Tests failed: {e}")
            self.logger.error(f"Error output: {e.stderr}")
            raise
    
    def create_git_hooks(self):
        """Set up Git hooks for code quality."""
        self.logger.info("Setting up Git hooks...")
        
        try:
            # Install pre-commit hooks
            subprocess.run(["pre-commit", "install"], 
                         cwd=self.project_root, check=True)
            
            self.logger.info("Git hooks installed")
            
        except subprocess.CalledProcessError as e:
            self.logger.warning(f"Could not install Git hooks: {e}")
        except FileNotFoundError:
            self.logger.warning("pre-commit not found, skipping Git hooks setup")
    
    def validate_setup(self) -> bool:
        """Validate that setup completed successfully."""
        self.logger.info("Validating setup...")
        
        validation_checks = [
            ("Directory structure", self._check_directories),
            ("Configuration files", self._check_config_files),
            ("Database setup", self._check_database),
            ("Python dependencies", self._check_python_deps)
        ]
        
        all_passed = True
        
        for check_name, check_func in validation_checks:
            try:
                if check_func():
                    self.logger.info(f"✓ {check_name} - OK")
                else:
                    self.logger.error(f"✗ {check_name} - FAILED")
                    all_passed = False
            except Exception as e:
                self.logger.error(f"✗ {check_name} - ERROR: {e}")
                all_passed = False
        
        return all_passed
    
    def _check_directories(self) -> bool:
        """Check if all required directories exist."""
        for directory in self.directories:
            if not (self.project_root / directory).exists():
                return False
        return True
    
    def _check_config_files(self) -> bool:
        """Check if configuration files exist."""
        required_files = [
            "config/config.py",
            "config/base.yaml",
            f"config/{self.environment}.yaml",
            ".env"
        ]
        
        for file_path in required_files:
            if not (self.project_root / file_path).exists():
                return False
        return True
    
    def _check_database(self) -> bool:
        """Check if database is accessible."""
        if self.environment == "dev":
            db_path = self.project_root / "data" / "dev_supply_chain_alpha.db"
            return db_path.exists()
        else:
            # For production, just check if config exists
            return True
    
    def _check_python_deps(self) -> bool:
        """Check if critical Python dependencies are installed."""
        critical_deps = ["pandas", "numpy", "scikit-learn", "requests"]
        
        for dep in critical_deps:
            try:
                __import__(dep)
            except ImportError:
                return False
        return True
    
    def run_full_setup(self):
        """Run complete setup process."""
        self.logger.info(f"Starting full setup for {self.environment} environment...")
        
        try:
            self.create_directories()
            self.create_requirements_file()
            self.install_python_dependencies()
            self.setup_database()
            self.create_environment_file()
            self.download_sample_data()
            self.create_git_hooks()
            
            if self.validate_setup():
                self.logger.info("Setup completed successfully!")
                self._print_next_steps()
            else:
                self.logger.error("Setup validation failed")
                return False
                
        except Exception as e:
            self.logger.error(f"Setup failed: {e}")
            return False
        
        return True
    
    def _print_next_steps(self):
        """Print next steps for the user."""
        next_steps = f"""

🎉 Supply Chain Alpha setup completed successfully!

Next steps:
1. Update .env file with your actual API keys
2. Review configuration in config/{self.environment}.yaml
3. Run tests: python -m pytest tests/
4. Start development: python -m streamlit run dashboard/app.py

For more information, see the README.md file.
        """
        
        print(next_steps)


def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(
        description="Supply Chain Alpha Setup Script",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--env",
        choices=["dev", "staging", "prod"],
        default="dev",
        help="Target environment (default: dev)"
    )
    
    parser.add_argument(
        "--install-deps",
        action="store_true",
        help="Install Python dependencies"
    )
    
    parser.add_argument(
        "--setup-db",
        action="store_true",
        help="Set up database and tables"
    )
    
    parser.add_argument(
        "--create-config",
        action="store_true",
        help="Create configuration files"
    )
    
    parser.add_argument(
        "--download-data",
        action="store_true",
        help="Download sample data"
    )
    
    parser.add_argument(
        "--run-tests",
        action="store_true",
        help="Run test suite"
    )
    
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all setup steps"
    )
    
    args = parser.parse_args()
    
    # Initialize setup manager
    setup_manager = SupplyChainAlphaSetup(args.env)
    
    try:
        if args.all:
            success = setup_manager.run_full_setup()
            sys.exit(0 if success else 1)
        
        # Run individual steps
        if args.install_deps:
            setup_manager.create_directories()
            setup_manager.create_requirements_file()
            setup_manager.install_python_dependencies()
        
        if args.setup_db:
            setup_manager.setup_database()
        
        if args.create_config:
            setup_manager.create_environment_file()
        
        if args.download_data:
            setup_manager.download_sample_data()
        
        if args.run_tests:
            setup_manager.run_tests()
        
        # If no specific options, run full setup
        if not any([args.install_deps, args.setup_db, args.create_config, 
                   args.download_data, args.run_tests]):
            success = setup_manager.run_full_setup()
            sys.exit(0 if success else 1)
    
    except KeyboardInterrupt:
        print("\nSetup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Setup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()