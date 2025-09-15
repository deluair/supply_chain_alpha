#!/usr/bin/env python3
"""
Supply Chain Disruption Alpha Strategy
Main application entry point for the supply chain disruption prediction and trading system.

This system:
1. Scrapes shipping data, port congestion metrics, and freight rates
2. Builds predictive models for supply chain disruptions
3. Creates systematic long/short equity strategies
4. Integrates ESG and sustainability metrics

Author: Supply Chain Alpha Team
Date: 2024
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any

import click
import yaml
from dotenv import load_dotenv
from loguru import logger

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from data_scrapers.shipping_scraper import ShippingScraper
from data_scrapers.port_scraper import PortScraper
from data_scrapers.freight_scraper import FreightScraper
from models.disruption_predictor import DisruptionPredictor
from models.sector_impact_analyzer import SectorImpactAnalyzer
from strategies.long_short_equity import LongShortEquityStrategy
# from utils.config_manager import ConfigManager  # Module not found
from utils.database import DatabaseManager
# from utils.scheduler import TaskScheduler  # Module not found


class SupplyChainAlphaSystem:
    """Main system orchestrator for supply chain disruption alpha strategy."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the supply chain alpha system.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.get_config()
        self.setup_logging()
        
        # Initialize components
        self.db_manager = DatabaseManager(self.config['database'])
        self.scheduler = TaskScheduler()
        
        # Initialize data scrapers
        self.shipping_scraper = ShippingScraper(self.config['data_sources']['shipping'])
        self.port_scraper = PortScraper(self.config['data_sources']['ports'])
        self.freight_scraper = FreightScraper(self.config['data_sources']['freight_rates'])
        
        # Initialize models
        self.disruption_predictor = DisruptionPredictor(self.config['models']['disruption_predictor'])
        self.sector_impact_model = SectorImpactAnalyzer(self.config['models']['sector_impact'])
        
        # Initialize trading strategy
        self.equity_strategy = LongShortEquityStrategy(self.config['strategies']['long_short_equity'])
        
        logger.info("Supply Chain Alpha System initialized successfully")
    
    def setup_logging(self) -> None:
        """Setup logging configuration."""
        log_config = self.config.get('logging', {})
        log_level = log_config.get('level', 'INFO')
        log_file = log_config.get('file', 'logs/supply_chain_alpha.log')
        
        # Create logs directory if it doesn't exist
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # Configure loguru
        logger.remove()  # Remove default handler
        logger.add(
            sys.stdout,
            level=log_level,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
        )
        logger.add(
            log_file,
            level=log_level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            rotation="1 day",
            retention="30 days"
        )
    
    def collect_data(self) -> Dict[str, Any]:
        """Collect all required data from various sources.
        
        Returns:
            Dictionary containing collected data
        """
        logger.info("Starting data collection process")
        
        data = {
            'shipping': {},
            'ports': {},
            'freight_rates': {},
            'timestamp': None
        }
        
        try:
            # Collect shipping data
            logger.info("Collecting shipping data")
            data['shipping'] = self.shipping_scraper.collect_all_data()
            
            # Collect port congestion data
            logger.info("Collecting port congestion data")
            data['ports'] = self.port_scraper.collect_all_data()
            
            # Collect freight rates
            logger.info("Collecting freight rates data")
            data['freight_rates'] = self.freight_scraper.collect_all_data()
            
            # Store data in database
            self.db_manager.store_raw_data(data)
            
            logger.info("Data collection completed successfully")
            return data
            
        except Exception as e:
            logger.error(f"Error during data collection: {str(e)}")
            raise
    
    def train_models(self) -> None:
        """Train predictive models using collected data."""
        logger.info("Starting model training process")
        
        try:
            # Get training data from database
            training_data = self.db_manager.get_training_data()
            
            # Train disruption prediction model
            logger.info("Training disruption prediction model")
            self.disruption_predictor.train(training_data)
            
            # Train sector impact model
            logger.info("Training sector impact model")
            self.sector_impact_model.train(training_data)
            
            logger.info("Model training completed successfully")
            
        except Exception as e:
            logger.error(f"Error during model training: {str(e)}")
            raise
    
    def generate_predictions(self) -> Dict[str, Any]:
        """Generate predictions using trained models.
        
        Returns:
            Dictionary containing predictions
        """
        logger.info("Generating predictions")
        
        try:
            # Get latest data for predictions
            latest_data = self.db_manager.get_latest_data()
            
            # Generate disruption predictions
            disruption_predictions = self.disruption_predictor.predict(latest_data)
            
            # Generate sector impact predictions
            sector_predictions = self.sector_impact_model.predict(latest_data)
            
            predictions = {
                'disruption_scores': disruption_predictions,
                'sector_impacts': sector_predictions,
                'timestamp': latest_data.get('timestamp')
            }
            
            # Store predictions in database
            self.db_manager.store_predictions(predictions)
            
            logger.info("Predictions generated successfully")
            return predictions
            
        except Exception as e:
            logger.error(f"Error during prediction generation: {str(e)}")
            raise
    
    def execute_strategy(self) -> Dict[str, Any]:
        """Execute trading strategy based on predictions.
        
        Returns:
            Dictionary containing strategy execution results
        """
        logger.info("Executing trading strategy")
        
        try:
            # Get latest predictions
            predictions = self.db_manager.get_latest_predictions()
            
            # Execute long/short equity strategy
            strategy_results = self.equity_strategy.execute(predictions)
            
            # Store strategy results
            self.db_manager.store_strategy_results(strategy_results)
            
            logger.info("Trading strategy executed successfully")
            return strategy_results
            
        except Exception as e:
            logger.error(f"Error during strategy execution: {str(e)}")
            raise
    
    def run_full_pipeline(self) -> None:
        """Run the complete supply chain alpha pipeline."""
        logger.info("Starting full supply chain alpha pipeline")
        
        try:
            # Step 1: Collect data
            self.collect_data()
            
            # Step 2: Train models (if needed)
            if self.should_retrain_models():
                self.train_models()
            
            # Step 3: Generate predictions
            self.generate_predictions()
            
            # Step 4: Execute strategy
            self.execute_strategy()
            
            logger.info("Full pipeline completed successfully")
            
        except Exception as e:
            logger.error(f"Error in full pipeline: {str(e)}")
            raise
    
    def should_retrain_models(self) -> bool:
        """Check if models should be retrained based on schedule or performance.
        
        Returns:
            True if models should be retrained
        """
        # Implementation would check last training time, model performance, etc.
        return self.db_manager.check_model_retraining_needed()
    
    def setup_scheduler(self) -> None:
        """Setup scheduled tasks for automated execution."""
        logger.info("Setting up task scheduler")
        
        schedule_config = self.config.get('scheduling', {})
        
        # Schedule data collection tasks
        data_schedule = schedule_config.get('data_collection', {})
        if data_schedule:
            self.scheduler.add_job(
                'collect_shipping_data',
                self.shipping_scraper.collect_all_data,
                data_schedule.get('shipping_data', '0 */6 * * *')
            )
            
            self.scheduler.add_job(
                'collect_port_data',
                self.port_scraper.collect_all_data,
                data_schedule.get('port_data', '0 */4 * * *')
            )
            
            self.scheduler.add_job(
                'collect_freight_data',
                self.freight_scraper.collect_all_data,
                data_schedule.get('freight_rates', '0 9 * * *')
            )
        
        # Schedule model training
        model_schedule = schedule_config.get('model_training', {})
        if model_schedule:
            self.scheduler.add_job(
                'train_models',
                self.train_models,
                model_schedule.get('disruption_model', '0 3 * * 1')
            )
        
        # Schedule strategy execution
        strategy_schedule = schedule_config.get('strategy_execution', {})
        if strategy_schedule:
            self.scheduler.add_job(
                'execute_strategy',
                self.execute_strategy,
                strategy_schedule.get('rebalancing', '0 16 * * 1')
            )
        
        logger.info("Task scheduler setup completed")


@click.group()
@click.option('--config', default='config/config.yaml', help='Configuration file path')
@click.pass_context
def cli(ctx, config):
    """Supply Chain Disruption Alpha Strategy CLI."""
    ctx.ensure_object(dict)
    ctx.obj['config'] = config


@cli.command()
@click.pass_context
def collect_data(ctx):
    """Collect data from all configured sources."""
    system = SupplyChainAlphaSystem(ctx.obj['config'])
    system.collect_data()


@cli.command()
@click.pass_context
def train_models(ctx):
    """Train predictive models."""
    system = SupplyChainAlphaSystem(ctx.obj['config'])
    system.train_models()


@cli.command()
@click.pass_context
def predict(ctx):
    """Generate predictions using trained models."""
    system = SupplyChainAlphaSystem(ctx.obj['config'])
    predictions = system.generate_predictions()
    click.echo(f"Predictions generated: {predictions}")


@cli.command()
@click.pass_context
def execute_strategy(ctx):
    """Execute trading strategy."""
    system = SupplyChainAlphaSystem(ctx.obj['config'])
    results = system.execute_strategy()
    click.echo(f"Strategy executed: {results}")


@cli.command()
@click.pass_context
def run_pipeline(ctx):
    """Run the complete pipeline."""
    system = SupplyChainAlphaSystem(ctx.obj['config'])
    system.run_full_pipeline()


@cli.command()
@click.pass_context
def start_scheduler(ctx):
    """Start the task scheduler for automated execution."""
    system = SupplyChainAlphaSystem(ctx.obj['config'])
    system.setup_scheduler()
    system.scheduler.start()
    click.echo("Scheduler started. Press Ctrl+C to stop.")
    try:
        system.scheduler.keep_alive()
    except KeyboardInterrupt:
        click.echo("Stopping scheduler...")
        system.scheduler.stop()


if __name__ == '__main__':
    # Load environment variables
    load_dotenv()
    
    # Run CLI
    cli()