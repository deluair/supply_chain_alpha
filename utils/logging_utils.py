#!/usr/bin/env python3
"""
Logging Utilities
Provides comprehensive logging capabilities for the supply chain disruption analysis system.

Features:
- Structured logging with JSON format
- Multiple log levels and handlers
- Performance monitoring
- Error tracking and alerting
- Log rotation and archival
- Real-time log streaming
- Custom formatters and filters

Author: Supply Chain Alpha Team
Version: 1.0.0
"""

import logging
import logging.handlers
import json
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field, asdict
from pathlib import Path
import traceback
import threading
import time
from contextlib import contextmanager

# Additional libraries
import colorlog
from pythonjsonlogger import jsonlogger
from enum import Enum


class LogLevel(Enum):
    """Enumeration of available log levels."""
    DEBUG = 'DEBUG'
    INFO = 'INFO'
    WARNING = 'WARNING'
    ERROR = 'ERROR'
    CRITICAL = 'CRITICAL'


@dataclass
class LogConfig:
    """Logging configuration settings."""
    # Basic settings
    log_level: str = 'INFO'
    log_format: str = 'json'  # 'json', 'text', 'colored'
    
    # File logging
    log_file: Optional[str] = 'logs/supply_chain_alpha.log'
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    
    # Console logging
    console_logging: bool = True
    console_level: str = 'INFO'
    
    # Performance logging
    performance_logging: bool = True
    performance_file: Optional[str] = 'logs/performance.log'
    
    # Error logging
    error_file: Optional[str] = 'logs/errors.log'
    
    # Structured logging
    include_timestamp: bool = True
    include_level: bool = True
    include_module: bool = True
    include_function: bool = True
    include_line_number: bool = True
    
    # Additional metadata
    service_name: str = 'supply_chain_alpha'
    environment: str = 'development'
    version: str = '1.0.0'
    
    # Log filtering
    exclude_modules: List[str] = field(default_factory=lambda: ['urllib3', 'requests'])
    
    # Remote logging (optional)
    remote_logging: bool = False
    remote_endpoint: Optional[str] = None
    remote_api_key: Optional[str] = None


@dataclass
class LogEntry:
    """Structured log entry."""
    timestamp: datetime
    level: str
    message: str
    module: str
    function: str
    line_number: int
    service_name: str
    environment: str
    version: str
    thread_id: str
    process_id: int
    extra_data: Dict[str, Any] = field(default_factory=dict)
    exception_info: Optional[Dict[str, Any]] = None
    performance_data: Optional[Dict[str, Any]] = None


class PerformanceTracker:
    """Track performance metrics for logging."""
    
    def __init__(self):
        self.start_times = {}
        self.metrics = {}
        self.lock = threading.Lock()
    
    def start_timer(self, operation_name: str) -> str:
        """Start timing an operation.
        
        Args:
            operation_name: Name of the operation
            
        Returns:
            Timer ID
        """
        timer_id = f"{operation_name}_{int(time.time() * 1000000)}"
        with self.lock:
            self.start_times[timer_id] = time.time()
        return timer_id
    
    def end_timer(self, timer_id: str) -> float:
        """End timing an operation.
        
        Args:
            timer_id: Timer ID from start_timer
            
        Returns:
            Elapsed time in seconds
        """
        end_time = time.time()
        with self.lock:
            start_time = self.start_times.pop(timer_id, end_time)
            elapsed = end_time - start_time
            
            # Extract operation name
            operation_name = timer_id.rsplit('_', 1)[0]
            
            # Update metrics
            if operation_name not in self.metrics:
                self.metrics[operation_name] = {
                    'count': 0,
                    'total_time': 0,
                    'min_time': float('inf'),
                    'max_time': 0,
                    'avg_time': 0
                }
            
            metrics = self.metrics[operation_name]
            metrics['count'] += 1
            metrics['total_time'] += elapsed
            metrics['min_time'] = min(metrics['min_time'], elapsed)
            metrics['max_time'] = max(metrics['max_time'], elapsed)
            metrics['avg_time'] = metrics['total_time'] / metrics['count']
            
            return elapsed
    
    def get_metrics(self) -> Dict[str, Dict[str, float]]:
        """Get performance metrics.
        
        Returns:
            Dictionary of performance metrics
        """
        with self.lock:
            return self.metrics.copy()
    
    def reset_metrics(self):
        """Reset all performance metrics."""
        with self.lock:
            self.metrics.clear()
            self.start_times.clear()


@dataclass
class PerformanceMetrics:
    """Performance metrics data structure."""
    operation_name: str
    count: int
    total_time: float
    min_time: float
    max_time: float
    avg_time: float
    timestamp: datetime = field(default_factory=datetime.now)


class JSONFormatter(jsonlogger.JsonFormatter):
    """Custom JSON formatter for structured logging."""
    
    def __init__(self, config: LogConfig):
        self.config = config
        
        # Define fields to include
        fields = []
        if config.include_timestamp:
            fields.append('timestamp')
        if config.include_level:
            fields.append('levelname')
        fields.append('message')
        if config.include_module:
            fields.append('module')
        if config.include_function:
            fields.append('funcName')
        if config.include_line_number:
            fields.append('lineno')
        
        format_string = ' '.join([f'%({field})s' for field in fields])
        super().__init__(format_string)
    
    def add_fields(self, log_record, record, message_dict):
        """Add custom fields to log record."""
        super().add_fields(log_record, record, message_dict)
        
        # Add service metadata
        log_record['service_name'] = self.config.service_name
        log_record['environment'] = self.config.environment
        log_record['version'] = self.config.version
        log_record['thread_id'] = record.thread
        log_record['process_id'] = record.process
        
        # Add timestamp in ISO format
        if 'timestamp' not in log_record:
            log_record['timestamp'] = datetime.utcnow().isoformat()
        
        # Add module name
        if 'module' not in log_record:
            log_record['module'] = record.module
        
        # Add exception information if present
        if record.exc_info:
            log_record['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }
        
        # Add extra data if present
        if hasattr(record, 'extra_data'):
            log_record['extra_data'] = record.extra_data
        
        # Add performance data if present
        if hasattr(record, 'performance_data'):
            log_record['performance_data'] = record.performance_data


class ColoredFormatter(colorlog.ColoredFormatter):
    """Custom colored formatter for console output."""
    
    def __init__(self, config: LogConfig):
        self.config = config
        
        # Define format string
        format_parts = []
        if config.include_timestamp:
            format_parts.append('%(asctime)s')
        format_parts.append('%(log_color)s%(levelname)-8s%(reset)s')
        if config.include_module:
            format_parts.append('%(cyan)s%(module)s%(reset)s')
        if config.include_function:
            format_parts.append('%(blue)s%(funcName)s%(reset)s')
        if config.include_line_number:
            format_parts.append('%(yellow)s:%(lineno)d%(reset)s')
        format_parts.append('%(message)s')
        
        format_string = ' - '.join(format_parts)
        
        super().__init__(
            format_string,
            datefmt='%Y-%m-%d %H:%M:%S',
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white',
            }
        )


class LogFilter(logging.Filter):
    """Custom log filter to exclude certain modules."""
    
    def __init__(self, exclude_modules: List[str]):
        super().__init__()
        self.exclude_modules = exclude_modules
    
    def filter(self, record):
        """Filter log records.
        
        Args:
            record: Log record
            
        Returns:
            True if record should be logged
        """
        return record.module not in self.exclude_modules


class SupplyChainLogger:
    """Main logger class for the supply chain system."""
    
    def __init__(self, config: Optional[LogConfig] = None):
        """Initialize logger.
        
        Args:
            config: Logging configuration
        """
        self.config = config or LogConfig()
        self.performance_tracker = PerformanceTracker()
        self.loggers = {}
        
        # Setup logging
        self._setup_logging()
        
        # Get main logger
        self.logger = self.get_logger('supply_chain_alpha')
        
        # Log startup
        self.logger.info("Supply Chain Alpha logging system initialized", extra={
            'extra_data': {
                'config': asdict(self.config),
                'startup_time': datetime.utcnow().isoformat()
            }
        })
    
    def _setup_logging(self):
        """Setup logging configuration."""
        # Create log directory
        if self.config.log_file:
            log_dir = Path(self.config.log_file).parent
            log_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, self.config.log_level.upper()))
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Setup file handler
        if self.config.log_file:
            file_handler = logging.handlers.RotatingFileHandler(
                self.config.log_file,
                maxBytes=self.config.max_file_size,
                backupCount=self.config.backup_count
            )
            
            if self.config.log_format == 'json':
                file_formatter = JSONFormatter(self.config)
            else:
                file_formatter = logging.Formatter(
                    '%(asctime)s - %(levelname)s - %(module)s - %(funcName)s:%(lineno)d - %(message)s'
                )
            
            file_handler.setFormatter(file_formatter)
            file_handler.setLevel(getattr(logging, self.config.log_level.upper()))
            
            # Add filter
            if self.config.exclude_modules:
                file_handler.addFilter(LogFilter(self.config.exclude_modules))
            
            root_logger.addHandler(file_handler)
        
        # Setup console handler
        if self.config.console_logging:
            console_handler = logging.StreamHandler(sys.stdout)
            
            if self.config.log_format == 'colored':
                console_formatter = ColoredFormatter(self.config)
            elif self.config.log_format == 'json':
                console_formatter = JSONFormatter(self.config)
            else:
                console_formatter = logging.Formatter(
                    '%(asctime)s - %(levelname)s - %(module)s - %(message)s'
                )
            
            console_handler.setFormatter(console_formatter)
            console_handler.setLevel(getattr(logging, self.config.console_level.upper()))
            
            # Add filter
            if self.config.exclude_modules:
                console_handler.addFilter(LogFilter(self.config.exclude_modules))
            
            root_logger.addHandler(console_handler)
        
        # Setup error file handler
        if self.config.error_file:
            error_dir = Path(self.config.error_file).parent
            error_dir.mkdir(parents=True, exist_ok=True)
            
            error_handler = logging.handlers.RotatingFileHandler(
                self.config.error_file,
                maxBytes=self.config.max_file_size,
                backupCount=self.config.backup_count
            )
            error_handler.setLevel(logging.ERROR)
            error_handler.setFormatter(JSONFormatter(self.config))
            root_logger.addHandler(error_handler)
        
        # Setup performance file handler
        if self.config.performance_logging and self.config.performance_file:
            perf_dir = Path(self.config.performance_file).parent
            perf_dir.mkdir(parents=True, exist_ok=True)
            
            perf_handler = logging.handlers.RotatingFileHandler(
                self.config.performance_file,
                maxBytes=self.config.max_file_size,
                backupCount=self.config.backup_count
            )
            perf_handler.setLevel(logging.INFO)
            perf_handler.setFormatter(JSONFormatter(self.config))
            
            # Create performance logger
            perf_logger = logging.getLogger('performance')
            perf_logger.addHandler(perf_handler)
            perf_logger.setLevel(logging.INFO)
            perf_logger.propagate = False
    
    def get_logger(self, name: str) -> logging.Logger:
        """Get or create a logger for a specific module.
        
        Args:
            name: Logger name (usually module name)
            
        Returns:
            Logger instance
        """
        if name not in self.loggers:
            logger = logging.getLogger(name)
            self.loggers[name] = logger
        
        return self.loggers[name]
    
    def log_performance(self, operation: str, duration: float, 
                       extra_data: Optional[Dict[str, Any]] = None):
        """Log performance metrics.
        
        Args:
            operation: Operation name
            duration: Duration in seconds
            extra_data: Additional performance data
        """
        if not self.config.performance_logging:
            return
        
        perf_logger = logging.getLogger('performance')
        
        performance_data = {
            'operation': operation,
            'duration_seconds': duration,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        if extra_data:
            performance_data.update(extra_data)
        
        perf_logger.info(f"Performance: {operation}", extra={
            'performance_data': performance_data
        })
    
    def log_error(self, message: str, exception: Optional[Exception] = None, 
                  extra_data: Optional[Dict[str, Any]] = None):
        """Log error with detailed information.
        
        Args:
            message: Error message
            exception: Exception object
            extra_data: Additional error context
        """
        error_data = {
            'error_message': message,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        if exception:
            error_data.update({
                'exception_type': type(exception).__name__,
                'exception_message': str(exception),
                'traceback': traceback.format_exc()
            })
        
        if extra_data:
            error_data.update(extra_data)
        
        self.logger.error(message, exc_info=exception is not None, extra={
            'extra_data': error_data
        })
    
    def log_business_event(self, event_type: str, event_data: Dict[str, Any]):
        """Log business-specific events.
        
        Args:
            event_type: Type of business event
            event_data: Event-specific data
        """
        business_logger = self.get_logger('business_events')
        
        event_log = {
            'event_type': event_type,
            'timestamp': datetime.utcnow().isoformat(),
            'event_data': event_data
        }
        
        business_logger.info(f"Business Event: {event_type}", extra={
            'extra_data': event_log
        })
    
    @contextmanager
    def performance_context(self, operation_name: str, 
                           extra_data: Optional[Dict[str, Any]] = None):
        """Context manager for performance tracking.
        
        Args:
            operation_name: Name of the operation
            extra_data: Additional performance data
        """
        timer_id = self.performance_tracker.start_timer(operation_name)
        start_time = time.time()
        
        try:
            yield
        except Exception as e:
            # Log the exception
            self.log_error(f"Error in {operation_name}", e, extra_data)
            raise
        finally:
            duration = self.performance_tracker.end_timer(timer_id)
            
            # Log performance
            perf_data = {'operation_name': operation_name}
            if extra_data:
                perf_data.update(extra_data)
            
            self.log_performance(operation_name, duration, perf_data)
    
    def log_data_quality(self, dataset_name: str, quality_metrics: Dict[str, Any]):
        """Log data quality metrics.
        
        Args:
            dataset_name: Name of the dataset
            quality_metrics: Quality metrics dictionary
        """
        quality_logger = self.get_logger('data_quality')
        
        quality_log = {
            'dataset_name': dataset_name,
            'timestamp': datetime.utcnow().isoformat(),
            'quality_metrics': quality_metrics
        }
        
        quality_logger.info(f"Data Quality: {dataset_name}", extra={
            'extra_data': quality_log
        })
    
    def log_model_performance(self, model_name: str, metrics: Dict[str, float], 
                             dataset_info: Optional[Dict[str, Any]] = None):
        """Log model performance metrics.
        
        Args:
            model_name: Name of the model
            metrics: Performance metrics
            dataset_info: Information about the dataset used
        """
        model_logger = self.get_logger('model_performance')
        
        model_log = {
            'model_name': model_name,
            'timestamp': datetime.utcnow().isoformat(),
            'performance_metrics': metrics
        }
        
        if dataset_info:
            model_log['dataset_info'] = dataset_info
        
        model_logger.info(f"Model Performance: {model_name}", extra={
            'extra_data': model_log
        })
    
    def log_trading_signal(self, signal_type: str, symbol: str, 
                          signal_data: Dict[str, Any]):
        """Log trading signals.
        
        Args:
            signal_type: Type of trading signal
            symbol: Stock/asset symbol
            signal_data: Signal-specific data
        """
        trading_logger = self.get_logger('trading_signals')
        
        signal_log = {
            'signal_type': signal_type,
            'symbol': symbol,
            'timestamp': datetime.utcnow().isoformat(),
            'signal_data': signal_data
        }
        
        trading_logger.info(f"Trading Signal: {signal_type} - {symbol}", extra={
            'extra_data': signal_log
        })
    
    def log_system_health(self, component: str, health_data: Dict[str, Any]):
        """Log system health metrics.
        
        Args:
            component: System component name
            health_data: Health metrics and status
        """
        health_logger = self.get_logger('system_health')
        
        health_log = {
            'component': component,
            'timestamp': datetime.utcnow().isoformat(),
            'health_data': health_data
        }
        
        # Determine log level based on health status
        status = health_data.get('status', 'unknown').lower()
        if status in ['error', 'critical', 'down']:
            log_level = logging.ERROR
        elif status in ['warning', 'degraded']:
            log_level = logging.WARNING
        else:
            log_level = logging.INFO
        
        health_logger.log(log_level, f"System Health: {component}", extra={
            'extra_data': health_log
        })
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance metrics summary.
        
        Returns:
            Dictionary with performance summary
        """
        metrics = self.performance_tracker.get_metrics()
        
        summary = {
            'timestamp': datetime.utcnow().isoformat(),
            'total_operations': len(metrics),
            'operations': metrics
        }
        
        # Calculate overall statistics
        if metrics:
            all_avg_times = [op['avg_time'] for op in metrics.values()]
            summary['overall_stats'] = {
                'fastest_avg_operation': min(all_avg_times),
                'slowest_avg_operation': max(all_avg_times),
                'mean_avg_time': sum(all_avg_times) / len(all_avg_times)
            }
        
        return summary
    
    def export_logs(self, start_date: datetime, end_date: datetime, 
                   output_file: str, log_level: Optional[str] = None) -> bool:
        """Export logs for a specific time period.
        
        Args:
            start_date: Start date for log export
            end_date: End date for log export
            output_file: Output file path
            log_level: Optional log level filter
            
        Returns:
            True if export successful
        """
        try:
            # This is a simplified implementation
            # In a real system, you'd parse the log files and filter by date/level
            
            exported_logs = []
            
            # Read main log file
            if self.config.log_file and os.path.exists(self.config.log_file):
                with open(self.config.log_file, 'r') as f:
                    for line in f:
                        try:
                            if self.config.log_format == 'json':
                                log_entry = json.loads(line.strip())
                                log_time = datetime.fromisoformat(log_entry.get('timestamp', ''))
                                
                                if start_date <= log_time <= end_date:
                                    if not log_level or log_entry.get('levelname') == log_level:
                                        exported_logs.append(log_entry)
                        except (json.JSONDecodeError, ValueError):
                            continue
            
            # Write exported logs
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w') as f:
                for log_entry in exported_logs:
                    f.write(json.dumps(log_entry) + '\n')
            
            self.logger.info(f"Exported {len(exported_logs)} log entries to {output_file}")
            return True
            
        except Exception as e:
            self.log_error(f"Failed to export logs: {e}", e)
            return False
    
    def cleanup_old_logs(self, days_to_keep: int = 30) -> bool:
        """Clean up old log files.
        
        Args:
            days_to_keep: Number of days to keep logs
            
        Returns:
            True if cleanup successful
        """
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)
            
            # Clean up main log files
            for log_file in [self.config.log_file, self.config.error_file, self.config.performance_file]:
                if log_file:
                    log_dir = Path(log_file).parent
                    if log_dir.exists():
                        for file_path in log_dir.glob(f"{Path(log_file).stem}.*"):
                            if file_path.stat().st_mtime < cutoff_date.timestamp():
                                file_path.unlink()
                                self.logger.info(f"Deleted old log file: {file_path}")
            
            return True
            
        except Exception as e:
            self.log_error(f"Failed to cleanup old logs: {e}", e)
            return False
    
    def set_log_level(self, level: str):
        """Change log level at runtime.
        
        Args:
            level: New log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        try:
            new_level = getattr(logging, level.upper())
            
            # Update all loggers
            for logger in self.loggers.values():
                logger.setLevel(new_level)
            
            # Update root logger
            logging.getLogger().setLevel(new_level)
            
            # Update config
            self.config.log_level = level.upper()
            
            self.logger.info(f"Log level changed to {level.upper()}")
            
        except AttributeError:
            self.log_error(f"Invalid log level: {level}")
    
    def get_log_stats(self) -> Dict[str, Any]:
        """Get logging statistics.
        
        Returns:
            Dictionary with logging statistics
        """
        stats = {
            'timestamp': datetime.utcnow().isoformat(),
            'config': asdict(self.config),
            'active_loggers': list(self.loggers.keys()),
            'performance_metrics': self.get_performance_summary()
        }
        
        # Add file sizes if files exist
        for log_type, log_file in [
            ('main_log', self.config.log_file),
            ('error_log', self.config.error_file),
            ('performance_log', self.config.performance_file)
        ]:
            if log_file and os.path.exists(log_file):
                stats[f'{log_type}_size_bytes'] = os.path.getsize(log_file)
        
        return stats


# Global logger instance
_global_logger: Optional[SupplyChainLogger] = None


def get_logger(name: str = 'supply_chain_alpha') -> logging.Logger:
    """Get a logger instance.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    global _global_logger
    
    if _global_logger is None:
        _global_logger = SupplyChainLogger()
    
    return _global_logger.get_logger(name)


def setup_logging(config: Optional[LogConfig] = None) -> SupplyChainLogger:
    """Setup global logging configuration.
    
    Args:
        config: Logging configuration
        
    Returns:
        SupplyChainLogger instance
    """
    global _global_logger
    _global_logger = SupplyChainLogger(config)
    return _global_logger


def log_performance(operation: str, duration: float, 
                   extra_data: Optional[Dict[str, Any]] = None):
    """Log performance metrics using global logger.
    
    Args:
        operation: Operation name
        duration: Duration in seconds
        extra_data: Additional performance data
    """
    global _global_logger
    if _global_logger:
        _global_logger.log_performance(operation, duration, extra_data)


@contextmanager
def performance_context(operation_name: str, 
                      extra_data: Optional[Dict[str, Any]] = None):
    """Context manager for performance tracking using global logger.
    
    Args:
        operation_name: Name of the operation
        extra_data: Additional performance data
    """
    global _global_logger
    if _global_logger:
        with _global_logger.performance_context(operation_name, extra_data):
            yield
    else:
        yield