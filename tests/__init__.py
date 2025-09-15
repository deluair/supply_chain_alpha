#!/usr/bin/env python3
"""
Test Suite for Supply Chain Alpha System

This package contains comprehensive tests for all components of the supply chain
disruption analysis and trading system.

Test Structure:
- Unit tests for individual components
- Integration tests for system interactions
- Performance tests for scalability
- End-to-end tests for complete workflows

Author: Supply Chain Alpha Team
Version: 1.0.0
"""

import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Test configuration
TEST_CONFIG = {
    'database': {
        'type': 'sqlite',
        'path': ':memory:',  # Use in-memory database for tests
    },
    'logging': {
        'level': 'WARNING',  # Reduce log noise during tests
        'console_logging': False,
        'file_logging': False,
    },
    'data_sources': {
        'mock_mode': True,  # Use mock data for tests
    },
    'performance': {
        'timeout_seconds': 30,  # Test timeout
    }
}

# Test data directory
TEST_DATA_DIR = Path(__file__).parent / 'test_data'
TEST_DATA_DIR.mkdir(exist_ok=True)

# Mock API responses directory
MOCK_DATA_DIR = Path(__file__).parent / 'mock_data'
MOCK_DATA_DIR.mkdir(exist_ok=True)