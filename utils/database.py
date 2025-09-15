#!/usr/bin/env python3
"""
Database Management Utilities
Handles data storage and retrieval for the supply chain disruption analysis system.

Features:
- Multi-database support (PostgreSQL, MongoDB, SQLite)
- Connection pooling and management
- Data validation and serialization
- Query optimization
- Backup and recovery utilities
- Data archiving and cleanup

Author: Supply Chain Alpha Team
Version: 1.0.0
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
from contextlib import asynccontextmanager
import os

# Database libraries
import asyncpg
import motor.motor_asyncio
import aiosqlite
import redis.asyncio as redis

# Data processing
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

# Utilities
import hashlib
import pickle
import gzip
from pathlib import Path


@dataclass
class DatabaseConfig:
    """Database configuration settings."""
    db_type: str  # 'postgresql', 'mongodb', 'sqlite', 'redis'
    host: str
    port: int
    database: str
    username: Optional[str] = None
    password: Optional[str] = None
    ssl_mode: str = 'prefer'
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30
    connection_timeout: int = 60
    
    # MongoDB specific
    replica_set: Optional[str] = None
    auth_source: str = 'admin'
    
    # SQLite specific
    file_path: Optional[str] = None
    
    # Redis specific
    redis_db: int = 0
    
    # Connection string override
    connection_string: Optional[str] = None


class DatabaseManager:
    """Manages database connections and operations."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize database manager.
        
        Args:
            config: Database configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        if config:
            self.config = DatabaseConfig(**config)
        else:
            self.config = self._load_default_config()
        
        # Connection pools
        self.pg_pool = None
        self.mongo_client = None
        self.mongo_db = None
        self.sqlite_connections = {}
        self.redis_client = None
        
        # Connection status
        self.is_connected = False
        self.connection_health = {}
        
        # Query cache
        self.query_cache = {}
        self.cache_ttl = 300  # 5 minutes
        
        # Performance metrics
        self.query_stats = {
            'total_queries': 0,
            'avg_query_time': 0,
            'slow_queries': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
    
    def _load_default_config(self) -> DatabaseConfig:
        """Load default database configuration.
        
        Returns:
            DatabaseConfig object
        """
        return DatabaseConfig(
            db_type=os.getenv('DB_TYPE', 'sqlite'),
            host=os.getenv('DB_HOST', 'localhost'),
            port=int(os.getenv('DB_PORT', '5432')),
            database=os.getenv('DB_NAME', 'supply_chain_alpha'),
            username=os.getenv('DB_USER'),
            password=os.getenv('DB_PASSWORD'),
            file_path=os.getenv('SQLITE_PATH', 'data/supply_chain_alpha.db')
        )
    
    async def connect(self) -> bool:
        """Establish database connections.
        
        Returns:
            True if successful
        """
        try:
            self.logger.info(f"Connecting to {self.config.db_type} database")
            
            if self.config.db_type == 'postgresql':
                await self._connect_postgresql()
            elif self.config.db_type == 'mongodb':
                await self._connect_mongodb()
            elif self.config.db_type == 'sqlite':
                await self._connect_sqlite()
            elif self.config.db_type == 'redis':
                await self._connect_redis()
            else:
                raise ValueError(f"Unsupported database type: {self.config.db_type}")
            
            self.is_connected = True
            self.logger.info("Database connection established successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to database: {e}")
            self.is_connected = False
            return False
    
    async def _connect_postgresql(self):
        """Connect to PostgreSQL database."""
        try:
            if self.config.connection_string:
                dsn = self.config.connection_string
            else:
                dsn = f"postgresql://{self.config.username}:{self.config.password}@{self.config.host}:{self.config.port}/{self.config.database}"
            
            self.pg_pool = await asyncpg.create_pool(
                dsn,
                min_size=1,
                max_size=self.config.pool_size,
                command_timeout=self.config.connection_timeout
            )
            
            # Test connection
            async with self.pg_pool.acquire() as conn:
                await conn.execute('SELECT 1')
            
            self.connection_health['postgresql'] = True
            
        except Exception as e:
            self.logger.error(f"PostgreSQL connection failed: {e}")
            raise
    
    async def _connect_mongodb(self):
        """Connect to MongoDB database."""
        try:
            if self.config.connection_string:
                connection_string = self.config.connection_string
            else:
                auth_part = ""
                if self.config.username and self.config.password:
                    auth_part = f"{self.config.username}:{self.config.password}@"
                
                replica_part = ""
                if self.config.replica_set:
                    replica_part = f"?replicaSet={self.config.replica_set}"
                
                connection_string = f"mongodb://{auth_part}{self.config.host}:{self.config.port}/{self.config.database}{replica_part}"
            
            self.mongo_client = motor.motor_asyncio.AsyncIOMotorClient(
                connection_string,
                serverSelectionTimeoutMS=self.config.connection_timeout * 1000
            )
            
            self.mongo_db = self.mongo_client[self.config.database]
            
            # Test connection
            await self.mongo_client.admin.command('ping')
            
            self.connection_health['mongodb'] = True
            
        except Exception as e:
            self.logger.error(f"MongoDB connection failed: {e}")
            raise
    
    async def _connect_sqlite(self):
        """Connect to SQLite database."""
        try:
            # Ensure directory exists
            if self.config.file_path:
                db_path = Path(self.config.file_path)
                db_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Test connection
                async with aiosqlite.connect(self.config.file_path) as conn:
                    await conn.execute('SELECT 1')
                
                self.connection_health['sqlite'] = True
            else:
                raise ValueError("SQLite file path not specified")
            
        except Exception as e:
            self.logger.error(f"SQLite connection failed: {e}")
            raise
    
    async def _connect_redis(self):
        """Connect to Redis database."""
        try:
            if self.config.connection_string:
                self.redis_client = redis.from_url(self.config.connection_string)
            else:
                self.redis_client = redis.Redis(
                    host=self.config.host,
                    port=self.config.port,
                    db=self.config.redis_db,
                    password=self.config.password,
                    socket_timeout=self.config.connection_timeout
                )
            
            # Test connection
            await self.redis_client.ping()
            
            self.connection_health['redis'] = True
            
        except Exception as e:
            self.logger.error(f"Redis connection failed: {e}")
            raise
    
    async def disconnect(self):
        """Close database connections."""
        try:
            if self.pg_pool:
                await self.pg_pool.close()
                self.pg_pool = None
            
            if self.mongo_client:
                self.mongo_client.close()
                self.mongo_client = None
                self.mongo_db = None
            
            for conn in self.sqlite_connections.values():
                await conn.close()
            self.sqlite_connections.clear()
            
            if self.redis_client:
                await self.redis_client.close()
                self.redis_client = None
            
            self.is_connected = False
            self.logger.info("Database connections closed")
            
        except Exception as e:
            self.logger.error(f"Error closing database connections: {e}")
    
    async def store_data(self, table_name: str, data: Union[Dict[str, Any], List[Dict[str, Any]]], 
                        upsert: bool = False) -> bool:
        """Store data in the database.
        
        Args:
            table_name: Name of the table/collection
            data: Data to store (single record or list of records)
            upsert: Whether to update existing records
            
        Returns:
            True if successful
        """
        try:
            start_time = datetime.utcnow()
            
            if not self.is_connected:
                await self.connect()
            
            # Ensure data is a list
            if isinstance(data, dict):
                data = [data]
            
            # Add timestamps if not present
            for record in data:
                if 'created_at' not in record:
                    record['created_at'] = datetime.utcnow()
                if 'updated_at' not in record:
                    record['updated_at'] = datetime.utcnow()
            
            success = False
            if self.config.db_type == 'postgresql':
                success = await self._store_postgresql(table_name, data, upsert)
            elif self.config.db_type == 'mongodb':
                success = await self._store_mongodb(table_name, data, upsert)
            elif self.config.db_type == 'sqlite':
                success = await self._store_sqlite(table_name, data, upsert)
            
            # Update performance metrics
            query_time = (datetime.utcnow() - start_time).total_seconds()
            self._update_query_stats(query_time)
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error storing data in {table_name}: {e}")
            return False
    
    async def _store_postgresql(self, table_name: str, data: List[Dict[str, Any]], upsert: bool) -> bool:
        """Store data in PostgreSQL."""
        try:
            async with self.pg_pool.acquire() as conn:
                # Create table if it doesn't exist
                await self._ensure_table_exists_pg(conn, table_name, data[0])
                
                for record in data:
                    columns = list(record.keys())
                    values = list(record.values())
                    placeholders = [f'${i+1}' for i in range(len(values))]
                    
                    if upsert:
                        # Use ON CONFLICT for upsert
                        conflict_columns = ['symbol'] if 'symbol' in columns else ['id']
                        update_clause = ', '.join([f"{col} = EXCLUDED.{col}" for col in columns if col not in conflict_columns])
                        
                        query = f"""
                            INSERT INTO {table_name} ({', '.join(columns)})
                            VALUES ({', '.join(placeholders)})
                            ON CONFLICT ({', '.join(conflict_columns)}) DO UPDATE SET
                            {update_clause}
                        """
                    else:
                        query = f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES ({', '.join(placeholders)})"
                    
                    await conn.execute(query, *values)
            
            return True
            
        except Exception as e:
            self.logger.error(f"PostgreSQL store error: {e}")
            return False
    
    async def _store_mongodb(self, table_name: str, data: List[Dict[str, Any]], upsert: bool) -> bool:
        """Store data in MongoDB."""
        try:
            collection = self.mongo_db[table_name]
            
            if upsert:
                # Use upsert operations
                operations = []
                for record in data:
                    filter_key = {'symbol': record['symbol']} if 'symbol' in record else {'_id': record.get('_id')}
                    operations.append(
                        {
                            'updateOne': {
                                'filter': filter_key,
                                'update': {'$set': record},
                                'upsert': True
                            }
                        }
                    )
                
                if operations:
                    await collection.bulk_write(operations)
            else:
                # Simple insert
                if len(data) == 1:
                    await collection.insert_one(data[0])
                else:
                    await collection.insert_many(data)
            
            return True
            
        except Exception as e:
            self.logger.error(f"MongoDB store error: {e}")
            return False
    
    async def _store_sqlite(self, table_name: str, data: List[Dict[str, Any]], upsert: bool) -> bool:
        """Store data in SQLite."""
        try:
            async with aiosqlite.connect(self.config.file_path) as conn:
                # Create table if it doesn't exist
                await self._ensure_table_exists_sqlite(conn, table_name, data[0])
                
                for record in data:
                    columns = list(record.keys())
                    values = list(record.values())
                    placeholders = ['?' for _ in values]
                    
                    if upsert:
                        # Use INSERT OR REPLACE
                        query = f"INSERT OR REPLACE INTO {table_name} ({', '.join(columns)}) VALUES ({', '.join(placeholders)})"
                    else:
                        query = f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES ({', '.join(placeholders)})"
                    
                    await conn.execute(query, values)
                
                await conn.commit()
            
            return True
            
        except Exception as e:
            self.logger.error(f"SQLite store error: {e}")
            return False
    
    async def query_data(self, table_name: str, 
                        filters: Optional[Dict[str, Any]] = None,
                        projection: Optional[List[str]] = None,
                        sort: Optional[List[Tuple[str, int]]] = None,
                        limit: Optional[int] = None,
                        offset: Optional[int] = None) -> List[Dict[str, Any]]:
        """Query data from the database.
        
        Args:
            table_name: Name of the table/collection
            filters: Filter conditions
            projection: Fields to include in results
            sort: Sort criteria [(field, direction)]
            limit: Maximum number of records
            offset: Number of records to skip
            
        Returns:
            List of matching records
        """
        try:
            start_time = datetime.utcnow()
            
            if not self.is_connected:
                await self.connect()
            
            # Check cache first
            cache_key = self._generate_cache_key(table_name, filters, projection, sort, limit, offset)
            cached_result = self._get_cached_result(cache_key)
            if cached_result is not None:
                self.query_stats['cache_hits'] += 1
                return cached_result
            
            self.query_stats['cache_misses'] += 1
            
            results = []
            if self.config.db_type == 'postgresql':
                results = await self._query_postgresql(table_name, filters, projection, sort, limit, offset)
            elif self.config.db_type == 'mongodb':
                results = await self._query_mongodb(table_name, filters, projection, sort, limit, offset)
            elif self.config.db_type == 'sqlite':
                results = await self._query_sqlite(table_name, filters, projection, sort, limit, offset)
            
            # Cache the result
            self._cache_result(cache_key, results)
            
            # Update performance metrics
            query_time = (datetime.utcnow() - start_time).total_seconds()
            self._update_query_stats(query_time)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error querying data from {table_name}: {e}")
            return []
    
    async def _query_postgresql(self, table_name: str, filters: Optional[Dict[str, Any]], 
                               projection: Optional[List[str]], sort: Optional[List[Tuple[str, int]]],
                               limit: Optional[int], offset: Optional[int]) -> List[Dict[str, Any]]:
        """Query data from PostgreSQL."""
        try:
            async with self.pg_pool.acquire() as conn:
                # Build query
                select_clause = ', '.join(projection) if projection else '*'
                query = f"SELECT {select_clause} FROM {table_name}"
                params = []
                
                # Add WHERE clause
                if filters:
                    where_conditions = []
                    param_count = 1
                    for key, value in filters.items():
                        if isinstance(value, dict):
                            # Handle operators like {'$gte': value}
                            for op, op_value in value.items():
                                sql_op = self._convert_mongo_op_to_sql(op)
                                where_conditions.append(f"{key} {sql_op} ${param_count}")
                                params.append(op_value)
                                param_count += 1
                        else:
                            where_conditions.append(f"{key} = ${param_count}")
                            params.append(value)
                            param_count += 1
                    
                    query += f" WHERE {' AND '.join(where_conditions)}"
                
                # Add ORDER BY clause
                if sort:
                    order_clauses = []
                    for field, direction in sort:
                        order_dir = 'ASC' if direction == 1 else 'DESC'
                        order_clauses.append(f"{field} {order_dir}")
                    query += f" ORDER BY {', '.join(order_clauses)}"
                
                # Add LIMIT and OFFSET
                if limit:
                    query += f" LIMIT {limit}"
                if offset:
                    query += f" OFFSET {offset}"
                
                # Execute query
                rows = await conn.fetch(query, *params)
                return [dict(row) for row in rows]
            
        except Exception as e:
            self.logger.error(f"PostgreSQL query error: {e}")
            return []
    
    async def _query_mongodb(self, table_name: str, filters: Optional[Dict[str, Any]], 
                            projection: Optional[List[str]], sort: Optional[List[Tuple[str, int]]],
                            limit: Optional[int], offset: Optional[int]) -> List[Dict[str, Any]]:
        """Query data from MongoDB."""
        try:
            collection = self.mongo_db[table_name]
            
            # Build query
            query_filter = filters or {}
            
            # Build projection
            projection_dict = None
            if projection:
                projection_dict = {field: 1 for field in projection}
            
            # Execute query
            cursor = collection.find(query_filter, projection_dict)
            
            # Add sort
            if sort:
                cursor = cursor.sort(sort)
            
            # Add skip and limit
            if offset:
                cursor = cursor.skip(offset)
            if limit:
                cursor = cursor.limit(limit)
            
            # Convert to list
            results = await cursor.to_list(length=None)
            
            # Convert ObjectId to string
            for result in results:
                if '_id' in result:
                    result['_id'] = str(result['_id'])
            
            return results
            
        except Exception as e:
            self.logger.error(f"MongoDB query error: {e}")
            return []
    
    async def _query_sqlite(self, table_name: str, filters: Optional[Dict[str, Any]], 
                           projection: Optional[List[str]], sort: Optional[List[Tuple[str, int]]],
                           limit: Optional[int], offset: Optional[int]) -> List[Dict[str, Any]]:
        """Query data from SQLite."""
        try:
            async with aiosqlite.connect(self.config.file_path) as conn:
                conn.row_factory = aiosqlite.Row
                
                # Build query
                select_clause = ', '.join(projection) if projection else '*'
                query = f"SELECT {select_clause} FROM {table_name}"
                params = []
                
                # Add WHERE clause
                if filters:
                    where_conditions = []
                    for key, value in filters.items():
                        if isinstance(value, dict):
                            # Handle operators
                            for op, op_value in value.items():
                                sql_op = self._convert_mongo_op_to_sql(op)
                                where_conditions.append(f"{key} {sql_op} ?")
                                params.append(op_value)
                        else:
                            where_conditions.append(f"{key} = ?")
                            params.append(value)
                    
                    query += f" WHERE {' AND '.join(where_conditions)}"
                
                # Add ORDER BY clause
                if sort:
                    order_clauses = []
                    for field, direction in sort:
                        order_dir = 'ASC' if direction == 1 else 'DESC'
                        order_clauses.append(f"{field} {order_dir}")
                    query += f" ORDER BY {', '.join(order_clauses)}"
                
                # Add LIMIT and OFFSET
                if limit:
                    query += f" LIMIT {limit}"
                if offset:
                    query += f" OFFSET {offset}"
                
                # Execute query
                cursor = await conn.execute(query, params)
                rows = await cursor.fetchall()
                
                return [dict(row) for row in rows]
            
        except Exception as e:
            self.logger.error(f"SQLite query error: {e}")
            return []
    
    def _convert_mongo_op_to_sql(self, mongo_op: str) -> str:
        """Convert MongoDB operators to SQL operators.
        
        Args:
            mongo_op: MongoDB operator (e.g., '$gte')
            
        Returns:
            SQL operator
        """
        op_mapping = {
            '$eq': '=',
            '$ne': '!=',
            '$gt': '>',
            '$gte': '>=',
            '$lt': '<',
            '$lte': '<=',
            '$in': 'IN',
            '$nin': 'NOT IN'
        }
        
        return op_mapping.get(mongo_op, '=')
    
    async def delete_data(self, table_name: str, filters: Dict[str, Any]) -> int:
        """Delete data from the database.
        
        Args:
            table_name: Name of the table/collection
            filters: Filter conditions for deletion
            
        Returns:
            Number of deleted records
        """
        try:
            if not self.is_connected:
                await self.connect()
            
            deleted_count = 0
            if self.config.db_type == 'postgresql':
                deleted_count = await self._delete_postgresql(table_name, filters)
            elif self.config.db_type == 'mongodb':
                deleted_count = await self._delete_mongodb(table_name, filters)
            elif self.config.db_type == 'sqlite':
                deleted_count = await self._delete_sqlite(table_name, filters)
            
            return deleted_count
            
        except Exception as e:
            self.logger.error(f"Error deleting data from {table_name}: {e}")
            return 0
    
    async def _delete_postgresql(self, table_name: str, filters: Dict[str, Any]) -> int:
        """Delete data from PostgreSQL."""
        try:
            async with self.pg_pool.acquire() as conn:
                where_conditions = []
                params = []
                param_count = 1
                
                for key, value in filters.items():
                    if isinstance(value, dict):
                        for op, op_value in value.items():
                            sql_op = self._convert_mongo_op_to_sql(op)
                            where_conditions.append(f"{key} {sql_op} ${param_count}")
                            params.append(op_value)
                            param_count += 1
                    else:
                        where_conditions.append(f"{key} = ${param_count}")
                        params.append(value)
                        param_count += 1
                
                query = f"DELETE FROM {table_name} WHERE {' AND '.join(where_conditions)}"
                result = await conn.execute(query, *params)
                
                # Extract number of deleted rows from result
                return int(result.split()[-1]) if result else 0
            
        except Exception as e:
            self.logger.error(f"PostgreSQL delete error: {e}")
            return 0
    
    async def _delete_mongodb(self, table_name: str, filters: Dict[str, Any]) -> int:
        """Delete data from MongoDB."""
        try:
            collection = self.mongo_db[table_name]
            result = await collection.delete_many(filters)
            return result.deleted_count
            
        except Exception as e:
            self.logger.error(f"MongoDB delete error: {e}")
            return 0
    
    async def _delete_sqlite(self, table_name: str, filters: Dict[str, Any]) -> int:
        """Delete data from SQLite."""
        try:
            async with aiosqlite.connect(self.config.file_path) as conn:
                where_conditions = []
                params = []
                
                for key, value in filters.items():
                    if isinstance(value, dict):
                        for op, op_value in value.items():
                            sql_op = self._convert_mongo_op_to_sql(op)
                            where_conditions.append(f"{key} {sql_op} ?")
                            params.append(op_value)
                    else:
                        where_conditions.append(f"{key} = ?")
                        params.append(value)
                
                query = f"DELETE FROM {table_name} WHERE {' AND '.join(where_conditions)}"
                cursor = await conn.execute(query, params)
                await conn.commit()
                
                return cursor.rowcount
            
        except Exception as e:
            self.logger.error(f"SQLite delete error: {e}")
            return 0
    
    async def _ensure_table_exists_pg(self, conn, table_name: str, sample_data: Dict[str, Any]):
        """Ensure PostgreSQL table exists."""
        try:
            # Check if table exists
            exists = await conn.fetchval(
                "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = $1)",
                table_name
            )
            
            if not exists:
                # Create table based on sample data
                columns = []
                for key, value in sample_data.items():
                    if isinstance(value, str):
                        col_type = 'TEXT'
                    elif isinstance(value, int):
                        col_type = 'INTEGER'
                    elif isinstance(value, float):
                        col_type = 'REAL'
                    elif isinstance(value, bool):
                        col_type = 'BOOLEAN'
                    elif isinstance(value, datetime):
                        col_type = 'TIMESTAMP'
                    else:
                        col_type = 'TEXT'  # Default to TEXT for complex types
                    
                    columns.append(f"{key} {col_type}")
                
                create_query = f"CREATE TABLE {table_name} ({', '.join(columns)})"
                await conn.execute(create_query)
                
                # Create index on symbol if it exists
                if 'symbol' in sample_data:
                    await conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_symbol ON {table_name} (symbol)")
                
                # Create index on timestamp if it exists
                if 'timestamp' in sample_data:
                    await conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_timestamp ON {table_name} (timestamp)")
            
        except Exception as e:
            self.logger.error(f"Error ensuring PostgreSQL table exists: {e}")
            raise
    
    async def _ensure_table_exists_sqlite(self, conn, table_name: str, sample_data: Dict[str, Any]):
        """Ensure SQLite table exists."""
        try:
            # Check if table exists
            cursor = await conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (table_name,)
            )
            exists = await cursor.fetchone()
            
            if not exists:
                # Create table based on sample data
                columns = []
                for key, value in sample_data.items():
                    if isinstance(value, str):
                        col_type = 'TEXT'
                    elif isinstance(value, int):
                        col_type = 'INTEGER'
                    elif isinstance(value, float):
                        col_type = 'REAL'
                    elif isinstance(value, bool):
                        col_type = 'INTEGER'  # SQLite doesn't have native boolean
                    elif isinstance(value, datetime):
                        col_type = 'TEXT'  # Store as ISO string
                    else:
                        col_type = 'TEXT'  # Default to TEXT
                    
                    columns.append(f"{key} {col_type}")
                
                create_query = f"CREATE TABLE {table_name} ({', '.join(columns)})"
                await conn.execute(create_query)
                
                # Create indexes
                if 'symbol' in sample_data:
                    await conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_symbol ON {table_name} (symbol)")
                
                if 'timestamp' in sample_data:
                    await conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_timestamp ON {table_name} (timestamp)")
                
                await conn.commit()
            
        except Exception as e:
            self.logger.error(f"Error ensuring SQLite table exists: {e}")
            raise
    
    def _generate_cache_key(self, table_name: str, filters: Optional[Dict[str, Any]], 
                           projection: Optional[List[str]], sort: Optional[List[Tuple[str, int]]],
                           limit: Optional[int], offset: Optional[int]) -> str:
        """Generate cache key for query."""
        key_data = {
            'table': table_name,
            'filters': filters,
            'projection': projection,
            'sort': sort,
            'limit': limit,
            'offset': offset
        }
        
        key_string = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _get_cached_result(self, cache_key: str) -> Optional[List[Dict[str, Any]]]:
        """Get cached query result."""
        if cache_key in self.query_cache:
            cached_data, timestamp = self.query_cache[cache_key]
            if (datetime.utcnow() - timestamp).total_seconds() < self.cache_ttl:
                return cached_data
            else:
                # Remove expired cache entry
                del self.query_cache[cache_key]
        
        return None
    
    def _cache_result(self, cache_key: str, result: List[Dict[str, Any]]):
        """Cache query result."""
        self.query_cache[cache_key] = (result, datetime.utcnow())
        
        # Limit cache size
        if len(self.query_cache) > 1000:
            # Remove oldest entries
            sorted_cache = sorted(self.query_cache.items(), key=lambda x: x[1][1])
            for key, _ in sorted_cache[:100]:  # Remove oldest 100 entries
                del self.query_cache[key]
    
    def _update_query_stats(self, query_time: float):
        """Update query performance statistics."""
        self.query_stats['total_queries'] += 1
        
        # Update average query time
        total_time = self.query_stats['avg_query_time'] * (self.query_stats['total_queries'] - 1)
        self.query_stats['avg_query_time'] = (total_time + query_time) / self.query_stats['total_queries']
        
        # Track slow queries (> 1 second)
        if query_time > 1.0:
            self.query_stats['slow_queries'] += 1
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get database health status.
        
        Returns:
            Dictionary with health information
        """
        try:
            health_status = {
                'connected': self.is_connected,
                'database_type': self.config.db_type,
                'connection_health': self.connection_health.copy(),
                'query_stats': self.query_stats.copy(),
                'cache_size': len(self.query_cache)
            }
            
            # Test connections
            if self.config.db_type == 'postgresql' and self.pg_pool:
                try:
                    async with self.pg_pool.acquire() as conn:
                        await conn.execute('SELECT 1')
                    health_status['postgresql_status'] = 'healthy'
                except Exception as e:
                    health_status['postgresql_status'] = f'error: {str(e)}'
            
            if self.config.db_type == 'mongodb' and self.mongo_client:
                try:
                    await self.mongo_client.admin.command('ping')
                    health_status['mongodb_status'] = 'healthy'
                except Exception as e:
                    health_status['mongodb_status'] = f'error: {str(e)}'
            
            if self.config.db_type == 'redis' and self.redis_client:
                try:
                    await self.redis_client.ping()
                    health_status['redis_status'] = 'healthy'
                except Exception as e:
                    health_status['redis_status'] = f'error: {str(e)}'
            
            return health_status
            
        except Exception as e:
            self.logger.error(f"Error getting health status: {e}")
            return {'error': str(e)}
    
    async def backup_data(self, backup_path: str, tables: Optional[List[str]] = None) -> bool:
        """Create database backup.
        
        Args:
            backup_path: Path to store backup
            tables: List of tables to backup (None for all)
            
        Returns:
            True if successful
        """
        try:
            self.logger.info(f"Creating database backup to {backup_path}")
            
            backup_data = {
                'timestamp': datetime.utcnow(),
                'database_type': self.config.db_type,
                'tables': {}
            }
            
            # Get list of tables to backup
            if tables is None:
                tables = await self._get_all_tables()
            
            # Backup each table
            for table in tables:
                try:
                    data = await self.query_data(table)
                    backup_data['tables'][table] = data
                    self.logger.info(f"Backed up {len(data)} records from {table}")
                except Exception as e:
                    self.logger.error(f"Error backing up table {table}: {e}")
            
            # Save backup file
            backup_path_obj = Path(backup_path)
            backup_path_obj.parent.mkdir(parents=True, exist_ok=True)
            
            # Compress and save
            with gzip.open(backup_path, 'wb') as f:
                pickle.dump(backup_data, f)
            
            self.logger.info(f"Database backup completed: {backup_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating database backup: {e}")
            return False
    
    async def restore_data(self, backup_path: str, overwrite: bool = False) -> bool:
        """Restore database from backup.
        
        Args:
            backup_path: Path to backup file
            overwrite: Whether to overwrite existing data
            
        Returns:
            True if successful
        """
        try:
            self.logger.info(f"Restoring database from {backup_path}")
            
            # Load backup file
            with gzip.open(backup_path, 'rb') as f:
                backup_data = pickle.load(f)
            
            # Restore each table
            for table_name, table_data in backup_data['tables'].items():
                try:
                    if overwrite:
                        # Clear existing data
                        await self.delete_data(table_name, {})
                    
                    # Insert backup data
                    if table_data:
                        await self.store_data(table_name, table_data)
                    
                    self.logger.info(f"Restored {len(table_data)} records to {table_name}")
                    
                except Exception as e:
                    self.logger.error(f"Error restoring table {table_name}: {e}")
            
            self.logger.info("Database restore completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Error restoring database: {e}")
            return False
    
    async def _get_all_tables(self) -> List[str]:
        """Get list of all tables in the database.
        
        Returns:
            List of table names
        """
        try:
            if self.config.db_type == 'postgresql':
                async with self.pg_pool.acquire() as conn:
                    rows = await conn.fetch(
                        "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
                    )
                    return [row['table_name'] for row in rows]
            
            elif self.config.db_type == 'mongodb':
                return await self.mongo_db.list_collection_names()
            
            elif self.config.db_type == 'sqlite':
                async with aiosqlite.connect(self.config.file_path) as conn:
                    cursor = await conn.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    )
                    rows = await cursor.fetchall()
                    return [row[0] for row in rows]
            
            return []
            
        except Exception as e:
            self.logger.error(f"Error getting table list: {e}")
            return []
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get database performance statistics.
        
        Returns:
            Dictionary with performance metrics
        """
        stats = self.query_stats.copy()
        
        # Calculate additional metrics
        if stats['total_queries'] > 0:
            stats['cache_hit_rate'] = stats['cache_hits'] / (stats['cache_hits'] + stats['cache_misses'])
            stats['slow_query_rate'] = stats['slow_queries'] / stats['total_queries']
        else:
            stats['cache_hit_rate'] = 0
            stats['slow_query_rate'] = 0
        
        return stats
    
    async def optimize_database(self) -> bool:
        """Optimize database performance.
        
        Returns:
            True if successful
        """
        try:
            self.logger.info("Optimizing database performance")
            
            if self.config.db_type == 'postgresql':
                async with self.pg_pool.acquire() as conn:
                    # Run VACUUM and ANALYZE
                    await conn.execute('VACUUM ANALYZE')
            
            elif self.config.db_type == 'sqlite':
                async with aiosqlite.connect(self.config.file_path) as conn:
                    # Run VACUUM and ANALYZE
                    await conn.execute('VACUUM')
                    await conn.execute('ANALYZE')
                    await conn.commit()
            
            # Clear query cache to free memory
            self.query_cache.clear()
            
            self.logger.info("Database optimization completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Error optimizing database: {e}")
            return False
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()