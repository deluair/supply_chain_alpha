"""Rate limiting utilities for API calls and data collection.

This module provides rate limiting functionality to ensure compliance with
API rate limits and prevent overwhelming external services.
"""

import asyncio
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, Optional, Any
from datetime import datetime, timedelta


@dataclass
class RateLimit:
    """Rate limit configuration."""
    requests_per_second: float
    requests_per_minute: Optional[float] = None
    requests_per_hour: Optional[float] = None
    burst_size: Optional[int] = None


class RateLimiter:
    """Token bucket rate limiter with multiple time windows."""
    
    def __init__(self, rate_limit: RateLimit):
        """Initialize rate limiter.
        
        Args:
            rate_limit: Rate limit configuration
        """
        self.rate_limit = rate_limit
        self.tokens = rate_limit.burst_size or int(rate_limit.requests_per_second)
        self.max_tokens = self.tokens
        self.last_update = time.time()
        
        # Track requests for different time windows
        self.request_times: deque = deque()
        self.minute_requests: deque = deque()
        self.hour_requests: deque = deque()
        
    async def acquire(self, tokens: int = 1) -> bool:
        """Acquire tokens from the rate limiter.
        
        Args:
            tokens: Number of tokens to acquire
            
        Returns:
            True if tokens were acquired, False otherwise
        """
        current_time = time.time()
        
        # Update token bucket
        self._update_tokens(current_time)
        
        # Check all rate limits
        if not self._check_rate_limits(current_time):
            return False
            
        if self.tokens >= tokens:
            self.tokens -= tokens
            self._record_request(current_time)
            return True
            
        return False
        
    async def wait_for_tokens(self, tokens: int = 1) -> None:
        """Wait until tokens are available.
        
        Args:
            tokens: Number of tokens needed
        """
        while not await self.acquire(tokens):
            await asyncio.sleep(0.1)
            
    def _update_tokens(self, current_time: float) -> None:
        """Update token bucket based on elapsed time."""
        elapsed = current_time - self.last_update
        self.tokens = min(
            self.max_tokens,
            self.tokens + elapsed * self.rate_limit.requests_per_second
        )
        self.last_update = current_time
        
    def _check_rate_limits(self, current_time: float) -> bool:
        """Check if request would violate any rate limits."""
        # Clean old requests
        self._clean_old_requests(current_time)
        
        # Check per-minute limit
        if (self.rate_limit.requests_per_minute and 
            len(self.minute_requests) >= self.rate_limit.requests_per_minute):
            return False
            
        # Check per-hour limit
        if (self.rate_limit.requests_per_hour and 
            len(self.hour_requests) >= self.rate_limit.requests_per_hour):
            return False
            
        return True
        
    def _record_request(self, current_time: float) -> None:
        """Record a request timestamp."""
        self.request_times.append(current_time)
        self.minute_requests.append(current_time)
        self.hour_requests.append(current_time)
        
    def _clean_old_requests(self, current_time: float) -> None:
        """Remove old request timestamps."""
        # Remove requests older than 1 second
        while (self.request_times and 
               current_time - self.request_times[0] > 1.0):
            self.request_times.popleft()
            
        # Remove requests older than 1 minute
        while (self.minute_requests and 
               current_time - self.minute_requests[0] > 60.0):
            self.minute_requests.popleft()
            
        # Remove requests older than 1 hour
        while (self.hour_requests and 
               current_time - self.hour_requests[0] > 3600.0):
            self.hour_requests.popleft()
            
    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics."""
        current_time = time.time()
        self._clean_old_requests(current_time)
        
        return {
            'available_tokens': int(self.tokens),
            'max_tokens': self.max_tokens,
            'requests_last_second': len(self.request_times),
            'requests_last_minute': len(self.minute_requests),
            'requests_last_hour': len(self.hour_requests),
            'rate_limit': {
                'requests_per_second': self.rate_limit.requests_per_second,
                'requests_per_minute': self.rate_limit.requests_per_minute,
                'requests_per_hour': self.rate_limit.requests_per_hour,
                'burst_size': self.rate_limit.burst_size
            }
        }


class MultiServiceRateLimiter:
    """Rate limiter for multiple services with different limits."""
    
    def __init__(self):
        """Initialize multi-service rate limiter."""
        self.limiters: Dict[str, RateLimiter] = {}
        
    def add_service(self, service_name: str, rate_limit: RateLimit) -> None:
        """Add a service with its rate limit.
        
        Args:
            service_name: Name of the service
            rate_limit: Rate limit configuration
        """
        self.limiters[service_name] = RateLimiter(rate_limit)
        
    async def acquire(self, service_name: str, tokens: int = 1) -> bool:
        """Acquire tokens for a specific service.
        
        Args:
            service_name: Name of the service
            tokens: Number of tokens to acquire
            
        Returns:
            True if tokens were acquired, False otherwise
        """
        if service_name not in self.limiters:
            return True  # No rate limit configured
            
        return await self.limiters[service_name].acquire(tokens)
        
    async def wait_for_tokens(self, service_name: str, tokens: int = 1) -> None:
        """Wait for tokens for a specific service.
        
        Args:
            service_name: Name of the service
            tokens: Number of tokens needed
        """
        if service_name not in self.limiters:
            return  # No rate limit configured
            
        await self.limiters[service_name].wait_for_tokens(tokens)
        
    def get_stats(self, service_name: Optional[str] = None) -> Dict[str, Any]:
        """Get rate limiter statistics.
        
        Args:
            service_name: Specific service name, or None for all services
            
        Returns:
            Statistics for the service(s)
        """
        if service_name:
            if service_name in self.limiters:
                return {service_name: self.limiters[service_name].get_stats()}
            else:
                return {service_name: 'No rate limit configured'}
        else:
            return {
                name: limiter.get_stats() 
                for name, limiter in self.limiters.items()
            }


class AdaptiveRateLimiter(RateLimiter):
    """Rate limiter that adapts based on response times and errors."""
    
    def __init__(self, rate_limit: RateLimit, adaptation_factor: float = 0.1):
        """Initialize adaptive rate limiter.
        
        Args:
            rate_limit: Initial rate limit configuration
            adaptation_factor: How quickly to adapt (0.0 to 1.0)
        """
        super().__init__(rate_limit)
        self.adaptation_factor = adaptation_factor
        self.base_rate = rate_limit.requests_per_second
        self.current_rate = rate_limit.requests_per_second
        self.response_times: deque = deque(maxlen=100)
        self.error_count = 0
        self.success_count = 0
        
    def record_response(self, response_time: float, success: bool) -> None:
        """Record API response for adaptation.
        
        Args:
            response_time: Response time in seconds
            success: Whether the request was successful
        """
        self.response_times.append(response_time)
        
        if success:
            self.success_count += 1
        else:
            self.error_count += 1
            
        # Adapt rate based on performance
        self._adapt_rate()
        
    def _adapt_rate(self) -> None:
        """Adapt the rate limit based on recent performance."""
        if len(self.response_times) < 10:
            return
            
        avg_response_time = sum(self.response_times) / len(self.response_times)
        total_requests = self.success_count + self.error_count
        error_rate = self.error_count / total_requests if total_requests > 0 else 0
        
        # Decrease rate if high error rate or slow responses
        if error_rate > 0.1 or avg_response_time > 2.0:
            self.current_rate *= (1 - self.adaptation_factor)
        # Increase rate if low error rate and fast responses
        elif error_rate < 0.01 and avg_response_time < 0.5:
            self.current_rate *= (1 + self.adaptation_factor)
            
        # Keep within reasonable bounds
        self.current_rate = max(0.1, min(self.base_rate * 2, self.current_rate))
        
        # Update rate limit
        self.rate_limit.requests_per_second = self.current_rate
        
    def get_adaptation_stats(self) -> Dict[str, Any]:
        """Get adaptation statistics."""
        total_requests = self.success_count + self.error_count
        avg_response_time = (
            sum(self.response_times) / len(self.response_times) 
            if self.response_times else 0
        )
        
        return {
            'base_rate': self.base_rate,
            'current_rate': self.current_rate,
            'adaptation_factor': self.adaptation_factor,
            'total_requests': total_requests,
            'success_count': self.success_count,
            'error_count': self.error_count,
            'error_rate': self.error_count / total_requests if total_requests > 0 else 0,
            'avg_response_time': avg_response_time,
            'recent_response_times': list(self.response_times)[-10:]
        }


# Default rate limiters for common services
DEFAULT_RATE_LIMITS = {
    'marine_traffic': RateLimit(
        requests_per_second=2.0,
        requests_per_minute=100,
        requests_per_hour=5000,
        burst_size=5
    ),
    'vessel_finder': RateLimit(
        requests_per_second=1.0,
        requests_per_minute=50,
        requests_per_hour=2000,
        burst_size=3
    ),
    'port_authority': RateLimit(
        requests_per_second=0.5,
        requests_per_minute=20,
        requests_per_hour=500,
        burst_size=2
    ),
    'freight_apis': RateLimit(
        requests_per_second=1.5,
        requests_per_minute=75,
        requests_per_hour=3000,
        burst_size=4
    )
}


def create_default_rate_limiter() -> MultiServiceRateLimiter:
    """Create a rate limiter with default service configurations.
    
    Returns:
        Configured multi-service rate limiter
    """
    limiter = MultiServiceRateLimiter()
    
    for service_name, rate_limit in DEFAULT_RATE_LIMITS.items():
        limiter.add_service(service_name, rate_limit)
        
    return limiter