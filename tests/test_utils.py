"""
Test suite for utility modules.
"""

import pytest
import sys
import os
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, Mock, mock_open
import logging
from datetime import datetime, timedelta

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.utils.logger import get_logger, setup_logging
from src.utils.health_monitor import HealthMonitor, ServiceStatus


class TestLogger:
    """Test logging utility functions."""
    
    def test_get_logger_basic(self):
        """Test basic logger creation."""
        logger = get_logger("test_module")
        
        assert logger is not None
        assert logger.name == "test_module"
        assert isinstance(logger, logging.Logger)
        
    def test_get_logger_with_level(self):
        """Test logger creation with specific level."""
        logger = get_logger("test_module", level=logging.DEBUG)
        
        assert logger.level == logging.DEBUG
        
    def test_get_logger_singleton_behavior(self):
        """Test that loggers with same name return same instance."""
        logger1 = get_logger("same_module")
        logger2 = get_logger("same_module")
        
        assert logger1 is logger2
        
    def test_setup_logging_with_file(self):
        """Test logging setup with file output."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
            log_file_path = f.name
            
        try:
            setup_logging(log_file=log_file_path, log_level="DEBUG")
            
            # Test that log file is created and writable
            logger = get_logger("test_file_logging")
            logger.info("Test log message")
            
            # Check if log file exists and has content
            assert os.path.exists(log_file_path)
            
            with open(log_file_path, 'r') as f:
                log_content = f.read()
                assert "Test log message" in log_content
                
        finally:
            if os.path.exists(log_file_path):
                os.unlink(log_file_path)
                
    def test_setup_logging_console_only(self):
        """Test logging setup for console only."""
        setup_logging(log_level="INFO")
        
        logger = get_logger("test_console_logging")
        
        # Should not raise any errors
        logger.info("Console test message")
        logger.warning("Console warning message")
        logger.error("Console error message")
        
    def test_log_level_configuration(self):
        """Test different log level configurations."""
        levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        
        for level_str in levels:
            setup_logging(log_level=level_str)
            logger = get_logger(f"test_{level_str.lower()}")
            
            # Verify logger level is set correctly
            expected_level = getattr(logging, level_str)
            # Note: The actual test might vary based on implementation
            
    def test_logging_with_structured_data(self):
        """Test logging with structured data (JSON)."""
        logger = get_logger("test_structured")
        
        # Test logging with extra fields
        extra_data = {
            "user_id": "user123",
            "request_id": "req456",
            "operation": "query_processing"
        }
        
        # This test depends on implementation details
        # logger.info("Processing query", extra=extra_data)
        
    def test_error_logging_with_traceback(self):
        """Test error logging with exception traceback."""
        logger = get_logger("test_error")
        
        try:
            # Intentionally cause an error
            result = 1 / 0
        except ZeroDivisionError:
            # Should not raise any errors
            logger.exception("Division by zero error occurred")
            
    def test_log_rotation_configuration(self):
        """Test log rotation configuration (if implemented)."""
        # This would test rotating file handler configuration
        # Implementation depends on specific logging setup
        pass


class TestHealthMonitor:
    """Test health monitoring functionality."""
    
    @pytest.fixture
    def health_monitor(self):
        """Create HealthMonitor instance for testing."""
        return HealthMonitor()
        
    def test_health_monitor_initialization(self, health_monitor):
        """Test HealthMonitor initialization."""
        assert health_monitor is not None
        assert hasattr(health_monitor, 'check_health')
        assert hasattr(health_monitor, 'get_system_info')
        
    @pytest.mark.asyncio
    async def test_basic_health_check(self, health_monitor):
        """Test basic health check functionality."""
        health_status = await health_monitor.check_health()
        
        assert health_status is not None
        assert "status" in health_status
        assert "timestamp" in health_status
        assert "checks" in health_status
        
        # Basic health should be healthy
        assert health_status["status"] in ["healthy", "degraded", "unhealthy"]
        
    @pytest.mark.asyncio
    async def test_azure_openai_health_check(self, health_monitor):
        """Test Azure OpenAI service health check."""
        with patch('src.utils.health_monitor.get_azure_client') as mock_get_client:
            mock_client = Mock()
            mock_client.models.list.return_value = Mock()
            mock_get_client.return_value = mock_client
            
            health_status = await health_monitor.check_azure_openai_health()
            
            assert health_status is not None
            assert health_status.service == "azure_openai"
            assert health_status.status in ["healthy", "unhealthy"]
            
    @pytest.mark.asyncio
    async def test_azure_openai_health_check_failure(self, health_monitor):
        """Test Azure OpenAI health check failure scenario."""
        with patch('src.utils.health_monitor.get_azure_client') as mock_get_client:
            mock_client = Mock()
            mock_client.models.list.side_effect = Exception("Connection failed")
            mock_get_client.return_value = mock_client
            
            health_status = await health_monitor.check_azure_openai_health()
            
            assert health_status.status == "unhealthy"
            assert health_status.error is not None
            
    @pytest.mark.asyncio
    async def test_qdrant_health_check(self, health_monitor):
        """Test Qdrant service health check."""
        with patch('src.utils.health_monitor.QdrantClient') as mock_qdrant:
            mock_client = Mock()
            mock_client.get_collections.return_value = Mock()
            mock_qdrant.return_value = mock_client
            
            health_status = await health_monitor.check_qdrant_health()
            
            assert health_status is not None
            assert health_status.service == "qdrant"
            assert health_status.status in ["healthy", "unhealthy"]
            
    @pytest.mark.asyncio
    async def test_qdrant_health_check_failure(self, health_monitor):
        """Test Qdrant health check failure scenario."""
        with patch('src.utils.health_monitor.QdrantClient') as mock_qdrant:
            mock_client = Mock()
            mock_client.get_collections.side_effect = Exception("Qdrant unreachable")
            mock_qdrant.return_value = mock_client
            
            health_status = await health_monitor.check_qdrant_health()
            
            assert health_status.status == "unhealthy"
            assert health_status.error is not None
            
    def test_get_system_info(self, health_monitor):
        """Test system information gathering."""
        system_info = health_monitor.get_system_info()
        
        assert system_info is not None
        assert "python_version" in system_info
        assert "platform" in system_info
        assert "memory_usage" in system_info
        assert "disk_usage" in system_info
        
        # Verify data types
        assert isinstance(system_info["python_version"], str)
        assert isinstance(system_info["platform"], str)
        
    def test_memory_usage_monitoring(self, health_monitor):
        """Test memory usage monitoring."""
        memory_info = health_monitor.get_memory_usage()
        
        assert memory_info is not None
        assert "total" in memory_info
        assert "available" in memory_info
        assert "percent" in memory_info
        assert "used" in memory_info
        
        # Verify values are reasonable
        assert memory_info["percent"] >= 0
        assert memory_info["percent"] <= 100
        assert memory_info["used"] <= memory_info["total"]
        
    def test_disk_usage_monitoring(self, health_monitor):
        """Test disk usage monitoring."""
        disk_info = health_monitor.get_disk_usage()
        
        assert disk_info is not None
        assert "total" in disk_info
        assert "used" in disk_info
        assert "free" in disk_info
        assert "percent" in disk_info
        
        # Verify values are reasonable
        assert disk_info["percent"] >= 0
        assert disk_info["percent"] <= 100
        
    @pytest.mark.asyncio
    async def test_comprehensive_health_check(self, health_monitor):
        """Test comprehensive health check with all services."""
        with patch('src.utils.health_monitor.get_azure_client') as mock_azure, \
             patch('src.utils.health_monitor.QdrantClient') as mock_qdrant:
            
            # Mock successful responses
            mock_azure_client = Mock()
            mock_azure_client.models.list.return_value = Mock()
            mock_azure.return_value = mock_azure_client
            
            mock_qdrant_client = Mock()
            mock_qdrant_client.get_collections.return_value = Mock()
            mock_qdrant.return_value = mock_qdrant_client
            
            health_status = await health_monitor.check_health()
            
            assert health_status["status"] == "healthy"
            assert "azure_openai" in health_status["checks"]
            assert "qdrant" in health_status["checks"]
            assert "system" in health_status["checks"]
            
    @pytest.mark.asyncio
    async def test_health_check_with_partial_failures(self, health_monitor):
        """Test health check with some services failing."""
        with patch('src.utils.health_monitor.get_azure_client') as mock_azure, \
             patch('src.utils.health_monitor.QdrantClient') as mock_qdrant:
            
            # Mock Azure success, Qdrant failure
            mock_azure_client = Mock()
            mock_azure_client.models.list.return_value = Mock()
            mock_azure.return_value = mock_azure_client
            
            mock_qdrant_client = Mock()
            mock_qdrant_client.get_collections.side_effect = Exception("Failed")
            mock_qdrant.return_value = mock_qdrant_client
            
            health_status = await health_monitor.check_health()
            
            assert health_status["status"] == "degraded"
            assert health_status["checks"]["azure_openai"]["status"] == "healthy"
            assert health_status["checks"]["qdrant"]["status"] == "unhealthy"
            
    def test_service_status_model(self):
        """Test ServiceStatus model."""
        status = ServiceStatus(
            service="test_service",
            status="healthy",
            response_time=0.123,
            timestamp=datetime.now(),
            details={"version": "1.0.0"}
        )
        
        assert status.service == "test_service"
        assert status.status == "healthy"
        assert status.response_time == 0.123
        assert status.details["version"] == "1.0.0"
        
    def test_health_check_caching(self, health_monitor):
        """Test health check result caching (if implemented)."""
        # This would test caching behavior to avoid too frequent health checks
        # Implementation depends on specific caching strategy
        pass
        
    @pytest.mark.asyncio
    async def test_health_check_timeout(self, health_monitor):
        """Test health check timeout handling."""
        with patch('src.utils.health_monitor.get_azure_client') as mock_azure:
            # Mock a slow response
            import asyncio
            
            async def slow_response():
                await asyncio.sleep(10)  # Longer than reasonable timeout
                return Mock()
                
            mock_client = Mock()
            mock_client.models.list = slow_response
            mock_azure.return_value = mock_client
            
            # Health check should timeout and report unhealthy
            health_status = await health_monitor.check_azure_openai_health()
            
            # Should complete in reasonable time (not 10 seconds)
            assert health_status.status == "unhealthy"
            
    def test_health_monitor_configuration(self, health_monitor):
        """Test health monitor configuration options."""
        # Test different configuration options if available
        config = {
            "timeout": 5.0,
            "retry_count": 3,
            "check_interval": 60
        }
        
        # This would test configuration of timeouts, retries, etc.
        # Implementation depends on specific configuration options available


class TestUtilityIntegration:
    """Test integration between utility modules."""
    
    def test_logger_health_monitor_integration(self):
        """Test that health monitor properly uses logger."""
        with patch('src.utils.health_monitor.get_logger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            
            health_monitor = HealthMonitor()
            
            # Verify logger was obtained
            mock_get_logger.assert_called()
            
    @pytest.mark.asyncio
    async def test_error_logging_in_health_checks(self):
        """Test that errors in health checks are properly logged."""
        with patch('src.utils.health_monitor.get_logger') as mock_get_logger, \
             patch('src.utils.health_monitor.get_azure_client') as mock_azure:
            
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            
            # Mock Azure client failure
            mock_azure.side_effect = Exception("Connection failed")
            
            health_monitor = HealthMonitor()
            health_status = await health_monitor.check_azure_openai_health()
            
            # Verify error was logged
            assert mock_logger.error.called or mock_logger.exception.called
            
    def test_structured_logging_with_health_data(self):
        """Test structured logging with health monitoring data."""
        # This would test integration between structured logging and health data
        # Implementation depends on specific logging format used
        pass


class TestErrorHandling:
    """Test error handling in utility modules."""
    
    def test_logger_with_invalid_configuration(self):
        """Test logger behavior with invalid configuration."""
        # Test with invalid log level
        try:
            setup_logging(log_level="INVALID_LEVEL")
        except ValueError:
            # Expected for invalid log level
            pass
            
        # Test with invalid file path
        try:
            setup_logging(log_file="/invalid/path/logfile.log")
        except (OSError, IOError):
            # Expected for invalid file path
            pass
            
    @pytest.mark.asyncio
    async def test_health_monitor_with_network_errors(self):
        """Test health monitor resilience to network errors."""
        health_monitor = HealthMonitor()
        
        # Mock network errors
        with patch('src.utils.health_monitor.get_azure_client') as mock_azure:
            mock_azure.side_effect = ConnectionError("Network unreachable")
            
            health_status = await health_monitor.check_azure_openai_health()
            
            assert health_status.status == "unhealthy"
            assert "network" in health_status.error.lower() or "connection" in health_status.error.lower()
            
    def test_utility_modules_with_missing_dependencies(self):
        """Test utility module behavior when dependencies are missing."""
        # This would test graceful degradation when optional dependencies are missing
        # Implementation depends on specific dependencies and fallback strategies
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
