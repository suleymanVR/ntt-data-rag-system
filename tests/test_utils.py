"""
Comprehensive tests for utility modules (health_monitor and logger).
Tests the real API methods and functionality.
"""

import pytest
import time
import tempfile
import logging
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.utils.health_monitor import HealthMonitor
from src.utils.logger import (
    setup_logging, get_logger, LoggerMixin, 
    StructuredLogger, PerformanceLogger, ErrorTracker
)


class TestHealthMonitor:
    """Test HealthMonitor functionality."""
    
    def setup_method(self):
        """Set up test health monitor."""
        self.monitor = HealthMonitor(max_history=10)
    
    def test_health_monitor_initialization(self):
        """Test HealthMonitor initialization."""
        monitor = HealthMonitor(max_history=5)
        assert monitor.max_history == 5
        assert len(monitor.question_metrics) == 0
        assert len(monitor.batch_metrics) == 0
        assert len(monitor.error_counts) == 0
    
    def test_record_question_metric(self):
        """Test recording question metrics with correct parameters."""
        self.monitor.record_question_metric(
            request_id="test_123",
            question_length=50,
            chunks_found=3,
            processing_time_ms=125.5
        )
        
        # Check that metric was recorded
        assert len(self.monitor.question_metrics) == 1
        metric = list(self.monitor.question_metrics)[0]
        assert metric['request_id'] == "test_123"
        assert metric['question_length'] == 50
        assert metric['chunks_found'] == 3
        assert metric['processing_time_ms'] == 125.5
        assert metric['success'] == True  # chunks_found > 0
    
    def test_record_batch_metric(self):
        """Test recording batch metrics."""
        self.monitor.record_batch_metric(
            batch_id="batch_001",
            total_questions=10,
            successful_questions=8
        )
        
        assert len(self.monitor.batch_metrics) == 1
        metric = list(self.monitor.batch_metrics)[0]
        assert metric['batch_id'] == "batch_001"
        assert metric['total_questions'] == 10
        assert metric['successful_questions'] == 8
        assert metric['success_rate'] == 0.8
    
    def test_record_error(self):
        """Test recording errors."""
        self.monitor.record_error("ValidationError", "Test error message")
        self.monitor.record_error("ValidationError", "Another error")
        self.monitor.record_error("NetworkError", "Connection failed")
        
        assert self.monitor.error_counts["ValidationError"] == 2
        assert self.monitor.error_counts["NetworkError"] == 1
    
    def test_get_health_metrics(self):
        """Test getting comprehensive health metrics."""
        # Record some test data
        self.monitor.record_question_metric("req1", 30, 2, 100.0)
        self.monitor.record_question_metric("req2", 45, 0, 200.0)
        self.monitor.record_error("TestError", "Test message")
        
        metrics = self.monitor.get_health_metrics()
        
        assert "uptime_seconds" in metrics
        assert "total_questions_processed" in metrics
        assert "question_processing" in metrics
        assert "errors" in metrics
        assert metrics["total_questions_processed"] == 2
    
    def test_get_error_summary(self):
        """Test getting error summary."""
        self.monitor.record_error("ValidationError", "Test error 1")
        self.monitor.record_error("ValidationError", "Test error 2")
        self.monitor.record_error("NetworkError", "Test error 3")
        
        summary = self.monitor.get_error_summary()
        
        assert "total_errors" in summary
        assert "error_breakdown" in summary  # The actual key name
        assert "most_common_error" in summary
        assert summary["total_errors"] == 3
        assert summary["error_breakdown"]["ValidationError"] == 2
        assert summary["error_breakdown"]["NetworkError"] == 1
        assert summary["most_common_error"] == "ValidationError"
    
    def test_reset_metrics(self):
        """Test resetting all metrics."""
        # Add some data
        self.monitor.record_question_metric("req1", 30, 2, 100.0)
        self.monitor.record_error("TestError", "Test message")
        
        # Reset and verify
        self.monitor.reset_metrics()
        
        assert len(self.monitor.question_metrics) == 0
        assert len(self.monitor.batch_metrics) == 0
        assert len(self.monitor.error_counts) == 0
    
    def test_max_history_limit(self):
        """Test that metrics respect max_history limit."""
        monitor = HealthMonitor(max_history=3)
        
        # Add more metrics than the limit
        for i in range(5):
            monitor.record_question_metric(f"req_{i}", 30, 2, 100.0)
        
        # Should only keep the last 3
        assert len(monitor.question_metrics) == 3
    
    def test_get_recent_questions(self):
        """Test getting recent question metrics."""
        # Record some metrics
        self.monitor.record_question_metric("req1", 30, 2, 100.0)
        self.monitor.record_question_metric("req2", 45, 3, 150.0)
        
        recent = self.monitor.get_recent_questions(limit=5)
        
        assert len(recent) == 2
        assert recent[0]['request_id'] == "req1"
        assert recent[1]['request_id'] == "req2"
    
    def test_export_metrics(self):
        """Test exporting all metrics."""
        self.monitor.record_question_metric("req1", 30, 2, 100.0)
        self.monitor.record_error("TestError", "Test message")
        
        exported = self.monitor.export_metrics()
        
        assert "timestamp" in exported
        assert "question_metrics" in exported
        assert "error_counts" in exported
        assert len(exported["question_metrics"]) == 1


class TestLoggingUtils:
    """Test logging utilities."""
    
    def test_setup_logging(self):
        """Test logging setup with default parameters."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "test.log"
            
            # Setup logging
            setup_logging(str(log_file))
            
            # Test that we can get a logger and log messages
            logger = logging.getLogger("test_logger")
            assert logger is not None
            
            # Log a test message
            logger.info("Test message")
            
            # Verify log file was created
            assert log_file.exists()
            
            # Close all handlers to release file handles
            for handler in logging.getLogger().handlers[:]:
                handler.close()
                logging.getLogger().removeHandler(handler)
    
    def test_get_logger(self):
        """Test get_logger function."""
        logger1 = get_logger("test.module1")
        logger2 = get_logger("test.module2")
        logger3 = get_logger("test.module1")  # Same name
        
        assert logger1.name == "test.module1"
        assert logger2.name == "test.module2"
        assert logger1 is logger3  # Should be the same instance
    
    def test_logger_mixin(self):
        """Test LoggerMixin functionality."""
        class TestClass(LoggerMixin):
            def test_method(self):
                return self.logger
        
        obj = TestClass()
        logger = obj.test_method()
        
        assert isinstance(logger, logging.Logger)
        assert logger.name.endswith("TestClass")
    
    def test_structured_logger(self):
        """Test StructuredLogger functionality."""
        with patch('logging.getLogger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            
            struct_logger = StructuredLogger("test_struct")
            
            # Test structured logging methods
            struct_logger.info_structured("Test message", key1="value1", key2="value2")
            struct_logger.error_structured("Error message", error_code=500)
            struct_logger.warning_structured("Warning message")
            struct_logger.debug_structured("Debug message", debug_level=3)
            
            # Verify that the underlying logger methods were called
            assert mock_logger.info.called
            assert mock_logger.error.called
            assert mock_logger.warning.called
            assert mock_logger.debug.called
    
    def test_performance_logger(self):
        """Test PerformanceLogger functionality."""
        with patch('logging.getLogger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            
            perf_logger = PerformanceLogger("test_perf")
            
            # Test timing and throughput logging
            perf_logger.log_timing("operation1", 123.45, context="test")
            perf_logger.log_throughput("operation2", 100, 1000.0, batch_size=10)
            
            # Verify logger calls
            assert mock_logger.info.call_count == 2
    
    def test_error_tracker(self):
        """Test ErrorTracker functionality."""
        with patch('logging.getLogger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            
            error_tracker = ErrorTracker("test_errors")
            
            # Track some errors
            error_tracker.track_error("ValidationError", "Invalid input", request_id="123")
            error_tracker.track_error("ValidationError", "Missing field", request_id="124")
            error_tracker.track_error("NetworkError", "Connection timeout", server="api1")
            
            # Check error counts
            summary = error_tracker.get_error_summary()
            assert summary["ValidationError"] == 2
            assert summary["NetworkError"] == 1
            
            # Verify logger calls
            assert mock_logger.error.call_count == 3


class TestUtilityIntegration:
    """Test integration between utility components."""
    
    def setup_method(self):
        """Set up test components."""
        self.monitor = HealthMonitor(max_history=100)
        self.logger = get_logger("integration_test")
    
    def test_monitor_and_logger_integration(self):
        """Test health monitor and logger working together."""
        # Record metrics while logging
        self.logger.info("Recording test metric")
        self.monitor.record_question_metric("test_123", 50, 3, 150.0)
        
        # Verify both worked
        assert len(self.monitor.question_metrics) == 1
    
    def test_performance_monitoring_with_logging(self):
        """Test performance monitoring combined with logging."""
        start_time = time.time()
        self.logger.info("Starting performance test")
        
        # Simulate some work
        time.sleep(0.1)
        processing_time = (time.time() - start_time) * 1000  # Convert to ms
        
        self.monitor.record_question_metric(
            request_id="perf_test_001",
            question_length=100,
            chunks_found=5,
            processing_time_ms=processing_time
        )
        
        self.logger.info(f"Performance test completed in {processing_time:.2f}ms")
        
        # Verify metrics were recorded
        metrics = self.monitor.get_health_metrics()
        assert metrics["total_questions_processed"] == 1
        
        # Verify processing time is reasonable (should be > 100ms due to sleep)
        recorded_metric = list(self.monitor.question_metrics)[0]
        assert recorded_metric["processing_time_ms"] > 100
    
    def test_error_handling_integration(self):
        """Test error handling across components."""
        try:
            # Simulate an error condition
            raise ValueError("Test error for integration testing")
        except ValueError as e:
            # Log the error
            self.logger.error(f"Caught error: {e}")
            
            # Record it in health monitor
            self.monitor.record_error("ValueError", str(e))
        
        # Verify error was recorded
        error_summary = self.monitor.get_error_summary()
        assert error_summary["total_errors"] == 1
        assert "ValueError" in error_summary["error_breakdown"]
