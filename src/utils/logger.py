"""
Logging configuration and utilities.
Centralized logging setup with proper formatting and levels.
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional

from ..config.settings import settings


def setup_logging(log_file: Optional[str] = None) -> None:
    """
    Setup application logging with proper configuration.
    
    Args:
        log_file: Optional log file path. If None, uses settings default.
    """
    
    # Create logs directory if it doesn't exist
    logs_dir = Path(settings.directories.logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine log file path
    if log_file is None:
        log_file = logs_dir / "ntt_rag_system.log"
    else:
        log_file = Path(log_file)
    
    # Create formatter
    formatter = logging.Formatter(
        fmt=settings.logging.log_format,
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, settings.logging.log_level))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, settings.logging.log_level))
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        filename=log_file,
        maxBytes=settings.logging.log_file_max_size,
        backupCount=settings.logging.log_file_backup_count,
        encoding='utf-8'
    )
    file_handler.setLevel(getattr(logging, settings.logging.log_level))
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    # Configure specific loggers
    configure_third_party_loggers()
    
    # Log the setup completion
    logger = logging.getLogger(__name__)
    logger.info("📝 Logging system initialized")
    logger.info(f"🗂️  Log level: {settings.logging.log_level}")
    logger.info(f"📄 Log file: {log_file}")


def configure_third_party_loggers() -> None:
    """Configure logging levels for third-party libraries."""
    
    # Disable AutoGen core events if requested
    if settings.logging.disable_autogen_logs:
        logging.getLogger('autogen_core.events').setLevel(logging.WARNING)
        logging.getLogger('autogen_core').setLevel(logging.WARNING)
        logging.getLogger('autogen_agentchat').setLevel(logging.INFO)
    
    # Configure other third-party loggers
    third_party_loggers = {
        'httpx': logging.WARNING,
        'httpcore': logging.WARNING,
        'urllib3': logging.WARNING,
        'requests': logging.WARNING,
        'openai': logging.INFO,
        'azure': logging.WARNING,
        'sklearn': logging.WARNING,
        'numpy': logging.WARNING,
        'uvicorn.access': logging.INFO,
        'uvicorn.error': logging.INFO,
        'fastapi': logging.INFO
    }
    
    for logger_name, level in third_party_loggers.items():
        logging.getLogger(logger_name).setLevel(level)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the specified name.
    
    Args:
        name: Logger name (usually __name__)
    
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


class LoggerMixin:
    """Mixin class to add logging capabilities to any class."""
    
    @property
    def logger(self) -> logging.Logger:
        """Get logger for this class."""
        return logging.getLogger(self.__class__.__module__ + '.' + self.__class__.__name__)


class StructuredLogger:
    """
    Structured logger for better log analysis and monitoring.
    Provides methods for logging with structured data.
    """
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
    
    def info_structured(self, message: str, **kwargs):
        """Log info message with structured data."""
        self.logger.info(f"{message} | {self._format_data(kwargs)}")
    
    def error_structured(self, message: str, **kwargs):
        """Log error message with structured data."""
        self.logger.error(f"{message} | {self._format_data(kwargs)}")
    
    def warning_structured(self, message: str, **kwargs):
        """Log warning message with structured data."""
        self.logger.warning(f"{message} | {self._format_data(kwargs)}")
    
    def debug_structured(self, message: str, **kwargs):
        """Log debug message with structured data."""
        self.logger.debug(f"{message} | {self._format_data(kwargs)}")
    
    def _format_data(self, data: dict) -> str:
        """Format structured data for logging."""
        if not data:
            return ""
        
        formatted_items = []
        for key, value in data.items():
            formatted_items.append(f"{key}={value}")
        
        return " ".join(formatted_items)


# Performance logging utilities
class PerformanceLogger:
    """Logger for performance metrics and timing."""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(f"performance.{name}")
    
    def log_timing(self, operation: str, duration_ms: float, **context):
        """Log operation timing."""
        context_str = " ".join(f"{k}={v}" for k, v in context.items())
        self.logger.info(f"TIMING {operation} {duration_ms:.2f}ms {context_str}")
    
    def log_throughput(self, operation: str, count: int, duration_ms: float, **context):
        """Log throughput metrics."""
        rate = count / (duration_ms / 1000) if duration_ms > 0 else 0
        context_str = " ".join(f"{k}={v}" for k, v in context.items())
        self.logger.info(f"THROUGHPUT {operation} {count} items {rate:.2f}/sec {context_str}")


# Error tracking utilities
class ErrorTracker:
    """Track and log errors with context."""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(f"errors.{name}")
        self.error_counts = {}
    
    def track_error(self, error_type: str, error_message: str, **context):
        """Track and log an error."""
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        
        context_str = " ".join(f"{k}={v}" for k, v in context.items())
        self.logger.error(f"ERROR {error_type} count={self.error_counts[error_type]} message={error_message} {context_str}")
    
    def get_error_summary(self) -> dict:
        """Get summary of tracked errors."""
        return dict(self.error_counts)