import json
import logging
import logging.handlers
import sys
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict

from orderflow_pro.config.settings import settings


class LogLevel(str, Enum):
    """Log level enumeration."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured JSON logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON."""

        # Base log data
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add extra fields from record
        extra_fields = {
            k: v
            for k, v in record.__dict__.items()
            if k
            not in {
                "name",
                "msg",
                "args",
                "levelname",
                "levelno",
                "pathname",
                "filename",
                "module",
                "exc_info",
                "exc_text",
                "stack_info",
                "lineno",
                "funcName",
                "created",
                "msecs",
                "relativeCreated",
                "thread",
                "threadName",
                "processName",
                "process",
                "message",
            }
        }

        if extra_fields:
            log_data["extra"] = extra_fields

        return json.dumps(log_data, ensure_ascii=False)


class ColoredFormatter(logging.Formatter):
    """Colored formatter for console output."""

    # ANSI color codes
    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
        "RESET": "\033[0m",  # Reset
    }

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors."""

        # Add colors to levelname
        if record.levelname in self.COLORS:
            record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{self.COLORS['RESET']}"

        return super().format(record)


class OrderFlowLogger:
    """OrderFlow Pro logger with structured logging capabilities."""

    def __init__(self, name: str = "orderflow_pro"):
        self.name = name
        self.logger = logging.getLogger(name)
        self._setup_logger()

    def _setup_logger(self):
        """Setup logger with handlers and formatters."""

        # Clear existing handlers
        self.logger.handlers.clear()

        # Set log level
        log_level = getattr(logging, settings.loglevel.upper(), logging.INFO)
        self.logger.setLevel(log_level)

        # Prevent duplicate logs
        self.logger.propagate = False

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)

        if settings.is_production():
            # Production: Structured JSON logging
            console_formatter = StructuredFormatter()
        else:
            # Development: Colored human-readable logging
            console_formatter = ColoredFormatter(
                fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s", datefmt="%H:%M:%S"
            )

        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)

        # File handler (if configured)
        if settings.is_production():
            self._setup_file_handler()

    def _setup_file_handler(self):
        """Setup rotating file handler for production."""

        # Create logs directory
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        # Setup rotating file handler
        file_handler = logging.handlers.RotatingFileHandler(
            filename=log_dir / "orderflow_pro.log",
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding="utf-8",
        )

        file_handler.setLevel(logging.DEBUG)  # File gets all logs
        file_formatter = StructuredFormatter()
        file_handler.setFormatter(file_formatter)

        self.logger.addHandler(file_handler)

    def debug(self, message: str, **kwargs):
        """Log debug message with optional extra data."""
        self.logger.debug(message, extra=kwargs)

    def info(self, message: str, **kwargs):
        """Log info message with optional extra data."""
        self.logger.info(message, extra=kwargs)

    def warning(self, message: str, **kwargs):
        """Log warning message with optional extra data."""
        self.logger.warning(message, extra=kwargs)

    def error(self, message: str, **kwargs):
        """Log error message with optional extra data."""
        self.logger.error(message, extra=kwargs)

    def critical(self, message: str, **kwargs):
        """Log critical message with optional extra data."""
        self.logger.critical(message, extra=kwargs)

    def exception(self, message: str, **kwargs):
        """Log exception with traceback."""
        self.logger.exception(message, extra=kwargs)

    def log_exchange_event(self, exchange: str, event: str, symbol: str = None, **data):
        """Log exchange-specific events."""
        extra_data = {"exchange": exchange, "event": event, "component": "exchange", **data}

        if symbol:
            extra_data["symbol"] = symbol

        self.info(f"Exchange event: {event}", **extra_data)

    def log_pattern_detected(self, pattern_type: str, symbol: str, exchange: str, confidence: float, **pattern_data):
        """Log pattern detection events."""
        extra_data = {
            "pattern_type": pattern_type,
            "symbol": symbol,
            "exchange": exchange,
            "confidence": confidence,
            "component": "pattern_detection",
            **pattern_data,
        }

        self.info(f"Pattern detected: {pattern_type}", **extra_data)

    def log_alert_sent(self, alert_type: str, symbol: str, exchange: str, priority: str, **alert_data):
        """Log alert sending events."""
        extra_data = {
            "alert_type": alert_type,
            "symbol": symbol,
            "exchange": exchange,
            "priority": priority,
            "component": "alerts",
            **alert_data,
        }

        self.info(f"Alert sent: {alert_type}", **extra_data)

    def log_volume_analysis(
        self, symbol: str, exchange: str, current_volume: float, average_volume: float, spike_ratio: float
    ):
        """Log volume analysis events."""
        extra_data = {
            "symbol": symbol,
            "exchange": exchange,
            "current_volume": current_volume,
            "average_volume": average_volume,
            "spike_ratio": spike_ratio,
            "component": "volume_analysis",
        }

        if spike_ratio >= 1.5:
            self.info(f"Volume spike detected: {spike_ratio:.2f}x", **extra_data)
        else:
            self.debug(f"Volume analysis: {spike_ratio:.2f}x", **extra_data)

    def log_order_book_update(self, symbol: str, exchange: str, bid_count: int, ask_count: int, spread: float):
        """Log order book update events."""
        extra_data = {
            "symbol": symbol,
            "exchange": exchange,
            "bid_levels": bid_count,
            "ask_levels": ask_count,
            "spread": spread,
            "component": "order_book",
        }

        self.debug("Order book updated", **extra_data)

    def log_whale_order(self, symbol: str, exchange: str, side: str, price: float, volume: float, value: float):
        """Log whale order detection."""
        extra_data = {
            "symbol": symbol,
            "exchange": exchange,
            "side": side,
            "price": price,
            "volume": volume,
            "notional_value": value,
            "component": "whale_detection",
        }

        self.info(f"Whale order detected: {side} ${value:,.0f}", **extra_data)

    def log_error_with_context(self, error: Exception, context: Dict[str, Any]):
        """Log error with additional context."""
        extra_data = {"error_type": type(error).__name__, "component": "error_handler", **context}

        self.exception(f"Error occurred: {str(error)}", **extra_data)

    def log_performance_metric(self, metric_name: str, value: float, unit: str = "ms", **context):
        """Log performance metrics."""
        extra_data = {"metric": metric_name, "value": value, "unit": unit, "component": "performance", **context}

        self.debug(f"Performance: {metric_name} = {value}{unit}", **extra_data)

    def log_startup(self, component: str, config: Dict[str, Any] = None):
        """Log component startup."""
        extra_data = {"component": component, "event": "startup"}

        if config:
            # Remove sensitive data
            safe_config = {
                k: v
                for k, v in config.items()
                if not any(sensitive in k.lower() for sensitive in ["key", "secret", "token", "password"])
            }
            extra_data["config"] = safe_config

        self.info(f"Component started: {component}", **extra_data)

    def log_shutdown(self, component: str, reason: str = "normal"):
        """Log component shutdown."""
        extra_data = {"component": component, "event": "shutdown", "reason": reason}

        self.info(f"Component shutdown: {component}", **extra_data)


# Create global logger instances
main_logger = OrderFlowLogger("orderflow_pro")
exchange_logger = OrderFlowLogger("orderflow_pro.exchange")
pattern_logger = OrderFlowLogger("orderflow_pro.patterns")
alert_logger = OrderFlowLogger("orderflow_pro.alerts")
telegram_logger = OrderFlowLogger("orderflow_pro.telegram")


def get_logger(name: str = "orderflow_pro") -> OrderFlowLogger:
    """Get a logger instance for a specific component."""
    return OrderFlowLogger(name)


# Convenience functions for common logging patterns
def log_startup(component: str, **config):
    """Log component startup."""
    main_logger.log_startup(component, config)


def log_shutdown(component: str, reason: str = "normal"):
    """Log component shutdown."""
    main_logger.log_shutdown(component, reason)


def log_error(error: Exception, **context):
    """Log error with context."""
    main_logger.log_error_with_context(error, context)


def log_performance(metric: str, value: float, unit: str = "ms", **context):
    """Log performance metric."""
    main_logger.log_performance_metric(metric, value, unit, **context)


# Setup function for initializing logging
def setup_logging():
    """Initialize logging configuration."""

    # Set root logger level to prevent duplicate logs
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.WARNING)

    # Suppress noisy third-party loggers
    logging.getLogger("ccxt").setLevel(logging.WARNING)
    logging.getLogger("aiohttp").setLevel(logging.WARNING)
    logging.getLogger("telegram").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)

    main_logger.info(
        "Logging initialized",
        level=settings.loglevel,
        environment=settings.environment,
        production=settings.is_production(),
    )


# Context manager for logging function execution
class LogExecutionTime:
    """Context manager to log function execution time."""

    def __init__(self, operation: str, logger: OrderFlowLogger = None, log_level: str = "debug", **context):
        self.operation = operation
        self.logger = logger or main_logger
        self.log_level = log_level.lower()
        self.context = context
        self.start_time = None

    def __enter__(self):
        self.start_time = datetime.utcnow()
        getattr(self.logger, self.log_level)(f"Starting: {self.operation}", **self.context)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = datetime.utcnow()
        duration_ms = (end_time - self.start_time).total_seconds() * 1000

        if exc_type is None:
            # Success
            self.logger.log_performance_metric(f"{self.operation}_duration", duration_ms, "ms", **self.context)
            getattr(self.logger, self.log_level)(f"Completed: {self.operation} ({duration_ms:.2f}ms)", **self.context)
        else:
            # Error
            self.logger.error(f"Failed: {self.operation} ({duration_ms:.2f}ms)", error=str(exc_val), **self.context)


# Decorator for logging function calls
def log_function_call(logger: OrderFlowLogger = None, level: str = "debug"):
    """Decorator to log function calls with execution time."""

    def decorator(func):
        def wrapper(*args, **kwargs):
            func_logger = logger or main_logger
            operation = f"{func.__module__}.{func.__name__}"

            with LogExecutionTime(operation, func_logger, level):
                return func(*args, **kwargs)

        return wrapper

    return decorator
