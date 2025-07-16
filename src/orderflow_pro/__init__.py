"""
OrderFlow Pro - Professional Order Book Analysis System

A comprehensive cryptocurrency order book monitoring and alert system
that detects trading patterns and sends real-time notifications.

Features:
- Multi-exchange order book monitoring
- Pattern detection (walls, whales, imbalances, volume spikes)
- Real-time Telegram alerts
- Professional logging and monitoring
- Configurable thresholds and alerts

Author: Jonathan Griffiths
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Jonathan Griffiths"
__email__ = "griffj147@gmail.com"

# Core imports
from orderflow_pro.alerts.system import alert_system
from orderflow_pro.alerts.telegram import telegram_bot

# Analyzer imports
from orderflow_pro.analyzers.bid_ask_walls import BidAskWallAnalyzer
from orderflow_pro.analyzers.imbalances import OrderBookImbalanceAnalyzer
from orderflow_pro.analyzers.volume_spikes import VolumeSpikeAnalyzer
from orderflow_pro.analyzers.whale_orders import WhaleOrderAnalyzer
from orderflow_pro.config.settings import settings

# Component imports
from orderflow_pro.exchanges.manager import exchange_manager
from orderflow_pro.main import app, main
from orderflow_pro.models.alerts import AlertPriority, AlertType, BaseAlert

# Model imports
from orderflow_pro.models.order_book import OrderBookLevel, OrderBookSnapshot
from orderflow_pro.models.patterns import BasePattern, PatternSignificance

# Utility imports
from orderflow_pro.utils.logger import get_logger, setup_logging

__all__ = [
    # Core
    "settings",
    "app",
    "main",
    # Components
    "exchange_manager",
    "alert_system",
    "telegram_bot",
    # Models
    "OrderBookSnapshot",
    "OrderBookLevel",
    "BaseAlert",
    "AlertType",
    "AlertPriority",
    "BasePattern",
    "PatternSignificance",
    # Analyzers
    "BidAskWallAnalyzer",
    "WhaleOrderAnalyzer",
    "OrderBookImbalanceAnalyzer",
    "VolumeSpikeAnalyzer",
    # Utilities
    "get_logger",
    "setup_logging",
]
