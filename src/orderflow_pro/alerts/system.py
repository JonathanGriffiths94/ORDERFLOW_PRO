"""
OrderFlow Pro - Alert System

Converts detected patterns into actionable alerts and manages alert lifecycle.
"""

import asyncio
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Callable, Dict, List, Optional

from orderflow_pro.config.settings import settings
from orderflow_pro.models.alerts import (
    AlertManager,
    AlertPriority,
    AlertStatus,
    AlertType,
    BaseAlert,
    BidAskWallAlert,
    OrderImbalanceAlert,
    VolumeSpikeAlert,
    WhaleOrderAlert,
)
from orderflow_pro.models.order_book import OrderBookSnapshot
from orderflow_pro.models.patterns import (
    BasePattern,
    ImbalancePattern,
    PatternSignificance,
    VolumePattern,
    WallPattern,
    WhaleOrderPattern,
)
from orderflow_pro.utils.logger import get_logger

logger = get_logger("orderflow_pro.alerts")


@dataclass
class AlertConfig:
    """Configuration for alert system."""

    default_cooldown_seconds: int = settings.alert_cooldown
    priority_cooldowns: Dict[AlertPriority, int] = None
    max_alerts_per_minute: int = 10
    enable_alert_filtering: bool = True
    min_confidence_threshold: Decimal = Decimal("0.6")

    def __post_init__(self):
        if self.priority_cooldowns is None:
            self.priority_cooldowns = {
                AlertPriority.LOW: self.default_cooldown_seconds,
                AlertPriority.MEDIUM: self.default_cooldown_seconds // 2,
                AlertPriority.HIGH: self.default_cooldown_seconds // 4,
                AlertPriority.CRITICAL: self.default_cooldown_seconds // 8,
            }


class AlertSystem:
    """Main alert system that processes patterns and generates alerts."""

    def __init__(self, config: AlertConfig = None):
        self.config = config or AlertConfig()
        self.alert_manager = AlertManager()
        self.alert_callbacks: List[Callable] = []
        self.alert_queue: asyncio.Queue = asyncio.Queue()
        self.processing_task: Optional[asyncio.Task] = None
        self.running = False

        # Statistics
        self.stats = {
            "total_alerts_generated": 0,
            "alerts_sent": 0,
            "alerts_filtered": 0,
            "alerts_on_cooldown": 0,
            "alerts_by_type": {},
        }

    async def start(self):
        """Start the alert processing system."""
        if self.running:
            logger.warning("Alert system already running")
            return

        self.running = True
        self.processing_task = asyncio.create_task(self._process_alert_queue())
        logger.info("Alert system started")

    async def stop(self):
        """Stop the alert processing system."""
        if not self.running:
            return

        self.running = False

        if self.processing_task:
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass

        logger.info("Alert system stopped")

    def register_alert_callback(self, callback: Callable):
        """Register a callback for alert notifications."""
        self.alert_callbacks.append(callback)
        logger.info(f"Registered alert callback: {callback.__name__}")

    def unregister_alert_callback(self, callback: Callable):
        """Unregister an alert callback."""
        if callback in self.alert_callbacks:
            self.alert_callbacks.remove(callback)
            logger.info(f"Unregistered alert callback: {callback.__name__}")

    # Pattern to Alert Conversion Methods

    def process_wall_pattern(self, pattern: WallPattern) -> Optional[BaseAlert]:
        """Convert wall pattern to alert."""

        if not self._should_generate_alert(pattern):
            return None

        alert = BidAskWallAlert(
            alert_id=str(uuid.uuid4()),
            priority=self._pattern_to_priority(pattern),
            exchange=pattern.exchange,
            symbol=pattern.symbol,
            timestamp=datetime.utcnow(),
            wall_side=pattern.wall_side.value,
            wall_price=pattern.wall_price,
            wall_volume=pattern.wall_volume,
            wall_value=pattern.wall_value,
            order_count=pattern.order_count,
            data={
                "pattern_id": pattern.pattern_id,
                "depth_rank": pattern.depth_rank,
                "confidence": float(pattern.confidence),
                "significance": pattern.significance.value,
                "wall_type": pattern.wall_type,
                "price_distance_pct": float(pattern.price_distance_percentage),
            },
        )

        self._update_stats("bid_ask_wall")
        return alert

    def process_whale_pattern(self, pattern: WhaleOrderPattern) -> Optional[BaseAlert]:
        """Convert whale order pattern to alert."""

        if not self._should_generate_alert(pattern):
            return None

        alert = WhaleOrderAlert(
            alert_id=str(uuid.uuid4()),
            priority=self._pattern_to_priority(pattern),
            exchange=pattern.exchange,
            symbol=pattern.symbol,
            timestamp=datetime.utcnow(),
            order_side=pattern.order_side.value,
            order_price=pattern.order_price,
            order_volume=pattern.order_volume,
            notional_value=pattern.notional_value,
            data={
                "pattern_id": pattern.pattern_id,
                "confidence": float(pattern.confidence),
                "significance": pattern.significance.value,
                "market_impact": float(pattern.market_impact) if pattern.market_impact else None,
                "order_type": pattern.order_type,
            },
        )

        self._update_stats("whale_order")
        return alert

    def process_imbalance_pattern(self, pattern: ImbalancePattern) -> Optional[BaseAlert]:
        """Convert imbalance pattern to alert."""

        if not self._should_generate_alert(pattern):
            return None

        alert = OrderImbalanceAlert(
            alert_id=str(uuid.uuid4()),
            priority=self._pattern_to_priority(pattern),
            exchange=pattern.exchange,
            symbol=pattern.symbol,
            timestamp=datetime.utcnow(),
            imbalance_ratio=pattern.imbalance_ratio,
            bid_volume=pattern.bid_volume,
            ask_volume=pattern.ask_volume,
            depth_analyzed=pattern.depth_analyzed,
            data={
                "pattern_id": pattern.pattern_id,
                "confidence": float(pattern.confidence),
                "significance": pattern.significance.value,
                "imbalance_strength": float(pattern.imbalance_strength),
                "bias_direction": pattern.bias_direction,
                "bias_percentage": float(pattern.bias_percentage),
            },
        )

        self._update_stats("order_imbalance")
        return alert

    def process_volume_pattern(self, pattern: VolumePattern) -> Optional[BaseAlert]:
        """Convert volume pattern to alert."""

        if not self._should_generate_alert(pattern):
            return None

        alert = VolumeSpikeAlert(
            alert_id=str(uuid.uuid4()),
            priority=self._pattern_to_priority(pattern),
            exchange=pattern.exchange,
            symbol=pattern.symbol,
            timestamp=datetime.utcnow(),
            current_volume=pattern.current_volume,
            average_volume=pattern.average_volume,
            spike_ratio=pattern.spike_ratio,
            data={
                "pattern_id": pattern.pattern_id,
                "confidence": float(pattern.confidence),
                "significance": pattern.significance.value,
                "spike_magnitude": float(pattern.spike_magnitude),
                "spike_percentage": float(pattern.spike_percentage),
                "timeframe": pattern.timeframe,
            },
        )

        self._update_stats("volume_spike")
        return alert

    def process_pattern(self, pattern: BasePattern) -> Optional[BaseAlert]:
        """Process any pattern type and convert to appropriate alert."""

        if isinstance(pattern, WallPattern):
            return self.process_wall_pattern(pattern)
        elif isinstance(pattern, WhaleOrderPattern):
            return self.process_whale_pattern(pattern)
        elif isinstance(pattern, ImbalancePattern):
            return self.process_imbalance_pattern(pattern)
        elif isinstance(pattern, VolumePattern):
            return self.process_volume_pattern(pattern)
        else:
            logger.warning(f"Unknown pattern type: {type(pattern)}")
            return None

    async def queue_alert(self, alert: BaseAlert):
        """Queue an alert for processing."""
        await self.alert_queue.put(alert)
        logger.debug(f"Alert queued: {alert.alert_type.value}")

    async def process_patterns(self, patterns: List[BasePattern]):
        """Process multiple patterns and queue alerts."""

        alerts_queued = 0

        for pattern in patterns:
            alert = self.process_pattern(pattern)
            if alert:
                await self.queue_alert(alert)
                alerts_queued += 1

        if alerts_queued > 0:
            logger.info(f"Queued {alerts_queued} alerts from {len(patterns)} patterns")

    async def _process_alert_queue(self):
        """Process alerts from the queue."""

        logger.info("Alert queue processing started")

        while self.running:
            try:
                # Get alert from queue with timeout
                alert = await asyncio.wait_for(self.alert_queue.get(), timeout=1.0)

                # Process the alert
                await self._process_single_alert(alert)

            except asyncio.TimeoutError:
                # No alert in queue, continue
                continue
            except Exception as e:
                logger.error(f"Error processing alert queue: {e}")
                await asyncio.sleep(1)

        logger.info("Alert queue processing stopped")

    async def _process_single_alert(self, alert: BaseAlert):
        """Process a single alert."""

        try:
            # Check if alert can be sent (not on cooldown)
            if not self.alert_manager.can_send_alert(alert):
                logger.debug(f"Alert on cooldown: {alert.alert_type.value}")
                self.stats["alerts_on_cooldown"] += 1
                return

            # Check rate limiting
            if not self._check_rate_limit():
                logger.warning("Alert rate limit exceeded")
                return

            # Send alert to callbacks
            await self._send_alert_to_callbacks(alert)

            # Mark as sent and add cooldown
            cooldown_seconds = self.config.priority_cooldowns.get(alert.priority, self.config.default_cooldown_seconds)
            self.alert_manager.mark_alert_sent(alert, cooldown_seconds)

            # Update statistics
            self.stats["alerts_sent"] += 1

            logger.log_alert_sent(
                alert.alert_type.value, alert.symbol, alert.exchange, alert.priority.value, alert_id=alert.alert_id
            )

        except Exception as e:
            logger.error(f"Error processing alert: {e}", alert_id=alert.alert_id, alert_type=alert.alert_type.value)
            alert.status = AlertStatus.FAILED

    async def _send_alert_to_callbacks(self, alert: BaseAlert):
        """Send alert to all registered callbacks."""

        for callback in self.alert_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(alert)
                else:
                    callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {callback.__name__}", error=str(e), alert_id=alert.alert_id)

    def _should_generate_alert(self, pattern: BasePattern) -> bool:
        """Check if pattern should generate an alert."""

        # Check confidence threshold
        if pattern.confidence < self.config.min_confidence_threshold:
            self.stats["alerts_filtered"] += 1
            return False

        # Check if pattern is fresh (not too old)
        if not pattern.is_fresh:
            self.stats["alerts_filtered"] += 1
            return False

        # Additional filtering can be added here

        return True

    def _pattern_to_priority(self, pattern: BasePattern) -> AlertPriority:
        """Convert pattern significance to alert priority."""

        mapping = {
            PatternSignificance.LOW: AlertPriority.LOW,
            PatternSignificance.MEDIUM: AlertPriority.MEDIUM,
            PatternSignificance.HIGH: AlertPriority.HIGH,
            PatternSignificance.CRITICAL: AlertPriority.CRITICAL,
        }

        return mapping.get(pattern.significance, AlertPriority.MEDIUM)

    def _check_rate_limit(self) -> bool:
        """Check if we're within rate limits."""

        # Get alerts sent in last minute
        recent_alerts = self.alert_manager.get_recent_alerts(hours=0)  # Last hour

        # Filter to last minute
        one_minute_ago = datetime.utcnow() - timedelta(minutes=1)
        recent_minute_alerts = [alert for alert in recent_alerts if alert.sent_at and alert.sent_at >= one_minute_ago]

        return len(recent_minute_alerts) < self.config.max_alerts_per_minute

    def _update_stats(self, alert_type: str):
        """Update alert statistics."""

        self.stats["total_alerts_generated"] += 1

        if alert_type not in self.stats["alerts_by_type"]:
            self.stats["alerts_by_type"][alert_type] = 0
        self.stats["alerts_by_type"][alert_type] += 1

    # Convenience methods for direct pattern processing

    async def process_order_book(self, snapshot: OrderBookSnapshot, patterns: List[BasePattern]):
        """Process order book snapshot and associated patterns."""

        if not patterns:
            return

        # Add current price to patterns that need it
        for pattern in patterns:
            if pattern.current_price == Decimal("0"):
                pattern.current_price = snapshot.mid_price

        await self.process_patterns(patterns)

    async def send_immediate_alert(self, alert: BaseAlert) -> bool:
        """Send an alert immediately (bypass queue)."""

        try:
            # Check cooldown
            if not self.alert_manager.can_send_alert(alert):
                logger.debug(f"Immediate alert blocked by cooldown: {alert.alert_type.value}")
                return False

            # Send to callbacks
            await self._send_alert_to_callbacks(alert)

            # Mark as sent
            self.alert_manager.mark_alert_sent(alert)

            logger.info(f"Immediate alert sent: {alert.alert_type.value}")
            return True

        except Exception as e:
            logger.error(f"Error sending immediate alert: {e}")
            return False

    # Alert management methods

    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get alert system statistics."""

        return {
            **self.stats,
            "queue_size": self.alert_queue.qsize(),
            "active_cooldowns": len(self.alert_manager.cooldowns),
            "recent_alerts_1h": len(self.alert_manager.get_recent_alerts(1)),
            "recent_alerts_24h": len(self.alert_manager.get_recent_alerts(24)),
            "system_running": self.running,
        }

    def get_recent_alerts(self, hours: int = 24) -> List[BaseAlert]:
        """Get recent alerts."""
        return self.alert_manager.get_recent_alerts(hours)

    def get_cooldown_status(self, alert_type: AlertType, exchange: str, symbol: str) -> Dict[str, Any]:
        """Get cooldown status for specific alert type."""

        key = self.alert_manager.generate_alert_key(alert_type, exchange, symbol)
        cooldown = self.alert_manager.cooldowns.get(key)

        if not cooldown:
            return {"on_cooldown": False, "remaining_seconds": 0}

        return {
            "on_cooldown": cooldown.is_on_cooldown,
            "remaining_seconds": cooldown.remaining_cooldown_seconds,
            "last_sent": cooldown.last_sent.isoformat(),
        }

    def clear_cooldowns(self, alert_type: AlertType = None, exchange: str = None, symbol: str = None):
        """Clear cooldowns (for testing/debugging)."""

        if alert_type and exchange and symbol:
            # Clear specific cooldown
            key = self.alert_manager.generate_alert_key(alert_type, exchange, symbol)
            if key in self.alert_manager.cooldowns:
                del self.alert_manager.cooldowns[key]
                logger.info(f"Cleared cooldown: {key}")
        else:
            # Clear all cooldowns
            self.alert_manager.cooldowns.clear()
            logger.info("Cleared all cooldowns")

    def reset_statistics(self):
        """Reset alert statistics."""

        self.stats = {
            "total_alerts_generated": 0,
            "alerts_sent": 0,
            "alerts_filtered": 0,
            "alerts_on_cooldown": 0,
            "alerts_by_type": {},
        }

        logger.info("Alert statistics reset")

    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()


# Global alert system instance
alert_system = AlertSystem()


# Convenience functions
async def process_patterns(patterns: List[BasePattern]):
    """Process patterns through the global alert system."""
    await alert_system.process_patterns(patterns)


async def send_alert(alert: BaseAlert) -> bool:
    """Send an alert through the global alert system."""
    return await alert_system.send_immediate_alert(alert)


def register_alert_callback(callback: Callable):
    """Register callback for alerts."""
    alert_system.register_alert_callback(callback)


def get_alert_stats() -> Dict[str, Any]:
    """Get alert system statistics."""
    return alert_system.get_alert_statistics()
