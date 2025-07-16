"""
OrderFlow Pro - Volume Spikes Analyzer

Detects volume spikes that indicate unusual trading activity.
"""

import uuid
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Deque, Dict, List, Optional

from orderflow_pro.config.settings import settings
from orderflow_pro.models.order_book import OrderBookSnapshot
from orderflow_pro.models.patterns import PatternSignificance, VolumePattern
from orderflow_pro.utils.logger import get_logger

logger = get_logger("orderflow_pro.analyzers.volume")


@dataclass
class VolumeDetectionConfig:
    """Configuration for volume spike detection."""

    spike_threshold: Decimal = Decimal(str(settings.volume_spike_threshold))
    rolling_window_size: int = 20  # Number of periods for rolling average
    min_volume_threshold: Decimal = Decimal("1000")  # Minimum volume to analyze
    timeframe: str = "5s"  # Analysis timeframe
    significance_thresholds: Dict[str, Decimal] = None

    def __post_init__(self):
        if self.significance_thresholds is None:
            self.significance_thresholds = {
                "low": self.spike_threshold,  # 1.5x
                "medium": self.spike_threshold * 2,  # 3.0x
                "high": self.spike_threshold * 3,  # 4.5x
                "critical": self.spike_threshold * 5,  # 7.5x
            }


@dataclass
class VolumeDataPoint:
    """Volume data point for rolling calculations."""

    timestamp: datetime
    volume: Decimal
    exchange: str
    symbol: str


class VolumeSpikeAnalyzer:
    """Analyzes volume data for spike patterns."""

    def __init__(self, config: VolumeDetectionConfig = None):
        self.config = config or VolumeDetectionConfig()
        self.volume_history: Dict[str, Deque[VolumeDataPoint]] = {}
        self.detected_spikes: Dict[str, List[VolumePattern]] = {}
        self.rolling_averages: Dict[str, Decimal] = {}

    def analyze_volume(self, exchange: str, symbol: str, current_volume: Decimal) -> Optional[VolumePattern]:
        """Analyze volume for spike patterns."""

        key = f"{exchange}:{symbol}"

        # Initialize history if not exists
        if key not in self.volume_history:
            self.volume_history[key] = deque(maxlen=self.config.rolling_window_size)
            self.detected_spikes[key] = []

        # Add current volume to history
        volume_point = VolumeDataPoint(
            timestamp=datetime.utcnow(), volume=current_volume, exchange=exchange, symbol=symbol
        )
        self.volume_history[key].append(volume_point)

        # Calculate rolling average
        average_volume = self._calculate_rolling_average(key)
        self.rolling_averages[key] = average_volume

        # Check for volume spike
        if self._is_volume_spike(current_volume, average_volume):
            spike_pattern = self._create_volume_spike_pattern(exchange, symbol, current_volume, average_volume)

            # Store detected spike
            self.detected_spikes[key].append(spike_pattern)

            logger.log_volume_analysis(
                symbol, exchange, float(current_volume), float(average_volume), float(current_volume / average_volume)
            )

            return spike_pattern

        return None

    def analyze_order_book_volume(self, snapshot: OrderBookSnapshot) -> Optional[VolumePattern]:
        """Analyze order book volume for spikes."""

        # Calculate total order book volume
        total_volume = snapshot.get_total_volume(snapshot.OrderSide.BID, 20) + snapshot.get_total_volume(
            snapshot.OrderSide.ASK, 20
        )

        return self.analyze_volume(snapshot.exchange, snapshot.symbol, total_volume)

    def _calculate_rolling_average(self, key: str) -> Decimal:
        """Calculate rolling average volume."""

        history = self.volume_history[key]

        if len(history) < 2:
            return Decimal("0")

        # Use all available data if we don't have full window yet
        volumes = [point.volume for point in history]

        # Remove current volume from average calculation
        if len(volumes) > 1:
            volumes = volumes[:-1]

        return sum(volumes) / len(volumes) if volumes else Decimal("0")

    def _is_volume_spike(self, current_volume: Decimal, average_volume: Decimal) -> bool:
        """Check if current volume constitutes a spike."""

        # Skip if volume is too low
        if current_volume < self.config.min_volume_threshold:
            return False

        # Skip if no average available
        if average_volume <= 0:
            return False

        # Calculate spike ratio
        spike_ratio = current_volume / average_volume

        return spike_ratio >= self.config.spike_threshold

    def _create_volume_spike_pattern(
        self, exchange: str, symbol: str, current_volume: Decimal, average_volume: Decimal
    ) -> VolumePattern:
        """Create volume spike pattern."""

        spike_ratio = current_volume / average_volume
        spike_magnitude = current_volume - average_volume

        # Calculate significance
        significance = self._calculate_significance(spike_ratio)

        # Calculate confidence
        confidence = self._calculate_spike_confidence(spike_ratio, current_volume, average_volume)

        pattern = VolumePattern(
            pattern_id=str(uuid.uuid4()),
            current_volume=current_volume,
            average_volume=average_volume,
            spike_ratio=spike_ratio,
            spike_magnitude=spike_magnitude,
            timeframe=self.config.timeframe,
            significance=significance,
            confidence=confidence,
            exchange=exchange,
            symbol=symbol,
            timestamp=datetime.utcnow(),
            current_price=Decimal("0"),  # Will be updated by caller if needed
            raw_data={
                "spike_percentage": float((spike_ratio - 1) * 100),
                "volume_threshold": float(self.config.min_volume_threshold),
                "rolling_window_size": self.config.rolling_window_size,
                "detection_method": "rolling_average",
            },
        )

        logger.log_pattern_detected(
            "volume_spike",
            symbol,
            exchange,
            float(confidence),
            spike_ratio=float(spike_ratio),
            current_volume=float(current_volume),
            average_volume=float(average_volume),
        )

        return pattern

    def _calculate_significance(self, spike_ratio: Decimal) -> PatternSignificance:
        """Calculate spike significance based on ratio."""

        thresholds = self.config.significance_thresholds

        if spike_ratio >= thresholds["critical"]:
            return PatternSignificance.CRITICAL
        elif spike_ratio >= thresholds["high"]:
            return PatternSignificance.HIGH
        elif spike_ratio >= thresholds["medium"]:
            return PatternSignificance.MEDIUM
        else:
            return PatternSignificance.LOW

    def _calculate_spike_confidence(
        self, spike_ratio: Decimal, current_volume: Decimal, average_volume: Decimal
    ) -> Decimal:
        """Calculate confidence score for volume spike."""

        confidence = Decimal("0.5")  # Base confidence

        # Spike magnitude factor (larger spikes = higher confidence)
        magnitude_factor = min(spike_ratio / self.config.spike_threshold, 10) / 10
        confidence += magnitude_factor * Decimal("0.3")

        # Absolute volume factor (higher volume = higher confidence)
        volume_factor = min(current_volume / (self.config.min_volume_threshold * 10), 1)
        confidence += volume_factor * Decimal("0.15")

        # Consistency factor (how stable the average is)
        consistency_factor = self._calculate_average_consistency()
        confidence += consistency_factor * Decimal("0.05")

        return min(confidence, Decimal("1.0"))

    def _calculate_average_consistency(self) -> Decimal:
        """Calculate how consistent the rolling average is."""

        # Simple implementation - can be enhanced
        # For now, assume higher consistency with more data points
        total_points = sum(len(history) for history in self.volume_history.values())
        max_possible = len(self.volume_history) * self.config.rolling_window_size

        if max_possible == 0:
            return Decimal("0")

        return Decimal(str(min(total_points / max_possible, 1)))

    def get_recent_spikes(self, exchange: str, symbol: str, hours: int = 1) -> List[VolumePattern]:
        """Get recent volume spikes."""

        key = f"{exchange}:{symbol}"
        spikes = self.detected_spikes.get(key, [])

        cutoff_time = datetime.utcnow().timestamp() - (hours * 3600)

        return [spike for spike in spikes if spike.timestamp.timestamp() > cutoff_time]

    def get_largest_spikes(self, exchange: str, symbol: str, limit: int = 10) -> List[VolumePattern]:
        """Get largest volume spikes by ratio."""

        key = f"{exchange}:{symbol}"
        spikes = self.detected_spikes.get(key, [])

        return sorted(spikes, key=lambda x: x.spike_ratio, reverse=True)[:limit]

    def get_current_volume_metrics(self, exchange: str, symbol: str) -> Optional[Dict]:
        """Get current volume metrics."""

        key = f"{exchange}:{symbol}"

        history = self.volume_history.get(key)
        if not history:
            return None

        current_volume = history[-1].volume if history else Decimal("0")
        average_volume = self.rolling_averages.get(key, Decimal("0"))

        if average_volume > 0:
            current_ratio = current_volume / average_volume
        else:
            current_ratio = Decimal("1")

        return {
            "current_volume": float(current_volume),
            "average_volume": float(average_volume),
            "current_ratio": float(current_ratio),
            "is_spike": current_ratio >= self.config.spike_threshold,
            "data_points": len(history),
            "window_size": self.config.rolling_window_size,
        }

    def get_volume_trend(self, exchange: str, symbol: str, periods: int = 10) -> Optional[Dict]:
        """Get volume trend over recent periods."""

        key = f"{exchange}:{symbol}"
        history = self.volume_history.get(key)

        if not history or len(history) < periods:
            return None

        recent_volumes = [point.volume for point in list(history)[-periods:]]

        # Calculate trend
        first_half = recent_volumes[: len(recent_volumes) // 2]
        second_half = recent_volumes[len(recent_volumes) // 2 :]

        first_avg = sum(first_half) / len(first_half)
        second_avg = sum(second_half) / len(second_half)

        trend_direction = "increasing" if second_avg > first_avg else "decreasing"
        trend_strength = abs(second_avg - first_avg) / first_avg if first_avg > 0 else 0

        return {
            "periods_analyzed": periods,
            "trend_direction": trend_direction,
            "trend_strength": float(trend_strength),
            "first_half_avg": float(first_avg),
            "second_half_avg": float(second_avg),
            "latest_volume": float(recent_volumes[-1]),
        }

    def get_spike_frequency(self, exchange: str, symbol: str, hours: int = 24) -> Dict:
        """Get spike frequency statistics."""

        key = f"{exchange}:{symbol}"
        spikes = self.detected_spikes.get(key, [])

        cutoff_time = datetime.utcnow().timestamp() - (hours * 3600)
        recent_spikes = [spike for spike in spikes if spike.timestamp.timestamp() > cutoff_time]

        if not recent_spikes:
            return {
                "total_spikes": 0,
                "spikes_per_hour": 0,
                "average_spike_ratio": 0,
                "largest_spike": 0,
                "critical_spikes": 0,
            }

        # Calculate statistics
        total_spikes = len(recent_spikes)
        spikes_per_hour = total_spikes / hours
        average_ratio = sum(spike.spike_ratio for spike in recent_spikes) / total_spikes
        largest_spike = max(spike.spike_ratio for spike in recent_spikes)
        critical_spikes = len([spike for spike in recent_spikes if spike.significance == PatternSignificance.CRITICAL])

        return {
            "total_spikes": total_spikes,
            "spikes_per_hour": float(spikes_per_hour),
            "average_spike_ratio": float(average_ratio),
            "largest_spike": float(largest_spike),
            "critical_spikes": critical_spikes,
            "hours_analyzed": hours,
        }

    def get_volume_summary(self, exchange: str, symbol: str) -> Dict:
        """Get comprehensive volume analysis summary."""

        key = f"{exchange}:{symbol}"

        # Get current metrics
        current_metrics = self.get_current_volume_metrics(exchange, symbol)

        # Get recent spikes
        recent_spikes = self.get_recent_spikes(exchange, symbol, 1)

        # Get spike frequency
        spike_freq = self.get_spike_frequency(exchange, symbol, 24)

        # Get volume trend
        volume_trend = self.get_volume_trend(exchange, symbol, 10)

        return {
            "current_metrics": current_metrics,
            "recent_spikes": len(recent_spikes),
            "spike_frequency": spike_freq,
            "volume_trend": volume_trend,
            "last_updated": datetime.utcnow().isoformat(),
        }

    def is_volume_elevated(self, exchange: str, symbol: str, threshold_multiplier: Decimal = Decimal("1.2")) -> bool:
        """Check if current volume is elevated above normal."""

        metrics = self.get_current_volume_metrics(exchange, symbol)

        if not metrics:
            return False

        return metrics["current_ratio"] >= float(threshold_multiplier)

    def get_volume_percentile(self, exchange: str, symbol: str, target_volume: Decimal = None) -> Optional[float]:
        """Get percentile ranking of volume."""

        key = f"{exchange}:{symbol}"
        history = self.volume_history.get(key)

        if not history:
            return None

        volumes = [point.volume for point in history]

        if target_volume is None:
            target_volume = volumes[-1] if volumes else Decimal("0")

        if not volumes:
            return None

        # Calculate percentile
        below_target = sum(1 for vol in volumes if vol < target_volume)
        percentile = (below_target / len(volumes)) * 100

        return float(percentile)

    def clear_old_spikes(self, max_age_seconds: int = 3600):
        """Clear volume spikes older than specified age."""

        cutoff_time = datetime.utcnow().timestamp() - max_age_seconds

        for key in list(self.detected_spikes.keys()):
            self.detected_spikes[key] = [
                spike for spike in self.detected_spikes[key] if spike.timestamp.timestamp() > cutoff_time
            ]

            # Remove empty entries
            if not self.detected_spikes[key]:
                del self.detected_spikes[key]

    def clear_old_history(self, max_age_seconds: int = 7200):
        """Clear volume history older than specified age."""

        cutoff_time = datetime.utcnow().timestamp() - max_age_seconds

        for key in list(self.volume_history.keys()):
            history = self.volume_history[key]

            # Filter out old entries
            filtered_history = deque(
                [point for point in history if point.timestamp.timestamp() > cutoff_time],
                maxlen=self.config.rolling_window_size,
            )

            if filtered_history:
                self.volume_history[key] = filtered_history
            else:
                del self.volume_history[key]

    def reset_analyzer(self):
        """Reset analyzer state."""
        self.volume_history.clear()
        self.detected_spikes.clear()
        self.rolling_averages.clear()
        logger.info("Volume spike analyzer reset")

    def get_analyzer_stats(self) -> Dict:
        """Get analyzer statistics."""

        total_symbols = len(self.volume_history)
        total_data_points = sum(len(history) for history in self.volume_history.values())
        total_spikes = sum(len(spikes) for spikes in self.detected_spikes.values())

        return {
            "symbols_tracked": total_symbols,
            "total_data_points": total_data_points,
            "total_spikes_detected": total_spikes,
            "average_data_points_per_symbol": total_data_points / total_symbols if total_symbols > 0 else 0,
            "rolling_window_size": self.config.rolling_window_size,
            "spike_threshold": float(self.config.spike_threshold),
        }
