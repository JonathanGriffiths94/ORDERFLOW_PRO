import uuid
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional

from orderflow_pro.config.settings import settings
from orderflow_pro.models.order_book import OrderBookSnapshot, OrderSide
from orderflow_pro.models.patterns import ImbalancePattern, PatternSignificance
from orderflow_pro.utils.logger import get_logger

logger = get_logger("orderflow_pro.analyzers.imbalances")


@dataclass
class ImbalanceDetectionConfig:
    """Configuration for imbalance detection."""

    min_imbalance_ratio: Decimal = Decimal(str(settings.imbalance_threshold))
    max_depth: int = settings.order_book_depth
    analysis_depths: List[int] = None  # Different depths to analyze
    volume_weighted: bool = True  # Use volume-weighted analysis
    min_total_volume: Decimal = Decimal("1000")  # Minimum volume for analysis
    significance_thresholds: Dict[str, Decimal] = None

    def __post_init__(self):
        if self.analysis_depths is None:
            self.analysis_depths = [5, 10, 20, 50]

        if self.significance_thresholds is None:
            # More extreme imbalances are more significant
            self.significance_thresholds = {
                "low": Decimal("0.65"),  # 65% imbalance
                "medium": Decimal("0.75"),  # 75% imbalance
                "high": Decimal("0.85"),  # 85% imbalance
                "critical": Decimal("0.95"),  # 95% imbalance
            }


@dataclass
class ImbalanceMetrics:
    """Metrics for order book imbalance analysis."""

    depth: int
    bid_volume: Decimal
    ask_volume: Decimal
    total_volume: Decimal
    imbalance_ratio: Decimal
    imbalance_strength: Decimal
    bias_direction: str
    confidence: Decimal


class OrderBookImbalanceAnalyzer:
    """Analyzes order book data for bid/ask imbalances."""

    def __init__(self, config: ImbalanceDetectionConfig = None):
        self.config = config or ImbalanceDetectionConfig()
        self.detected_imbalances: Dict[str, List[ImbalancePattern]] = {}
        self.imbalance_history: Dict[str, List[ImbalanceMetrics]] = {}

    def analyze_order_book(self, snapshot: OrderBookSnapshot) -> List[ImbalancePattern]:
        """Analyze order book for imbalance patterns."""

        imbalances = []

        # Analyze imbalances at different depths
        for depth in self.config.analysis_depths:
            if depth <= len(snapshot.bids) and depth <= len(snapshot.asks):
                metrics = self._calculate_imbalance_metrics(snapshot, depth)

                # Check if imbalance is significant
                if self._is_significant_imbalance(metrics):
                    imbalance_pattern = self._create_imbalance_pattern(snapshot, metrics)
                    imbalances.append(imbalance_pattern)

        # Store detected imbalances
        key = f"{snapshot.exchange}:{snapshot.symbol}"
        self.detected_imbalances[key] = imbalances

        # Update imbalance history
        self._update_imbalance_history(snapshot)

        # Log imbalance detection
        if imbalances:
            logger.info(
                f"Detected {len(imbalances)} imbalance patterns",
                exchange=snapshot.exchange,
                symbol=snapshot.symbol,
                depths_analyzed=len(self.config.analysis_depths),
            )

        return imbalances

    def _calculate_imbalance_metrics(self, snapshot: OrderBookSnapshot, depth: int) -> ImbalanceMetrics:
        """Calculate imbalance metrics for a specific depth."""

        # Get bid and ask volumes
        bid_volume = snapshot.get_total_volume(OrderSide.BID, depth)
        ask_volume = snapshot.get_total_volume(OrderSide.ASK, depth)
        total_volume = bid_volume + ask_volume

        # Calculate imbalance ratio (bid volume / total volume)
        imbalance_ratio = bid_volume / total_volume if total_volume > 0 else Decimal("0.5")

        # Calculate imbalance strength (how far from neutral 50%)
        imbalance_strength = abs(imbalance_ratio - Decimal("0.5"))

        # Determine bias direction
        bias_direction = "bullish" if imbalance_ratio > Decimal("0.5") else "bearish"

        # Calculate confidence based on volume and consistency
        confidence = self._calculate_imbalance_confidence(imbalance_ratio, total_volume, depth, snapshot)

        return ImbalanceMetrics(
            depth=depth,
            bid_volume=bid_volume,
            ask_volume=ask_volume,
            total_volume=total_volume,
            imbalance_ratio=imbalance_ratio,
            imbalance_strength=imbalance_strength,
            bias_direction=bias_direction,
            confidence=confidence,
        )

    def _is_significant_imbalance(self, metrics: ImbalanceMetrics) -> bool:
        """Check if imbalance is significant enough to report."""

        # Check minimum volume threshold
        if metrics.total_volume < self.config.min_total_volume:
            return False

        # Check imbalance threshold
        if metrics.imbalance_ratio >= self.config.min_imbalance_ratio:
            return True  # Bullish imbalance

        if metrics.imbalance_ratio <= (1 - self.config.min_imbalance_ratio):
            return True  # Bearish imbalance

        return False

    def _create_imbalance_pattern(self, snapshot: OrderBookSnapshot, metrics: ImbalanceMetrics) -> ImbalancePattern:
        """Create imbalance pattern from metrics."""

        # Calculate significance
        significance = self._calculate_significance(metrics.imbalance_ratio)

        # Create pattern
        pattern = ImbalancePattern(
            pattern_id=str(uuid.uuid4()),
            imbalance_ratio=metrics.imbalance_ratio,
            bid_volume=metrics.bid_volume,
            ask_volume=metrics.ask_volume,
            depth_analyzed=metrics.depth,
            imbalance_strength=metrics.imbalance_strength,
            significance=significance,
            confidence=metrics.confidence,
            exchange=snapshot.exchange,
            symbol=snapshot.symbol,
            timestamp=datetime.utcnow(),
            current_price=snapshot.mid_price,
            raw_data={
                "total_volume": float(metrics.total_volume),
                "bias_direction": metrics.bias_direction,
                "imbalance_percentage": float(metrics.imbalance_ratio * 100),
                "analysis_depth": metrics.depth,
            },
        )

        logger.log_pattern_detected(
            f"{metrics.bias_direction}_imbalance",
            snapshot.symbol,
            snapshot.exchange,
            float(metrics.confidence),
            imbalance_ratio=float(metrics.imbalance_ratio),
            depth=metrics.depth,
            total_volume=float(metrics.total_volume),
        )

        return pattern

    def _calculate_significance(self, imbalance_ratio: Decimal) -> PatternSignificance:
        """Calculate imbalance significance based on ratio."""

        thresholds = self.config.significance_thresholds

        # Check both bullish and bearish extremes
        if imbalance_ratio >= thresholds["critical"] or imbalance_ratio <= (1 - thresholds["critical"]):
            return PatternSignificance.CRITICAL
        elif imbalance_ratio >= thresholds["high"] or imbalance_ratio <= (1 - thresholds["high"]):
            return PatternSignificance.HIGH
        elif imbalance_ratio >= thresholds["medium"] or imbalance_ratio <= (1 - thresholds["medium"]):
            return PatternSignificance.MEDIUM
        else:
            return PatternSignificance.LOW

    def _calculate_imbalance_confidence(
        self, imbalance_ratio: Decimal, total_volume: Decimal, depth: int, snapshot: OrderBookSnapshot
    ) -> Decimal:
        """Calculate confidence score for imbalance."""

        confidence = Decimal("0.5")  # Base confidence

        # Imbalance strength factor (more extreme = higher confidence)
        imbalance_strength = abs(imbalance_ratio - Decimal("0.5"))
        strength_factor = min(imbalance_strength * 2, 1)  # Normalize to 0-1
        confidence += Decimal(str(strength_factor)) * Decimal("0.3")

        # Volume factor (more volume = higher confidence)
        volume_factor = min(total_volume / (self.config.min_total_volume * 5), 1)
        confidence += Decimal(str(volume_factor)) * Decimal("0.15")

        # Depth factor (deeper analysis = higher confidence)
        depth_factor = min(depth / 50, 1)  # Normalize to 0-1
        confidence += Decimal(str(depth_factor)) * Decimal("0.1")

        # Consistency factor (check if imbalance is consistent across depths)
        consistency_factor = self._calculate_consistency_factor(snapshot)
        confidence += consistency_factor * Decimal("0.05")

        return min(confidence, Decimal("1.0"))

    def _calculate_consistency_factor(self, snapshot: OrderBookSnapshot) -> Decimal:
        """Calculate how consistent the imbalance is across different depths."""

        # Calculate imbalance ratios at different depths
        ratios = []
        for depth in [5, 10, 20]:
            if depth <= len(snapshot.bids) and depth <= len(snapshot.asks):
                bid_vol = snapshot.get_total_volume(OrderSide.BID, depth)
                ask_vol = snapshot.get_total_volume(OrderSide.ASK, depth)
                total_vol = bid_vol + ask_vol

                if total_vol > 0:
                    ratio = bid_vol / total_vol
                    ratios.append(ratio)

        if len(ratios) < 2:
            return Decimal("0")

        # Calculate variance in ratios
        avg_ratio = sum(ratios) / len(ratios)
        variance = sum((r - avg_ratio) ** 2 for r in ratios) / len(ratios)

        # Lower variance = higher consistency
        consistency = max(0, 1 - float(variance) * 10)  # Scale variance
        return Decimal(str(consistency))

    def _update_imbalance_history(self, snapshot: OrderBookSnapshot):
        """Update imbalance history for trend analysis."""

        key = f"{snapshot.exchange}:{snapshot.symbol}"

        if key not in self.imbalance_history:
            self.imbalance_history[key] = []

        # Calculate metrics for primary depth (20 levels)
        primary_metrics = self._calculate_imbalance_metrics(snapshot, 20)

        # Add to history
        self.imbalance_history[key].append(primary_metrics)

        # Keep only recent history (last 100 updates)
        if len(self.imbalance_history[key]) > 100:
            self.imbalance_history[key] = self.imbalance_history[key][-100:]

    def get_imbalance_trend(self, exchange: str, symbol: str, periods: int = 10) -> Optional[Dict]:
        """Get imbalance trend over recent periods."""

        key = f"{exchange}:{symbol}"
        history = self.imbalance_history.get(key, [])

        if len(history) < periods:
            return None

        recent_history = history[-periods:]

        # Calculate trend metrics
        ratios = [h.imbalance_ratio for h in recent_history]
        avg_ratio = sum(ratios) / len(ratios)

        # Calculate trend direction
        first_half = ratios[: len(ratios) // 2]
        second_half = ratios[len(ratios) // 2 :]

        first_avg = sum(first_half) / len(first_half)
        second_avg = sum(second_half) / len(second_half)

        trend_direction = "strengthening" if second_avg > first_avg else "weakening"
        trend_strength = abs(second_avg - first_avg)

        return {
            "periods_analyzed": periods,
            "average_ratio": float(avg_ratio),
            "trend_direction": trend_direction,
            "trend_strength": float(trend_strength),
            "current_ratio": float(ratios[-1]),
            "bias": "bullish" if avg_ratio > 0.5 else "bearish",
        }

    def get_extreme_imbalances(
        self, exchange: str, symbol: str, min_ratio: Decimal = Decimal("0.8")
    ) -> List[ImbalancePattern]:
        """Get extreme imbalance patterns."""

        key = f"{exchange}:{symbol}"
        imbalances = self.detected_imbalances.get(key, [])

        extreme_imbalances = []

        for imbalance in imbalances:
            if imbalance.imbalance_ratio >= min_ratio or imbalance.imbalance_ratio <= (1 - min_ratio):
                extreme_imbalances.append(imbalance)

        return sorted(extreme_imbalances, key=lambda x: x.imbalance_strength, reverse=True)

    def get_consistent_imbalances(
        self, exchange: str, symbol: str, min_confidence: Decimal = Decimal("0.8")
    ) -> List[ImbalancePattern]:
        """Get highly confident imbalance patterns."""

        key = f"{exchange}:{symbol}"
        imbalances = self.detected_imbalances.get(key, [])

        return [i for i in imbalances if i.confidence >= min_confidence]

    def get_imbalance_by_depth(self, exchange: str, symbol: str, depth: int) -> Optional[ImbalancePattern]:
        """Get imbalance pattern for specific depth."""

        key = f"{exchange}:{symbol}"
        imbalances = self.detected_imbalances.get(key, [])

        for imbalance in imbalances:
            if imbalance.depth_analyzed == depth:
                return imbalance

        return None

    def get_imbalance_summary(self, exchange: str, symbol: str) -> Dict:
        """Get summary of detected imbalances."""

        key = f"{exchange}:{symbol}"
        imbalances = self.detected_imbalances.get(key, [])

        if not imbalances:
            return {
                "total_imbalances": 0,
                "bullish_imbalances": 0,
                "bearish_imbalances": 0,
                "strongest_imbalance": None,
                "average_confidence": 0,
            }

        bullish_imbalances = [i for i in imbalances if i.imbalance_ratio > Decimal("0.5")]
        bearish_imbalances = [i for i in imbalances if i.imbalance_ratio <= Decimal("0.5")]

        strongest = max(imbalances, key=lambda x: x.imbalance_strength)
        avg_confidence = sum(i.confidence for i in imbalances) / len(imbalances)

        return {
            "total_imbalances": len(imbalances),
            "bullish_imbalances": len(bullish_imbalances),
            "bearish_imbalances": len(bearish_imbalances),
            "strongest_imbalance": {
                "ratio": float(strongest.imbalance_ratio),
                "strength": float(strongest.imbalance_strength),
                "depth": strongest.depth_analyzed,
                "bias": strongest.bias_direction,
            },
            "average_confidence": float(avg_confidence),
            "critical_imbalances": len([i for i in imbalances if i.significance == PatternSignificance.CRITICAL]),
        }

    def clear_old_imbalances(self, max_age_seconds: int = 300):
        """Clear imbalances older than specified age."""

        cutoff_time = datetime.utcnow().timestamp() - max_age_seconds

        for key in list(self.detected_imbalances.keys()):
            self.detected_imbalances[key] = [
                imbalance
                for imbalance in self.detected_imbalances[key]
                if imbalance.timestamp.timestamp() > cutoff_time
            ]

            # Remove empty entries
            if not self.detected_imbalances[key]:
                del self.detected_imbalances[key]
