import uuid
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Dict, List

from orderflow_pro.config.settings import settings
from orderflow_pro.models.order_book import OrderBookLevel, OrderBookSnapshot, OrderSide
from orderflow_pro.models.patterns import PatternSignificance, WhaleOrderPattern
from orderflow_pro.utils.logger import get_logger

logger = get_logger("orderflow_pro.analyzers.whale")


@dataclass
class WhaleDetectionConfig:
    """Configuration for whale order detection."""

    min_whale_value: Decimal = Decimal(str(settings.whale_order_threshold))
    max_depth: int = settings.order_book_depth
    market_impact_threshold: Decimal = Decimal("0.01")  # 1% market impact
    significance_thresholds: Dict[str, Decimal] = None
    tracking_window_seconds: int = 300  # 5 minutes

    def __post_init__(self):
        if self.significance_thresholds is None:
            self.significance_thresholds = {
                "low": self.min_whale_value,
                "medium": self.min_whale_value * 2,
                "high": self.min_whale_value * 5,
                "critical": self.min_whale_value * 10,
            }


@dataclass
class WhaleOrderTracking:
    """Tracking information for whale orders."""

    order_id: str
    exchange: str
    symbol: str
    side: OrderSide
    price: Decimal
    volume: Decimal
    notional_value: Decimal
    first_seen: datetime
    last_seen: datetime
    appearances: int = 1

    def update_tracking(self):
        """Update tracking information."""
        self.last_seen = datetime.utcnow()
        self.appearances += 1


class WhaleOrderAnalyzer:
    """Analyzes order book data for whale orders."""

    def __init__(self, config: WhaleDetectionConfig = None):
        self.config = config or WhaleDetectionConfig()
        self.detected_whales: Dict[str, List[WhaleOrderPattern]] = {}
        self.whale_tracking: Dict[str, WhaleOrderTracking] = {}

    def analyze_order_book(self, snapshot: OrderBookSnapshot) -> List[WhaleOrderPattern]:
        """Analyze order book for whale order patterns."""

        whales = []

        # Analyze bid side (whale buy orders)
        bid_whales = self._detect_whale_orders(snapshot, OrderSide.BID)
        whales.extend(bid_whales)

        # Analyze ask side (whale sell orders)
        ask_whales = self._detect_whale_orders(snapshot, OrderSide.ASK)
        whales.extend(ask_whales)

        # Update tracking
        self._update_whale_tracking(snapshot, whales)

        # Store detected whales
        key = f"{snapshot.exchange}:{snapshot.symbol}"
        self.detected_whales[key] = whales

        # Log whale detection
        if whales:
            total_value = sum(float(w.notional_value) for w in whales)
            logger.info(
                f"Detected {len(whales)} whale orders",
                exchange=snapshot.exchange,
                symbol=snapshot.symbol,
                total_value=total_value,
                bid_whales=len(bid_whales),
                ask_whales=len(ask_whales),
            )

        return whales

    def _detect_whale_orders(self, snapshot: OrderBookSnapshot, side: OrderSide) -> List[WhaleOrderPattern]:
        """Detect whale orders on a specific side."""

        levels = snapshot.get_levels(side, self.config.max_depth)
        if not levels:
            return []

        whales = []

        for i, level in enumerate(levels):
            if level.notional_value >= self.config.min_whale_value:
                # Calculate significance
                significance = self._calculate_significance(level.notional_value)

                # Calculate confidence
                confidence = self._calculate_whale_confidence(level, i, levels, snapshot)

                # Estimate market impact
                market_impact = self._estimate_market_impact(level, snapshot, side)

                # Create whale order pattern
                whale = WhaleOrderPattern(
                    pattern_id=str(uuid.uuid4()),
                    order_side=side,
                    order_price=level.price,
                    order_volume=level.volume,
                    notional_value=level.notional_value,
                    market_impact=market_impact,
                    significance=significance,
                    confidence=confidence,
                    exchange=snapshot.exchange,
                    symbol=snapshot.symbol,
                    timestamp=datetime.utcnow(),
                    current_price=snapshot.mid_price,
                    raw_data={
                        "level_index": i,
                        "depth_rank": i + 1,
                        "total_levels": len(levels),
                        "price_distance_pct": self._calculate_price_distance_pct(level.price, snapshot.mid_price),
                        "order_count": level.count,
                    },
                )

                whales.append(whale)

                logger.log_whale_order(
                    snapshot.symbol,
                    snapshot.exchange,
                    side.value,
                    float(level.price),
                    float(level.volume),
                    float(level.notional_value),
                )

        return whales

    def _calculate_significance(self, notional_value: Decimal) -> PatternSignificance:
        """Calculate whale order significance based on value."""

        thresholds = self.config.significance_thresholds

        if notional_value >= thresholds["critical"]:
            return PatternSignificance.CRITICAL
        elif notional_value >= thresholds["high"]:
            return PatternSignificance.HIGH
        elif notional_value >= thresholds["medium"]:
            return PatternSignificance.MEDIUM
        else:
            return PatternSignificance.LOW

    def _calculate_whale_confidence(
        self, level: OrderBookLevel, index: int, all_levels: List[OrderBookLevel], snapshot: OrderBookSnapshot
    ) -> Decimal:
        """Calculate confidence score for whale order."""

        confidence = Decimal("0.5")  # Base confidence

        # Size factor (larger orders = higher confidence)
        size_factor = min(level.notional_value / self.config.min_whale_value, 10) / 10
        confidence += size_factor * Decimal("0.3")

        # Position factor (closer to market = higher confidence)
        position_factor = max(0, 1 - (index / len(all_levels)))
        confidence += position_factor * Decimal("0.15")

        # Relative size factor (compared to surrounding levels)
        relative_factor = self._calculate_relative_size_factor(level, index, all_levels)
        confidence += relative_factor * Decimal("0.1")

        # Market context factor (how it fits with overall market)
        market_factor = self._calculate_market_context_factor(level, snapshot)
        confidence += market_factor * Decimal("0.05")

        return min(confidence, Decimal("1.0"))

    def _calculate_relative_size_factor(
        self, level: OrderBookLevel, index: int, all_levels: List[OrderBookLevel]
    ) -> Decimal:
        """Calculate how large this order is relative to surrounding levels."""

        if len(all_levels) <= 1:
            return Decimal("0")

        # Get surrounding levels
        surrounding_levels = []

        # Previous levels
        for i in range(max(0, index - 2), index):
            surrounding_levels.append(all_levels[i])

        # Next levels
        for i in range(index + 1, min(len(all_levels), index + 3)):
            surrounding_levels.append(all_levels[i])

        if not surrounding_levels:
            return Decimal("0")

        # Calculate average of surrounding levels
        avg_surrounding = sum(l.notional_value for l in surrounding_levels) / len(surrounding_levels)

        if avg_surrounding <= 0:
            return Decimal("0")

        # Calculate relative size
        relative_size = level.notional_value / avg_surrounding

        # Normalize to 0-1 range (cap at 10x)
        return min(relative_size / 10, Decimal("1"))

    def _calculate_market_context_factor(self, level: OrderBookLevel, snapshot: OrderBookSnapshot) -> Decimal:
        """Calculate how the whale order fits within market context."""

        # Check if order is at significant price level
        price_distance = abs(level.price - snapshot.mid_price)
        distance_pct = (price_distance / snapshot.mid_price) * 100

        # Orders closer to market are more significant
        if distance_pct <= 1:  # Within 1% of market
            return Decimal("1")
        elif distance_pct <= 2:  # Within 2% of market
            return Decimal("0.7")
        elif distance_pct <= 5:  # Within 5% of market
            return Decimal("0.4")
        else:
            return Decimal("0.1")

    def _estimate_market_impact(self, level: OrderBookLevel, snapshot: OrderBookSnapshot, side: OrderSide) -> Decimal:
        """Estimate potential market impact of whale order."""

        # Get opposite side for impact calculation
        opposite_side = OrderSide.ASK if side == OrderSide.BID else OrderSide.BID
        opposite_levels = snapshot.get_levels(opposite_side, 20)

        if not opposite_levels:
            return Decimal("0")

        # Calculate how much volume would be consumed
        remaining_volume = level.volume
        consumed_levels = 0

        for opp_level in opposite_levels:
            if remaining_volume <= 0:
                break

            if remaining_volume >= opp_level.volume:
                remaining_volume -= opp_level.volume
                consumed_levels += 1
            else:
                consumed_levels += remaining_volume / opp_level.volume
                remaining_volume = Decimal("0")

        # Calculate price impact
        if consumed_levels > 0 and consumed_levels < len(opposite_levels):
            start_price = opposite_levels[0].price
            end_price = opposite_levels[int(consumed_levels)].price
            price_impact = abs(end_price - start_price) / start_price
            return price_impact

        return Decimal("0")

    def _calculate_price_distance_pct(self, order_price: Decimal, mid_price: Decimal) -> float:
        """Calculate percentage distance from mid price."""

        if mid_price == 0:
            return 0.0

        distance = abs(order_price - mid_price)
        return float((distance / mid_price) * 100)

    def _update_whale_tracking(self, snapshot: OrderBookSnapshot, whales: List[WhaleOrderPattern]):
        """Update whale order tracking."""

        current_time = datetime.utcnow()

        # Create tracking entries for new whales
        for whale in whales:
            tracking_key = f"{whale.exchange}:{whale.symbol}:{whale.order_side.value}:{whale.order_price}"

            if tracking_key in self.whale_tracking:
                # Update existing tracking
                self.whale_tracking[tracking_key].update_tracking()
            else:
                # Create new tracking
                self.whale_tracking[tracking_key] = WhaleOrderTracking(
                    order_id=whale.pattern_id,
                    exchange=whale.exchange,
                    symbol=whale.symbol,
                    side=whale.order_side,
                    price=whale.order_price,
                    volume=whale.order_volume,
                    notional_value=whale.notional_value,
                    first_seen=current_time,
                    last_seen=current_time,
                )

        # Clean up old tracking entries
        self._cleanup_old_tracking()

    def _cleanup_old_tracking(self):
        """Remove old whale tracking entries."""

        cutoff_time = datetime.utcnow().timestamp() - self.config.tracking_window_seconds

        keys_to_remove = []
        for key, tracking in self.whale_tracking.items():
            if tracking.last_seen.timestamp() < cutoff_time:
                keys_to_remove.append(key)

        for key in keys_to_remove:
            del self.whale_tracking[key]

    def get_whale_by_value(self, exchange: str, symbol: str, min_value: Decimal = None) -> List[WhaleOrderPattern]:
        """Get whale orders above a certain value."""

        key = f"{exchange}:{symbol}"
        whales = self.detected_whales.get(key, [])

        if min_value:
            whales = [w for w in whales if w.notional_value >= min_value]

        return sorted(whales, key=lambda w: w.notional_value, reverse=True)

    def get_whale_by_side(self, exchange: str, symbol: str, side: OrderSide) -> List[WhaleOrderPattern]:
        """Get whale orders on a specific side."""

        key = f"{exchange}:{symbol}"
        whales = self.detected_whales.get(key, [])

        return [w for w in whales if w.order_side == side]

    def get_persistent_whales(self, exchange: str, symbol: str, min_appearances: int = 3) -> List[WhaleOrderTracking]:
        """Get whale orders that have appeared multiple times."""

        persistent = []

        for tracking in self.whale_tracking.values():
            if tracking.exchange == exchange and tracking.symbol == symbol and tracking.appearances >= min_appearances:
                persistent.append(tracking)

        return sorted(persistent, key=lambda w: w.notional_value, reverse=True)

    def get_market_impact_whales(
        self, exchange: str, symbol: str, min_impact: Decimal = None
    ) -> List[WhaleOrderPattern]:
        """Get whale orders with significant market impact."""

        key = f"{exchange}:{symbol}"
        whales = self.detected_whales.get(key, [])

        impact_whales = []
        for whale in whales:
            if whale.market_impact and whale.market_impact >= (min_impact or self.config.market_impact_threshold):
                impact_whales.append(whale)

        return sorted(impact_whales, key=lambda w: w.market_impact, reverse=True)

    def get_whale_summary(self, exchange: str, symbol: str) -> Dict:
        """Get summary of detected whale orders."""

        key = f"{exchange}:{symbol}"
        whales = self.detected_whales.get(key, [])

        if not whales:
            return {
                "total_whales": 0,
                "total_value": 0,
                "bid_whales": 0,
                "ask_whales": 0,
                "largest_whale": 0,
                "critical_whales": 0,
            }

        bid_whales = [w for w in whales if w.order_side == OrderSide.BID]
        ask_whales = [w for w in whales if w.order_side == OrderSide.ASK]

        return {
            "total_whales": len(whales),
            "total_value": sum(w.notional_value for w in whales),
            "bid_whales": len(bid_whales),
            "ask_whales": len(ask_whales),
            "largest_whale": max(w.notional_value for w in whales),
            "critical_whales": len([w for w in whales if w.significance == PatternSignificance.CRITICAL]),
            "high_impact_whales": len(
                [w for w in whales if w.market_impact and w.market_impact >= self.config.market_impact_threshold]
            ),
        }

    def clear_old_whales(self, max_age_seconds: int = 300):
        """Clear whale orders older than specified age."""

        cutoff_time = datetime.utcnow().timestamp() - max_age_seconds

        for key in list(self.detected_whales.keys()):
            self.detected_whales[key] = [
                whale for whale in self.detected_whales[key] if whale.timestamp.timestamp() > cutoff_time
            ]

            # Remove empty entries
            if not self.detected_whales[key]:
                del self.detected_whales[key]
