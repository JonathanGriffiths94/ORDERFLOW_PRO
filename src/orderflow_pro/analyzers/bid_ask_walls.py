import uuid
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Dict, List

from orderflow_pro.config.settings import settings
from orderflow_pro.models.order_book import OrderBookLevel, OrderBookSnapshot, OrderSide
from orderflow_pro.models.patterns import PatternSignificance, WallPattern
from orderflow_pro.utils.logger import get_logger

logger = get_logger("orderflow_pro.analyzers.walls")


@dataclass
class WallDetectionConfig:
    """Configuration for wall detection."""

    min_wall_value: Decimal = Decimal(str(settings.bid_ask_wall_threshold))
    max_depth: int = settings.order_book_depth
    wall_cluster_distance: Decimal = Decimal("0.005")  # 0.5% price clustering
    min_orders_for_wall: int = 1  # Minimum orders to consider a wall
    significance_thresholds: Dict[str, Decimal] = None

    def __post_init__(self):
        if self.significance_thresholds is None:
            self.significance_thresholds = {
                "low": self.min_wall_value,
                "medium": self.min_wall_value * 2,
                "high": self.min_wall_value * 5,
                "critical": self.min_wall_value * 10,
            }


class BidAskWallAnalyzer:
    """Analyzes order book data for bid/ask walls."""

    def __init__(self, config: WallDetectionConfig = None):
        self.config = config or WallDetectionConfig()
        self.detected_walls: Dict[str, List[WallPattern]] = {}

    def analyze_order_book(self, snapshot: OrderBookSnapshot) -> List[WallPattern]:
        """Analyze order book for wall patterns."""

        walls = []

        # Analyze bid walls (support)
        bid_walls = self._detect_walls(snapshot, OrderSide.BID)
        walls.extend(bid_walls)

        # Analyze ask walls (resistance)
        ask_walls = self._detect_walls(snapshot, OrderSide.ASK)
        walls.extend(ask_walls)

        # Store detected walls
        key = f"{snapshot.exchange}:{snapshot.symbol}"
        self.detected_walls[key] = walls

        # Log wall detection
        if walls:
            logger.info(
                f"Detected {len(walls)} walls",
                exchange=snapshot.exchange,
                symbol=snapshot.symbol,
                bid_walls=len(bid_walls),
                ask_walls=len(ask_walls),
            )

        return walls

    def _detect_walls(self, snapshot: OrderBookSnapshot, side: OrderSide) -> List[WallPattern]:
        """Detect walls on a specific side of the order book."""

        levels = snapshot.get_levels(side, self.config.max_depth)
        if not levels:
            return []

        walls = []

        # Method 1: Single large level walls
        single_walls = self._detect_single_level_walls(snapshot, levels, side)
        walls.extend(single_walls)

        # Method 2: Clustered walls (multiple levels close together)
        clustered_walls = self._detect_clustered_walls(snapshot, levels, side)
        walls.extend(clustered_walls)

        return walls

    def _detect_single_level_walls(
        self, snapshot: OrderBookSnapshot, levels: List[OrderBookLevel], side: OrderSide
    ) -> List[WallPattern]:
        """Detect walls formed by single large levels."""

        walls = []

        for i, level in enumerate(levels):
            if level.notional_value >= self.config.min_wall_value:
                # Calculate significance
                significance = self._calculate_significance(level.notional_value)

                # Calculate confidence based on size and position
                confidence = self._calculate_wall_confidence(level, i, levels)

                # Create wall pattern
                wall = WallPattern(
                    pattern_id=str(uuid.uuid4()),
                    wall_side=side,
                    wall_price=level.price,
                    wall_volume=level.volume,
                    wall_value=level.notional_value,
                    order_count=level.count,
                    depth_rank=i + 1,
                    significance=significance,
                    confidence=confidence,
                    exchange=snapshot.exchange,
                    symbol=snapshot.symbol,
                    timestamp=datetime.utcnow(),
                    current_price=snapshot.mid_price,
                    raw_data={"level_index": i, "total_levels": len(levels), "detection_method": "single_level"},
                )

                walls.append(wall)

                logger.log_pattern_detected(
                    f"{side.value}_wall",
                    snapshot.symbol,
                    snapshot.exchange,
                    float(confidence),
                    wall_value=float(level.notional_value),
                    wall_price=float(level.price),
                    depth_rank=i + 1,
                )

        return walls

    def _detect_clustered_walls(
        self, snapshot: OrderBookSnapshot, levels: List[OrderBookLevel], side: OrderSide
    ) -> List[WallPattern]:
        """Detect walls formed by clusters of levels."""

        if len(levels) < 2:
            return []

        walls = []
        clusters = self._find_price_clusters(levels, side)

        for cluster in clusters:
            cluster_value = sum(level.notional_value for level in cluster["levels"])

            if cluster_value >= self.config.min_wall_value:
                # Calculate cluster metrics
                cluster_volume = sum(level.volume for level in cluster["levels"])
                avg_price = sum(level.price * level.volume for level in cluster["levels"]) / cluster_volume

                # Calculate significance
                significance = self._calculate_significance(cluster_value)

                # Calculate confidence
                confidence = self._calculate_cluster_confidence(cluster, levels)

                # Create clustered wall pattern
                wall = WallPattern(
                    pattern_id=str(uuid.uuid4()),
                    wall_side=side,
                    wall_price=avg_price,
                    wall_volume=cluster_volume,
                    wall_value=cluster_value,
                    order_count=len(cluster["levels"]),
                    depth_rank=cluster["start_index"] + 1,
                    significance=significance,
                    confidence=confidence,
                    exchange=snapshot.exchange,
                    symbol=snapshot.symbol,
                    timestamp=datetime.utcnow(),
                    current_price=snapshot.mid_price,
                    raw_data={
                        "cluster_size": len(cluster["levels"]),
                        "price_range": float(cluster["price_range"]),
                        "detection_method": "clustered",
                        "start_index": cluster["start_index"],
                        "end_index": cluster["end_index"],
                    },
                )

                walls.append(wall)

                logger.log_pattern_detected(
                    f"{side.value}_cluster_wall",
                    snapshot.symbol,
                    snapshot.exchange,
                    float(confidence),
                    wall_value=float(cluster_value),
                    cluster_size=len(cluster["levels"]),
                    price_range=float(cluster["price_range"]),
                )

        return walls

    def _find_price_clusters(self, levels: List[OrderBookLevel], side: OrderSide) -> List[Dict]:
        """Find clusters of orders at similar price levels."""

        if not levels:
            return []

        clusters = []
        current_cluster = []
        cluster_start_idx = 0

        for i, level in enumerate(levels):
            if not current_cluster:
                # Start new cluster
                current_cluster = [level]
                cluster_start_idx = i
            else:
                # Check if level belongs to current cluster
                if self._is_in_cluster(level, current_cluster, side):
                    current_cluster.append(level)
                else:
                    # Finalize current cluster if it has multiple levels
                    if len(current_cluster) >= 2:
                        clusters.append(
                            {
                                "levels": current_cluster,
                                "start_index": cluster_start_idx,
                                "end_index": i - 1,
                                "price_range": self._calculate_price_range(current_cluster),
                            }
                        )

                    # Start new cluster
                    current_cluster = [level]
                    cluster_start_idx = i

        # Handle last cluster
        if len(current_cluster) >= 2:
            clusters.append(
                {
                    "levels": current_cluster,
                    "start_index": cluster_start_idx,
                    "end_index": len(levels) - 1,
                    "price_range": self._calculate_price_range(current_cluster),
                }
            )

        return clusters

    def _is_in_cluster(self, level: OrderBookLevel, cluster: List[OrderBookLevel], side: OrderSide) -> bool:
        """Check if a level belongs to the current cluster."""

        if not cluster:
            return False

        # Get reference price (first level in cluster)
        ref_price = cluster[0].price

        # Calculate percentage distance
        price_diff = abs(level.price - ref_price)
        percentage_diff = price_diff / ref_price

        return percentage_diff <= self.config.wall_cluster_distance

    def _calculate_price_range(self, levels: List[OrderBookLevel]) -> Decimal:
        """Calculate price range for a cluster."""

        if not levels:
            return Decimal("0")

        prices = [level.price for level in levels]
        return max(prices) - min(prices)

    def _calculate_significance(self, wall_value: Decimal) -> PatternSignificance:
        """Calculate wall significance based on value."""

        thresholds = self.config.significance_thresholds

        if wall_value >= thresholds["critical"]:
            return PatternSignificance.CRITICAL
        elif wall_value >= thresholds["high"]:
            return PatternSignificance.HIGH
        elif wall_value >= thresholds["medium"]:
            return PatternSignificance.MEDIUM
        else:
            return PatternSignificance.LOW

    def _calculate_wall_confidence(
        self, level: OrderBookLevel, index: int, all_levels: List[OrderBookLevel]
    ) -> Decimal:
        """Calculate confidence score for a single-level wall."""

        confidence = Decimal("0.5")  # Base confidence

        # Size factor (larger walls = higher confidence)
        size_factor = min(level.notional_value / self.config.min_wall_value, 5) / 5
        confidence += size_factor * Decimal("0.3")

        # Position factor (closer to best price = higher confidence)
        position_factor = max(0, 1 - (index / len(all_levels)))
        confidence += position_factor * Decimal("0.2")

        # Relative size factor (compared to surrounding levels)
        if index > 0 and index < len(all_levels) - 1:
            prev_level = all_levels[index - 1]
            next_level = all_levels[index + 1]
            avg_surrounding = (prev_level.notional_value + next_level.notional_value) / 2

            if avg_surrounding > 0:
                relative_size = level.notional_value / avg_surrounding
                relative_factor = min(relative_size / 3, 1)  # Cap at 3x
                confidence += relative_factor * Decimal("0.1")

        return min(confidence, Decimal("1.0"))

    def _calculate_cluster_confidence(self, cluster: Dict, all_levels: List[OrderBookLevel]) -> Decimal:
        """Calculate confidence score for a clustered wall."""

        confidence = Decimal("0.6")  # Base confidence (higher for clusters)

        # Cluster size factor
        cluster_size = len(cluster["levels"])
        size_factor = min(cluster_size / 5, 1)  # Cap at 5 levels
        confidence += size_factor * Decimal("0.2")

        # Density factor (how tight the cluster is)
        cluster_value = sum(level.notional_value for level in cluster["levels"])
        density_factor = min(cluster_value / (self.config.min_wall_value * 2), 1)
        confidence += density_factor * Decimal("0.15")

        # Position factor
        position_factor = max(0, 1 - (cluster["start_index"] / len(all_levels)))
        confidence += position_factor * Decimal("0.05")

        return min(confidence, Decimal("1.0"))

    def get_walls_near_price(
        self, exchange: str, symbol: str, target_price: Decimal, max_distance_pct: Decimal = Decimal("2")
    ) -> List[WallPattern]:
        """Get walls near a specific price level."""

        key = f"{exchange}:{symbol}"
        walls = self.detected_walls.get(key, [])

        nearby_walls = []

        for wall in walls:
            price_diff = abs(wall.wall_price - target_price)
            distance_pct = (price_diff / target_price) * 100

            if distance_pct <= max_distance_pct:
                nearby_walls.append(wall)

        return nearby_walls

    def get_strongest_walls(
        self, exchange: str, symbol: str, side: OrderSide = None, limit: int = 5
    ) -> List[WallPattern]:
        """Get strongest walls by value."""

        key = f"{exchange}:{symbol}"
        walls = self.detected_walls.get(key, [])

        if side:
            walls = [w for w in walls if w.wall_side == side]

        # Sort by wall value (descending)
        walls.sort(key=lambda w: w.wall_value, reverse=True)

        return walls[:limit]

    def clear_old_walls(self, max_age_seconds: int = 300):
        """Clear walls older than specified age."""

        cutoff_time = datetime.utcnow().timestamp() - max_age_seconds

        for key in list(self.detected_walls.keys()):
            self.detected_walls[key] = [
                wall for wall in self.detected_walls[key] if wall.timestamp.timestamp() > cutoff_time
            ]

            # Remove empty entries
            if not self.detected_walls[key]:
                del self.detected_walls[key]

    def get_wall_summary(self, exchange: str, symbol: str) -> Dict:
        """Get summary of detected walls."""

        key = f"{exchange}:{symbol}"
        walls = self.detected_walls.get(key, [])

        bid_walls = [w for w in walls if w.wall_side == OrderSide.BID]
        ask_walls = [w for w in walls if w.wall_side == OrderSide.ASK]

        return {
            "total_walls": len(walls),
            "bid_walls": len(bid_walls),
            "ask_walls": len(ask_walls),
            "strongest_bid": max(bid_walls, key=lambda w: w.wall_value).wall_value if bid_walls else 0,
            "strongest_ask": max(ask_walls, key=lambda w: w.wall_value).wall_value if ask_walls else 0,
            "critical_walls": len([w for w in walls if w.significance == PatternSignificance.CRITICAL]),
            "high_confidence_walls": len([w for w in walls if w.confidence > Decimal("0.8")]),
        }
