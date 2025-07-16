from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, computed_field

from .order_book import OrderSide


class PatternType(str, Enum):
    """Pattern type enumeration."""

    BID_WALL = "bid_wall"
    ASK_WALL = "ask_wall"
    WHALE_BID = "whale_bid"
    WHALE_ASK = "whale_ask"
    BULLISH_IMBALANCE = "bullish_imbalance"
    BEARISH_IMBALANCE = "bearish_imbalance"
    VOLUME_SPIKE = "volume_spike"
    LIQUIDITY_GAP = "liquidity_gap"


class PatternSignificance(str, Enum):
    """Pattern significance levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class BasePattern(BaseModel):
    """Base pattern model with common fields."""

    # Pattern identification
    pattern_id: str = Field(..., description="Unique pattern identifier")
    pattern_type: PatternType = Field(..., description="Type of pattern detected")
    significance: PatternSignificance = Field(..., description="Pattern significance level")

    # Market context
    exchange: str = Field(..., description="Exchange where pattern was detected")
    symbol: str = Field(..., description="Trading pair symbol")
    timestamp: datetime = Field(..., description="Pattern detection timestamp")

    # Pattern data
    confidence: Decimal = Field(..., ge=0, le=1, description="Pattern confidence score (0-1)")
    description: str = Field(..., description="Human-readable pattern description")
    raw_data: Dict[str, Any] = Field(..., description="Raw pattern data")

    # Analysis context
    current_price: Decimal = Field(..., description="Current market price")
    price_distance: Optional[Decimal] = Field(default=None, description="Distance from current price")

    @computed_field
    @property
    def age_seconds(self) -> int:
        """Get pattern age in seconds."""
        return int((datetime.utcnow() - self.timestamp).total_seconds())

    @computed_field
    @property
    def is_fresh(self) -> bool:
        """Check if pattern is fresh (less than 5 minutes old)."""
        return self.age_seconds < 300

    def get_alert_priority(self) -> str:
        """Convert significance to alert priority."""
        mapping = {
            PatternSignificance.LOW: "low",
            PatternSignificance.MEDIUM: "medium",
            PatternSignificance.HIGH: "high",
            PatternSignificance.CRITICAL: "critical",
        }
        return mapping[self.significance]


class WallPattern(BasePattern):
    """Order book wall pattern (bid or ask wall)."""

    # Wall-specific data
    wall_side: OrderSide = Field(..., description="Wall side (bid/ask)")
    wall_price: Decimal = Field(..., description="Wall price level")
    wall_volume: Decimal = Field(..., description="Total wall volume")
    wall_value: Decimal = Field(..., description="Wall value in USD")
    order_count: Optional[int] = Field(default=None, description="Number of orders in wall")
    depth_rank: int = Field(..., description="Wall position in order book (1=best)")

    @computed_field
    @property
    def wall_type(self) -> str:
        """Get wall type description."""
        return "Support Wall" if self.wall_side == OrderSide.BID else "Resistance Wall"

    @computed_field
    @property
    def price_distance_percentage(self) -> Decimal:
        """Calculate percentage distance from current price."""
        if self.current_price == 0:
            return Decimal("0")

        distance = abs(self.wall_price - self.current_price)
        return (distance / self.current_price) * 100

    def __init__(self, **data):
        # Auto-set pattern type based on wall side
        if "pattern_type" not in data:
            data["pattern_type"] = PatternType.BID_WALL if data["wall_side"] == OrderSide.BID else PatternType.ASK_WALL

        # Auto-generate description if not provided
        if "description" not in data:
            wall_type = "Support" if data["wall_side"] == OrderSide.BID else "Resistance"
            data["description"] = (
                f"{wall_type} wall of ${data['wall_value']:,.0f} "
                f"at ${data['wall_price']:,.2f} "
                f"({data['wall_volume']:,.4f} volume)"
            )

        super().__init__(**data)


class WhaleOrderPattern(BasePattern):
    """Large order (whale) pattern."""

    # Whale order specific data
    order_side: OrderSide = Field(..., description="Order side (bid/ask)")
    order_price: Decimal = Field(..., description="Order price")
    order_volume: Decimal = Field(..., description="Order volume")
    notional_value: Decimal = Field(..., description="Order value in USD")
    market_impact: Optional[Decimal] = Field(default=None, description="Estimated market impact")

    @computed_field
    @property
    def order_type(self) -> str:
        """Get order type description."""
        return "Whale Bid" if self.order_side == OrderSide.BID else "Whale Ask"

    def __init__(self, **data):
        # Auto-set pattern type based on order side
        if "pattern_type" not in data:
            data["pattern_type"] = (
                PatternType.WHALE_BID if data["order_side"] == OrderSide.BID else PatternType.WHALE_ASK
            )

        # Auto-generate description if not provided
        if "description" not in data:
            order_type = "bid" if data["order_side"] == OrderSide.BID else "ask"
            data["description"] = (
                f"Large {order_type} order: ${data['notional_value']:,.0f} "
                f"at ${data['order_price']:,.2f} "
                f"({data['order_volume']:,.4f} volume)"
            )

        super().__init__(**data)


class ImbalancePattern(BasePattern):
    """Order book imbalance pattern."""

    # Imbalance specific data
    imbalance_ratio: Decimal = Field(..., description="Bid/ask volume ratio")
    bid_volume: Decimal = Field(..., description="Total bid volume analyzed")
    ask_volume: Decimal = Field(..., description="Total ask volume analyzed")
    depth_analyzed: int = Field(..., description="Order book depth analyzed")
    imbalance_strength: Decimal = Field(..., description="Strength of imbalance (0-1)")

    @computed_field
    @property
    def bias_direction(self) -> str:
        """Get market bias direction."""
        return "Bullish" if self.imbalance_ratio > Decimal("0.5") else "Bearish"

    @computed_field
    @property
    def bias_percentage(self) -> Decimal:
        """Get bias percentage."""
        if self.imbalance_ratio > Decimal("0.5"):
            return self.imbalance_ratio * 100
        else:
            return (1 - self.imbalance_ratio) * 100

    def __init__(self, **data):
        # Auto-set pattern type based on imbalance direction
        if "pattern_type" not in data:
            ratio = data["imbalance_ratio"]
            data["pattern_type"] = (
                PatternType.BULLISH_IMBALANCE if ratio > Decimal("0.5") else PatternType.BEARISH_IMBALANCE
            )

        # Auto-generate description if not provided
        if "description" not in data:
            ratio = data["imbalance_ratio"]
            if ratio > Decimal("0.5"):
                bias = "Bullish"
                percentage = ratio * 100
            else:
                bias = "Bearish"
                percentage = (1 - ratio) * 100

            data["description"] = (
                f"{bias} order book imbalance: {percentage:.1f}% {bias.lower()} bias "
                f"(Depth: {data['depth_analyzed']} levels)"
            )

        super().__init__(**data)


class VolumePattern(BasePattern):
    """Volume spike pattern."""

    pattern_type: PatternType = Field(default=PatternType.VOLUME_SPIKE, frozen=True)

    # Volume specific data
    current_volume: Decimal = Field(..., description="Current volume")
    average_volume: Decimal = Field(..., description="Average volume baseline")
    spike_ratio: Decimal = Field(..., description="Volume spike ratio")
    spike_magnitude: Decimal = Field(..., description="Spike magnitude (current - average)")
    timeframe: str = Field(..., description="Volume timeframe analyzed")

    @computed_field
    @property
    def spike_percentage(self) -> Decimal:
        """Get spike percentage above average."""
        return (self.spike_ratio - 1) * 100

    def __init__(self, **data):
        # Auto-generate description if not provided
        if "description" not in data:
            spike_pct = (data["spike_ratio"] - 1) * 100
            data["description"] = (
                f"Volume spike: {spike_pct:.1f}% above {data['timeframe']} average "
                f"(Current: {data['current_volume']:,.0f}, "
                f"Average: {data['average_volume']:,.0f})"
            )

        super().__init__(**data)


class LiquidityGapPattern(BasePattern):
    """Liquidity gap pattern."""

    pattern_type: PatternType = Field(default=PatternType.LIQUIDITY_GAP, frozen=True)

    # Gap specific data
    gap_side: OrderSide = Field(..., description="Gap side (bid/ask)")
    gap_start_price: Decimal = Field(..., description="Gap start price")
    gap_end_price: Decimal = Field(..., description="Gap end price")
    gap_size_percentage: Decimal = Field(..., description="Gap size as percentage")
    gap_volume_missing: Decimal = Field(..., description="Volume that should be in gap")

    @computed_field
    @property
    def gap_range(self) -> Decimal:
        """Get absolute gap range."""
        return abs(self.gap_end_price - self.gap_start_price)

    @computed_field
    @property
    def gap_midpoint(self) -> Decimal:
        """Get gap midpoint price."""
        return (self.gap_start_price + self.gap_end_price) / 2

    def __init__(self, **data):
        # Auto-generate description if not provided
        if "description" not in data:
            side_name = "bid" if data["gap_side"] == OrderSide.BID else "ask"
            data["description"] = (
                f"Liquidity gap on {side_name} side: "
                f"${data['gap_start_price']:,.2f} - ${data['gap_end_price']:,.2f} "
                f"({data['gap_size_percentage']:.2f}% gap)"
            )

        super().__init__(**data)


class PatternCollection(BaseModel):
    """Collection of detected patterns for a symbol."""

    symbol: str = Field(..., description="Trading pair symbol")
    exchange: str = Field(..., description="Exchange name")
    timestamp: datetime = Field(..., description="Collection timestamp")

    # Pattern collections
    walls: List[WallPattern] = Field(default_factory=list, description="Wall patterns")
    whale_orders: List[WhaleOrderPattern] = Field(default_factory=list, description="Whale order patterns")
    imbalances: List[ImbalancePattern] = Field(default_factory=list, description="Imbalance patterns")
    volume_spikes: List[VolumePattern] = Field(default_factory=list, description="Volume spike patterns")
    liquidity_gaps: List[LiquidityGapPattern] = Field(default_factory=list, description="Liquidity gap patterns")

    @computed_field
    @property
    def total_patterns(self) -> int:
        """Get total number of patterns detected."""
        return (
            len(self.walls)
            + len(self.whale_orders)
            + len(self.imbalances)
            + len(self.volume_spikes)
            + len(self.liquidity_gaps)
        )

    @computed_field
    @property
    def critical_patterns(self) -> List[BasePattern]:
        """Get all critical significance patterns."""
        all_patterns = self.get_all_patterns()
        return [p for p in all_patterns if p.significance == PatternSignificance.CRITICAL]

    @computed_field
    @property
    def high_confidence_patterns(self) -> List[BasePattern]:
        """Get patterns with high confidence (>0.8)."""
        all_patterns = self.get_all_patterns()
        return [p for p in all_patterns if p.confidence > Decimal("0.8")]

    def get_all_patterns(self) -> List[BasePattern]:
        """Get all patterns as a single list."""
        return self.walls + self.whale_orders + self.imbalances + self.volume_spikes + self.liquidity_gaps

    def get_patterns_by_type(self, pattern_type: PatternType) -> List[BasePattern]:
        """Get patterns of a specific type."""
        all_patterns = self.get_all_patterns()
        return [p for p in all_patterns if p.pattern_type == pattern_type]

    def get_patterns_by_significance(self, significance: PatternSignificance) -> List[BasePattern]:
        """Get patterns of a specific significance level."""
        all_patterns = self.get_all_patterns()
        return [p for p in all_patterns if p.significance == significance]

    def add_pattern(self, pattern: BasePattern):
        """Add a pattern to the appropriate collection."""
        if isinstance(pattern, WallPattern):
            self.walls.append(pattern)
        elif isinstance(pattern, WhaleOrderPattern):
            self.whale_orders.append(pattern)
        elif isinstance(pattern, ImbalancePattern):
            self.imbalances.append(pattern)
        elif isinstance(pattern, VolumePattern):
            self.volume_spikes.append(pattern)
        elif isinstance(pattern, LiquidityGapPattern):
            self.liquidity_gaps.append(pattern)

    def clear_old_patterns(self, max_age_seconds: int = 300):
        """Remove patterns older than specified age."""
        cutoff_time = datetime.utcnow().timestamp() - max_age_seconds

        self.walls = [p for p in self.walls if p.timestamp.timestamp() > cutoff_time]
        self.whale_orders = [p for p in self.whale_orders if p.timestamp.timestamp() > cutoff_time]
        self.imbalances = [p for p in self.imbalances if p.timestamp.timestamp() > cutoff_time]
        self.volume_spikes = [p for p in self.volume_spikes if p.timestamp.timestamp() > cutoff_time]
        self.liquidity_gaps = [p for p in self.liquidity_gaps if p.timestamp.timestamp() > cutoff_time]
