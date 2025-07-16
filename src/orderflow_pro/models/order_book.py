from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, computed_field


class OrderSide(str, Enum):
    """Order side enumeration."""

    BID = "bid"
    ASK = "ask"


class OrderBookLevel(BaseModel):
    """Represents a single level in the order book."""

    price: Decimal = Field(..., description="Price level")
    volume: Decimal = Field(..., description="Volume at this price level")
    count: Optional[int] = Field(default=None, description="Number of orders (if available)")

    @computed_field
    @property
    def notional_value(self) -> Decimal:
        """Calculate notional value (price * volume)."""
        return self.price * self.volume

    def __str__(self) -> str:
        return f"Level(price={self.price}, volume={self.volume}, value=${self.notional_value:,.2f})"


class OrderBookSnapshot(BaseModel):
    """Complete order book snapshot from an exchange."""

    # Metadata
    exchange: str = Field(..., description="Exchange name")
    symbol: str = Field(..., description="Trading pair symbol")
    timestamp: datetime = Field(..., description="Snapshot timestamp")

    # Order book data
    bids: List[OrderBookLevel] = Field(..., description="Bid levels (buy orders)")
    asks: List[OrderBookLevel] = Field(..., description="Ask levels (sell orders)")

    # Optional metadata
    nonce: Optional[int] = Field(default=None, description="Exchange nonce/sequence")

    @computed_field
    @property
    def spread(self) -> Decimal:
        """Calculate bid-ask spread."""
        if not self.bids or not self.asks:
            return Decimal("0")
        return self.asks[0].price - self.bids[0].price

    @computed_field
    @property
    def spread_percentage(self) -> Decimal:
        """Calculate spread as percentage of mid price."""
        if not self.bids or not self.asks:
            return Decimal("0")
        mid_price = (self.bids[0].price + self.asks[0].price) / 2
        return (self.spread / mid_price) * 100

    @computed_field
    @property
    def mid_price(self) -> Decimal:
        """Calculate mid price."""
        if not self.bids or not self.asks:
            return Decimal("0")
        return (self.bids[0].price + self.asks[0].price) / 2

    def get_levels(self, side: OrderSide, depth: int = 10) -> List[OrderBookLevel]:
        """Get order book levels for a specific side."""
        if side == OrderSide.BID:
            return self.bids[:depth]
        return self.asks[:depth]

    def get_total_volume(self, side: OrderSide, depth: int = 10) -> Decimal:
        """Get total volume for a side up to specified depth."""
        levels = self.get_levels(side, depth)
        return sum(level.volume for level in levels)

    def get_total_notional(self, side: OrderSide, depth: int = 10) -> Decimal:
        """Get total notional value for a side up to specified depth."""
        levels = self.get_levels(side, depth)
        return sum(level.notional_value for level in levels)

    def get_imbalance_ratio(self, depth: int = 10) -> Decimal:
        """Calculate bid/ask imbalance ratio."""
        bid_volume = self.get_total_volume(OrderSide.BID, depth)
        ask_volume = self.get_total_volume(OrderSide.ASK, depth)

        total_volume = bid_volume + ask_volume
        if total_volume == 0:
            return Decimal("0.5")  # Neutral

        return bid_volume / total_volume

    def find_large_orders(self, threshold_usd: Decimal, depth: int = 50) -> List[Tuple[OrderSide, OrderBookLevel]]:
        """Find orders above USD threshold."""
        large_orders = []

        # Check bids
        for level in self.get_levels(OrderSide.BID, depth):
            if level.notional_value >= threshold_usd:
                large_orders.append((OrderSide.BID, level))

        # Check asks
        for level in self.get_levels(OrderSide.ASK, depth):
            if level.notional_value >= threshold_usd:
                large_orders.append((OrderSide.ASK, level))

        return large_orders

    def find_walls(self, threshold_usd: Decimal, depth: int = 50) -> Dict[OrderSide, List[OrderBookLevel]]:
        """Find bid/ask walls (clusters of large orders)."""
        walls = {OrderSide.BID: [], OrderSide.ASK: []}

        for side in [OrderSide.BID, OrderSide.ASK]:
            levels = self.get_levels(side, depth)
            for level in levels:
                if level.notional_value >= threshold_usd:
                    walls[side].append(level)

        return walls

    def get_liquidity_gaps(
        self, min_gap_percentage: Decimal = Decimal("0.1")
    ) -> List[Tuple[OrderSide, Decimal, Decimal]]:
        """Find liquidity gaps (large price gaps between levels)."""
        gaps = []

        # Check bid gaps
        for i in range(len(self.bids) - 1):
            price_diff = self.bids[i].price - self.bids[i + 1].price
            gap_percentage = (price_diff / self.bids[i].price) * 100
            if gap_percentage >= min_gap_percentage:
                gaps.append((OrderSide.BID, self.bids[i + 1].price, self.bids[i].price))

        # Check ask gaps
        for i in range(len(self.asks) - 1):
            price_diff = self.asks[i + 1].price - self.asks[i].price
            gap_percentage = (price_diff / self.asks[i].price) * 100
            if gap_percentage >= min_gap_percentage:
                gaps.append((OrderSide.ASK, self.asks[i].price, self.asks[i + 1].price))

        return gaps


class AggregatedOrderBook(BaseModel):
    """Aggregated order book from multiple exchanges."""

    symbol: str = Field(..., description="Trading pair symbol")
    timestamp: datetime = Field(..., description="Aggregation timestamp")

    # Exchange snapshots
    snapshots: Dict[str, OrderBookSnapshot] = Field(..., description="Individual exchange snapshots")

    # Aggregated data
    aggregated_bids: List[OrderBookLevel] = Field(..., description="Volume-weighted aggregated bids")
    aggregated_asks: List[OrderBookLevel] = Field(..., description="Volume-weighted aggregated asks")

    @computed_field
    @property
    def total_exchanges(self) -> int:
        """Number of exchanges in aggregation."""
        return len(self.snapshots)

    @computed_field
    @property
    def best_bid(self) -> Optional[OrderBookLevel]:
        """Best aggregated bid."""
        return self.aggregated_bids[0] if self.aggregated_bids else None

    @computed_field
    @property
    def best_ask(self) -> Optional[OrderBookLevel]:
        """Best aggregated ask."""
        return self.aggregated_asks[0] if self.aggregated_asks else None

    @computed_field
    @property
    def aggregated_spread(self) -> Decimal:
        """Aggregated spread."""
        if not self.best_bid or not self.best_ask:
            return Decimal("0")
        return self.best_ask.price - self.best_bid.price

    def get_exchange_comparison(self) -> Dict[str, Dict[str, Decimal]]:
        """Compare key metrics across exchanges."""
        comparison = {}

        for exchange, snapshot in self.snapshots.items():
            comparison[exchange] = {
                "best_bid": snapshot.bids[0].price if snapshot.bids else Decimal("0"),
                "best_ask": snapshot.asks[0].price if snapshot.asks else Decimal("0"),
                "spread": snapshot.spread,
                "bid_volume_10": snapshot.get_total_volume(OrderSide.BID, 10),
                "ask_volume_10": snapshot.get_total_volume(OrderSide.ASK, 10),
                "imbalance": snapshot.get_imbalance_ratio(10),
            }

        return comparison

    def find_arbitrage_opportunities(self, min_profit_bps: int = 10) -> List[Dict]:
        """Find arbitrage opportunities between exchanges."""
        opportunities = []

        exchanges = list(self.snapshots.keys())
        for i, buy_exchange in enumerate(exchanges):
            for sell_exchange in exchanges[i + 1 :]:
                buy_snapshot = self.snapshots[buy_exchange]
                sell_snapshot = self.snapshots[sell_exchange]

                if not buy_snapshot.asks or not sell_snapshot.bids:
                    continue

                buy_price = buy_snapshot.asks[0].price
                sell_price = sell_snapshot.bids[0].price

                if sell_price > buy_price:
                    profit_bps = int(((sell_price - buy_price) / buy_price) * 10000)
                    if profit_bps >= min_profit_bps:
                        opportunities.append(
                            {
                                "buy_exchange": buy_exchange,
                                "sell_exchange": sell_exchange,
                                "buy_price": buy_price,
                                "sell_price": sell_price,
                                "profit_bps": profit_bps,
                                "profit_percentage": profit_bps / 100,
                            }
                        )

        return sorted(opportunities, key=lambda x: x["profit_bps"], reverse=True)


class VolumeData(BaseModel):
    """Volume data for pattern analysis."""

    exchange: str = Field(..., description="Exchange name")
    symbol: str = Field(..., description="Trading pair symbol")
    timestamp: datetime = Field(..., description="Data timestamp")

    # Volume metrics
    volume_24h: Decimal = Field(..., description="24-hour volume")
    volume_1h: Decimal = Field(..., description="1-hour volume")
    volume_current: Decimal = Field(..., description="Current period volume")

    # Volume analysis
    volume_average_20: Optional[Decimal] = Field(default=None, description="20-period volume average")
    volume_spike_ratio: Optional[Decimal] = Field(default=None, description="Current vs average ratio")

    @computed_field
    @property
    def is_volume_spike(self) -> bool:
        """Check if current volume is a spike."""
        if not self.volume_average_20 or not self.volume_spike_ratio:
            return False
        return self.volume_spike_ratio >= Decimal("1.5")  # 150% threshold

    def calculate_spike_ratio(self, current_volume: Decimal, average_volume: Decimal) -> Decimal:
        """Calculate volume spike ratio."""
        if average_volume == 0:
            return Decimal("1")
        return current_volume / average_volume
