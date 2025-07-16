from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class AlertType(str, Enum):
    """Alert type enumeration."""

    VOLUME_SPIKE = "volume_spike"
    WHALE_ORDER = "whale_order"
    BID_ASK_WALL = "bid_ask_wall"
    ORDER_IMBALANCE = "order_imbalance"
    LIQUIDITY_GAP = "liquidity_gap"
    ARBITRAGE = "arbitrage"


class AlertPriority(str, Enum):
    """Alert priority levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertStatus(str, Enum):
    """Alert status enumeration."""

    PENDING = "pending"
    SENT = "sent"
    FAILED = "failed"
    COOLDOWN = "cooldown"


class BaseAlert(BaseModel):
    """Base alert model with common fields."""

    # Identification
    alert_id: str = Field(..., description="Unique alert identifier")
    alert_type: AlertType = Field(..., description="Type of alert")
    priority: AlertPriority = Field(..., description="Alert priority level")

    # Market context
    exchange: str = Field(..., description="Exchange name")
    symbol: str = Field(..., description="Trading pair symbol")
    timestamp: datetime = Field(..., description="Alert timestamp")

    # Alert data
    message: str = Field(..., description="Human-readable alert message")
    data: Dict[str, Any] = Field(..., description="Raw alert data")

    # Status tracking
    status: AlertStatus = Field(default=AlertStatus.PENDING, description="Alert status")
    sent_at: Optional[datetime] = Field(default=None, description="Time alert was sent")

    def to_telegram_message(self) -> str:
        """Convert alert to Telegram message format."""
        priority_emoji = {
            AlertPriority.LOW: "ℹ️",
            AlertPriority.MEDIUM: "⚠️",
            AlertPriority.HIGH: "🚨",
            AlertPriority.CRITICAL: "🔥",
        }

        emoji = priority_emoji.get(self.priority, "📊")

        return f"""
{emoji} **{self.alert_type.value.upper().replace("_", " ")}**

**Exchange:** {self.exchange.upper()}
**Symbol:** {self.symbol}
**Time:** {self.timestamp.strftime("%H:%M:%S UTC")}

{self.message}
        """.strip()


class VolumeSpikeAlert(BaseAlert):
    """Volume spike alert."""

    alert_type: AlertType = Field(default=AlertType.VOLUME_SPIKE, frozen=True)

    # Volume spike specific data
    current_volume: Decimal = Field(..., description="Current volume")
    average_volume: Decimal = Field(..., description="Average volume")
    spike_ratio: Decimal = Field(..., description="Volume spike ratio")

    def __init__(self, **data):
        # Auto-generate message if not provided
        if "message" not in data:
            spike_percentage = (data["spike_ratio"] - 1) * 100
            data["message"] = (
                f"Volume spike detected: {spike_percentage:.1f}% above average\n"
                f"Current: {data['current_volume']:,.0f}\n"
                f"Average: {data['average_volume']:,.0f}\n"
                f"Ratio: {data['spike_ratio']:.2f}x"
            )
        super().__init__(**data)


class WhaleOrderAlert(BaseAlert):
    """Large order (whale) alert."""

    alert_type: AlertType = Field(default=AlertType.WHALE_ORDER, frozen=True)

    # Whale order specific data
    order_side: str = Field(..., description="Order side (bid/ask)")
    order_price: Decimal = Field(..., description="Order price")
    order_volume: Decimal = Field(..., description="Order volume")
    notional_value: Decimal = Field(..., description="Order notional value in USD")

    def __init__(self, **data):
        if "message" not in data:
            side_emoji = "🟢" if data["order_side"] == "bid" else "🔴"
            data["message"] = (
                f"{side_emoji} Large {data['order_side']} order detected\n"
                f"Price: ${data['order_price']:,.2f}\n"
                f"Volume: {data['order_volume']:,.4f}\n"
                f"Value: ${data['notional_value']:,.0f}"
            )
        super().__init__(**data)


class BidAskWallAlert(BaseAlert):
    """Bid/Ask wall alert."""

    alert_type: AlertType = Field(default=AlertType.BID_ASK_WALL, frozen=True)

    # Wall specific data
    wall_side: str = Field(..., description="Wall side (bid/ask)")
    wall_price: Decimal = Field(..., description="Wall price level")
    wall_volume: Decimal = Field(..., description="Total wall volume")
    wall_value: Decimal = Field(..., description="Total wall value in USD")
    order_count: Optional[int] = Field(default=None, description="Number of orders in wall")

    def __init__(self, **data):
        if "message" not in data:
            side_emoji = "🛡️" if data["wall_side"] == "bid" else "🚧"
            wall_type = "Support" if data["wall_side"] == "bid" else "Resistance"
            data["message"] = (
                f"{side_emoji} {wall_type} wall detected\n"
                f"Price: ${data['wall_price']:,.2f}\n"
                f"Volume: {data['wall_volume']:,.4f}\n"
                f"Value: ${data['wall_value']:,.0f}"
            )
        super().__init__(**data)


class OrderImbalanceAlert(BaseAlert):
    """Order book imbalance alert."""

    alert_type: AlertType = Field(default=AlertType.ORDER_IMBALANCE, frozen=True)

    # Imbalance specific data
    imbalance_ratio: Decimal = Field(..., description="Bid/ask imbalance ratio")
    bid_volume: Decimal = Field(..., description="Total bid volume")
    ask_volume: Decimal = Field(..., description="Total ask volume")
    depth_analyzed: int = Field(..., description="Order book depth analyzed")

    def __init__(self, **data):
        if "message" not in data:
            ratio = data["imbalance_ratio"]
            if ratio > Decimal("0.7"):
                bias = "Bullish"
                emoji = "📈"
                percentage = ratio * 100
            else:
                bias = "Bearish"
                emoji = "📉"
                percentage = (1 - ratio) * 100

            data["message"] = (
                f"{emoji} {bias} order book imbalance\n"
                f"Bias: {percentage:.1f}% {bias.lower()}\n"
                f"Bid Volume: {data['bid_volume']:,.0f}\n"
                f"Ask Volume: {data['ask_volume']:,.0f}\n"
                f"Depth: {data['depth_analyzed']} levels"
            )
        super().__init__(**data)


class LiquidityGapAlert(BaseAlert):
    """Liquidity gap alert."""

    alert_type: AlertType = Field(default=AlertType.LIQUIDITY_GAP, frozen=True)

    # Gap specific data
    gap_side: str = Field(..., description="Gap side (bid/ask)")
    gap_start: Decimal = Field(..., description="Gap start price")
    gap_end: Decimal = Field(..., description="Gap end price")
    gap_size: Decimal = Field(..., description="Gap size percentage")

    def __init__(self, **data):
        if "message" not in data:
            data["message"] = (
                f"🕳️ Liquidity gap detected on {data['gap_side']} side\n"
                f"Range: ${data['gap_start']:,.2f} - ${data['gap_end']:,.2f}\n"
                f"Gap Size: {data['gap_size']:.2f}%\n"
                f"⚡ Potential for rapid price movement"
            )
        super().__init__(**data)


class ArbitrageAlert(BaseAlert):
    """Arbitrage opportunity alert."""

    alert_type: AlertType = Field(default=AlertType.ARBITRAGE, frozen=True)

    # Arbitrage specific data
    buy_exchange: str = Field(..., description="Exchange to buy from")
    sell_exchange: str = Field(..., description="Exchange to sell to")
    buy_price: Decimal = Field(..., description="Buy price")
    sell_price: Decimal = Field(..., description="Sell price")
    profit_bps: int = Field(..., description="Profit in basis points")

    def __init__(self, **data):
        if "message" not in data:
            profit_pct = data["profit_bps"] / 100
            data["message"] = (
                f"💰 Arbitrage opportunity detected\n"
                f"Buy: {data['buy_exchange'].upper()} @ ${data['buy_price']:,.2f}\n"
                f"Sell: {data['sell_exchange'].upper()} @ ${data['sell_price']:,.2f}\n"
                f"Profit: {profit_pct:.2f}% ({data['profit_bps']} bps)"
            )
        super().__init__(**data)


class AlertCooldown(BaseModel):
    """Alert cooldown tracking."""

    alert_key: str = Field(..., description="Alert identifier key")
    alert_type: AlertType = Field(..., description="Type of alert")
    exchange: str = Field(..., description="Exchange name")
    symbol: str = Field(..., description="Trading pair symbol")

    last_sent: datetime = Field(..., description="Last time alert was sent")
    cooldown_until: datetime = Field(..., description="Cooldown end time")

    @property
    def is_on_cooldown(self) -> bool:
        """Check if alert is still on cooldown."""
        return datetime.utcnow() < self.cooldown_until

    @property
    def remaining_cooldown_seconds(self) -> int:
        """Get remaining cooldown time in seconds."""
        if not self.is_on_cooldown:
            return 0
        return int((self.cooldown_until - datetime.utcnow()).total_seconds())


class AlertManager(BaseModel):
    """Alert management system."""

    cooldowns: Dict[str, AlertCooldown] = Field(default_factory=dict, description="Active cooldowns")
    sent_alerts: List[BaseAlert] = Field(default_factory=list, description="Sent alerts history")

    def generate_alert_key(self, alert_type: AlertType, exchange: str, symbol: str) -> str:
        """Generate unique alert key for cooldown tracking."""
        return f"{alert_type.value}:{exchange}:{symbol}"

    def is_on_cooldown(self, alert_type: AlertType, exchange: str, symbol: str) -> bool:
        """Check if alert type is on cooldown for this symbol/exchange."""
        key = self.generate_alert_key(alert_type, exchange, symbol)
        cooldown = self.cooldowns.get(key)

        if not cooldown:
            return False

        if cooldown.is_on_cooldown:
            return True

        # Remove expired cooldown
        del self.cooldowns[key]
        return False

    def add_cooldown(self, alert: BaseAlert, cooldown_seconds: int = 300):
        """Add alert to cooldown."""
        key = self.generate_alert_key(alert.alert_type, alert.exchange, alert.symbol)
        cooldown_until = datetime.utcnow().replace(microsecond=0)
        cooldown_until = cooldown_until.replace(second=cooldown_until.second + cooldown_seconds)

        self.cooldowns[key] = AlertCooldown(
            alert_key=key,
            alert_type=alert.alert_type,
            exchange=alert.exchange,
            symbol=alert.symbol,
            last_sent=alert.sent_at or datetime.utcnow(),
            cooldown_until=cooldown_until,
        )

    def can_send_alert(self, alert: BaseAlert) -> bool:
        """Check if alert can be sent (not on cooldown)."""
        return not self.is_on_cooldown(alert.alert_type, alert.exchange, alert.symbol)

    def mark_alert_sent(self, alert: BaseAlert, cooldown_seconds: int = 300):
        """Mark alert as sent and add to cooldown."""
        alert.status = AlertStatus.SENT
        alert.sent_at = datetime.utcnow()
        self.sent_alerts.append(alert)
        self.add_cooldown(alert, cooldown_seconds)

    def get_recent_alerts(self, hours: int = 24) -> List[BaseAlert]:
        """Get alerts sent in the last N hours."""
        cutoff = datetime.utcnow().replace(microsecond=0)
        cutoff = cutoff.replace(hour=cutoff.hour - hours)

        return [alert for alert in self.sent_alerts if alert.sent_at and alert.sent_at >= cutoff]
