import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

from telegram import Bot, Update
from telegram.constants import ParseMode
from telegram.error import NetworkError, RetryAfter, TelegramError
from telegram.ext import Application, CommandHandler, ContextTypes

from orderflow_pro.config.settings import settings
from orderflow_pro.models.alerts import AlertPriority, AlertType, BaseAlert
from orderflow_pro.utils.logger import get_logger

logger = get_logger("orderflow_pro.telegram")


@dataclass
class TelegramConfig:
    """Configuration for Telegram integration."""

    bot_token: str = settings.telegram_bot_token
    chat_id: str = settings.telegram_chat_id
    max_retries: int = 3
    retry_delay: float = 1.0
    enable_formatting: bool = True
    enable_status_updates: bool = True


class TelegramBot:
    """Telegram bot for sending OrderFlow Pro alerts."""

    def __init__(self, config: TelegramConfig = None):
        self.config = config or TelegramConfig()
        self.bot: Optional[Bot] = None
        self.application: Optional[Application] = None
        self.running = False

        # Statistics
        self.stats = {
            "messages_sent": 0,
            "messages_failed": 0,
            "retries_attempted": 0,
            "last_message_time": None,
            "uptime_start": None,
        }

        # Initialize bot
        self._initialize_bot()

    def _initialize_bot(self):
        """Initialize Telegram bot and application."""

        try:
            # Create bot instance
            self.bot = Bot(token=self.config.bot_token)

            # Create application
            self.application = Application.builder().token(self.config.bot_token).build()

            # Add command handlers
            self.application.add_handler(CommandHandler("start", self._handle_start))
            self.application.add_handler(CommandHandler("status", self._handle_status))
            self.application.add_handler(CommandHandler("stats", self._handle_stats))
            self.application.add_handler(CommandHandler("help", self._handle_help))

            logger.info("Telegram bot initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Telegram bot: {e}")
            raise

    async def start(self):
        """Start the Telegram bot."""

        if self.running:
            logger.warning("Telegram bot already running")
            return

        try:
            # Test connection
            await self._test_connection()

            # Start application
            await self.application.initialize()
            await self.application.start()

            self.running = True
            self.stats["uptime_start"] = datetime.utcnow()

            logger.info("Telegram bot started successfully")

            # Send startup message
            if self.config.enable_status_updates:
                await self._send_startup_message()

        except Exception as e:
            logger.error(f"Failed to start Telegram bot: {e}")
            raise

    async def stop(self):
        """Stop the Telegram bot."""

        if not self.running:
            return

        try:
            # Send shutdown message
            if self.config.enable_status_updates:
                await self._send_shutdown_message()

            # Stop application
            await self.application.stop()
            await self.application.shutdown()

            self.running = False

            logger.info("Telegram bot stopped")

        except Exception as e:
            logger.error(f"Error stopping Telegram bot: {e}")

    async def _test_connection(self):
        """Test Telegram bot connection."""

        try:
            bot_info = await self.bot.get_me()
            logger.info(f"Connected to Telegram bot: @{bot_info.username}")

            # Test chat access
            chat_info = await self.bot.get_chat(self.config.chat_id)
            logger.info(f"Connected to chat: {chat_info.type}")

        except Exception as e:
            logger.error(f"Telegram connection test failed: {e}")
            raise

    async def send_alert(self, alert: BaseAlert) -> bool:
        """Send an alert to Telegram."""

        if not self.running:
            logger.warning("Telegram bot not running, cannot send alert")
            return False

        try:
            # Format alert message
            message = self._format_alert_message(alert)

            # Send message with retry logic
            success = await self._send_message_with_retry(message)

            if success:
                self.stats["messages_sent"] += 1
                self.stats["last_message_time"] = datetime.utcnow()

                logger.info(f"Alert sent to Telegram: {alert.alert_type.value}")
            else:
                self.stats["messages_failed"] += 1
                logger.error(f"Failed to send alert to Telegram: {alert.alert_type.value}")

            return success

        except Exception as e:
            logger.error(f"Error sending alert to Telegram: {e}")
            self.stats["messages_failed"] += 1
            return False

    def _format_alert_message(self, alert: BaseAlert) -> str:
        """Format alert for Telegram message."""

        if not self.config.enable_formatting:
            return alert.message

        # Get priority emoji
        priority_emoji = self._get_priority_emoji(alert.priority)

        # Get alert type emoji
        type_emoji = self._get_alert_type_emoji(alert.alert_type)

        # Format header
        header = f"{priority_emoji} {type_emoji} **{alert.alert_type.value.upper().replace('_', ' ')}**"

        # Format exchange and symbol
        exchange_info = f"**Exchange:** {alert.exchange.upper()}"
        symbol_info = f"**Symbol:** {alert.symbol}"
        time_info = f"**Time:** {alert.timestamp.strftime('%H:%M:%S UTC')}"

        # Format main message
        main_message = alert.message

        # Add confidence if available
        confidence_info = ""
        if "confidence" in alert.data:
            confidence = alert.data["confidence"]
            confidence_info = f"**Confidence:** {confidence:.1%}"

        # Build final message
        parts = [header, "", exchange_info, symbol_info, time_info]

        if confidence_info:
            parts.append(confidence_info)

        parts.extend(["", main_message])

        # Add footer for critical alerts
        if alert.priority == AlertPriority.CRITICAL:
            parts.extend(["", "🚨 **CRITICAL ALERT** 🚨"])

        return "\n".join(parts)

    def _get_priority_emoji(self, priority: AlertPriority) -> str:
        """Get emoji for alert priority."""

        emoji_map = {
            AlertPriority.LOW: "ℹ️",
            AlertPriority.MEDIUM: "⚠️",
            AlertPriority.HIGH: "🚨",
            AlertPriority.CRITICAL: "🔥",
        }

        return emoji_map.get(priority, "📊")

    def _get_alert_type_emoji(self, alert_type: AlertType) -> str:
        """Get emoji for alert type."""

        emoji_map = {
            AlertType.VOLUME_SPIKE: "⚡",
            AlertType.WHALE_ORDER: "🐋",
            AlertType.BID_ASK_WALL: "🏗️",
            AlertType.ORDER_IMBALANCE: "⚖️",
            AlertType.LIQUIDITY_GAP: "🕳️",
            AlertType.ARBITRAGE: "💰",
        }

        return emoji_map.get(alert_type, "📊")

    async def _send_message_with_retry(self, message: str) -> bool:
        """Send message with retry logic."""

        for attempt in range(self.config.max_retries):
            try:
                # First try with HTML formatting
                await self.bot.send_message(chat_id=self.config.chat_id, text=message, parse_mode=ParseMode.HTML)
                return True

            except RetryAfter as e:
                # Rate limited, wait and retry
                wait_time = e.retry_after + 1
                logger.warning(f"Rate limited, waiting {wait_time} seconds")
                await asyncio.sleep(wait_time)
                self.stats["retries_attempted"] += 1

            except NetworkError as e:
                # Network error, wait and retry
                logger.warning(f"Network error: {e}, retrying in {self.config.retry_delay} seconds")
                await asyncio.sleep(self.config.retry_delay)
                self.stats["retries_attempted"] += 1

            except TelegramError as e:
                # Telegram API error
                logger.error(f"Telegram API error: {e}")

                # If it's a formatting error, try sending without formatting
                if "can't parse" in str(e).lower() or "reserved" in str(e).lower():
                    try:
                        # Remove HTML tags and send as plain text
                        plain_message = (
                            message.replace("<b>", "").replace("</b>", "").replace("<i>", "").replace("</i>", "")
                        )
                        await self.bot.send_message(chat_id=self.config.chat_id, text=plain_message)
                        logger.info("Message sent as plain text")
                        return True
                    except Exception as e2:
                        logger.error(f"Failed to send message without formatting: {e2}")

                # For other errors, don't retry
                return False

            except Exception as e:
                logger.error(f"Unexpected error sending message: {e}")
                return False

        logger.error(f"Failed to send message after {self.config.max_retries} attempts")
        return False

    async def _send_startup_message(self):
        """Send startup notification."""

        message = "🚀 <b>OrderFlow Pro Started</b>\n\nMonitoring order books for trading opportunities..."
        await self._send_message_with_retry(message)

    async def _send_shutdown_message(self):
        """Send shutdown notification."""

        if self.stats["uptime_start"]:
            uptime = datetime.utcnow() - self.stats["uptime_start"]
            message = (
                f"🛑 <b>OrderFlow Pro Stopped</b>\n\nUptime: {uptime}\nMessages sent: {self.stats['messages_sent']}"
            )
        else:
            message = "🛑 <b>OrderFlow Pro Stopped</b>"
        await self._send_message_with_retry(message)

    async def _send_shutdown_message(self):
        """Send shutdown notification."""

        uptime = datetime.utcnow() - self.stats["uptime_start"]
        message = f"🛑 **OrderFlow Pro Stopped**\n\nUptime: {uptime}\nMessages sent: {self.stats['messages_sent']}"
        await self._send_message_with_retry(message)

    async def send_status_update(self, status_info: Dict[str, Any]):
        """Send status update message."""

        if not self.config.enable_status_updates:
            return

        message = self._format_status_message(status_info)
        await self._send_message_with_retry(message)

    def _format_status_message(self, status_info: Dict[str, Any]) -> str:
        """Format status update message."""

        lines = ["📊 **OrderFlow Pro Status**", ""]

        # Exchange status
        if "exchanges" in status_info:
            lines.append("**Exchanges:**")
            for exchange, connected in status_info["exchanges"].items():
                status = "✅" if connected else "❌"
                lines.append(f"  {status} {exchange.upper()}")
            lines.append("")

        # Alert statistics
        if "alerts" in status_info:
            alerts = status_info["alerts"]
            lines.append("**Alerts (24h):**")
            lines.append(f"  📤 Sent: {alerts.get('sent', 0)}")
            lines.append(f"  🔄 Queued: {alerts.get('queued', 0)}")
            lines.append(f"  ⏰ Cooldown: {alerts.get('cooldown', 0)}")

        return "\n".join(lines)

    # Command handlers

    async def _handle_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command."""

        welcome_message = """
🚀 **Welcome to OrderFlow Pro!**

I'll send you real-time alerts about:
• 🐋 Whale orders
• 🏗️ Bid/Ask walls  
• ⚖️ Order imbalances
• ⚡ Volume spikes

**Commands:**
/status - System status
/stats - Statistics
/help - This help message
        """

        await update.message.reply_text(welcome_message, parse_mode=ParseMode.MARKDOWN_V2)

    async def _handle_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /status command."""

        # Get system status (this would come from main application)
        status_message = f"""
📊 **OrderFlow Pro Status**

**Bot Status:** {"🟢 Online" if self.running else "🔴 Offline"}
**Uptime:** {datetime.utcnow() - self.stats["uptime_start"] if self.stats["uptime_start"] else "N/A"}
**Messages Sent:** {self.stats["messages_sent"]}
**Last Message:** {self.stats["last_message_time"].strftime("%H:%M:%S") if self.stats["last_message_time"] else "None"}
        """

        await update.message.reply_text(status_message, parse_mode=ParseMode.MARKDOWN_V2)

    async def _handle_stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /stats command."""

        stats_message = f"""
📈 **OrderFlow Pro Statistics**

**Messages:**
• Sent: {self.stats["messages_sent"]}
• Failed: {self.stats["messages_failed"]}
• Retries: {self.stats["retries_attempted"]}

**Success Rate:** {(self.stats["messages_sent"] / max(1, self.stats["messages_sent"] + self.stats["messages_failed"])) * 100:.1f}%
        """

        await update.message.reply_text(stats_message, parse_mode=ParseMode.MARKDOWN_V2)

    async def _handle_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command."""

        help_message = """
🆘 **OrderFlow Pro Help**

**Alert Types:**
• 🐋 **Whale Orders** - Large orders above $100k
• 🏗️ **Bid/Ask Walls** - Support/resistance levels
• ⚖️ **Order Imbalances** - Directional bias >70%
• ⚡ **Volume Spikes** - Volume >150% of average

**Commands:**
• /start - Welcome message
• /status - Current system status
• /stats - Message statistics
• /help - This help message

**Priority Levels:**
• ℹ️ Low - Informational
• ⚠️ Medium - Notable
• 🚨 High - Important
• 🔥 Critical - Urgent action
        """

        await update.message.reply_text(help_message, parse_mode=ParseMode.MARKDOWN_V2)

    def get_statistics(self) -> Dict[str, Any]:
        """Get Telegram bot statistics."""

        uptime = None
        if self.stats["uptime_start"]:
            uptime = datetime.utcnow() - self.stats["uptime_start"]

        return {
            **self.stats,
            "running": self.running,
            "uptime": str(uptime) if uptime else None,
            "success_rate": self._calculate_success_rate(),
            "chat_id": self.config.chat_id,
            "bot_token_configured": bool(self.config.bot_token),
        }

    def _calculate_success_rate(self) -> float:
        """Calculate message success rate."""

        total_attempts = self.stats["messages_sent"] + self.stats["messages_failed"]
        if total_attempts == 0:
            return 0.0

        return (self.stats["messages_sent"] / total_attempts) * 100

    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()


# Global Telegram bot instance
telegram_bot = TelegramBot()


# Alert callback function for the alert system
async def telegram_alert_callback(alert: BaseAlert):
    """Callback function to send alerts to Telegram."""
    await telegram_bot.send_alert(alert)


# Convenience functions
async def send_telegram_alert(alert: BaseAlert) -> bool:
    """Send alert to Telegram."""
    return await telegram_bot.send_alert(alert)


async def send_status_update(status_info: Dict[str, Any]):
    """Send status update to Telegram."""
    await telegram_bot.send_status_update(status_info)


def get_telegram_stats() -> Dict[str, Any]:
    """Get Telegram bot statistics."""
    return telegram_bot.get_statistics()


# Setup function to register with alert system
def setup_telegram_alerts():
    """Setup Telegram integration with alert system."""

    from orderflow_pro.alerts.system import alert_system

    # Register telegram callback
    alert_system.register_alert_callback(telegram_alert_callback)

    logger.info("Telegram alerts configured")
