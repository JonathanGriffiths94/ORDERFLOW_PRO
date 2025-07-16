import asyncio
import signal
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List

from orderflow_pro.alerts.system import alert_system
from orderflow_pro.alerts.telegram import setup_telegram_alerts, telegram_bot
from orderflow_pro.analyzers.bid_ask_walls import BidAskWallAnalyzer
from orderflow_pro.analyzers.imbalances import OrderBookImbalanceAnalyzer
from orderflow_pro.analyzers.volume_spikes import VolumeSpikeAnalyzer
from orderflow_pro.analyzers.whale_orders import WhaleOrderAnalyzer
from orderflow_pro.config.settings import settings
from orderflow_pro.exchanges.manager import exchange_manager
from orderflow_pro.models.order_book import OrderBookSnapshot
from orderflow_pro.utils.logger import log_shutdown, log_startup, main_logger, setup_logging


@dataclass
class ApplicationStats:
    """Application statistics."""

    start_time: datetime
    order_books_processed: int = 0
    patterns_detected: int = 0
    alerts_sent: int = 0
    errors_encountered: int = 0
    uptime_seconds: int = 0

    def update_uptime(self):
        """Update uptime calculation."""
        self.uptime_seconds = int((datetime.utcnow() - self.start_time).total_seconds())


class OrderFlowProApp:
    """Main OrderFlow Pro application."""

    def __init__(self):
        self.running = False
        self.shutdown_event = asyncio.Event()

        # Initialize analyzers
        self.wall_analyzer = BidAskWallAnalyzer()
        self.whale_analyzer = WhaleOrderAnalyzer()
        self.imbalance_analyzer = OrderBookImbalanceAnalyzer()
        self.volume_analyzer = VolumeSpikeAnalyzer()

        # Statistics
        self.stats = ApplicationStats(start_time=datetime.utcnow())

        # Background tasks
        self.background_tasks: List[asyncio.Task] = []

        # Setup signal handlers
        self._setup_signal_handlers()

    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""

        def signal_handler(signum, frame):
            """Handle shutdown signals."""
            main_logger.info(f"Received signal {signum}, initiating shutdown...")

            # Create new event loop if needed
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            # Schedule shutdown
            loop.create_task(self._shutdown_signal_handler())

        # Register signal handlers
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    async def _shutdown_signal_handler(self):
        """Handle shutdown signal asynchronously."""
        self.shutdown_event.set()

    async def start(self):
        """Start the OrderFlow Pro application."""

        if self.running:
            main_logger.warning("Application already running")
            return

        main_logger.info("Starting OrderFlow Pro application...")

        try:
            # Setup logging
            setup_logging()

            # Log startup
            log_startup(
                "OrderFlow Pro",
                {
                    "version": "1.0.0",
                    "exchanges": settings.exchanges,
                    "trading_pairs": settings.trading_pairs,
                    "alert_cooldown": settings.alert_cooldown,
                    "environment": settings.environment,
                },
            )

            # Start components in order
            await self._start_components()

            # Setup data flow
            self._setup_data_flow()

            # Start background tasks
            self._start_background_tasks()

            self.running = True
            main_logger.info("OrderFlow Pro application started successfully")

            # Send startup notification
            await self._send_startup_notification()

        except Exception as e:
            main_logger.error(f"Failed to start application: {e}")
            await self.stop()
            raise

    async def _start_components(self):
        """Start all application components."""

        main_logger.info("Starting application components...")

        # Start exchange manager
        main_logger.info("Starting exchange manager...")
        await exchange_manager.start()

        # Start alert system
        main_logger.info("Starting alert system...")
        await alert_system.start()

        # Start Telegram bot
        main_logger.info("Starting Telegram bot...")
        await telegram_bot.start()

        # Setup Telegram alerts
        setup_telegram_alerts()

        main_logger.info("All components started successfully")

    def _setup_data_flow(self):
        """Setup the data flow pipeline."""

        main_logger.info("Setting up data flow pipeline...")

        # Register order book callback
        exchange_manager.register_order_book_callback(self._process_order_book)

        main_logger.info("Data flow pipeline configured")

    def _start_background_tasks(self):
        """Start background maintenance tasks."""

        main_logger.info("Starting background tasks...")

        # Statistics update task
        stats_task = asyncio.create_task(self._stats_update_loop())
        self.background_tasks.append(stats_task)

        # Cleanup task
        cleanup_task = asyncio.create_task(self._cleanup_loop())
        self.background_tasks.append(cleanup_task)

        # Health check task
        health_task = asyncio.create_task(self._health_check_loop())
        self.background_tasks.append(health_task)

        main_logger.info(f"Started {len(self.background_tasks)} background tasks")

    async def _process_order_book(self, snapshot: OrderBookSnapshot):
        """Process order book snapshot through the analysis pipeline."""

        try:
            # Update statistics
            self.stats.order_books_processed += 1

            # Run all analyzers
            all_patterns = []

            # Analyze bid/ask walls
            wall_patterns = self.wall_analyzer.analyze_order_book(snapshot)
            all_patterns.extend(wall_patterns)

            # Analyze whale orders
            whale_patterns = self.whale_analyzer.analyze_order_book(snapshot)
            all_patterns.extend(whale_patterns)

            # Analyze order book imbalances
            imbalance_patterns = self.imbalance_analyzer.analyze_order_book(snapshot)
            all_patterns.extend(imbalance_patterns)

            # Analyze volume (if available)
            # Note: Volume analysis typically requires separate volume data
            # This is a placeholder for integration with volume data

            # Update pattern statistics
            self.stats.patterns_detected += len(all_patterns)

            # Send patterns to alert system
            if all_patterns:
                await alert_system.process_order_book(snapshot, all_patterns)

                main_logger.debug(
                    f"Processed {len(all_patterns)} patterns", exchange=snapshot.exchange, symbol=snapshot.symbol
                )

        except Exception as e:
            main_logger.error(f"Error processing order book: {e}", exchange=snapshot.exchange, symbol=snapshot.symbol)
            self.stats.errors_encountered += 1

    async def _stats_update_loop(self):
        """Background task to update statistics."""

        while self.running:
            try:
                # Update uptime
                self.stats.update_uptime()

                # Log periodic statistics
                if self.stats.uptime_seconds % 300 == 0:  # Every 5 minutes
                    await self._log_periodic_stats()

                await asyncio.sleep(1)

            except Exception as e:
                main_logger.error(f"Error in stats update loop: {e}")
                await asyncio.sleep(5)

    async def _cleanup_loop(self):
        """Background task for cleanup operations."""

        while self.running:
            try:
                # Clean up old patterns
                self.wall_analyzer.clear_old_walls()
                self.whale_analyzer.clear_old_whales()
                self.imbalance_analyzer.clear_old_imbalances()
                self.volume_analyzer.clear_old_spikes()

                # Clean up old volume history
                self.volume_analyzer.clear_old_history()

                main_logger.debug("Cleanup operations completed")

                # Run cleanup every 5 minutes
                await asyncio.sleep(300)

            except Exception as e:
                main_logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(60)

    async def _health_check_loop(self):
        """Background task for health monitoring."""

        while self.running:
            try:
                # Check component health
                health_status = await self._check_component_health()

                # Log health issues
                for component, status in health_status.items():
                    if not status["healthy"]:
                        main_logger.warning(f"Component unhealthy: {component}", **status)

                # Send health updates periodically
                if self.stats.uptime_seconds % 3600 == 0:  # Every hour
                    await self._send_health_update(health_status)

                await asyncio.sleep(30)

            except Exception as e:
                main_logger.error(f"Error in health check loop: {e}")
                await asyncio.sleep(60)

    async def _check_component_health(self) -> Dict[str, Dict[str, Any]]:
        """Check health of all components."""

        health_status = {}

        # Exchange manager health
        connected_exchanges = exchange_manager.get_connected_exchanges()
        health_status["exchange_manager"] = {
            "healthy": len(connected_exchanges) > 0,
            "connected_exchanges": len(connected_exchanges),
            "total_exchanges": len(settings.exchanges),
            "details": exchange_manager.get_exchange_status(),
        }

        # Alert system health
        alert_stats = alert_system.get_alert_statistics()
        health_status["alert_system"] = {
            "healthy": alert_stats["system_running"],
            "queue_size": alert_stats["queue_size"],
            "recent_alerts": alert_stats["recent_alerts_1h"],
            "details": alert_stats,
        }

        # Telegram bot health
        telegram_stats = telegram_bot.get_statistics()
        health_status["telegram_bot"] = {
            "healthy": telegram_stats["running"],
            "success_rate": telegram_stats["success_rate"],
            "messages_sent": telegram_stats["messages_sent"],
            "details": telegram_stats,
        }

        return health_status

    async def _log_periodic_stats(self):
        """Log periodic statistics."""

        main_logger.info(
            "Periodic statistics",
            uptime_seconds=self.stats.uptime_seconds,
            order_books_processed=self.stats.order_books_processed,
            patterns_detected=self.stats.patterns_detected,
            alerts_sent=self.stats.alerts_sent,
            errors_encountered=self.stats.errors_encountered,
        )

    async def _send_startup_notification(self):
        """Send startup notification via Telegram."""

        try:
            status_info = {
                "exchanges": {
                    exchange: exchange_manager.is_exchange_connected(exchange) for exchange in settings.exchanges
                },
                "trading_pairs": settings.trading_pairs,
                "start_time": self.stats.start_time.strftime("%H:%M:%S UTC"),
            }

            await telegram_bot.send_status_update(status_info)

        except Exception as e:
            main_logger.error(f"Error sending startup notification: {e}")

    async def _send_health_update(self, health_status: Dict[str, Dict[str, Any]]):
        """Send health update via Telegram."""

        try:
            # Format health status for Telegram
            status_info = {
                "exchanges": {
                    exchange: status["connected"]
                    for exchange, status in health_status["exchange_manager"]["details"].items()
                },
                "alerts": {
                    "sent": self.stats.alerts_sent,
                    "queued": health_status["alert_system"]["queue_size"],
                    "cooldown": len(alert_system.alert_manager.cooldowns),
                },
                "uptime": f"{self.stats.uptime_seconds // 3600}h {(self.stats.uptime_seconds % 3600) // 60}m",
            }

            await telegram_bot.send_status_update(status_info)

        except Exception as e:
            main_logger.error(f"Error sending health update: {e}")

    async def stop(self):
        """Stop the OrderFlow Pro application."""

        if not self.running:
            return

        main_logger.info("Stopping OrderFlow Pro application...")

        try:
            # Set running flag to False
            self.running = False

            # Cancel background tasks
            for task in self.background_tasks:
                task.cancel()

            # Wait for background tasks to complete
            if self.background_tasks:
                await asyncio.gather(*self.background_tasks, return_exceptions=True)

            # Stop components in reverse order
            await self._stop_components()

            # Log shutdown
            log_shutdown("OrderFlow Pro", "normal")

            main_logger.info("OrderFlow Pro application stopped successfully")

        except Exception as e:
            main_logger.error(f"Error during shutdown: {e}")

    async def _stop_components(self):
        """Stop all application components."""

        main_logger.info("Stopping application components...")

        # Stop Telegram bot
        try:
            await telegram_bot.stop()
        except Exception as e:
            main_logger.error(f"Error stopping Telegram bot: {e}")

        # Stop alert system
        try:
            await alert_system.stop()
        except Exception as e:
            main_logger.error(f"Error stopping alert system: {e}")

        # Stop exchange manager
        try:
            await exchange_manager.stop()
        except Exception as e:
            main_logger.error(f"Error stopping exchange manager: {e}")

        main_logger.info("All components stopped")

    async def run(self):
        """Run the application until shutdown signal."""

        await self.start()

        try:
            # Wait for shutdown signal
            await self.shutdown_event.wait()
        except KeyboardInterrupt:
            main_logger.info("Received keyboard interrupt")
        finally:
            await self.stop()

    def get_application_status(self) -> Dict[str, Any]:
        """Get comprehensive application status."""

        return {
            "running": self.running,
            "uptime_seconds": self.stats.uptime_seconds,
            "start_time": self.stats.start_time.isoformat(),
            "statistics": {
                "order_books_processed": self.stats.order_books_processed,
                "patterns_detected": self.stats.patterns_detected,
                "alerts_sent": self.stats.alerts_sent,
                "errors_encountered": self.stats.errors_encountered,
            },
            "components": {
                "exchange_manager": exchange_manager.get_exchange_status(),
                "alert_system": alert_system.get_alert_statistics(),
                "telegram_bot": telegram_bot.get_statistics(),
            },
            "configuration": {
                "exchanges": settings.exchanges,
                "trading_pairs": settings.trading_pairs,
                "environment": settings.environment,
            },
        }

    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()


# Global application instance
app = OrderFlowProApp()


async def main():
    """Main entry point."""

    try:
        await app.run()
    except Exception as e:
        main_logger.critical(f"Critical error in main: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Run the application
    asyncio.run(main())
