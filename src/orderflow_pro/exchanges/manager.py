import asyncio
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any, Callable, Dict, List, Optional

import ccxt.pro as ccxt

from orderflow_pro.config.settings import settings
from orderflow_pro.models.order_book import AggregatedOrderBook, OrderBookLevel, OrderBookSnapshot
from orderflow_pro.utils.logger import LogExecutionTime, get_logger

logger = get_logger("orderflow_pro.exchange")


@dataclass
class ExchangeStatus:
    """Exchange connection status."""

    name: str
    connected: bool
    last_update: Optional[datetime] = None
    error_count: int = 0
    last_error: Optional[str] = None


class ExchangeManager:
    """Manages multiple exchange connections and order book monitoring."""

    def __init__(self):
        self.exchanges: Dict[str, ccxt.Exchange] = {}
        self.exchange_status: Dict[str, ExchangeStatus] = {}
        self.order_book_callbacks: List[Callable] = []
        self.running = False
        self.tasks: List[asyncio.Task] = []

        # Initialize exchanges
        self._initialize_exchanges()

    def _initialize_exchanges(self):
        """Initialize exchange instances with ccxt."""

        logger.info("Initializing exchanges", exchanges=settings.exchanges)

        for exchange_name in settings.exchanges:
            try:
                # Get exchange configuration
                config = settings.get_exchange_config(exchange_name)

                # Create exchange instance
                exchange_class = getattr(ccxt, exchange_name)
                exchange = exchange_class(config)

                # Store exchange instance
                self.exchanges[exchange_name] = exchange
                self.exchange_status[exchange_name] = ExchangeStatus(name=exchange_name, connected=False)

                logger.info(
                    f"Exchange initialized: {exchange_name}",
                    exchange=exchange_name,
                    config=self._sanitize_config(config),
                )

            except Exception as e:
                logger.error(f"Failed to initialize exchange: {exchange_name}", exchange=exchange_name, error=str(e))
                self.exchange_status[exchange_name] = ExchangeStatus(
                    name=exchange_name, connected=False, error_count=1, last_error=str(e)
                )

    def _sanitize_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Remove sensitive data from config for logging."""
        sanitized = config.copy()

        # Remove sensitive fields
        sensitive_fields = ["apiKey", "secret", "passphrase"]
        for field in sensitive_fields:
            if field in sanitized:
                sanitized[field] = "***"

        return sanitized

    async def start(self):
        """Start monitoring all exchanges."""
        if self.running:
            logger.warning("Exchange manager already running")
            return

        self.running = True
        logger.info("Starting exchange manager")

        # Start monitoring tasks for each exchange
        for exchange_name in self.exchanges:
            for symbol in settings.trading_pairs:
                task = asyncio.create_task(self._monitor_exchange_symbol(exchange_name, symbol))
                self.tasks.append(task)

        # Start health check task
        health_task = asyncio.create_task(self._health_check_loop())
        self.tasks.append(health_task)

        logger.info(
            "Exchange manager started",
            exchanges=len(self.exchanges),
            symbols=len(settings.trading_pairs),
            tasks=len(self.tasks),
        )

    async def stop(self):
        """Stop monitoring all exchanges."""
        if not self.running:
            return

        logger.info("Stopping exchange manager")
        self.running = False

        # Cancel all tasks
        for task in self.tasks:
            task.cancel()

        # Wait for tasks to complete
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)

        # Close exchange connections
        for exchange_name, exchange in self.exchanges.items():
            try:
                await exchange.close()
                logger.info(f"Exchange connection closed: {exchange_name}")
            except Exception as e:
                logger.error(f"Error closing exchange: {exchange_name}", error=str(e))

        self.tasks.clear()
        logger.info("Exchange manager stopped")

    async def _monitor_exchange_symbol(self, exchange_name: str, symbol: str):
        """Monitor order book for a specific exchange and symbol."""

        exchange = self.exchanges[exchange_name]
        status = self.exchange_status[exchange_name]

        logger.info(f"Starting monitoring: {exchange_name} {symbol}")

        while self.running:
            try:
                with LogExecutionTime(f"fetch_order_book_{exchange_name}_{symbol}", logger, "debug"):
                    # Fetch order book
                    order_book = await exchange.watch_order_book(symbol, settings.order_book_depth)

                    # Convert to our format
                    snapshot = self._convert_order_book(exchange_name, symbol, order_book)

                    # Update status
                    status.connected = True
                    status.last_update = datetime.utcnow()
                    status.error_count = 0
                    status.last_error = None

                    # Log order book update
                    logger.log_order_book_update(
                        symbol, exchange_name, len(snapshot.bids), len(snapshot.asks), float(snapshot.spread)
                    )

                    # Notify callbacks
                    await self._notify_callbacks(snapshot)

                    # Small delay to prevent overwhelming
                    await asyncio.sleep(0.1)

            except ccxt.NetworkError as e:
                logger.warning(f"Network error for {exchange_name} {symbol}: {str(e)}")
                status.connected = False
                status.error_count += 1
                status.last_error = str(e)

                # Exponential backoff
                await asyncio.sleep(min(5 * status.error_count, 60))

            except ccxt.ExchangeError as e:
                logger.error(f"Exchange error for {exchange_name} {symbol}: {str(e)}")
                status.connected = False
                status.error_count += 1
                status.last_error = str(e)

                # Longer delay for exchange errors
                await asyncio.sleep(min(10 * status.error_count, 120))

            except Exception as e:
                logger.exception(f"Unexpected error for {exchange_name} {symbol}")
                status.connected = False
                status.error_count += 1
                status.last_error = str(e)

                # Even longer delay for unexpected errors
                await asyncio.sleep(min(15 * status.error_count, 180))

    def _convert_order_book(self, exchange_name: str, symbol: str, raw_order_book: Dict) -> OrderBookSnapshot:
        """Convert ccxt order book to our format."""

        # Convert bids
        bids = []
        for price, volume in raw_order_book["bids"]:
            bids.append(OrderBookLevel(price=Decimal(str(price)), volume=Decimal(str(volume))))

        # Convert asks
        asks = []
        for price, volume in raw_order_book["asks"]:
            asks.append(OrderBookLevel(price=Decimal(str(price)), volume=Decimal(str(volume))))

        # Create snapshot
        return OrderBookSnapshot(
            exchange=exchange_name,
            symbol=symbol,
            timestamp=datetime.utcnow(),
            bids=bids,
            asks=asks,
            nonce=raw_order_book.get("nonce"),
        )

    async def _notify_callbacks(self, snapshot: OrderBookSnapshot):
        """Notify all registered callbacks of order book updates."""

        for callback in self.order_book_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(snapshot)
                else:
                    callback(snapshot)
            except Exception as e:
                logger.error("Error in order book callback", callback=callback.__name__, error=str(e))

    async def _health_check_loop(self):
        """Monitor exchange health and log status."""

        while self.running:
            try:
                # Check each exchange status
                for exchange_name, status in self.exchange_status.items():
                    if not status.connected:
                        logger.warning(
                            f"Exchange disconnected: {exchange_name}",
                            exchange=exchange_name,
                            error_count=status.error_count,
                            last_error=status.last_error,
                        )

                    # Check if exchange is stale (no updates for 30 seconds)
                    if status.last_update:
                        seconds_since_update = (datetime.utcnow() - status.last_update).total_seconds()
                        if seconds_since_update > 30:
                            logger.warning(
                                f"Exchange stale: {exchange_name}",
                                exchange=exchange_name,
                                seconds_since_update=seconds_since_update,
                            )

                # Log overall health
                connected_count = sum(1 for status in self.exchange_status.values() if status.connected)
                total_count = len(self.exchange_status)

                logger.debug(
                    f"Exchange health check: {connected_count}/{total_count} connected",
                    connected=connected_count,
                    total=total_count,
                )

                # Wait before next check
                await asyncio.sleep(30)

            except Exception:
                logger.exception("Error in health check loop")
                await asyncio.sleep(10)

    def register_order_book_callback(self, callback: Callable):
        """Register a callback for order book updates."""
        self.order_book_callbacks.append(callback)
        logger.info(f"Registered order book callback: {callback.__name__}")

    def unregister_order_book_callback(self, callback: Callable):
        """Unregister an order book callback."""
        if callback in self.order_book_callbacks:
            self.order_book_callbacks.remove(callback)
            logger.info(f"Unregistered order book callback: {callback.__name__}")

    async def get_order_book(self, exchange_name: str, symbol: str) -> Optional[OrderBookSnapshot]:
        """Get a single order book snapshot."""

        if exchange_name not in self.exchanges:
            logger.error(f"Exchange not found: {exchange_name}")
            return None

        exchange = self.exchanges[exchange_name]

        try:
            with LogExecutionTime(f"fetch_single_order_book_{exchange_name}_{symbol}", logger, "debug"):
                order_book = await exchange.fetch_order_book(symbol, settings.order_book_depth)
                return self._convert_order_book(exchange_name, symbol, order_book)

        except Exception as e:
            logger.error(
                f"Failed to fetch order book: {exchange_name} {symbol}",
                exchange=exchange_name,
                symbol=symbol,
                error=str(e),
            )
            return None

    async def get_aggregated_order_book(self, symbol: str) -> Optional[AggregatedOrderBook]:
        """Get aggregated order book from all exchanges."""

        snapshots = {}

        # Fetch from all exchanges
        for exchange_name in self.exchanges:
            snapshot = await self.get_order_book(exchange_name, symbol)
            if snapshot:
                snapshots[exchange_name] = snapshot

        if not snapshots:
            logger.warning(f"No order book data available for {symbol}")
            return None

        # Create aggregated order book
        return self._aggregate_order_books(symbol, snapshots)

    def _aggregate_order_books(self, symbol: str, snapshots: Dict[str, OrderBookSnapshot]) -> AggregatedOrderBook:
        """Aggregate order books from multiple exchanges."""

        all_bids = []
        all_asks = []

        # Collect all bids and asks
        for snapshot in snapshots.values():
            all_bids.extend(snapshot.bids)
            all_asks.extend(snapshot.asks)

        # Sort bids (highest price first)
        all_bids.sort(key=lambda x: x.price, reverse=True)

        # Sort asks (lowest price first)
        all_asks.sort(key=lambda x: x.price)

        # Aggregate volumes at same price levels
        aggregated_bids = self._aggregate_levels(all_bids)
        aggregated_asks = self._aggregate_levels(all_asks)

        return AggregatedOrderBook(
            symbol=symbol,
            timestamp=datetime.utcnow(),
            snapshots=snapshots,
            aggregated_bids=aggregated_bids,
            aggregated_asks=aggregated_asks,
        )

    def _aggregate_levels(self, levels: List[OrderBookLevel]) -> List[OrderBookLevel]:
        """Aggregate order book levels at same price."""

        if not levels:
            return []

        aggregated = []
        current_price = None
        current_volume = Decimal("0")

        for level in levels:
            if current_price is None or level.price != current_price:
                # New price level
                if current_price is not None:
                    aggregated.append(OrderBookLevel(price=current_price, volume=current_volume))

                current_price = level.price
                current_volume = level.volume
            else:
                # Same price level, add volume
                current_volume += level.volume

        # Add last level
        if current_price is not None:
            aggregated.append(OrderBookLevel(price=current_price, volume=current_volume))

        return aggregated

    def get_exchange_status(self, exchange_name: str = None) -> Dict[str, ExchangeStatus]:
        """Get exchange status information."""

        if exchange_name:
            return {exchange_name: self.exchange_status.get(exchange_name)}

        return self.exchange_status.copy()

    def get_connected_exchanges(self) -> List[str]:
        """Get list of currently connected exchanges."""

        return [name for name, status in self.exchange_status.items() if status.connected]

    def is_exchange_connected(self, exchange_name: str) -> bool:
        """Check if specific exchange is connected."""

        status = self.exchange_status.get(exchange_name)
        return status.connected if status else False

    async def reconnect_exchange(self, exchange_name: str):
        """Attempt to reconnect a specific exchange."""

        if exchange_name not in self.exchanges:
            logger.error(f"Exchange not found: {exchange_name}")
            return False

        logger.info(f"Attempting to reconnect: {exchange_name}")

        try:
            exchange = self.exchanges[exchange_name]

            # Close existing connection
            await exchange.close()

            # Reinitialize
            config = settings.get_exchange_config(exchange_name)
            exchange_class = getattr(ccxt, exchange_name)
            self.exchanges[exchange_name] = exchange_class(config)

            # Reset status
            self.exchange_status[exchange_name] = ExchangeStatus(name=exchange_name, connected=False)

            logger.info(f"Exchange reconnection initiated: {exchange_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to reconnect exchange: {exchange_name}", exchange=exchange_name, error=str(e))
            return False

    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()


# Global exchange manager instance
exchange_manager = ExchangeManager()


# Convenience functions
async def get_order_book(exchange: str, symbol: str) -> Optional[OrderBookSnapshot]:
    """Get order book snapshot from specific exchange."""
    return await exchange_manager.get_order_book(exchange, symbol)


async def get_aggregated_order_book(symbol: str) -> Optional[AggregatedOrderBook]:
    """Get aggregated order book from all exchanges."""
    return await exchange_manager.get_aggregated_order_book(symbol)


def register_order_book_callback(callback: Callable):
    """Register callback for order book updates."""
    exchange_manager.register_order_book_callback(callback)


def get_exchange_status() -> Dict[str, ExchangeStatus]:
    """Get status of all exchanges."""
    return exchange_manager.get_exchange_status()
