from typing import Any, List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # ==========================================
    # APPLICATION SETTINGS
    # ==========================================
    loglevel: str = Field(default="INFO", description="Logging level")
    debug: bool = Field(default=False, description="Debug mode")
    environment: str = Field(default="development", description="Environment")

    # ==========================================
    # TELEGRAM CONFIGURATION
    # ==========================================
    telegram_bot_token: str = Field(..., description="Telegram bot token from @BotFather")
    telegram_chat_id: str = Field(..., description="Your Telegram chat ID")

    # ==========================================
    # TRADING CONFIGURATION
    # ==========================================
    # These are defined as strings in env and parsed to lists
    trading_pairs: str = Field(default="BTC/USDT,ETH/USDT", description="Trading pairs to monitor")
    exchanges: str = Field(default="binance,coinbase,kraken,okx,bybit", description="Exchanges to monitor")

    # ==========================================
    # ALERT THRESHOLDS
    # ==========================================
    volume_spike_threshold: float = Field(default=1.5, ge=1.0, description="Volume spike threshold (150% = 1.5)")
    whale_order_threshold: float = Field(default=100000.0, ge=1000.0, description="Whale order threshold in USD")
    bid_ask_wall_threshold: float = Field(default=50000.0, ge=1000.0, description="Bid/ask wall threshold in USD")
    imbalance_threshold: float = Field(
        default=0.7, ge=0.5, le=1.0, description="Order book imbalance threshold (70% = 0.7)"
    )

    # ==========================================
    # MONITORING SETTINGS
    # ==========================================
    order_book_depth: int = Field(default=50, ge=10, le=1000, description="Number of order book levels to analyze")
    update_interval: int = Field(default=5, ge=1, le=60, description="Seconds between updates")
    alert_cooldown: int = Field(default=300, ge=60, description="Cooldown between similar alerts (seconds)")

    # ==========================================
    # CCXT CONFIGURATION
    # ==========================================
    ccxt_sandbox: bool = Field(default=False, description="Use sandbox/testnet")
    ccxt_rate_limit: bool = Field(default=True, description="Enable rate limiting")
    ccxt_timeout: int = Field(default=30000, description="Request timeout in ms")

    # ==========================================
    # OPTIONAL EXCHANGE API KEYS (for better rate limits)
    # ==========================================
    binance_api_key: Optional[str] = Field(default=None, description="Binance API key")
    binance_secret: Optional[str] = Field(default=None, description="Binance secret")

    coinbase_api_key: Optional[str] = Field(default=None, description="Coinbase API key")
    coinbase_secret: Optional[str] = Field(default=None, description="Coinbase secret")
    coinbase_passphrase: Optional[str] = Field(default=None, description="Coinbase passphrase")

    kraken_api_key: Optional[str] = Field(default=None, description="Kraken API key")
    kraken_secret: Optional[str] = Field(default=None, description="Kraken secret")

    okx_api_key: Optional[str] = Field(default=None, description="OKX API key")
    okx_secret: Optional[str] = Field(default=None, description="OKX secret")
    okx_passphrase: Optional[str] = Field(default=None, description="OKX passphrase")

    bybit_api_key: Optional[str] = Field(default=None, description="Bybit API key")
    bybit_secret: Optional[str] = Field(default=None, description="Bybit secret")

    # ==========================================
    # INFRASTRUCTURE
    # ==========================================
    sentry_dsn: Optional[str] = Field(default=None, description="Sentry DSN for error tracking")
    metrics_enabled: bool = Field(default=False, description="Enable metrics collection")

    # ==========================================
    # AWS DEPLOYMENT
    # ==========================================
    aws_region: str = Field(default="us-east-1", description="AWS region")
    aws_access_key_id: Optional[str] = Field(default=None, description="AWS access key")
    aws_secret_access_key: Optional[str] = Field(default=None, description="AWS secret key")

    # ==========================================
    # INTERNAL PARSED FIELDS
    # ==========================================
    _trading_pairs_list: List[str] = []
    _exchanges_list: List[str] = []

    # ==========================================
    # VALIDATORS
    # ==========================================
    @field_validator("loglevel")
    @classmethod
    def validate_loglevel(cls, v):
        """Validate log level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"loglevel must be one of {valid_levels}")
        return v.upper()

    def model_post_init(self, __context: Any) -> None:
        """Parse comma-separated fields after initialization."""
        # Parse trading pairs
        self._trading_pairs_list = [p.strip() for p in self.trading_pairs.split(",") if p.strip()]
        # Validate trading pairs
        for pair in self._trading_pairs_list:
            if "/" not in pair:
                raise ValueError(f"Invalid trading pair format: {pair}. Use format: BTC/USDT")
        self._trading_pairs_list = [p.upper() for p in self._trading_pairs_list]

        # Parse exchanges
        self._exchanges_list = [e.strip().lower() for e in self.exchanges.split(",") if e.strip()]
        # Validate exchanges
        valid_exchanges = ["binance", "coinbase", "kraken", "okx", "bybit"]
        for exchange in self._exchanges_list:
            if exchange not in valid_exchanges:
                raise ValueError(f"Exchange {exchange} not supported. Valid: {valid_exchanges}")

    # ==========================================
    # CONFIGURATION
    # ==========================================
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ==========================================
    # PROPERTY ACCESSORS
    # ==========================================
    @property
    def trading_pairs_list(self) -> List[str]:
        """Get trading pairs as a list."""
        return self._trading_pairs_list

    @property
    def exchanges_list(self) -> List[str]:
        """Get exchanges as a list."""
        return self._exchanges_list

    # ==========================================
    # HELPER METHODS
    # ==========================================
    def get_exchange_config(self, exchange_name: str) -> dict:
        """Get configuration for a specific exchange."""
        exchange_name = exchange_name.lower()

        config = {
            "sandbox": self.ccxt_sandbox,
            "rateLimit": self.ccxt_rate_limit,
            "timeout": self.ccxt_timeout,
            "enableRateLimit": True,
        }

        # Add API keys if available
        if exchange_name == "binance" and self.binance_api_key:
            config.update(
                {
                    "apiKey": self.binance_api_key,
                    "secret": self.binance_secret,
                }
            )
        elif exchange_name == "coinbase" and self.coinbase_api_key:
            config.update(
                {
                    "apiKey": self.coinbase_api_key,
                    "secret": self.coinbase_secret,
                    "passphrase": self.coinbase_passphrase,
                }
            )
        elif exchange_name == "kraken" and self.kraken_api_key:
            config.update(
                {
                    "apiKey": self.kraken_api_key,
                    "secret": self.kraken_secret,
                }
            )
        elif exchange_name == "okx" and self.okx_api_key:
            config.update(
                {
                    "apiKey": self.okx_api_key,
                    "secret": self.okx_secret,
                    "passphrase": self.okx_passphrase,
                }
            )
        elif exchange_name == "bybit" and self.bybit_api_key:
            config.update(
                {
                    "apiKey": self.bybit_api_key,
                    "secret": self.bybit_secret,
                }
            )

        return config

    def is_production(self) -> bool:
        """Check if running in production."""
        return self.environment.lower() == "production"

    def get_log_config(self) -> dict:
        """Get logging configuration."""
        return {
            "level": self.loglevel,
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "filename": "logs/orderflow_pro.log" if self.is_production() else None,
        }


# Create global settings instance
settings = Settings()
