"""
RETINA ML Worker - Entry Point
==============================

Main entry point for the ML inference worker.

Usage
-----

From the command line (after installation):
    $ retina-worker

Or directly:
    $ python -m retina_worker.main

Or via Docker:
    $ docker compose up worker

Configuration
-------------

The worker is configured via environment variables:

- REDIS_URL: Redis connection string (default: redis://localhost:6379)
- WORKER_CONCURRENCY: Not yet used (default: 1)
- DEFAULT_UNSUPERVISED_MODEL: Default Stage 1 model (default: patchcore)
- DEBUG_MODE: Enable debug logging (default: true)
- MOCK_INFERENCE_DELAY_MS: Simulated delay for testing (default: 500)

See config.py for all configuration options.
"""

import logging

import structlog

from .config import get_settings
from .worker import Worker


def configure_logging(debug: bool = False) -> None:
    """
    Configure structured logging.

    Parameters
    ----------
    debug : bool
        Enable debug-level logging
    """
    logging.basicConfig(
        format="%(message)s",
        level=logging.DEBUG if debug else logging.INFO,
    )
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.dev.ConsoleRenderer() if debug else structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def main() -> None:
    """
    Main entry point for the worker.
    
    Initializes logging, loads configuration, and starts the worker loop.
    """
    # Load settings
    settings = get_settings()
    
    # Configure logging
    configure_logging(debug=settings.debug_mode)
    
    logger = structlog.get_logger()
    logger.info(
        "RETINA ML Worker starting",
        version="0.1.0",
        redis_url=settings.redis_url,
        consumer_name=settings.consumer_name,
        debug_mode=settings.debug_mode,
    )
    
    # Create and run worker
    worker = Worker(settings)
    
    try:
        worker.run()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    except Exception as e:
        logger.exception("Worker crashed", error=str(e))
        raise
    finally:
        logger.info("Worker exiting")


if __name__ == "__main__":
    main()
