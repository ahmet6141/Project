"""FighterCAD - Parametric supersonic dogfight fighter aircraft design tool."""

import logging

__version__ = "0.1.0"


def setup_logging(level: int = logging.INFO) -> None:
    """Configure console logging for FighterCAD."""
    logger = logging.getLogger("fightercad")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        ))
        logger.addHandler(handler)
    logger.setLevel(level)


# Auto-configure logging on import
setup_logging()
