"""
logger.py — Event Logger
========================
Logs every NIDS event (packet inspection result) to a persistent log file
with structured, timestamped entries. Also supports in-memory log retrieval.
"""

import os
import logging
from datetime import datetime
from typing import Optional

# ------------------------------------------
# Configuration
# ------------------------------------------
LOG_DIR      = "logs"
LOG_FILE     = os.path.join(LOG_DIR, "events.log")
ALERT_FILE   = os.path.join(LOG_DIR, "alerts.log")   # Separate file for attacks only
LOG_FORMAT   = "%(asctime)s | %(levelname)-8s | %(message)s"
DATE_FORMAT  = "%Y-%m-%d %H:%M:%S"


# ------------------------------------------
# Logger Setup
# ------------------------------------------

def _setup_logger(name: str, filepath: str, level=logging.DEBUG) -> logging.Logger:
    """
    Internal helper — create and configure a file logger.

    Args:
        name     : Logger name (must be unique per file).
        filepath : Destination log file path.
        level    : Minimum log level.

    Returns:
        logging.Logger: Configured logger instance.
    """
    os.makedirs(LOG_DIR, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent duplicate handlers if called multiple times
    if not logger.handlers:
        fh = logging.FileHandler(filepath, encoding="utf-8")
        fh.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
        logger.addHandler(fh)

    return logger


# Singleton loggers
_event_logger = _setup_logger("nids.events", LOG_FILE)
_alert_logger = _setup_logger("nids.alerts", ALERT_FILE, level=logging.WARNING)


# ------------------------------------------
# Public API
# ------------------------------------------

def log_event(
    src_ip     : str,
    dst_ip     : str,
    protocol   : str,
    packet_size: int,
    prediction : str,
    confidence : Optional[float] = None,
) -> None:
    """
    Log a single packet inspection event.

    Args:
        src_ip      : Source IP address.
        dst_ip      : Destination IP address.
        protocol    : Network protocol string (e.g. 'TCP').
        packet_size : Packet size in bytes.
        prediction  : 'NORMAL' or 'ATTACK'.
        confidence  : Model confidence score (0–1), optional.
    """
    conf_str = f"  confidence={confidence:.4f}" if confidence is not None else ""
    message  = (
        f"src={src_ip:<15} dst={dst_ip:<15} "
        f"proto={protocol:<5} size={packet_size:>5}B "
        f"result={prediction}{conf_str}"
    )

    if prediction == "ATTACK":
        _event_logger.warning(message)
        _alert_logger.warning(message)   # Mirror to alerts-only log
    else:
        _event_logger.info(message)


def log_system(message: str, level: str = "info") -> None:
    """
    Log a system-level message (startup, shutdown, errors).

    Args:
        message : Text to log.
        level   : One of 'debug', 'info', 'warning', 'error', 'critical'.
    """
    log_fn = getattr(_event_logger, level.lower(), _event_logger.info)
    log_fn(f"[SYSTEM] {message}")


def log_error(message: str, exc: Optional[Exception] = None) -> None:
    """
    Log an error with optional exception details.

    Args:
        message : Error description.
        exc     : Exception instance (optional).
    """
    if exc:
        _event_logger.exception(f"[ERROR] {message}")
    else:
        _event_logger.error(f"[ERROR] {message}")


def get_log_path() -> str:
    """Return the absolute path of the main event log file."""
    return os.path.abspath(LOG_FILE)


def get_alert_log_path() -> str:
    """Return the absolute path of the alerts-only log file."""
    return os.path.abspath(ALERT_FILE)


def tail_log(n: int = 20) -> list[str]:
    """
    Return the last `n` lines from the event log file.

    Args:
        n : Number of lines to return.

    Returns:
        list[str]: Last n log lines (newest last).
    """
    if not os.path.exists(LOG_FILE):
        return []
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()
    return [l.rstrip() for l in lines[-n:]]


# -- Quick standalone test ---------------------------------------------------
if __name__ == "__main__":
    log_system("NIDS Logger initialised")
    log_event("192.168.1.5", "10.0.0.1", "TCP", 512, "NORMAL", 0.97)
    log_event("203.0.113.9", "192.168.1.1", "ICMP", 1400, "ATTACK", 0.89)
    log_error("Simulated error")
    print("Recent log entries:")
    for line in tail_log(5):
        print(" ", line)
    print(f"\nLog file -> {get_log_path()}")
    print(f"Alert file -> {get_alert_log_path()}")
