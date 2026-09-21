"""
alert.py — Alert & Notification Engine
========================================
Handles visual and optional audio alerts when an attack is detected.
Provides colour-coded terminal output for real-time monitoring.
"""

import sys
import time
import threading
from datetime import datetime
from typing import Optional

# ------------------------------------------
# ANSI Colour Codes (cross-platform via colorama)
# ------------------------------------------
try:
    from colorama import init as _colorama_init, Fore, Back, Style
    _colorama_init(autoreset=True)          # Enables ANSI codes on Windows
    _HAS_COLORAMA = True
except ImportError:
    _HAS_COLORAMA = False

    # Minimal fallback shims so the rest of the code works without colorama
    class _FallbackFore:
        RED = GREEN = YELLOW = CYAN = WHITE = MAGENTA = BLUE = RESET = ""
    class _FallbackBack:
        RED = GREEN = YELLOW = RESET = ""
    class _FallbackStyle:
        BRIGHT = RESET_ALL = DIM = ""

    Fore  = _FallbackFore()
    Back  = _FallbackBack()
    Style = _FallbackStyle()


# ------------------------------------------
# Alert Statistics (in-memory counters)
# ------------------------------------------
_stats = {"total": 0, "normal": 0, "attack": 0}


def _beep() -> None:
    """
    Play a system beep sound (non-blocking).
    Falls back silently if the platform doesn't support it.
    """
    try:
        if sys.platform == "win32":
            import winsound
            winsound.Beep(1000, 300)   # Frequency=1000 Hz, Duration=300 ms
        else:
            sys.stdout.write("\a")     # ASCII bell character
            sys.stdout.flush()
    except Exception:
        pass   # Never crash on beep failure


# ------------------------------------------
# Core Alert Functions
# ------------------------------------------

def raise_attack_alert(
    src_ip     : str,
    dst_ip     : str,
    protocol   : str,
    packet_size: int,
    confidence : float = 1.0,
    beep       : bool  = True,
) -> None:
    """
    Display a RED terminal alert and optionally beep when an attack is detected.

    Args:
        src_ip      : Attacker's source IP address.
        dst_ip      : Target destination IP address.
        protocol    : Network protocol (e.g. 'TCP').
        packet_size : Packet size in bytes.
        confidence  : Model prediction confidence (0.0–1.0).
        beep        : Whether to play an audio alert.
    """
    _stats["attack"] += 1
    _stats["total"]  += 1

    timestamp = datetime.now().strftime("%H:%M:%S")

    border = f"{Fore.RED}{Style.BRIGHT}{'=' * 62}{Style.RESET_ALL}"
    print(border)
    print(
        f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}"
        f"  [!]  ATTACK DETECTED  [{timestamp}]"
        f"{Style.RESET_ALL}"
    )
    print(
        f"  {Fore.RED}>> Src IP    : {Style.BRIGHT}{src_ip}{Style.RESET_ALL}"
    )
    print(
        f"  {Fore.RED}>> Dst IP    : {Style.BRIGHT}{dst_ip}{Style.RESET_ALL}"
    )
    print(
        f"  {Fore.RED}>> Protocol  : {Style.BRIGHT}{protocol}{Style.RESET_ALL}"
    )
    print(
        f"  {Fore.RED}>> Pkt Size  : {Style.BRIGHT}{packet_size} bytes{Style.RESET_ALL}"
    )
    print(
        f"  {Fore.RED}>> Confidence: {Style.BRIGHT}{confidence * 100:.1f}%{Style.RESET_ALL}"
    )
    print(border)

    # Non-blocking beep so it doesn't stall the detection loop
    if beep:
        threading.Thread(target=_beep, daemon=True).start()


def raise_normal_status(
    src_ip     : str,
    dst_ip     : str,
    protocol   : str,
    packet_size: int,
    confidence : float = 1.0,
) -> None:
    """
    Display a GREEN status line for normal (benign) traffic.

    Args:
        src_ip      : Source IP.
        dst_ip      : Destination IP.
        protocol    : Network protocol.
        packet_size : Packet size in bytes.
        confidence  : Model prediction confidence (0.0–1.0).
    """
    _stats["normal"] += 1
    _stats["total"]  += 1

    timestamp = datetime.now().strftime("%H:%M:%S")

    print(
        f"{Fore.GREEN}[{timestamp}] ✔ NORMAL  "
        f"| {src_ip:<15} -> {dst_ip:<15} "
        f"| {protocol:<4} | {packet_size:>5}B "
        f"| conf={confidence * 100:.1f}%{Style.RESET_ALL}"
    )


def print_status_bar() -> None:
    """
    Print a live statistics summary bar to the terminal.
    Call this periodically (e.g. every N packets) for a dashboard feel.
    """
    total   = _stats["total"]
    normal  = _stats["normal"]
    attacks = _stats["attack"]
    pct     = (attacks / total * 100) if total else 0.0

    print(
        f"\n{Fore.CYAN}{Style.BRIGHT}"
        f"  +- NIDS Dashboard ---------------------------------+\n"
        f"  |  Packets Inspected : {total:<6}                       |\n"
        f"  |  {Fore.GREEN}Normal Traffic   : {normal:<6}{Fore.CYAN}                       |\n"
        f"  |  {Fore.RED}Attack Detected  : {attacks:<6}{Fore.CYAN}  ({pct:5.1f}% threat rate)  |\n"
        f"  +----------------------------------------------------+"
        f"{Style.RESET_ALL}\n"
    )


def print_banner() -> None:
    """Print the NIDS startup banner."""
    banner = f"""
{Fore.CYAN}{Style.BRIGHT}
  +==========================================================+
  |       AI-Based Network Intrusion Detection System        |
  |                     [NIDS v1.0]                          |
  |       Real-time ML-Powered Threat Detection Engine       |
  +==========================================================+
{Style.RESET_ALL}"""
    print(banner)


def print_info(message: str) -> None:
    """Print a cyan informational message."""
    print(f"{Fore.CYAN}[INFO] {message}{Style.RESET_ALL}")


def print_warning(message: str) -> None:
    """Print a yellow warning message."""
    print(f"{Fore.YELLOW}[WARN] {message}{Style.RESET_ALL}")


def print_error(message: str) -> None:
    """Print a red error message."""
    print(f"{Fore.RED}[ERR ] {message}{Style.RESET_ALL}")


def get_stats() -> dict:
    """Return current alert statistics dictionary."""
    return dict(_stats)


def reset_stats() -> None:
    """Reset in-memory packet counters (e.g. on session restart)."""
    _stats.update({"total": 0, "normal": 0, "attack": 0})


# -- Quick standalone test ---------------------------------------------------
if __name__ == "__main__":
    print_banner()
    print_info("Starting alert engine test…")
    raise_normal_status("192.168.1.5", "8.8.8.8",    "TCP",  512, 0.97)
    raise_normal_status("10.0.0.2",   "1.1.1.1",    "UDP",  256, 0.91)
    raise_attack_alert ("203.0.113.9","192.168.1.1", "ICMP", 1450, 0.88, beep=False)
    raise_attack_alert ("45.33.32.156","10.0.0.5",  "TCP",  64,  0.95, beep=False)
    print_status_bar()
