"""
main.py -- AI-Based Network Intrusion Detection System
=======================================================
Entry point. Wires together every module in the pipeline:

    capture -> preprocess -> detector -> alert -> logger

Runs a continuous monitoring loop with a live terminal dashboard.
Supports both simulated packets (default) and real Scapy capture.

Usage:
    python main.py                     # Simulator mode (no root required)
    python main.py --live              # Live capture (requires admin/root)
    python main.py --threshold 0.6     # Custom attack confidence threshold
    python main.py --attack-rate 0.4   # Simulator attack probability
    python main.py --train             # Force retrain the model then exit
"""

import sys
import io

# ── Force UTF-8 output on Windows (prevents cp1252 UnicodeEncodeError) ───────
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import time
import signal
import argparse
import threading
from datetime import datetime

# ------------------------------------------
# Local modules
# ------------------------------------------
from alert     import (
    print_banner, print_info, print_warning, print_error,
    raise_attack_alert, raise_normal_status,
    print_status_bar, get_stats,
)
from logger    import log_event, log_system, log_error, get_log_path
from detector  import NIDSDetector
from capture   import get_packet_stream
from model     import model_exists, ensure_trained


# ------------------------------------------
# CLI Arguments
# ------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        prog        = "nids",
        description = "AI-Based Network Intrusion Detection System",
    )
    p.add_argument(
        "--live",
        action  = "store_true",
        help    = "Use live packet capture (requires admin/root + Scapy).",
    )
    p.add_argument(
        "--threshold",
        type    = float,
        default = 0.50,
        metavar = "PROB",
        help    = "Attack confidence threshold (0.0–1.0). Default: 0.50",
    )
    p.add_argument(
        "--attack-rate",
        type    = float,
        default = 0.35,
        dest    = "attack_rate",
        metavar = "RATE",
        help    = "Simulated attack probability (0.0–1.0). Default: 0.35",
    )
    p.add_argument(
        "--interface",
        type    = str,
        default = None,
        metavar = "IFACE",
        help    = "Network interface for live capture (e.g. eth0, Wi-Fi).",
    )
    p.add_argument(
        "--train",
        action  = "store_true",
        help    = "Force retrain the model and exit.",
    )
    p.add_argument(
        "--delay",
        type    = float,
        default = 0.4,
        metavar = "SEC",
        help    = "Delay (seconds) between simulated packets. Default: 0.4",
    )
    p.add_argument(
        "--dashboard-interval",
        type    = int,
        default = 20,
        dest    = "dashboard_interval",
        metavar = "N",
        help    = "Print the stats dashboard every N packets. Default: 20",
    )
    return p.parse_args()


# ------------------------------------------
# Graceful Shutdown
# ------------------------------------------

_stop_event = threading.Event()


def _handle_sigint(signum, frame) -> None:
    """Handle Ctrl+C — trigger graceful shutdown."""
    print("\n")
    print_warning("Shutdown signal received (Ctrl+C). Stopping NIDS…")
    _stop_event.set()


signal.signal(signal.SIGINT, _handle_sigint)


# ------------------------------------------
# Training Sub-Command
# ------------------------------------------

def run_training() -> None:
    """Force-train (or retrain) the ML model."""
    from dataset    import generate_dataset
    from preprocess import preprocess_dataset
    from model      import train_model

    print_info("Generating dataset…")
    df = generate_dataset(save=True)

    print_info("Preprocessing…")
    X, y = preprocess_dataset(df, fit_scaler=True)

    print_info("Training RandomForestClassifier…")
    clf, metrics = train_model(X, y, save=True)

    print_info(
        f"Training complete! "
        f"Accuracy={metrics['accuracy'] * 100:.2f}%  "
        f"ROC-AUC={metrics['roc_auc']:.4f}"
    )


# ------------------------------------------
# Detection Loop
# ------------------------------------------

def run_detection_loop(args: argparse.Namespace) -> None:
    """
    Main detection loop — captures packets, classifies them, alerts & logs.

    Args:
        args : Parsed CLI arguments.
    """
    # -- 1. Ensure model is trained -----------------------------------------
    print_info("Initialising ML model…")
    detector = NIDSDetector(threshold=args.threshold)

    # -- 2. Choose packet source --------------------------------------------
    packet_stream = get_packet_stream(
        interface          = args.interface,
        simulate           = not args.live,
        attack_probability = args.attack_rate,
        stop_event         = _stop_event,
    )

    # -- 3. Log system start ------------------------------------------------
    mode = "LIVE" if args.live else "SIMULATOR"
    log_system(
        f"NIDS started | mode={mode} | threshold={args.threshold} "
        f"| log={get_log_path()}"
    )

    print_info(f"Monitoring started. Press Ctrl+C to stop.\n")

    # -- 4. Detection loop -------------------------------------------------
    packet_count = 0

    for packet_info in packet_stream:
        if _stop_event.is_set():
            break

        try:
            # - Classify -------------------------------------------------
            result = detector.predict(packet_info)

            src_ip      = result["src_ip"]
            dst_ip      = result["dst_ip"]
            protocol    = result["protocol"]
            packet_size = result["packet_size"]
            label       = result["label"]
            confidence  = result["confidence"]
            is_attack   = result["is_attack"]

            # - Alert -----------------------------------------------------
            if is_attack:
                raise_attack_alert(
                    src_ip      = src_ip,
                    dst_ip      = dst_ip,
                    protocol    = protocol,
                    packet_size = packet_size,
                    confidence  = confidence,
                    beep        = True,
                )
            else:
                raise_normal_status(
                    src_ip      = src_ip,
                    dst_ip      = dst_ip,
                    protocol    = protocol,
                    packet_size = packet_size,
                    confidence  = confidence,
                )

            # - Log -------------------------------------------------------
            log_event(
                src_ip      = src_ip,
                dst_ip      = dst_ip,
                protocol    = protocol,
                packet_size = packet_size,
                prediction  = label,
                confidence  = confidence,
            )

            # - Periodic dashboard -----------------------------------------
            packet_count += 1
            if packet_count % args.dashboard_interval == 0:
                print_status_bar()

        except KeyboardInterrupt:
            break
        except Exception as exc:
            print_error(f"Packet processing error: {exc}")
            log_error(str(exc), exc=exc)
            time.sleep(0.1)   # Brief back-off before retrying

    # -- 5. Final summary --------------------------------------------------
    stats = get_stats()
    log_system(
        f"NIDS stopped | packets={stats['total']} "
        f"normal={stats['normal']} attacks={stats['attack']}"
    )

    print("\n")
    print_status_bar()
    print_info(f"Event log saved -> {get_log_path()}")
    print_info("NIDS shutdown complete.")


# ------------------------------------------
# Entry Point
# ------------------------------------------

def main() -> None:
    """Main entry point — parse args, print banner, route to correct mode."""
    args = parse_args()

    print_banner()

    if args.train:
        run_training()
        return

    run_detection_loop(args)


if __name__ == "__main__":
    main()
