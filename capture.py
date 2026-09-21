"""
capture.py — Network Packet Capture Engine
============================================
Uses Scapy to sniff live network packets and extracts structured feature
dictionaries that feed directly into the preprocessing pipeline.

NOTE:
  • On Linux/macOS — requires root (sudo python main.py).
  • On Windows — requires running as Administrator and WinPcap/Npcap installed.
  • If live capture is unavailable, a simulator mode generates synthetic
    packet data so the rest of the pipeline can still be tested.
"""

import time
import random
import threading
from typing import Optional, Callable, Generator

# ------------------------------------------
# Scapy import guard (graceful fallback)
# ------------------------------------------
try:
    from scapy.all import sniff, IP, TCP, UDP, ICMP, Raw
    _SCAPY_AVAILABLE = True
except ImportError:
    _SCAPY_AVAILABLE = False

# ------------------------------------------
# Constants
# ------------------------------------------
CAPTURE_TIMEOUT  = 2          # Seconds per sniff batch
CAPTURE_COUNT    = 1          # Packets per sniff call  (1 = process immediately)
DEFAULT_IFACE    = None       # None -> Scapy picks the default interface


# ------------------------------------------
# Feature Extraction from Scapy Packet
# ------------------------------------------

def extract_features(packet) -> Optional[dict]:
    """
    Extract a structured feature dictionary from a raw Scapy packet object.

    Extracted features:
    - src_ip       : Source IP address string.
    - dst_ip       : Destination IP address string.
    - protocol     : 'TCP', 'UDP', 'ICMP', or 'OTHER'.
    - packet_size  : Total packet length in bytes.
    - src_port     : Source port (TCP/UDP only; 0 otherwise).
    - dst_port     : Destination port (TCP/UDP only; 0 otherwise).
    - duration     : Simulated session duration (seconds).
    - packet_count : Simulated packet count for this session.
    - timestamp    : Unix epoch of capture.

    Args:
        packet : Scapy packet object.

    Returns:
        dict | None: Feature dictionary, or None if the packet has no IP layer.
    """
    if not packet.haslayer(IP):
        return None   # Ignore non-IP frames (ARP, etc.)

    ip_layer = packet[IP]

    # Determine protocol and port numbers
    if packet.haslayer(TCP):
        protocol  = "TCP"
        src_port  = packet[TCP].sport
        dst_port  = packet[TCP].dport
    elif packet.haslayer(UDP):
        protocol  = "UDP"
        src_port  = packet[UDP].sport
        dst_port  = packet[UDP].dport
    elif packet.haslayer(ICMP):
        protocol  = "ICMP"
        src_port  = 0
        dst_port  = 0
    else:
        protocol  = "OTHER"
        src_port  = 0
        dst_port  = 0

    packet_size = len(packet)

    return {
        "src_ip"       : ip_layer.src,
        "dst_ip"       : ip_layer.dst,
        "protocol"     : protocol,
        "packet_size"  : packet_size,
        "src_port"     : src_port,
        "dst_port"     : dst_port,
        "duration"     : round(random.uniform(0.01, 5.0), 4),   # Approximate
        "packet_count" : random.randint(1, 50),
        "timestamp"    : time.time(),
    }


# ------------------------------------------
# Live Capture (Scapy)
# ------------------------------------------

def capture_one(interface: Optional[str] = DEFAULT_IFACE) -> Optional[dict]:
    """
    Capture a single network packet and return its feature dict.

    Args:
        interface : Network interface name (e.g. 'eth0', 'Wi-Fi').
                    Pass None to let Scapy choose the default.

    Returns:
        dict | None: Feature dictionary or None if no IP packet captured.
    """
    if not _SCAPY_AVAILABLE:
        raise RuntimeError(
            "Scapy is not installed. Run: pip install scapy"
        )

    packets = sniff(iface=interface, count=CAPTURE_COUNT, timeout=CAPTURE_TIMEOUT)
    for pkt in packets:
        features = extract_features(pkt)
        if features:
            return features
    return None


def capture_stream(
    interface : Optional[str]      = DEFAULT_IFACE,
    callback  : Optional[Callable] = None,
    stop_event: Optional[threading.Event] = None,
) -> Generator[dict, None, None]:
    """
    Continuously capture packets and yield feature dicts (generator).

    This is the primary function used by main.py for continuous monitoring.

    Args:
        interface  : Network interface name.
        callback   : Optional callable invoked for each captured packet dict.
        stop_event : threading.Event — set it to gracefully stop the loop.

    Yields:
        dict: Feature dictionary for each captured IP packet.
    """
    if not _SCAPY_AVAILABLE:
        raise RuntimeError("Scapy is not installed. Run: pip install scapy")

    def _packet_handler(pkt):
        features = extract_features(pkt)
        if features:
            if callback:
                callback(features)

    # We expose a generator API — sniff in a thread and yield via a queue
    import queue
    q: queue.Queue[dict] = queue.Queue()

    def _on_packet(pkt):
        f = extract_features(pkt)
        if f:
            q.put(f)

    def _sniffer():
        sniff(
            iface     = interface,
            prn       = _on_packet,
            store     = False,
            stop_filter = lambda _: (stop_event is not None and stop_event.is_set()),
        )

    t = threading.Thread(target=_sniffer, daemon=True)
    t.start()

    while not (stop_event and stop_event.is_set()):
        try:
            yield q.get(timeout=1.0)
        except queue.Empty:
            continue


# ------------------------------------------
# Simulator Mode (no Scapy / no privileges)
# ------------------------------------------

# Pre-defined attack and normal IP pools for realism
_NORMAL_IPS = ["192.168.1.{}".format(i) for i in range(1, 20)]
_ATTACK_IPS = ["203.0.113.{}".format(i) for i in range(1, 10)] + \
              ["45.33.32.156", "198.51.100.5", "172.16.254.1"]
_NORMAL_PORTS = [80, 443, 22, 8080, 53, 25, 110, 3306]
_ATTACK_PORTS = [23, 135, 445, 4444, 3389, 8888, 31337]


def simulate_packet(attack_probability: float = 0.35) -> dict:
    """
    Generate a realistic synthetic packet feature dict — no Scapy required.

    Used when:
    - Scapy is not installed.
    - Running without admin/root privileges.
    - Unit-testing the pipeline.

    Args:
        attack_probability : Probability (0–1) that the packet is an attack.

    Returns:
        dict: Synthetic packet feature dictionary.
    """
    is_attack = random.random() < attack_probability

    if is_attack:
        src_ip  = random.choice(_ATTACK_IPS)
        dst_ip  = random.choice(_NORMAL_IPS)
        proto   = random.choice(["ICMP", "TCP", "TCP"])
        size    = random.randint(1200, 1500)      # Large flood-style packets
        s_port  = random.randint(1024, 65535)
        d_port  = random.choice(_ATTACK_PORTS)
        count   = random.randint(200, 2000)       # Burst
        dur     = round(random.uniform(0.001, 0.5), 4)
    else:
        src_ip  = random.choice(_NORMAL_IPS)
        dst_ip  = "8.8.8.8" if random.random() > 0.5 else random.choice(_NORMAL_IPS)
        proto   = random.choice(["TCP", "TCP", "UDP"])
        size    = random.randint(64, 900)
        s_port  = random.randint(1024, 65535)
        d_port  = random.choice(_NORMAL_PORTS)
        count   = random.randint(1, 40)
        dur     = round(random.uniform(0.1, 10.0), 4)

    return {
        "src_ip"       : src_ip,
        "dst_ip"       : dst_ip,
        "protocol"     : proto,
        "packet_size"  : size,
        "src_port"     : s_port,
        "dst_port"     : d_port,
        "duration"     : dur,
        "packet_count" : count,
        "timestamp"    : time.time(),
    }


def simulate_stream(
    delay            : float = 0.4,
    attack_probability: float = 0.35,
    stop_event       : Optional[threading.Event] = None,
) -> Generator[dict, None, None]:
    """
    Yield simulated packet dicts indefinitely (or until stop_event is set).

    Args:
        delay             : Seconds between synthetic packets.
        attack_probability: Fraction of packets simulated as attacks.
        stop_event        : threading.Event to signal loop termination.

    Yields:
        dict: Synthetic packet feature dictionary.
    """
    while not (stop_event and stop_event.is_set()):
        yield simulate_packet(attack_probability)
        time.sleep(delay)


# ------------------------------------------
# Factory — auto-select capture mode
# ------------------------------------------

def get_packet_stream(
    interface         : Optional[str]           = DEFAULT_IFACE,
    simulate          : bool                    = False,
    attack_probability: float                   = 0.35,
    stop_event        : Optional[threading.Event] = None,
) -> Generator[dict, None, None]:
    """
    Return the appropriate packet stream generator.

    - If simulate=True  -> use synthetic simulator (no privileges needed).
    - If simulate=False -> attempt live Scapy capture; fall back to simulator
      automatically if Scapy is unavailable.

    Args:
        interface         : Network interface (live mode only).
        simulate          : Force simulator mode.
        attack_probability: Simulator attack rate (0–1).
        stop_event        : Stop signal threading.Event.

    Yields:
        dict: Packet feature dictionary.
    """
    if simulate or not _SCAPY_AVAILABLE:
        mode = "SIMULATOR" if simulate else "SIMULATOR (Scapy not found)"
        print(f"[capture] Mode: {mode} | attack_rate={attack_probability * 100:.0f}%")
        yield from simulate_stream(
            attack_probability=attack_probability,
            stop_event=stop_event,
        )
    else:
        print(f"[capture] Mode: LIVE CAPTURE | interface={interface or 'default'}")
        yield from capture_stream(interface=interface, stop_event=stop_event)


# -- Quick standalone test ---------------------------------------------------
if __name__ == "__main__":
    import itertools
    print("Simulating 5 packets…\n")
    for pkt in itertools.islice(simulate_stream(delay=0.1), 5):
        print(pkt)
