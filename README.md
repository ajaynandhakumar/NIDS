# AI-Based Network Intrusion Detection System (NIDS)

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Scikit-Learn](https://img.shields.io/badge/ML-scikit--learn-orange?logo=scikit-learn)
![Scapy](https://img.shields.io/badge/Network-Scapy-green)
![License](https://img.shields.io/badge/License-MIT-brightgreen)
![Status](https://img.shields.io/badge/Status-Active-success)

*A production-style, ML-powered network intrusion detection system with real-time packet analysis and colour-coded terminal alerts.*

</div>

---

## 📌 Project Overview

The **AI-Based Network Intrusion Detection System (NIDS)** is a modular Python application that monitors network traffic in real time and uses Machine Learning to distinguish between **normal** and **malicious (attack)** packets.

It is designed to demonstrate industry-level cybersecurity tooling — suitable for a professional GitHub portfolio, academic projects, or as a foundation for a production NIDS deployment.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🤖 ML Detection | RandomForestClassifier (scikit-learn) trained on traffic features |
| 📡 Dual Capture Mode | Live Scapy sniffing **or** built-in traffic simulator |
| 🎨 Colour-Coded Output | `🟢 GREEN` for normal, `🔴 RED` for attacks |
| 🔔 Audio Alerts | System beep on attack detection (Windows/Linux) |
| 📊 Live Dashboard | Rolling packet statistics printed every N packets |
| 📝 Structured Logging | Timestamped `logs/events.log` + `logs/alerts.log` |
| 🎛️ Adjustable Threshold | Fine-tune attack confidence threshold via CLI flag |
| 🔁 Auto-Train | Generates synthetic dataset and trains model if none exists |
| 📦 Modular Architecture | Clean separation of concerns across 8 focused modules |

---

## 📁 Project Structure

```
NIDS/
├── main.py          # Entry point — runs the full monitoring pipeline
├── capture.py       # Packet capture (Scapy) + traffic simulator
├── preprocess.py    # Feature engineering + normalisation pipeline
├── model.py         # ML model training, evaluation, and persistence
├── detector.py      # Real-time classification engine (NIDSDetector)
├── alert.py         # Colour-coded terminal alerts + audio notifications
├── logger.py        # Structured file-based event logging
├── dataset.py       # Synthetic dataset generator (5 000 samples)
├── requirements.txt # Python dependencies
├── README.md        # This file
├── data/
│   └── network_traffic.csv   # Generated training dataset
├── models/
│   ├── nids_model.pkl         # Saved RandomForest model
│   └── scaler.pkl             # Saved MinMaxScaler
└── logs/
    ├── events.log             # All events (normal + attacks)
    └── alerts.log             # Attack-only log
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| ML Framework | scikit-learn (RandomForestClassifier) |
| Data Processing | pandas, numpy |
| Model Persistence | joblib |
| Network Capture | Scapy |
| Terminal UI | colorama (ANSI colours) |

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/nids.git
cd nids
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. (Windows only) Install Npcap for live capture

Download and install **Npcap** from [https://npcap.com](https://npcap.com) if you want to use live packet capture mode.

---

## 🚀 How to Run

### ▶️ Default Mode — Simulator (no admin required)

```bash
python main.py
```

The simulator generates realistic synthetic network packets and runs the full ML pipeline without needing root/admin or a network interface.

---

### 📡 Live Capture Mode (requires admin / root + Npcap/libpcap)

**Windows** — run as Administrator:
```bash
python main.py --live
```

**Linux / macOS** — run with sudo:
```bash
sudo python main.py --live --interface eth0
```

---

### 🎛️ CLI Options

| Flag | Default | Description |
|---|---|---|
| `--live` | off | Enable live Scapy packet capture |
| `--interface IFACE` | auto | Network interface (e.g. `eth0`, `Wi-Fi`) |
| `--threshold PROB` | `0.50` | Attack confidence threshold (0.0–1.0) |
| `--attack-rate RATE` | `0.35` | Simulator attack probability |
| `--delay SEC` | `0.4` | Delay between simulated packets (seconds) |
| `--dashboard-interval N` | `20` | Print dashboard every N packets |
| `--train` | off | Force retrain the model and exit |

### Examples

```bash
# High-sensitivity detection (lower threshold)
python main.py --threshold 0.35

# Simulate heavy attack traffic (60% attack rate)
python main.py --attack-rate 0.6

# Retrain the ML model from scratch
python main.py --train

# Live capture on specific interface
python main.py --live --interface "Wi-Fi"
```

---

## 📊 Sample Output

```
  ╔══════════════════════════════════════════════════════════╗
  ║       AI-Based Network Intrusion Detection System        ║
  ║                     [NIDS v1.0]                          ║
  ║       Real-time ML-Powered Threat Detection Engine       ║
  ╚══════════════════════════════════════════════════════════╝

[INFO] Initialising ML model…
[model] Loaded model v1.0 with 200 trees.
[detector] ✅  Model ready | threshold=50%
[capture] Mode: SIMULATOR | attack_rate=35%
[INFO] Monitoring started. Press Ctrl+C to stop.

[20:15:01] ✔ NORMAL  | 192.168.1.7     → 8.8.8.8          | TCP  |   512B | conf=97.1%
[20:15:01] ✔ NORMAL  | 192.168.1.3     → 1.1.1.1          | UDP  |   256B | conf=91.3%
══════════════════════════════════════════════════════════════
  ⚠  ATTACK DETECTED  [20:15:02]
  ► Src IP    : 203.0.113.4
  ► Dst IP    : 192.168.1.12
  ► Protocol  : ICMP
  ► Pkt Size  : 1450 bytes
  ► Confidence: 88.5%
══════════════════════════════════════════════════════════════

  ┌─ NIDS Dashboard ──────────────────────────────────┐
  │  Packets Inspected : 20                            │
  │  Normal Traffic   : 13                             │
  │  Attack Detected  : 7      ( 35.0% threat rate)   │
  └────────────────────────────────────────────────────┘
```

---

## 🧠 ML Model Details

- **Algorithm**: RandomForestClassifier (200 trees, max_depth=15)
- **Features**: `packet_size`, `protocol`, `duration`, `src_port`, `dst_port`, `packet_count`, `byte_rate`
- **Training data**: 5,000 synthetic samples (60% normal / 40% attack)
- **Evaluation**: 80/20 train-test split + 5-fold cross-validation
- **Typical accuracy**: ~97–99% on synthetic data

---

## 📂 Log File Format

```
2026-09-21 20:15:01 | INFO     | src=192.168.1.7   dst=8.8.8.8      proto=TCP   size=  512B result=NORMAL  confidence=0.9710
2026-09-21 20:15:02 | WARNING  | src=203.0.113.4   dst=192.168.1.12 proto=ICMP  size= 1450B result=ATTACK  confidence=0.8850
```

---

## 🔒 Security Disclaimer

This tool is designed for **educational and research purposes**. Live packet capture requires appropriate authorization. Do **not** deploy on networks you do not own or have explicit permission to monitor.

---

## 📄 License

MIT License — free to use, modify, and distribute with attribution.

---

## 👤 Author

Built as a professional-grade cybersecurity portfolio project demonstrating full-stack Python engineering, ML integration, and real-time systems design.
