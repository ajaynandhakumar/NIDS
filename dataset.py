"""
dataset.py — Dummy Dataset Generator
======================================
Generates a synthetic network traffic dataset for training the ML model.
This simulates real-world NIDS datasets like KDD Cup 99 or CICIDS.
"""

import numpy as np
import pandas as pd
import os

# ------------------------------------------
# Constants
# ------------------------------------------
DATASET_PATH = "data/network_traffic.csv"
RANDOM_SEED  = 42
NUM_SAMPLES  = 5000   # Total records (normal + attack)


def generate_dataset(n_samples: int = NUM_SAMPLES, save: bool = True) -> pd.DataFrame:
    """
    Generate a synthetic network traffic dataset.

    Features generated:
    - packet_size  : Size of the packet in bytes
    - protocol     : Network protocol (0=TCP, 1=UDP, 2=ICMP)
    - duration     : Connection duration in seconds
    - src_port     : Source port number
    - dst_port     : Destination port number
    - packet_count : Number of packets in session
    - byte_rate    : Bytes per second (derived)
    - label        : 0 = NORMAL, 1 = ATTACK

    Returns:
        pd.DataFrame: The generated dataset.
    """
    np.random.seed(RANDOM_SEED)

    n_normal = int(n_samples * 0.6)   # 60% normal traffic
    n_attack = n_samples - n_normal   # 40% attack traffic

    # -- Normal traffic profile --------------------------
    normal = pd.DataFrame({
        "packet_size"  : np.random.normal(loc=500,  scale=150,  size=n_normal).clip(64, 1500),
        "protocol"     : np.random.choice([0, 1, 2], size=n_normal, p=[0.6, 0.3, 0.1]),
        "duration"     : np.random.exponential(scale=2.0, size=n_normal).clip(0.01, 60),
        "src_port"     : np.random.randint(1024, 65535, size=n_normal),
        "dst_port"     : np.random.choice([80, 443, 22, 8080, 3306], size=n_normal),
        "packet_count" : np.random.randint(1, 50, size=n_normal),
        "label"        : 0,  # NORMAL
    })

    # -- Attack traffic profile (anomalous behaviour) ----
    attack = pd.DataFrame({
        "packet_size"  : np.where(
                            np.random.rand(n_attack) > 0.5,
                            np.random.normal(1450, 30, n_attack),   # Large flood packets
                            np.random.normal(64,   10, n_attack),   # Tiny fragmented packets
                         ).clip(40, 1500),
        "protocol"     : np.random.choice([0, 1, 2], size=n_attack, p=[0.3, 0.2, 0.5]),
        "duration"     : np.random.exponential(scale=0.1, size=n_attack).clip(0.001, 5),
        "src_port"     : np.random.randint(1024, 65535, size=n_attack),
        "dst_port"     : np.random.choice([23, 135, 445, 3389, 4444], size=n_attack),
        "packet_count" : np.random.randint(100, 10000, size=n_attack),  # Burst traffic
        "label"        : 1,  # ATTACK
    })

    df = pd.concat([normal, attack], ignore_index=True)

    # Derived feature: bytes per second (avoid division by zero)
    df["byte_rate"] = (df["packet_size"] * df["packet_count"]) / (df["duration"] + 1e-6)

    # Shuffle the dataset
    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    if save:
        os.makedirs(os.path.dirname(DATASET_PATH), exist_ok=True)
        df.to_csv(DATASET_PATH, index=False)
        print(f"[dataset] [OK]  Dataset saved -> {DATASET_PATH}  ({len(df)} records)")

    return df


# -- Quick standalone test ---------------------------------------------------
if __name__ == "__main__":
    df = generate_dataset()
    print(df.head(10))
    print(f"\nLabel distribution:\n{df['label'].value_counts()}")
