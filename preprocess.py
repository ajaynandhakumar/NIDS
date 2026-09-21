"""
preprocess.py — Data Preprocessing Pipeline
=============================================
Cleans raw packet data captured from the network (or from the dummy dataset),
engineers features, encodes categoricals, and normalises values so the data
is ready to feed into any scikit-learn model.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import joblib
import os
from typing import Tuple

# ------------------------------------------
# Constants
# ------------------------------------------
SCALER_PATH    = "models/scaler.pkl"
FEATURE_COLS   = [
    "packet_size",
    "protocol",
    "duration",
    "src_port",
    "dst_port",
    "packet_count",
    "byte_rate",
]

# Map protocol strings -> integers (for live packets)
PROTOCOL_MAP: dict[str, int] = {
    "TCP"  : 0,
    "UDP"  : 1,
    "ICMP" : 2,
    "OTHER": 3,
}


# ------------------------------------------
# Raw Packet Dict -> DataFrame row
# ------------------------------------------

def packet_to_dataframe(packet_info: dict) -> pd.DataFrame:
    """
    Convert a single raw packet dict (from capture.py) into a one-row DataFrame
    that matches the training feature schema.

    Args:
        packet_info : Dict with keys: src_ip, dst_ip, protocol, packet_size,
                      src_port (optional), dst_port (optional), duration (optional).

    Returns:
        pd.DataFrame: Single-row DataFrame ready for preprocessing.
    """
    proto_str = str(packet_info.get("protocol", "OTHER")).upper()
    proto_int = PROTOCOL_MAP.get(proto_str, 3)

    packet_size  = int(packet_info.get("packet_size", 0))
    src_port     = int(packet_info.get("src_port",    random_port()))
    dst_port     = int(packet_info.get("dst_port",    80))
    duration     = float(packet_info.get("duration",  np.random.exponential(2.0)))
    packet_count = int(packet_info.get("packet_count", np.random.randint(1, 20)))
    byte_rate    = (packet_size * packet_count) / (duration + 1e-6)

    return pd.DataFrame([{
        "packet_size"  : packet_size,
        "protocol"     : proto_int,
        "duration"     : duration,
        "src_port"     : src_port,
        "dst_port"     : dst_port,
        "packet_count" : packet_count,
        "byte_rate"    : byte_rate,
    }])


def random_port() -> int:
    """Return a random ephemeral port number."""
    return int(np.random.randint(1024, 65535))


# ------------------------------------------
# Full Dataset Preprocessing (for training)
# ------------------------------------------

def preprocess_dataset(
    df        : pd.DataFrame,
    fit_scaler: bool = True,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Full preprocessing pipeline for a labeled dataset.

    Steps:
        1. Drop duplicates & rows with null values.
        2. Clip extreme outliers.
        3. Encode categorical features.
        4. Derive composite features.
        5. Normalise with MinMaxScaler.
        6. Return (X_features, y_labels).

    Args:
        df         : Raw DataFrame (must contain FEATURE_COLS + 'label').
        fit_scaler : If True, fit a new scaler and save it. If False, load saved.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: (X, y) ready for model training/eval.
    """
    df = df.copy()

    # -- 1. Clean ----------------------------------------------------------
    before = len(df)
    df.drop_duplicates(inplace=True)
    df.dropna(subset=FEATURE_COLS + ["label"], inplace=True)
    after  = len(df)
    if before != after:
        print(f"[preprocess] Removed {before - after} dirty rows (dups / nulls).")

    # -- 2. Clip outliers (IQR-based) ---------------------------------------
    for col in ["packet_size", "byte_rate", "packet_count", "duration"]:
        q1, q3 = df[col].quantile(0.01), df[col].quantile(0.99)
        df[col] = df[col].clip(q1, q3)

    # -- 3. Encode protocol if it is still a string -------------------------
    if df["protocol"].dtype == object:
        df["protocol"] = df["protocol"].str.upper().map(PROTOCOL_MAP).fillna(3).astype(int)

    # -- 4. Separate features and labels -----------------------------------
    # Accept both 0/1 and 'normal'/'attack' label formats
    y = df["label"].copy()
    if y.dtype == object:
        y = y.str.lower().map({"normal": 0, "attack": 1}).fillna(0).astype(int)
    else:
        y = y.astype(int)

    X = df[FEATURE_COLS].copy().astype(float)

    # -- 5. Normalise -------------------------------------------------------
    scaler = _get_scaler(X, fit_scaler)
    X_scaled = pd.DataFrame(scaler.transform(X), columns=FEATURE_COLS)

    print(
        f"[preprocess] [OK]  Preprocessing complete: "
        f"{len(X_scaled)} samples × {len(FEATURE_COLS)} features | "
        f"attacks={int(y.sum())} ({y.mean() * 100:.1f}%)"
    )
    return X_scaled, y


# ------------------------------------------
# Live Single-Packet Preprocessing
# ------------------------------------------

def preprocess_packet(packet_info: dict) -> np.ndarray:
    """
    Preprocess a single live packet dict for real-time prediction.

    Args:
        packet_info : Raw packet dictionary from capture.py.

    Returns:
        pd.DataFrame: Shape (1, n_features), normalised and model-ready.
    """
    df = packet_to_dataframe(packet_info)
    df = df.astype(float)

    scaler = _load_scaler()
    if scaler is None:
        # No saved scaler yet — return raw DataFrame (keeps column names)
        return df[FEATURE_COLS]

    scaled_values = scaler.transform(df[FEATURE_COLS])
    return pd.DataFrame(scaled_values, columns=FEATURE_COLS)


# ------------------------------------------
# Scaler Helpers
# ------------------------------------------

def _get_scaler(X: pd.DataFrame, fit: bool) -> MinMaxScaler:
    """
    Return a fitted MinMaxScaler, loading or fitting as directed.

    Args:
        X   : Feature DataFrame to fit on.
        fit : Whether to fit a new scaler.

    Returns:
        MinMaxScaler: Ready-to-use scaler.
    """
    if fit:
        scaler = MinMaxScaler()
        scaler.fit(X)
        os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)
        joblib.dump(scaler, SCALER_PATH)
        print(f"[preprocess] Scaler saved -> {SCALER_PATH}")
        return scaler

    saved = _load_scaler()
    if saved is None:
        raise FileNotFoundError(
            f"Scaler not found at {SCALER_PATH}. Run training first."
        )
    return saved


def _load_scaler() -> MinMaxScaler | None:
    """
    Load the saved MinMaxScaler from disk.

    Returns:
        MinMaxScaler | None: Loaded scaler, or None if file not found.
    """
    if os.path.exists(SCALER_PATH):
        return joblib.load(SCALER_PATH)
    return None


# -- Quick standalone test ---------------------------------------------------
if __name__ == "__main__":
    from dataset import generate_dataset
    df = generate_dataset(save=False)
    X, y = preprocess_dataset(df, fit_scaler=True)
    print(X.head())
    print(f"\nFeatures: {list(X.columns)}")
    print(f"Label counts:\n{y.value_counts()}")

    # Test live packet preprocessing
    sample_packet = {
        "src_ip": "192.168.1.5", "dst_ip": "10.0.0.1",
        "protocol": "TCP", "packet_size": 512,
        "src_port": 50000, "dst_port": 80,
        "duration": 1.5, "packet_count": 5,
    }
    arr = preprocess_packet(sample_packet)
    print(f"\nLive packet feature vector:\n{arr}")
