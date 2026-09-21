"""
detector.py — Real-Time Threat Detector
=========================================
Loads the trained RandomForest model and runs inference on preprocessed
packet feature vectors. Provides both single-packet and batch prediction
APIs, plus an anomaly threshold mechanism for fine-grained control.
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple

from model      import load_model, model_exists, ensure_trained
from preprocess import preprocess_packet, FEATURE_COLS

# ------------------------------------------
# Configuration
# ------------------------------------------
LABEL_MAP = {0: "NORMAL", 1: "ATTACK"}

# Anomaly threshold — packets with attack-probability ABOVE this are flagged.
# Lower values -> more sensitive (more false positives).
# Higher values -> less sensitive (fewer alerts).
DEFAULT_THRESHOLD = 0.50   # 50% confidence required to flag as ATTACK


# ------------------------------------------
# Detector Class
# ------------------------------------------

class NIDSDetector:
    """
    Network Intrusion Detector powered by a RandomForestClassifier.

    Usage:
        detector = NIDSDetector()
        result   = detector.predict(packet_info)
        print(result["label"], result["confidence"])
    """

    def __init__(self, threshold: float = DEFAULT_THRESHOLD):
        """
        Initialise the detector, loading (or training) the ML model.

        Args:
            threshold : Attack probability threshold (0.0–1.0).
                        Packets with P(attack) > threshold are flagged.
        """
        self.threshold = threshold
        self._clf      = None
        self._features = None
        self._load()

    # -- Private helpers ---------------------------------------------------

    def _load(self) -> None:
        """Load or train the model, then cache it in self._clf."""
        self._clf, self._features = ensure_trained()
        print(
            f"[detector] [OK]  Model ready | "
            f"threshold={self.threshold * 100:.0f}%"
        )

    # -- Core Prediction API -----------------------------------------------

    def predict(self, packet_info: dict) -> dict:
        """
        Classify a single packet as NORMAL or ATTACK.

        Args:
            packet_info : Raw packet feature dict (from capture.py).

        Returns:
            dict with keys:
            - 'label'         : 'NORMAL' or 'ATTACK'
            - 'confidence'    : Float 0–1 (model's attack probability)
            - 'is_attack'     : Boolean shortcut
            - 'src_ip'        : Pass-through from input
            - 'dst_ip'        : Pass-through from input
            - 'protocol'      : Pass-through from input
            - 'packet_size'   : Pass-through from input
        """
        # Preprocess the raw packet dict into a normalised feature array
        X = preprocess_packet(packet_info)

        # Get probability scores [P(normal), P(attack)]
        proba      = self._clf.predict_proba(X)[0]
        attack_prob = proba[1]

        # Apply custom threshold (overrides hard argmax decision)
        is_attack  = attack_prob > self.threshold
        label      = "ATTACK" if is_attack else "NORMAL"

        return {
            "label"       : label,
            "confidence"  : float(attack_prob),
            "is_attack"   : is_attack,
            "src_ip"      : packet_info.get("src_ip",      "?"),
            "dst_ip"      : packet_info.get("dst_ip",      "?"),
            "protocol"    : str(packet_info.get("protocol","?")),
            "packet_size" : int(packet_info.get("packet_size", 0)),
        }

    def predict_batch(self, packets: list[dict]) -> list[dict]:
        """
        Classify a batch of packet dicts efficiently.

        Internally vectorised: one model call for the entire batch.

        Args:
            packets : List of raw packet feature dicts.

        Returns:
            List[dict]: One result dict per input packet.
        """
        if not packets:
            return []

        from preprocess import packet_to_dataframe

        rows = []
        for pkt in packets:
            df_row = packet_to_dataframe(pkt).astype(float)
            rows.append(df_row)

        X_batch = pd.concat(rows, ignore_index=True)

        # Load scaler if available
        from preprocess import _load_scaler
        scaler = _load_scaler()
        if scaler is not None:
            X_batch[FEATURE_COLS] = scaler.transform(X_batch[FEATURE_COLS])

        probas       = self._clf.predict_proba(X_batch[FEATURE_COLS])
        attack_probs = probas[:, 1]

        results = []
        for i, pkt in enumerate(packets):
            ap        = attack_probs[i]
            is_attack = ap > self.threshold
            results.append({
                "label"       : "ATTACK" if is_attack else "NORMAL",
                "confidence"  : float(ap),
                "is_attack"   : is_attack,
                "src_ip"      : pkt.get("src_ip",      "?"),
                "dst_ip"      : pkt.get("dst_ip",      "?"),
                "protocol"    : str(pkt.get("protocol","?")),
                "packet_size" : int(pkt.get("packet_size", 0)),
            })
        return results

    # -- Threshold Management ----------------------------------------------

    def set_threshold(self, threshold: float) -> None:
        """
        Dynamically adjust the attack probability threshold.

        Args:
            threshold : New threshold value (0.0–1.0).
        """
        if not 0.0 < threshold < 1.0:
            raise ValueError("Threshold must be between 0.0 and 1.0 (exclusive).")
        self.threshold = threshold
        print(f"[detector] Threshold updated -> {threshold * 100:.0f}%")

    def get_threshold(self) -> float:
        """Return the current attack probability threshold."""
        return self.threshold


# ------------------------------------------
# Singleton detector instance (shared by main.py)
# ------------------------------------------

_detector_instance: Optional[NIDSDetector] = None


def get_detector(threshold: float = DEFAULT_THRESHOLD) -> NIDSDetector:
    """
    Return the shared NIDSDetector singleton, creating it on first call.

    Args:
        threshold : Initial threshold (only used on first call).

    Returns:
        NIDSDetector: Ready-to-use detector.
    """
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = NIDSDetector(threshold=threshold)
    return _detector_instance


# -- Quick standalone test ---------------------------------------------------
if __name__ == "__main__":
    from capture import simulate_packet

    detector = NIDSDetector()

    print("\n-- Single-packet predictions ---------------------------------")
    for _ in range(6):
        pkt    = simulate_packet(attack_probability=0.5)
        result = detector.predict(pkt)
        tag    = "🔴 ATTACK" if result["is_attack"] else "🟢 NORMAL"
        print(
            f"  {tag}  src={result['src_ip']:<15} "
            f"dst={result['dst_ip']:<15} "
            f"conf={result['confidence'] * 100:.1f}%"
        )

    print("\n-- Batch prediction (10 packets) -----------------------------")
    batch   = [simulate_packet(0.5) for _ in range(10)]
    results = detector.predict_batch(batch)
    attacks = sum(r["is_attack"] for r in results)
    print(f"  Attacks detected: {attacks}/{len(results)}")
