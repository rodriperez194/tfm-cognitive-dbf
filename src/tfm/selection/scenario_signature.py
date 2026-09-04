from __future__ import annotations

from collections import Counter
from typing import Any


# ============================================================
# 1. CLASS → LABEL MAPPING
# ============================================================

CLASS_TO_LABEL = {
    "AircraftTarget": "aircraft",
    "DroneTarget": "drone",
    "Dummy": "dummy",
    "StaticTarget": "static",
    "TruckRoadTarget": "truck",
    # Add more classes here if needed
}


def get_target_label(target: Any) -> str:
    """
    Map a target/jammer object to its canonical label.

    Parameters
    ----------
    target : Any
        Target or jammer instance.

    Returns
    -------
    str
        Canonical label such as "drone", "aircraft", etc.
    """
    class_name = target.__class__.__name__

    if class_name not in CLASS_TO_LABEL:
        raise ValueError(f"Unknown target class: {class_name}")

    return CLASS_TO_LABEL[class_name]


# ============================================================
# 2. BUILD SCENARIO SIGNATURE
# ============================================================

def build_target_signature(jammers: list[Any]) -> str:
    """
    Build a canonical target signature string from a jammer list.

    Examples
    --------
    [] -> "0x_none"
    [Dummy] -> "1x_dummy"
    [Drone, Drone, Truck] -> "2x_drone + 1x_truck"
    [Dummy, StaticTarget] -> "1x_dummy + 1x_static"

    Parameters
    ----------
    jammers : list[Any]
        List of jammer objects.

    Returns
    -------
    str
        Canonical signature.
    """
    if not jammers:
        return "0x_none"

    labels = [get_target_label(jammer) for jammer in jammers]
    counts = Counter(labels)

    # Stable ordering (alphabetical)
    sorted_items = sorted(counts.items(), key=lambda item: item[0])

    parts = [f"{count}x_{label}" for label, count in sorted_items]

    return " + ".join(parts)


# ============================================================
# 3. EXTRACT MINIMAL LOOKUP KEY
# ============================================================

def extract_lookup_key(jammers: list[Any]) -> dict[str, Any]:
    """
    Extract the minimal key used by the final lookup table.

    Parameters
    ----------
    jammers : list[Any]
        List of jammer objects.

    Returns
    -------
    dict[str, Any]
        Dictionary containing only the target signature.
    """
    return {
        "target_signature": build_target_signature(jammers),
    }