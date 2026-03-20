def decide(signal, intent=None, bbox=None, frame_height=720):
    """
    Unified Conflict-Aware Risk Index (UCARI)

    Features:
    - 10% spatial relevance threshold
    - No signal + no vehicle → SAFE
    - No signal + near vehicle → CAUTION
    - Continuous risk formulation
    """

    # ==================================================
    # CASE 1: No Traffic Signal Detected
    # ==================================================
    if signal == "UNKNOWN":

        # No vehicle detected
        if intent is None or bbox is None:
            return "SAFE", 0.0

        # Vehicle detected → check spatial relevance
        x1, y1, x2, y2 = bbox
        bbox_height = y2 - y1
        relative_height = bbox_height / frame_height

        # Far vehicle → safe
        if relative_height < 0.10:
            return "SAFE", 0.0
        else:
            # Near vehicle but no signal → cautious driving
            return "CAUTION", 0.4

    # ==================================================
    # CASE 2: No Vehicle Detected
    # ==================================================
    if intent is None or bbox is None:

        if signal == "GREEN":
            return "SAFE", 0.0

        if signal == "YELLOW":
            return "CAUTION", 0.4

        if signal == "RED":
            return "RISK", 0.7

        return "SAFE", 0.0

    # ==================================================
    # SIGNAL SEVERITY
    # ==================================================
    if signal == "RED":
        SS = 1.0
    elif signal == "YELLOW":
        SS = 0.6
    else:  # GREEN
        SS = 0.0

    brake = intent.get("brake", False)
    left = intent.get("left_indicator", False)
    right = intent.get("right_indicator", False)

    # ==================================================
    # DISTANCE + 10% SPATIAL THRESHOLD
    # ==================================================
    x1, y1, x2, y2 = bbox
    bbox_height = y2 - y1
    relative_height = bbox_height / frame_height

    # Minimum spatial relevance threshold (10%)
    if relative_height < 0.10:
        return "SAFE", 0.0

    # Normalized distance score
    D = min(relative_height * 2.0, 1.0)

    # ==================================================
    # RISK COMPONENTS
    # ==================================================
    BA = 1 if not brake else 0
    I = 1 if (left or right) else 0

    # Weights (interpretable)
    w1 = 0.5  # signal conflict weight
    w2 = 0.3  # proximity weight
    w3 = 0.2  # maneuver weight

    UCARI = (
        w1 * (SS * BA) +
        w2 * D +
        w3 * I
    )

    UCARI = min(UCARI, 1.0)

    # ==================================================
    # DECISION MAPPING
    # ==================================================
    if UCARI >= 0.7:
        decision = "RISK"
    elif UCARI >= 0.4:
        decision = "CAUTION"
    else:
        decision = "SAFE"

    return decision, round(UCARI, 3)
