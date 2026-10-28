def extract_features(hand, width, height):
    features = []
    
    wrist = hand[0]

    middle_base = hand[9]

    dx = (middle_base.x -wrist.x) * width
    dy = (middle_base.y -wrist.y) * height
    dz = (middle_base.z -wrist.z) * width

    scale = (dx**2 + dy**2 + dz**2)**0.5
    if scale < 1e-6:
        return None

    for point in hand:
        relative_x = (point.x - wrist.x) * width
        relative_y = (point.y - wrist.y) * height
        relative_z = (point.z - wrist.z) * width

        features.append(relative_x /scale)
        features.append(relative_y /scale)
        features.append(relative_z /scale)

    return features