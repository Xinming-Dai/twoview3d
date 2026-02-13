def pseudo_pointcloud_masked(depth):
    H, W = depth.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))

    X = (u - W / 2) / W
    Y = (v - H / 2) / H
    Z = depth

    mask = Z > 0  # or np.isfinite(Z)

    points = np.stack([X[mask], Y[mask], Z[mask]], axis=-1)
    return points