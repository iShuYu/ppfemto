import numpy as np
import awkward as ak


def gen_pair(X, field_map=None):
    """
    根据给定的字段映射生成 track pair 的 ak.combinations

    Parameters
    ----------
    X : ak.Array
        原始 awkward array
    field_map : dict
        {输出字段名: X 中的 branch 名}
        例如 {"px": "TRACK_PX", "py": "TRACK_PY", "pz": "TRACK_PZ"}

    Returns
    -------
    pair : ak.Array
        每个 event 内 tracks 的两两组合
    """
    if field_map is None:
        field_map = {
            "px": "TRACK_PX",
            "py": "TRACK_PY",
            "pz": "TRACK_PZ",
        }
    tracks = ak.zip(
        {k: X[v] for k, v in field_map.items()},
        depth_limit=2,
    )

    pair = ak.combinations(tracks, 2, fields=["a", "b"], axis=1)
    return pair


def shuffle_events(X, shuffle: int = 1):
    """
    按 (PVZ, nLongTracks) 排序后整体顺移一格

    Parameter:
    ----------
    X:
        输入的akarray

    Returns:
    --------
    输出根据PVZ和nLongTracks排序后的akarray和shuffle之后的akarray
    以用于后续的mixing，成为一个mixing pair
    """

    pvz = ak.to_numpy(ak.flatten(X["PVZ"]))
    ntrk = ak.to_numpy(X["nLongTracks"])

    order = np.lexsort((ntrk, pvz))

    X_sorted = X[order]

    X_shifted = ak.concatenate(
        [X_sorted[-shuffle:], X_sorted[:-shuffle]],
        axis=0,
    )

    return X_sorted, X_shifted


def gen_mixed_pair(X, field_map: dict = None, shuffle: int = 1):
    """
    对event进行mix，再将event i和event j的tracks进行cartesian product

    Parameters
    ----------
    X : ak.Array
        原始 awkward array
    field_map : dict
        {输出字段名: X 中的 branch 名}
        例如 {"px": "TRACK_PX", "py": "TRACK_PY", "pz": "TRACK_PZ"}

    Returns
    -------
    pair : ak.Array
        每个 event 内 tracks 的两两组合
    """
    if field_map is None:
        field_map = {
            "px": "TRACK_PX",
            "py": "TRACK_PY",
            "pz": "TRACK_PZ",
        }
    X_sort, X_shuffle = shuffle_events(X, shuffle=shuffle)

    a = ak.zip(
        {k: X_sort[v] for k, v in field_map.items()},
        depth_limit=2,
    )

    b = ak.zip(
        {k: X_shuffle[v] for k, v in field_map.items()},
        depth_limit=2,
    )

    pair = ak.cartesian(
        {"a": a, "b": b},
        axis=1,
        nested=True,
    )

    return pair
