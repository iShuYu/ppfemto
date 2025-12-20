import numpy as np
import awkward as ak
from iohelper import *


def event_mask(arr, cfg, block):
    """
    根据配置文件中指定的筛选EVENT ，在 Awkward Array event层面上构造布尔掩码。

    Parameters
    ----------
    arr : ak.Array
        输入的 Awkward Array，通常为 record 形式，
        字段名需与配置中对应 block 的 key 保持一致。
        例如 TRACK 级或 EVENT 级的数据结构。

    cfg : SimpleNamespace
        由 `load_config` 生成的配置对象。
        每个 block（如 TRACK、EVENT）是一个命名空间，
        其属性名为筛选变量名，属性值为 [low, high] 的区间定义。

    block : str
        配置中使用的筛选块名称，例如 "TRACK" 或 "EVENT"。
        函数将从 `cfg.<block>` 中读取对应的筛选条件。

    Returns
    -------
    ak.Array
        布尔类型的 Awkward Array，
        其形状与输入 `arr` 在事件和元素维度上完全对齐。
        该 mask 可直接用于 Awkward Array 的索引筛选，
        或与其他 mask 进行逻辑组合。
    """
    cuts = getattr(cfg, block)

    mask = ak.Array([True] * len(arr))
    for field, (lo, hi) in cuts.__dict__.items():
        x = arr[field]

        if lo is not None:
            mask = mask & (x >= lo)
        if hi is not None:
            mask = mask & (x <= hi)

    return mask


def track_mask(arr, cfg, block):
    """
    根据配置文件中指定的筛选 block，在 Awkward Array 向量branch上构造布尔掩码。

    Parameters
    ----------
    见event_mask()

    Returns
    -------
    ak.Array
        该 mask 可直接用于 Awkward Array 的索引 x 向量变量 筛选，
    """
    cuts = getattr(cfg, block)

    iter = 0
    for field, (lo, hi) in cuts.__dict__.items():

        x = arr[field]
        if iter > 0:
            if lo is not None:
                mask = mask & (x >= lo)
            if hi is not None:
                mask = mask & (x <= hi)
        else:
            mask = ak.full_like(x, True, dtype=bool)
        iter += 1
    return mask


def select(arr, mask):
    """
    根据掩码拿出arr中位置为True的数据

    Parameters
    ----------
    arr : ak.Array
        Awkward Array
    mask : ak.Array
        由event_mask或者track_mask生成的掩码

    Returns
    -------
    ak.Array
        通过筛选条件的 track 子集，保持原有的事件结构不变。
    """
    return arr[mask]


def apply_track_mask_by_prefix(tree: ak.Array, mask: ak.Array, prefix="TRACK"):
    """
    仅对 prefix 开头的 track-level branch 应用 mask，
    其余 event-level branch 保持不变
    筛选出特定的tracks

    Parameters
    ----------
    tree : ak.Array
        Awkward Array
    mask : ak.Array
        由event_mask或者track_mask生成的掩码
    prefix:
        对以prefix开头的branch进行筛选

    Returns
    -------
    ak.Array
        通过筛选条件的 track 子集，保持原有的事件结构不变。
    """
    out = {}

    for field in tree.fields:
        if field.startswith(prefix):
            out[field] = tree[field][mask]
        else:
            out[field] = tree[field]

    return ak.zip(out, depth_limit=1)


def unique_event(tree, key: str = "EVENTNUMBER"):
    """
    按 event-level 的 key 去重，只保留第一次出现的 event

    Parameters:
    ----------
    tree:
        需要筛选的ak.Array
    key:
        根据哪一个branch的unique进行筛选，默认使用EVENTNUMBER
    """
    ev = tree[key]
    ev_np = ak.to_numpy(ev)
    _, first_idx = np.unique(ev_np, return_index=True)
    first_idx = np.sort(first_idx)

    return tree[first_idx]


def reduce_raw(input_path, tree_path, cfg_select_path, cfg_io_path, output_path):
    """
    将一个未经处理的root文件，根据cfg_select.json里记录的筛选条件提取出有用的部分

    Parameters
    ----------
    root_path:
        root文件所处的路径
    tree_path:
        tree在root文件里所处的路径
    select_branch:
        读取的时候选定哪些branch，若为None则默认选取所有branch
    exclude_branch:
        读取的时候排除哪些branch，若为None则默认不排除任何branch
    cfg_select_path:
        记录有event cut和tracks cut的文件
    output_path:
        筛选后文件的储存路径
    Returns
    -------
    ak.Array
        通过筛选条件的 track 子集，保持原有的事件结构不变。
    """
    cfg_select = load_config(config_path=cfg_select_path)
    cfg_io = load_config(config_path=cfg_io_path)

    tree = load_tree(
        root_path=input_path,
        tree_path=tree_path,
        exclude_branch=cfg_io.data.exclude_branch,
    )

    # select event
    mask_event = event_mask(tree, cfg_select, "EVENT")
    tree = select(tree, mask_event)

    # select tracks
    mask_tracks = track_mask(tree, cfg_select, "TRACK")
    tree = apply_track_mask_by_prefix(tree, mask_tracks, prefix="TRACK")

    # select event with at least two proton
    tree = tree[ak.num(tree["TRACK_P"], axis=1) >= 2]

    # delete duplicate events
    tree = unique_event(tree)

    # ---------------- optional: 写出 ----------------
    # 如果你是写 root / parquet / npz，这里换成你自己的 writer
    save_tree(tree, output_path)

    return tree
