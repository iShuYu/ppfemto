import json
import uproot
import awkward as ak
from pathlib import Path
from types import SimpleNamespace
from uproot.interpretation.numerical import AsDtype


def load_tree(
    root_path: str,
    tree_path: str,
    select_branch: str | list[str] | None = None,
    exclude_branch: str | list[str] | None = None,
    only_scalar: bool = False,
    mother: str = "D0",
    add_branch_flag: bool=True,
) -> ak.Array:
    """
    从 ROOT 文件读取指定 tree，支持自动筛选单值 scalar branches。

    Parameters
    ----------
    root_path : str
        ROOT 文件路径。
    tree_path : str
        tree 在文件中的完整路径（例如 "JpsiTuple/DecayTree"）。
    select_branch : str, list[str], or None
        想要读取的 branch 名称或列表；None 表示读取全部。
    exclude_branch : str, list[str], or None
        想要排除的 branch 名称或列表。
    only_scalar : bool
        如果为 True，则自动筛选所有单值（非 array）branch。

    Returns
    -------
    ak.Array
        Awkward Array。
    """
    tree = uproot.open(f"{root_path}:{tree_path}")
    all_branches = set(tree.keys())

    # ========= 1. 自动筛选 scalar-only branches =========
    if only_scalar:
        scalar_branches = []
        for br in all_branches:
            interp = tree[br].interpretation
            if isinstance(interp, AsDtype):
                scalar_branches.append(br)
        all_scalar_set = set(scalar_branches)
    else:
        all_scalar_set = all_branches

    # ========= 2. select_branch  =========
    if select_branch is None:
        selected = set(all_scalar_set)
    else:
        if isinstance(select_branch, str):
            select_branch = [select_branch]
        selected = set(select_branch) & all_scalar_set

    # ========= 3. exclude_branch  =========
    if exclude_branch:
        if isinstance(exclude_branch, str):
            exclude_branch = [exclude_branch]
        selected -= set(exclude_branch)

    tree = tree.arrays(list(selected), library="ak")
    if add_branch_flag:
        tree = add_branch(tree=tree, mother=mother)
    return tree


def add_branch(tree: ak.Array, mother: str = "D0") -> ak.Array:
    """
    为 Awkward Array 按需添加派生字段。
    Parameters
    ----------
    tree : ak.Array
        输入的 Awkward Array。

    Returns
    -------
    ak.Array
        添加派生字段后的 Awkward Array。
    """
    if "TRACK_PIDp2K" not in tree.fields:
        tree["TRACK_PIDp2K"] = tree["TRACK_PIDp"] - tree["TRACK_PIDK"]

    if "TRACK_ETA_PHI_GHOST" not in tree.fields:
        tree["TRACK_ETA_PHI_GHOST"] = (
            1e6 * tree["TRACK_ETA"]
            + 1e6 * abs(tree["TRACK_PHI"])
            + tree["TRACK_GHOSTPROB"]
        )

    # PVNTRACKS这个变量需要自己构建，故 or True
    if ("PVNTRACKS" not in tree.fields) or True:
        tree["PVNTRACKS"] = (tree["PVCHI2"] / tree["PVCHI2DOF"] + 3) / 2

    # tracks的z坐标距离D0粒子的z坐标足够近的，它们为同源粒子
    if "TRACK_Z_to_PARTICLE_Z" not in tree.fields:
        tree["TRACK_Z_to_PARTICLE_Z"] = (
            tree[f"{mother}_OWNPV_Z"] - tree["TRACK_OWNPV_Z"]
        )

    return tree


def dict_to_namespace(d: dict) -> SimpleNamespace:
    """
    递归地将 dict 转换为 SimpleNamespace
    """
    ns = SimpleNamespace()
    for k, v in d.items():
        if isinstance(v, dict):
            setattr(ns, k, dict_to_namespace(v))
        else:
            setattr(ns, k, v)
    return ns


def load_config(config_path: str | Path) -> SimpleNamespace:
    """
    从 JSON 文件加载配置并转换为 Namespace。

    Parameters
    ----------
    config_path : str or Path
        配置文件路径 (例如 config.json)

    Returns
    -------
    cfg : SimpleNamespace
        可通过属性访问的配置对象
    """
    config_path = Path(config_path)
    with config_path.open("r") as f:
        data = json.load(f)
    return dict_to_namespace(data)


def save_tree(tree: ak.Array, output_path: str):
    """
    将ak array储存为parquet文件
    """
    ak.to_parquet(tree, output_path)
