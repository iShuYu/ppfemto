import uproot
import awkward as ak
from uproot.interpretation.numerical import AsDtype


def load_tree(
    root_path: str,
    tree_path: str,
    select_branch: str | list[str] | None = None,
    exclude_branch: str | list[str] | None = None,
    only_scalar: bool = False,
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

    return tree.arrays(list(selected), library="ak")
