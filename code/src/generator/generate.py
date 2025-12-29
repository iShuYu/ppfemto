import os
import awkward as ak

from selector import *
from kinematic import *
from combinator import *


def generate(
    input_dir: str,
    output_same_kstar: str,
    output_same_costheta: str,
    output_mix_kstar_dir: str,
    mixing_time: int = 20,
    m1: float = 938.272,
    m2: float = 938.272,
    min_angle: float = 0.0005,
    mother: str = "D0",
):
    """
    计算 same-event 与 mixed-event 的两体关联量（k*、cosθ）。

    功能：
    - 读取 input_dir 下所有 parquet（event → tracks）
    - 对同一 event 内不同 track 进行 pairing，计算 same-event k* 与 cosθ
    - 按 event mixing（排序 + cyclic shift）生成 mixed-event pair，
      重复 mixing_time 次，计算对应的 k*
    - 结果以 parquet 格式写出

    Parameters
    ----------
    input_dir : str
        输入 parquet 文件所在目录（支持 *.parquet）
    output_same_kstar : str
        same-event k* 的输出 parquet 路径
    output_same_costheta : str
        same-event cosθ 的输出 parquet 路径
    output_mix_kstar_dir : str
        mixed-event k* 的输出目录（每次 mixing 一个文件）
    mixing_time : int, default 21
        event mixing 的次数（对应不同 shift）
    min_angle:
        最小容忍夹角，小于此值认为是clonetracks

    Notes
    -----
    - same-event 与 mixed-event 的 pair 结构在 track-pair 层面保持一致
    - mixed-event 通过 event-level shift 实现，不引入 event 泄漏
    """

    # ---------- input ----------
    input_path = os.path.join(input_dir, "*.parquet")
    X = ak.from_parquet(input_path)

    # ------- remove clone tracks --------
    X = remove_clone_tracks_N(X=X, min_angle=min_angle)

    # ---------- ensure output dirs ----------
    os.makedirs(os.path.dirname(output_same_kstar), exist_ok=True)
    os.makedirs(os.path.dirname(output_same_costheta), exist_ok=True)
    os.makedirs(output_mix_kstar_dir, exist_ok=True)

    # ---------- same-event ----------
    same_pair = gen_pair(X)
    kstar_same = kstar(same_pair, m1=m1, m2=m2)
    cos_angle = cos_theta(same_pair)

    ak.to_parquet(kstar_same, output_same_kstar)
    ak.to_parquet(cos_angle, output_same_costheta)

    # ---------- mixed-event ----------
    for i in range(1, mixing_time + 1):
        mixing_pair = gen_mixed_pair(X, shuffle=i, mother=mother)
        kstar_mix = kstar(mixing_pair, m1=m1, m2=m2)
        cos_angle_mix = cos_theta(mixing_pair)

        out_kstar_path = os.path.join(
            output_mix_kstar_dir,
            f"kstar_mix{i}.parquet",
        )
        ak.to_parquet(kstar_mix, out_kstar_path)

        out_costheta_path = os.path.join(
            output_mix_kstar_dir,
            f"costheta_mix{i}.parquet",
        )
        ak.to_parquet(cos_angle_mix, out_costheta_path)
