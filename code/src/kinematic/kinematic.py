import numpy as np
import awkward as ak

"""
注意，使用这里的计算公式前，先阅读combinator，搞清楚combinator里如何搞定：
1. 相同event里的tracks的两两配对
2. mix event里的tracks的两两配对
3. zip的时候改了key name，所以这里写的pxpypz，原数据里并没有，不要误解

并搞清楚为什么这两者出来的pair，为什么都可以用于这里的函数
"""


def norm(v):
    """
    计算三维向量的模长（欧几里得范数）。

    Parameters
    ----------
    v : ak.Array
        包含 px、py、pz 字段的 Awkward Array，
        表示一个或一组三维向量。

    Returns
    -------
    ak.Array
        向量的模长，与输入向量在事件和元素维度上对齐。
    """
    return (v.px**2 + v.py**2 + v.pz**2) ** 0.5


def dot(a, b):
    """
    计算两组三维向量的点积。

    Parameters
    ----------
    a, b : ak.Array
        包含 px、py、pz 字段的 Awkward Array，
        表示两组形状可广播的三维向量。

    Returns
    -------
    ak.Array
        点积结果，与输入向量在事件和元素维度上对齐。
    """
    return a.px * b.px + a.py * b.py + a.pz * b.pz


def cos_theta(pairs):
    """
    计算两向量之间夹角的余弦值。

    Parameters
    ----------
    pairs : ak.Array
        由 ak.combinations 或类似算子生成的 Awkward Array，
        其中每个元素包含字段 a 和 b，分别表示两组三维向量，
        且 a、b 均包含 px、py、pz 字段。

    Returns
    -------
    ak.Array
        各向量对之间夹角的 cos(theta)，数值被限制在 [-1, 1] 区间内，
        以避免浮点误差导致的数值不稳定。
    """
    ct = dot(pairs.a, pairs.b) / (norm(pairs.a) * norm(pairs.b))
    return ak.where(ct > 1, 1, ct)


def kstar(pairs, m1: float = 938.272, m2: float = 938.272):
    """
    k*计算，根据两两配对后的结果计算

    Parameters
    ----------
    pairs : ak.Array
        将两个array combination之后的array
        这两个array各自都包含质子的px, py, pz
    m1, m2 : float
        用于计算k*的两个粒子的质量，这里都是质子的质量

    Returns
    -------
    ak.Array
        两两配对计算的k*值
    """
    # energies
    e1 = (m1**2 + pairs.a.px**2 + pairs.a.py**2 + pairs.a.pz**2) ** 0.5
    e2 = (m2**2 + pairs.b.px**2 + pairs.b.py**2 + pairs.b.pz**2) ** 0.5

    # q_inv = (p1 - p2)^2
    qinv = (
        (pairs.a.px - pairs.b.px) ** 2
        + (pairs.a.py - pairs.b.py) ** 2
        + (pairs.a.pz - pairs.b.pz) ** 2
        - (e1 - e2) ** 2
    )

    # w = (qinv + m1^2 + m2^2) / 2
    w = (qinv + m1**2 + m2**2) / 2.0

    # k*
    k = ((w**2 - m1**2 * m2**2) / (2.0 * w + m1**2 + m2**2)) ** 0.5

    return k


def cos_theta_two_ak(
    x1,
    x2,
    px: str = "TRACK_PX",
    py: str = "TRACK_PY",
    pz: str = "TRACK_PZ",
    px2: str = "TRACK_PX",
    py2: str = "TRACK_PY",
    pz2: str = "TRACK_PZ",
):
    """
    额外的一个函数，专门用于计算非“组合配对”的，相同形状的两个ak的每一条tracks的函数
    主要是为了算x和roll_tracks(x)的，
    因为去除clonetracks有一个更简单的办法：
    1. Clonetracks只发生在两个相近的tracks中，所以不需要两两配对计算所有夹角组合
    2. 取而代之，我们可以先sort ETA，如果ETA都不相邻，那不可能是tracks
    3. 所以去阅读roll_tracks和sort_tracks_by这两个函数

    """
    x1 = ak.values_astype(x1, np.float64)
    x2 = ak.values_astype(x2, np.float64)
    x1x2 = x1[px] * x2[px2] + x1[py] * x2[py2] + x1[pz] * x2[pz2]
    normx1 = (x1[px] * x1[px] + x1[py] * x1[py] + x1[pz] * x1[pz]) ** 0.5
    normx2 = (x2[px2] * x2[px2] + x2[py2] * x2[py2] + x2[pz2] * x2[pz2]) ** 0.5
    return x1x2 / normx1 / normx2
