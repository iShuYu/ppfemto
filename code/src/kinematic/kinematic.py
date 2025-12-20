import awkward as ak


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
