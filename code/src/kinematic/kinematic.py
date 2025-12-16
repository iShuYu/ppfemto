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
    return ak.sqrt(v.px**2 + v.py**2 + v.pz**2)


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
    return ak.clip(ct, -1.0, 1.0)
