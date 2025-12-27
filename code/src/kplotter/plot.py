import numpy as np
import awkward as ak
import matplotlib.pyplot as plt
from selector import *


def get_ratio(
    kstar_same: ak.Array,
    kstar_mix: ak.Array,
    bins: int,
    range,
):
    """
    计算 same-event 与 mixed-event 的 k* 分布比值（ratio）。

    在相同的 bin 和 range 下分别对 same / mix 填充直方图，
    使用 density 归一化后的结果计算比值，
    并基于原始 bin 内计数给出 Poisson 不确定度。

    Parameters
    ----------
    kstar_same : ak.Array
        same-event 的 k* 数组
    kstar_mix : ak.Array
        mixed-event 的 k* 数组
    costheta_same : ak.Array
        same-event 对应的 cos(theta)
    costheta_mix : ak.Array
        mixed-event 对应的 cos(theta)
    bins : int
        直方图 bin 数
    range : list[float]
        直方图范围 [min, max]

    Returns
    -------
    centers : np.ndarray
        每个 bin 的中心
    ratio : np.ndarray
        density 直方图的比值
    err : np.ndarray
        ratio 的 Poisson 不确定度
    """

    x1 = ak.flatten(kstar_same, axis=None).to_numpy()
    x2 = ak.flatten(kstar_mix, axis=None).to_numpy()

    h1, edges = np.histogram(x1, bins=bins, range=range, density=False)
    h2, _ = np.histogram(x2, bins=bins, range=range, density=False)

    bin_width = edges[1] - edges[0]
    N1, N2 = h1.sum(), h2.sum()

    d1 = h1 / (N1 * bin_width)
    d2 = h2 / (N2 * bin_width)

    ratio = d1 / d2

    # Poisson uncertainty
    err = ratio * np.sqrt(1 / h1 + 1 / h2)

    mask = (h1 > 0) & (h2 > 0)
    ratio[~mask] = np.nan
    err[~mask] = np.nan

    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, ratio, err


def plot_ratio(
    centers,
    ratio,
    err,
    xlim=None,
    ylim=None,
    ax=None,
    label=None,
    color=None,
):
    """
    绘制 ratio 及其 Poisson 误差条。

    输入为已经计算好的 bin center、ratio 和误差，
    自动忽略 NaN bin，并在图中绘制 y=1 的参考线。

    Parameters
    ----------
    centers : np.ndarray
        bin 中心
    ratio : np.ndarray
        每个 bin 的 ratio
    err : np.ndarray
        ratio 的不确定度
    ax : matplotlib.axes.Axes, optional
        目标坐标轴，若为 None 则新建图像
    label : str, optional
        图例标签
    color : str, optional
        绘图颜色

    Returns
    -------
    ax : matplotlib.axes.Axes
        绘图所使用的坐标轴
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))

    # 自动屏蔽 nan
    mask = np.isfinite(ratio) & np.isfinite(err)

    ax.errorbar(
        centers[mask],
        ratio[mask],
        yerr=err[mask],
        fmt="o",
        ms=4,
        capsize=2,
        color=color,
        label=label,
    )

    ax.axhline(1.0, ls="--", lw=1, color="gray")
    ax.set_xlabel(r"$k^*$")
    ax.set_ylabel(r"C($k^*)$")
    ax.grid(alpha=0.3)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    if label is not None:
        ax.legend()

    return ax
