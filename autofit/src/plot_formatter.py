"""
A collection of utility functions for formatting matplotlib plots in the AutoFit style
"""

# built-ins libraries
from typing import Optional

# external packages
import numpy as np
from matplotlib import ticker

# internal modules


def zero_out_axes(axes):
    if axes.get_xlim()[0] > 0:
        axes.set_xlim([0, axes.get_xlim()[1]])
    elif axes.get_xlim()[1] < 0:
        axes.set_xlim([axes.get_xlim()[0], 0])
    if axes.get_ylim()[0] > 0:
        axes.set_ylim([0, axes.get_ylim()[1]])
    elif axes.get_ylim()[1] < 0:
        axes.set_ylim([axes.get_ylim()[0], 0])


def set_xaxis_format_linear(
    axes,
    x_points: Optional[list[float]] = None,
    xminmax: Optional[tuple[float, float]] = None,
):
    """
    Formats the x-axis nicely
    """
    axes.set(xscale="linear")
    axes.spines["left"].set_position(("data", 0.0))
    axes.spines["right"].set_position(("data", 0.0))
    if xminmax is None:
        xmin, xmax = min(x_points), max(x_points)
    else:
        xmin, xmax = xminmax
    log_delta_x = np.log10(xmax - xmin if xmax > xmin else 10) // 1
    if log_delta_x > 4:
        axes.xaxis.set_major_formatter(ticker.FuncFormatter(_lambda_gt_4()))
    elif log_delta_x in [0, 1, 2, 3, 4]:
        axes.xaxis.set_major_formatter(ticker.FuncFormatter(_lambda_0_to_4()))
    elif log_delta_x == -1:
        axes.xaxis.set_major_formatter(ticker.FuncFormatter(_lambda_neg1()))
    elif log_delta_x == -2:
        axes.xaxis.set_major_formatter(ticker.FuncFormatter(_lambda_neg2()))
    elif log_delta_x == -3:
        axes.xaxis.set_major_formatter(ticker.FuncFormatter(_lambda_neg3()))
    elif log_delta_x == -4:
        axes.xaxis.set_major_formatter(ticker.FuncFormatter(_lambda_neg4()))
    else:
        axes.xaxis.set_major_formatter(ticker.FuncFormatter(_lambda_lt_neg4()))


def set_xaxis_format_log(
    axes,
    x_points: Optional[list[float]] = None,
    xminmax: Optional[tuple[float, float]] = None,
):
    """
    Formats the x-axis nicely for exponential data
    """
    if xminmax is None:
        log_min, log_max = np.log(min(x_points)), np.log(max(x_points))
    else:
        log_min, log_max = tuple(np.log(xminmax))
    axes.set_xlim(
        [
            np.exp(log_min - (log_max - log_min) / 10),
            np.exp(log_max + (log_max - log_min) / 10),
        ]
    )
    axes.set(xscale="log")
    axes.spines["right"].set_visible(False)


def set_yaxis_format_linear(
    axes,
    y_points: Optional[list[float]] = None,
    yminmax: Optional[tuple[float, float]] = None,
):
    """
    Formats the y-axis nicely for linear data
    """
    axes.set(yscale="linear")
    axes.spines["top"].set_position(("data", 0.0))
    axes.spines["bottom"].set_position(("data", 0.0))
    if yminmax is None:
        ymin, ymax = min(y_points), max(y_points)
    else:
        ymin, ymax = yminmax
    log_delta_y = np.log10(ymax - ymin if ymax > ymin else 10) // 1
    if log_delta_y > 4:
        axes.yaxis.set_major_formatter(ticker.FuncFormatter(_lambda_gt_4()))
    elif log_delta_y in [0, 1, 2, 3, 4]:
        axes.yaxis.set_major_formatter(ticker.FuncFormatter(_lambda_0_to_4()))
    elif log_delta_y == -1:
        axes.yaxis.set_major_formatter(ticker.FuncFormatter(_lambda_neg1()))
    elif log_delta_y == -2:
        axes.yaxis.set_major_formatter(ticker.FuncFormatter(_lambda_neg2()))
    elif log_delta_y == -3:
        axes.yaxis.set_major_formatter(ticker.FuncFormatter(_lambda_neg3()))
    elif log_delta_y == -4:
        axes.yaxis.set_major_formatter(ticker.FuncFormatter(_lambda_neg4()))
    else:
        axes.yaxis.set_major_formatter(ticker.FuncFormatter(_lambda_lt_neg4()))


def set_yaxis_format_log(
    axes,
    y_points: Optional[list[float]] = None,
    yminmax: Optional[tuple[float, float]] = None,
):
    """
    Formats the y-axis nicely for exponential data
    """
    axes.set(yscale="log")
    if yminmax is None:
        log_min, log_max = np.log(min(y_points)), np.log(max(y_points))
    else:
        log_min, log_max = tuple(np.log(yminmax))
    axes.set_ylim(
        [
            np.exp(log_min - (log_max - log_min) / 10),
            np.exp(log_max + (log_max - log_min) / 10),
        ]
    )
    axes.spines["top"].set_visible(False)


def fix_axes_labels(axes, xmin, xmax, ymin, ymax, xlabel):

    #  proportion between xmin and xmax where the zero lies
    # x(tx) = xmin + (xmax - xmin)*tx with 0<tx<1 so
    tx = max(0.0, -xmin / (xmax - xmin))
    ty = max(0.0, -ymin / (max(ymax - ymin, 1e-5)))

    # how much of the screen is taken by the x and y spines
    offset_x, offset_y = -0.07, -0.04

    axes.xaxis.set_label_coords(1.050 - 0.005 * len(xlabel), offset_y + ty)
    axes.yaxis.set_label_coords(offset_x + tx, +0.750)


def _lambda_gt_4():
    return lambda x, pos: "" if x == 0 else f"{x:.2E}"


def _lambda_0_to_4():
    return lambda x, pos: (
        "" if x == 0 else (f"{x:.1F}" if (x - np.round(x)) ** 2 > 1e-10 else f"{int(x)}")
    )


def _lambda_neg1():
    return lambda x, pos: "" if x == 0 else f"{x:.2F}"


def _lambda_neg2():
    return lambda x, pos: "" if x == 0 else f"{x:.3F}"


def _lambda_neg3():
    return lambda x, pos: "" if x == 0 else f"{x:.4F}"


def _lambda_neg4():
    return lambda x, pos: "" if x == 0 else f"{x:.5F}"


def _lambda_lt_neg4():
    return lambda x, pos: "" if x == 0 else f"{x:.2E}"
