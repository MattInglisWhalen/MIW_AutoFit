# default libraries
from typing import Callable

# external libraries


# internal classes
from autofit.src.composite_function import CompositeFunction
from autofit.src.package import logger, debug

IT_LIMIT = 25

def find_window_right(evaluator: Callable[[CompositeFunction],float], target: float,
                      model: CompositeFunction, idx) -> (float, float):
    """Finds an arg_idx such that evaluator(model(arg_0,...arg_idx,...arg_N)) > target.

    Assumes model.args give the minimum of evaluator, and from model.args[arg_idx], searches right.

    Returns (interval size, new_arg).
    """
    return _find_window_dir(evaluator, target, model, idx, +1)

def find_window_left(evaluator: Callable[[CompositeFunction],float], target: float,
                     model: CompositeFunction, idx) -> (float, float):
    """Finds an arg_idx such that evaluator(model(arg_0,...arg_idx,...arg_N)) > target.

    Assumes model.args give the minimum of evaluator, and from model.args[arg_idx], searches left.

    Returns (interval size, new_arg).
    """
    return _find_window_dir(evaluator, target, model, idx, -1)

def _find_window_dir(evaluator: Callable[[CompositeFunction], float], target: float,
                     model: CompositeFunction, idx: int, sign: int) -> (float, float):
    """Finds an arg_idx such that evaluator(model(arg_0,...arg_idx,...arg_N)) > target.

    Assumes model.args give the minimum of evaluator, and starting from x0 = model.args[arg_idx],
    it searches left or right depending on sign.

    Returns (interval size, new_arg).
    """
    tmp_model = model.copy()
    opt_val = evaluator(tmp_model)

    x0 = tmp_model.get_arg_i(idx)
    diff = max(abs(x0/2)-1e-5,1e-5)/2

    # Find domain of x -- [x0,x0+diff] or [x0-diff,x0] -- that produces a range containing target
    its = 0
    while True :

        outer_x = x0 + sign * diff

        tmp_model.set_arg_i(idx,outer_x)
        outer_eval = evaluator(tmp_model)

        debug(f"algorithms.find_window_dir(): {idx=} {sign=} {outer_eval=}")
        if outer_eval > target:
            break
        elif outer_eval + 1e-5 < opt_val :
            logger(f"algorithms.find_window_dir():\n"
                   f"We've improved on the previous minimum for {evaluator.__name__}!"
                   f" {outer_eval:.2F} at {outer_x:.3F}")
            return -1, outer_x

        its += 1
        if its > IT_LIMIT :
            logger("algorithms.find_window_dir(): ITERATION LIMIT REACHED v1")
            break

        diff *= 2

    return diff, 0

def bisect(evaluator: Callable[[CompositeFunction],float], target: float,
           model: CompositeFunction, idx: int, left_x: float, right_x: float) -> (float, int, float):
    """Finds an arg_idx such that evaluator(model(arg_0,...arg_idx,...arg_N)) = target.

    Assumes model.args give the minimum of evaluator, and does a binary search between left_x and right_x.

    Returns (x_target, error_code, new_arg).
    """
    tmp_model = model.copy()
    opt_val = evaluator(tmp_model)

    # guard against wrong ordering
    if left_x > right_x :
        left_x, right_x = right_x, left_x

    diff = right_x - left_x

    # check that it's a solvable problem
    tmp_model.set_arg_i(idx, left_x)
    left_val = evaluator(tmp_model)
    tmp_model.set_arg_i(idx, right_x)
    right_val = evaluator(tmp_model)
    debug(left_val , target , right_val )
    assert(left_val <= target <= right_val or left_val >= target >= right_val)
    sign = 1 if left_val < right_val else -1

    # do the bisection
    its = 0
    while diff > 1e-5:

        mid_x = (left_x + right_x) / 2


        tmp_model.set_arg_i(idx,mid_x)
        mid_eval = evaluator(tmp_model)


        if mid_eval + 1e-5 < opt_val:
            logger(f"algorithms.find_window_dir():\n"
                   f"We've improved on the previous minimum for {evaluator.__name__}!"
                   f" {mid_eval:.2F} at {mid_x:.3F}")
            return 0, 1, mid_x

        if sign*mid_eval > sign*target:
            right_x = mid_x
        else:
            left_x = mid_x

        diff = right_x - left_x

        its += 1
        if its > IT_LIMIT:
            break

    mid_x = (left_x + right_x) / 2
    # debug(f"algorithms.bisect(): ended with midx={mid_x} for arg {idx}")
    return mid_x, 0, 0

