"""
Evaluates the accuracy of autofit's model identification
"""

# built-in libraries
import random

# external packages
import numpy as np

# user-defined classes
from autofit.src.optimizer import Optimizer
from autofit.src.datum1D import Datum1D
from autofit.src.package import logger, performance


def get_success_failure(fn, base_optimizer) -> int:
    """
    Generates data from the functional form of fn, then creates a new optimizer with the
    same list of functional models as base_optimizer. The new optimizer fits the generated data
    to all possible models in the list, and if the top model is the same as fn the trial is
    rated as a success (return 1). A failure returns a 0. If no data could be generated, this
    returns -1.
    """
    # create dataset
    xs = (
        list(np.arange(-10, -1, 1))
        + list(np.arange(-1, 1 + 0.1, 0.101))
        + list(np.arange(1, 10 + 1, 1))
    )
    for degree in range(fn.dof):
        rng = 0
        this_node = fn.get_nodes_with_freedom()[degree]
        parent_node = this_node.parent
        parent_name = ""
        if parent_node is not None:
            parent_name = parent_node.prim.name

        while abs(rng) < 0.2:
            rng = random.random() * 10 - 5
            # certain primitives need to be positive

        # particular funtions need particular scaling to show up well in a plot
        # -- which is what a good data collector would ensure
        if this_node.prim.name in ["exp_dim1", "n_exp_dim2", "dim0_pow2"]:
            rng = abs(rng)
            if this_node.prim.name == "exp_dim1":
                rng = rng / 5
        if parent_name in ["my_sin", "my_cos"]:
            rng = np.sign(rng) * 0.2 + rng / 6
        if this_node.prim.name == "pow_neg1" and this_node.num_children() == 0:
            rng *= 0.1
        fn.set_arg_i(degree, val=rng)
    if fn.name == "Sigmoid":
        fn.set_arg_i(1, 9.5)
    if fn.name == "my_log(pow1·pow1)":
        fn.set_arg_i(1, abs(fn.args[1]))

    ys = [fn.eval_at(x) * (1 + (random.normalvariate(0, 1)) / 20) for x in xs]
    # miny, maxy = min(ys), max(ys)
    # errors = [abs(fn.eval_at(x))/20 + (maxy-miny)/20 for x in xs]
    logger(f"Testing {fn.name} with {fn.args}")
    # print(f" and \n{xs=} and \n{ys=}")

    end = len(ys) - 1
    for idx, y in enumerate(reversed(ys[:])):
        if np.isnan(y) or abs(y) > 1e5:
            del xs[end - idx]
            del ys[end - idx]

    if len(ys) == 0:
        print(f"Failed to get data for {fn}")
        return -1

    test_data = [Datum1D(x, y) for x, y in zip(xs, ys)]

    # fit fresh optimizer to dataset
    test_opt = Optimizer(data=test_data, regen=False)
    test_opt.composite_function_list = base_optimizer.composite_function_list
    test_opt.find_best_model_for_dataset()

    # check whether the top model is correct
    if test_opt.top_model.name == fn.name:
        return 1
    return 0


@performance
def find_accuracy_of_optimizer(max_depth=2, n_repeats=1):
    """
    Finds the accuracy of the backend optimizer across functional models
    with max_depth. Repeats for each trial n_repeats times.
    """

    opt_for_models = Optimizer(
        use_functions_dict=Optimizer.all_defaults_on_dict(), max_functions=max_depth
    )
    opt_for_models.build_composite_function_list()

    test_fns = opt_for_models.composite_function_list
    del test_fns[-2]  # Normal
    successes = []
    random.seed(42)
    print(f"\nChecking accuracy across {len(test_fns)} models")
    for fn in test_fns:

        fn_successes = []
        for _ in range(n_repeats):
            test_outcome = get_success_failure(fn, base_optimizer=opt_for_models)
            if test_outcome < 0:
                # this happens when there's no valid y values in the repeat
                continue
            fn_successes.append(test_outcome)

        successes.extend(fn_successes)
        print(
            f"{sum(successes)}/{len(successes)} = {np.mean(successes) * 100:.2f}% -- "
            f"{np.mean(fn_successes):.3f} : [{fn.name}]"
        )

    acc = np.mean(successes)
    print("Summary:")
    print(
        f"Total tests complete: "
        f"{len(successes)}/{len(test_fns)*n_repeats} "
        f"= {(100/n_repeats)*len(successes)/len(test_fns):.2f}%"
    )

    print(f"Total percent successful: {sum(successes)}/{len(successes)} = {acc*100:.2f}%")
    print(f"Binary-equivalent accuracy: {acc ** (1/(len(test_fns)-1))*100:.2f}%")


if __name__ == "__main__":
    for depth in range(4):
        find_accuracy_of_optimizer(depth + 1, n_repeats=5)
