import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import typing


def trans_affine(x, mu, std):
    return x * std + mu


def trans_scale(x, std, square=False):
    y = x * std**2 if square else x * std
    return y


def trans_bin(x, std_x, std_y, square=False):
    y = x * (std_y / std_x) ** 2 if square else x * std_y / std_x
    return y


def ale_plot(
    ale_params: dict,
    accum_effect_func: callable,
    feature: int,
    error: typing.Union[None, str] = None,
    scale_x: typing.Union[None, dict] = None,
    scale_y: typing.Union[None, dict] = None,
    ylim = None,
    ground_truth = None,
    title: typing.Union[None, str] = None,
    savefig: typing.Union[bool, str] = False,
    violin: bool = False,
    data_effect_lst = None,
):
    """

    Parameters
    ----------
    ale_params: Dict with ale parameters
    accum_effect_func: the accumulated effect function
    feature: which feature to plot
    error: None or in ['std', 'stderr', 'both']
     - if 'std' the accumulated standard deviation is shown
     - if 'stderr' the accumulated standard error of the mean is shown
     - if 'both', std and stderr is plot
    scale_x: None or Dict with ['std', 'mean']
    scale_y: None or Dict with ['std', 'mean']
    savefig: False or path to store figure
    """
    # assert ale_params contains needed quantities
    assert all(
        name in ale_params
        for name in ["limits", "dx", "bin_estimator_variance", "bin_effect"]
    )

    x = np.linspace(ale_params["limits"][0], ale_params["limits"][-1], 1000)
    y, std, std_err = accum_effect_func(feature, x, True)

    # transform
    x = x if scale_x is None else trans_affine(x, scale_x["mean"], scale_x["std"])
    y = y if scale_y is None else trans_affine(y, scale_y["mean"], scale_y["std"])
    std = std if scale_y is None else trans_scale(std, scale_y["std"])
    std_err = std_err if scale_y is None else trans_scale(std_err, scale_y["std"])
    limits = (
        ale_params["limits"]
        if scale_x is None
        else trans_affine(ale_params["limits"], scale_x["mean"], scale_x["std"])
    )
    dx = (
        ale_params["dx"]
        if scale_x is None
        else trans_scale(ale_params["dx"], scale_x["std"])
    )
    bin_variance = (
        ale_params["bin_variance"]
        if scale_y is None
        else trans_bin(ale_params["bin_variance"], scale_x["std"], scale_y["std"], True)
    )
    bin_est_var = (
        ale_params["bin_estimator_variance"]
        if scale_y is None
        else trans_bin(ale_params["bin_estimator_variance"], scale_x["std"], scale_y["std"], True)
    )
    bin_effect = (
        ale_params["bin_effect"]
        if scale_y is None
        else trans_bin(ale_params["bin_effect"], scale_x["std"], scale_y["std"])
    )
    data_effect_lst = [data_effect_lst[i] if scale_y is None else trans_bin(data_effect_lst[i], scale_x["std"], scale_y["std"]) for i in range(len(data_effect_lst))]

    # PLOT
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    if title is None:
        ax1.set_title("RHALE plot for: $x_{" + str(feature + 1) + "}$")
    else:
        ax1.set_title(title)

    # first subplot
    ale_curve(ax1, x, y, std_err, std, ylim, ground_truth=ground_truth, error=error)

    # second subplot
    ale_bins(ax2, bin_effect, bin_variance, bin_est_var, limits, dx, error, violin=violin, data_effect_lst=data_effect_lst)

    ax1.set_ylabel("$y$")
    ax2.set_xlabel("$x_{%d}$" % (feature + 1))
    ax2.set_ylabel("$\partial y / \partial x_{%d}$" % (feature + 1))

    if savefig:
        plt.savefig(savefig, bbox_inches="tight")
    plt.show(block=False)


def ale_curve(ax1, x, y, std_err, std, ylim, ground_truth, error=None):
    ax1.plot(x, y, "b--", label="RHALE estimation")

    if ground_truth is not None:
        ax1.plot(x, ground_truth(x), "r--", label="ground truth ALE")

    if ylim is not None:
        ax1.set_ylim(ylim)

    if error == "std":
        ax1.fill_between(
            x, y - std, y + std, color="blue", alpha=0.1, label="STD"
        )
    elif error == "stderr":
        ax1.fill_between(
            x,
            y - 2 * std_err,
            y + 2 * std_err,
            color="blue",
            alpha=0.1,
            label="SE",
        )
    elif error == "both":
        ax1.fill_between(x, y - std, y + std, color="blue", alpha=0.2, label="std")
        ax1.fill_between(
            x,
            y - np.sqrt(std_err),
            y + np.sqrt(std_err),
            color="blue",
            alpha=0.1,
            label="standard error",
        )
    ax1.legend()


def ale_bins(ax2, bin_effects, bin_variance, bin_est_var, limits, dx, error, violin=False, data_effect_lst=None):
    bin_centers = (limits[:-1] + limits[1:]) / 2
    if error == "std":
        yerr = np.sqrt(bin_variance)
    elif error == "stderr":
        yerr = np.sqrt(bin_est_var)
    elif error is None:
        yerr = None

    # mean_std = np.sum(np.sqrt(bin_variance) * dx) / np.sum(dx)

    yerr = np.sqrt(bin_variance) if error is not None else None
    ax2.bar(
        x=bin_centers,
        height=bin_effects,
        width=dx,
        color=(0.1, 0.1, 0.1, 0.1),
        edgecolor="blue",
        yerr=yerr,
        align="center",
        capsize=2,
        ecolor="green",
        label="bin effect",
    )
    if error is not None:
        ax2.plot([], [], "|", label="bin std", color="green")
        # ax2.scatter([], [], label="mean std: %.3f" % mean_std)

        # get handles and labels
        handles, labels = plt.gca().get_legend_handles_labels()

        # specify order of items in legend
        order = [0, 1]

        # add legend to plot
        ax2.legend([handles[idx] for idx in order], [labels[idx] for idx in order])
    else:
        ax2.legend()

    if violin:
        # xx = np.concatenate([[bin_centers[i]] * len(data_effect_lst[i]) for i in range(len(data_effect_lst))])
        # yy = np.concatenate(data_effect_lst)

        # if data_effect_lst is empty, skip
        bin_centers_1 = []
        data_effect_lst_1 = []
        lenghts = []
        for i in range(len(data_effect_lst)):
            if len(data_effect_lst[i]) > 0:
                bin_centers_1.append(bin_centers[i])
                data_effect_lst_1.append(data_effect_lst[i])
                lenghts.append(limits[i+1] - limits[i])
        bin_centers_1 = np.array(bin_centers_1)
        lenghts = np.array(lenghts)
        ax2.violinplot(dataset=data_effect_lst_1, positions=bin_centers_1, widths=0.5*lenghts, points=15, showmeans=True, showextrema=False, showmedians=False)




def plot_1d(x, feature, eval, confidence=None, title=None):
    plt.figure()
    plt.title(title)

    assert confidence in [None, "std", "stderr"]
    if confidence is None:
        plt.plot(x, eval(feature, x, uncertainty=False), "b-")
    elif confidence == "std":
        y, std, var_est = eval(feature, x, uncertainty=True)
        plt.plot(x, y, "b-", label="$\hat{f}_\mu$")
        plt.fill_between(
            x, y - std, y + std, color="red", alpha=0.4, label="$\hat{f}_{\sigma}$"
        )
    elif confidence == "stderr":
        y, std, var_est = eval(feature, x, uncertainty=True)
        stderr = 2*np.sqrt(var_est)
        plt.plot(x, y, "b-", label="mean PDP")
        plt.fill_between(x, y - stderr, y + stderr, color="red", alpha=0.4, label="std err")
    plt.show(block=False)


def plot_pdp_ice(x, feature, y_pdp, y_ice, title, normalize, scale_x=None, scale_y=None, ylim=None,
                 ground_truth=None, savefig=None):
    plt.figure()
    plt.title(title)
    if normalize:
        y_ice_outputs = [y_ice[i].eval(feature, x) for i in range(len(y_ice))]
        y_pdp_output = y_pdp.eval(feature, x)
    else:
        y_ice_outputs = [y_ice[i]._eval_unnorm(feature, x) for i in range(len(y_ice))]
        y_pdp_output = y_pdp._eval_unnorm(feature, x)
    y_ice_outputs = np.array(y_ice_outputs)
    y_pdp_output = np.array(y_pdp_output)

    x = x if scale_x is None else trans_affine(x, scale_x["mean"], scale_x["std"])
    y_pdp_output = (
        y_pdp_output
        if scale_y is None
        else trans_affine(y_pdp_output, scale_y["mean"], scale_y["std"])
    )
    y_ice_outputs = (
        y_ice_outputs
        if scale_y is None
        else trans_affine(y_ice_outputs, scale_y["mean"], scale_y["std"])
    )

    plt.plot(x, y_ice_outputs[0, :], color="green", alpha=0.1, label="ICE")
    plt.plot(x, y_ice_outputs.T, color="green", alpha=0.1)
    plt.plot(x, y_pdp_output, color="blue", label="PDP")

    if ground_truth is not None:
        plt.plot(x, ground_truth(x), "r--", label="ground truth PDP")

    plt.ylim(ylim)
    plt.xlabel("$x_{%d}$" % (feature + 1))
    plt.ylabel("$y$")
    plt.legend()
    if savefig is not None:
        plt.savefig(savefig, bbox_inches="tight")

    plt.show(block=False)

