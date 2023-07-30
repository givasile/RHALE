import sys
import os
import pythia
from pythia import visualization as vis
import numpy as np
import matplotlib.pyplot as plt

def add_parent_path():
    cwd = os.getcwd()
    if os.path.split(os.getcwd())[-1] == "mdale":
        path_to_examples = os.path.join(cwd, "examples/")
        sys.path.append(cwd)
    elif os.path.split(os.getcwd())[-1] == "examples":
        path_to_examples = cwd
        sys.path.append(os.path.split(cwd)[0])
    else:
        raise OSError

    return path_to_examples


def compare(dale_gt, dale):
    """Returns the mean error (error per bin) for mu, var, res

    :param dale_gt:
    :param dale:
    :returns:

    """
    limits_gt = dale_gt["limits"]
    limits = dale["limits"]
    ind = np.digitize(limits, limits_gt) - 1

    rho_list = []
    mu_err_list = []
    var_err_list = []
    for i in range(limits.shape[0] - 1):
        means = dale_gt["bin_effect"][ind[i]+1:ind[i+1]]
        variances = dale_gt["bin_variance"][ind[i]+1:ind[i+1]]

        rho_bin = means.std()
        rho_list.append(rho_bin)

        mu_gt = means.mean()
        mu_est = dale["bin_effect"][i]
        mu_err_list.append(np.abs(mu_gt - mu_est))

        var_gt = np.sqrt(variances.mean())
        var_est = np.sqrt(dale["bin_variance"][i])
        var_err_list.append(np.abs(var_gt - var_est))

    mean_rho = np.mean(rho_list)
    mean_mu_err = np.mean(mu_err_list)
    mean_var_err = np.mean(var_err_list)
    return mean_rho, mean_mu_err, mean_var_err


def measure_fixed_error(dale_gt, gen_dist, model, axis_limits, K_list, nof_iterations, nof_points):
    # fit approximation
    rho_mean = []
    rho_std = []

    mu_mean = []
    mu_std = []

    var_mean = []
    var_std = []

    # for fixed
    for i, k in enumerate(K_list):
        rho_tmp = []
        mu_tmp = []
        var_tmp = []
        for l in range(nof_iterations):
            X = gen_dist.generate(N=nof_points)
            dale = pythia.DALE(data=X,
                           model=model.predict,
                           model_jac=model.jacobian,
                           axis_limits=axis_limits)
            binning = pythia.binning_methods.Fixed(nof_bins=k, min_points_per_bin=2)

            try:
                dale.fit(features=[0], binning_method=binning)
                res_err, mu_err, var_err = compare(dale_gt.feature_effect["feature_0"],
                                                   dale.feature_effect["feature_0"])
                rho_tmp.append(res_err)
                mu_tmp.append(mu_err)
                var_tmp.append(var_err)
            except:
                print("exception")
                pass

        rho_mean.append(np.mean(rho_tmp))
        rho_std.append(np.std(rho_tmp))

        mu_mean.append(np.mean(mu_tmp))
        mu_std.append(np.std(mu_tmp))

        var_mean.append(np.mean(var_tmp))
        var_std.append(np.std(var_tmp))

    stats = {"rho_mean": rho_mean,
             "rho_std": rho_std,
             "mu_err_mean": mu_mean,
             "mu_err_std": mu_std,
             "var_err_mean": var_mean,
             "var_err_std": var_std}

    return stats


def measure_fixed_error_real_dataset(dale_gt,
                                     data,
                                     model,
                                     model_grad,
                                     axis_limits,
                                     K_list,
                                     nof_iterations,
                                     nof_points,
                                     feature):

    # fit approximation
    rho_mean = []
    rho_std = []

    mu_mean = []
    mu_std = []
    var_mean = []
    var_std = []

    # for fixed
    for i, k in enumerate(K_list):
        rho_tmp = []
        mu_tmp = []
        var_tmp = []
        for l in range(nof_iterations):
            ind = np.random.choice(data.shape[0], size=nof_points, replace=False)
            X = data[ind]
            dale = pythia.DALE(data=X,
                               model=model,
                               model_jac=model_grad,
                               axis_limits=axis_limits)
            binning = pythia.binning_methods.Fixed(nof_bins=k)
            dale.fit(features=[feature], binning_method=binning)

            try:
                dale.fit(features=[feature], binning_method=binning)
                res_err, mu_err, var_err = compare(dale_gt.feature_effect["feature_" + str(feature)],
                                                   dale.feature_effect["feature_" + str(feature)])
                rho_tmp.append(res_err)
                mu_tmp.append(mu_err)
                var_tmp.append(var_err)
            except:
                print("exception")
                pass

        rho_mean.append(np.mean(rho_tmp))
        rho_std.append(np.std(rho_tmp))

        mu_mean.append(np.mean(mu_tmp))
        mu_std.append(np.std(mu_tmp))

        var_mean.append(np.mean(var_tmp))
        var_std.append(np.std(var_tmp))

    stats = {"rho_mean": rho_mean,
             "rho_std": rho_std,
             "mu_err_mean": mu_mean,
             "mu_err_std": mu_std,
             "var_err_mean": var_mean,
             "var_err_std": var_std}

    return stats


def measure_auto_error(dale_gt, gen_dist, model, axis_limits, nof_iterations, nof_points):
    rho = []
    mu = []
    var = []
    for l in range(nof_iterations):
        X = gen_dist.generate(N=nof_points)
        dale = pythia.DALE(data=X,
                       model=model.predict,
                       model_jac=model.jacobian,
                       axis_limits=axis_limits)
        binning = pythia.binning_methods.DynamicProgramming(max_nof_bins=20, min_points_per_bin=10)
        dale.fit(features=[0], binning_method=binning)
        res_err, mu_err, var_err = compare(dale_gt.feature_effect["feature_0"],
                                           dale.feature_effect["feature_0"])

        rho.append(res_err)
        mu.append(mu_err)
        var.append(var_err)

    stats = {"rho_mean": np.mean(rho),
             "rho_std": np.std(rho),
             "mu_err_mean": np.mean(mu),
             "mu_err_std": np.std(mu),
             "var_err_mean": np.mean(var),
             "var_err_std": np.std(var)}
    return stats


def measure_auto_error_real_dataset(dale_gt,
                                    data,
                                    model,
                                    model_grad,
                                    axis_limits,
                                    nof_iterations,
                                    nof_points,
                                    feature):
    rho = []
    mu = []
    var = []
    for l in range(nof_iterations):
        ind = np.random.choice(data.shape[0], size=nof_points, replace=False)
        X = data[ind]
        dale = pythia.DALE(data=X,
                       model=model,
                       model_jac=model_grad,
                       axis_limits=axis_limits)

        binning = pythia.binning_methods.DynamicProgramming(max_nof_bins=20, min_points_per_bin=30)
        dale.fit(features=[feature], binning_method=binning)
        res_err, mu_err, var_err = compare(dale_gt.feature_effect["feature_" + str(feature)],
                                           dale.feature_effect["feature_" + str(feature)])

        rho.append(res_err)
        mu.append(mu_err)
        var.append(var_err)

    stats = {"rho_mean": np.mean(rho),
             "rho_std": np.std(rho),
             "mu_err_mean": np.mean(mu),
             "mu_err_std": np.std(mu),
             "var_err_mean": np.mean(var),
             "var_err_std": np.std(var)}
    return stats


def plot_fixed_vs_auto(K_list, fixed_mean, fixed_std, auto_mean, auto_std, metric,
                       scale_x = None,
                       scale_y = None,
                       savefig=None):

    def scale(x, scale_x, scale_y):
        return x * scale_y / scale_x

    fixed_mean = np.array(fixed_mean)
    fixed_mean = fixed_mean if scale_y is None else scale(fixed_mean, scale_x, scale_y)

    fixed_std = np.array(fixed_std)
    fixed_std = fixed_std if scale_y is None else scale(fixed_std, scale_x, scale_y)

    auto_mean = np.array(auto_mean)
    auto_mean = auto_mean if scale_y is None else scale(auto_mean, scale_x, scale_y)

    auto_std = np.array(auto_std)
    auto_std = auto_std if scale_y is None else scale(auto_std, scale_x, scale_y)

    plt.figure()
    assert metric in ["rho", "mu", "var"]
    if metric == "rho":
        plt.ylabel(r"$\log \mathcal{L}_{\mathtt{K}}^{\rho}$")
        plt.title(r"log mean bin-residual error($\log \mathcal{L}_{\mathtt{K}}^{\rho}$)")
    elif metric == "var":
        plt.ylabel("$\log \mathcal{L}_{\mathtt{K}}^{\sigma}$")
        plt.title("log mean bin-uncertainty error ($\log \mathcal{L}_{\mathtt{K}}^{\sigma}$)")
    elif metric == "mu":
        plt.ylabel("$\log \mathcal{L}_{\mathtt{K}}^{\mu}$")
        plt.title("log mean bin-effect error ($\log \mathcal{L}_{\mathtt{K}}^{\mu}$)")

    plt.plot(K_list, np.log(fixed_mean), 'x', label="ALE")
    plt.plot(K_list,
             np.log(np.repeat(auto_mean, len(K_list))),
             "r--",
             label="RHALE")

    # plt.errorbar(K_list, fixed_mean, yerr=fixed_std, fmt="", linestyle='', label="fixed-size")
    # plt.plot(K_list,
    #          np.repeat(auto_mean, len(K_list)),
    #          "r--",
    #          label="UALE (variable-size)")

    plt.xlabel("$K$")

    plt.legend()
    if savefig is not None:
        plt.savefig(savefig, bbox_inches="tight")
    plt.show(block=False)
