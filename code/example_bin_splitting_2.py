import sys
import os

cwd = os.getcwd()
last_part = os.path.basename(os.path.normpath(cwd))
if last_part == "code":
    uai_code_pardir = cwd
else:
    uai_code_pardir = os.path.join(cwd, "code")
    sys.path.append(uai_code_pardir)

import numpy as np
import pythia
import example_models.distributions as dist
import example_models.models as models
import utils

savefig = True
folder_name = "example_bin_splitting_2"
np.random.seed(23)

path = uai_code_pardir

# gen dist
gen_dist = dist.Correlated1(D=2, x1_min=0, x1_max=1, x2_sigma=.4)
axis_limits = gen_dist.axis_limits
model = models.SquareWithInteraction(b0=0, b1=4, b2=1, b3=1)

# fit ground-truth
X_0 = gen_dist.generate(N=1000000)
dale_gt = pythia.DALE(data=X_0,
                      model=model.predict,
                      model_jac=model.jacobian,
                      axis_limits=axis_limits)

binning = pythia.binning_methods.Fixed(nof_bins=1000)
dale_gt.fit(features=0, binning_method=binning)

# fit approx with auto binning
X = gen_dist.generate(N=500)
dale = pythia.DALE(data=X,
                   model=model.predict,
                   model_jac=model.jacobian,
                   axis_limits=axis_limits)

binning = pythia.binning_methods.DynamicProgramming(max_nof_bins=40, min_points_per_bin=10)
dale.fit(features=0, binning_method=binning)
if savefig:
    path2dir = os.path.join(path, folder_name)
    savepath = os.path.join(path2dir, "fig_1.pdf") if savefig else None
    dale.plot(title="RHALE plot for $x_1$", savefig=savepath, violin=True)
else:
    dale.plot(violin=True)


# fit approximation
K_list = list(range(2,11)) + list(range(15, 101, 5))
stats_fixed = utils.measure_fixed_error(dale_gt,
                                        gen_dist,
                                        model,
                                        axis_limits,
                                        K_list,
                                        nof_iterations=10,
                                        nof_points=500)
stats_auto = utils.measure_auto_error(dale_gt,
                                      gen_dist,
                                      model,
                                      axis_limits,
                                      nof_iterations=10,
                                      nof_points=500)

# plots
path2dir = os.path.join(path, folder_name)
savepath = os.path.join(path2dir, "fig_2.pdf") if savefig else None
utils.plot_fixed_vs_auto(K_list,
                         stats_fixed["mu_err_mean"],
                         stats_fixed["mu_err_std"],
                         stats_auto["mu_err_mean"],
                         stats_auto["mu_err_mean"],
                         "mu",
                         savefig=savepath)

savepath = os.path.join(path2dir, "fig_3.pdf") if savefig else None
utils.plot_fixed_vs_auto(K_list,
                         stats_fixed["var_err_mean"],
                         stats_fixed["var_err_std"],
                         stats_auto["var_err_mean"],
                         stats_auto["var_err_mean"],
                         "var",
                         savefig=savepath)

savepath = os.path.join(path2dir, "fig_4.pdf") if savefig else None
utils.plot_fixed_vs_auto(K_list,
                         stats_fixed["rho_mean"],
                         stats_fixed["rho_std"],
                         stats_auto["rho_mean"],
                         stats_auto["rho_mean"],
                         "rho",
                         savefig=savepath)
