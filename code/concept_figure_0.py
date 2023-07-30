import sys
import os
cwd = os.getcwd()
last_part = os.path.basename(os.path.normpath(cwd))
if last_part == "code":
    uai_code_pardir = cwd
else:
    uai_code_pardir = os.path.join(cwd, "code")
    sys.path.append(uai_code_pardir)

import pythia
import numpy as np
import matplotlib.pyplot as plt

def f(x):
    """Evaluate function:
    y = sin(2*pi*x1)*(if x1<0) - 2*sin(2*pi*x1)*(if x3<0) + x1*x2 + x2

    """
    y = np.zeros_like(x[:,0])

    y = y + 0.2*x[:, 0] - 5*x[:, 1]

    ind = x[:, 2] >= 0
    y[ind] += 10 * x[:, 1]

    # add some noise
    noise = np.random.normal(0, 1, size=y.shape)
    y += noise
    return y


def dfdx(x):
    """Evaluate jacobian of:
    y = sin(2*pi*x1)*(if x1<0) - 2*sin(2*pi*x1)*(if x3<0) + x1*x2 + x2

    dy/dx1 = 2*pi*x1*cos(2*pi*x1)*(if x1<0) - 4*pi*x1*cos(2*pi*x1)*(if x3<0) + x2
    dy/dx2 = x1 + 1
    dy/dx3 = 0
    """

    dydx = np.zeros_like(x)
    dydx[:, 0] = 0.2
    dydx[:, 1] = -5

    ind = x[:, 2] >= 0
    dydx[ind, 1] += 10
    return dydx


def generate_samples(N):
    def gen_unif():
        y = np.concatenate((np.array([-1]),
                            np.random.uniform(-1, 1, size=int(N - 2)),
                            np.array([1])))
        return y

    x1 = gen_unif()
    x2 = gen_unif()
    x3 = gen_unif()
    x = np.stack([x1, x2, x3], -1)
    return x


def simple_ale_plot(x, y, xlabel=None, title=None, savefig=None):
    plt.figure()
    plt.ylim(-10, 10)
    plt.plot(x, ale_gt(x), "r--", label="ground truth ALE")
    plt.plot(x, y, "b--", label="ALE estimation")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("$y$")
    plt.legend()
    if savefig:
        plt.savefig(savefig, bbox_inches="tight")
    plt.show(block=False)


def ale_gt(x):
    y = np.zeros_like(x)
    return y


# main part
# np.random.seed(seed=21)
# axis_limits = np.array([[-1, 1], [-1, 1], [-1, 1]]).T
# N = 100
# x = generate_samples(N)
# dirpath = os.path.realpath(os.path.join(uai_code_pardir, "concept_figure_0"))

# # DALE
# for feat in [0, 1, 2]: #, 1, 2]:
#     xx = np.linspace(-1, 1, 100)

#     # simple ale
#     dale = pythia.DALE(data=x, model=f, model_jac=dfdx, axis_limits=axis_limits)
#     binning = pythia.binning_methods.Fixed(nof_bins=100)
#     dale.fit([feat], binning_method=binning)
#     yy = dale.eval(feature=feat, x=xx, uncertainty=False)
#     simple_ale_plot(xx, yy, xlabel="$x_" + str(feat + 1) + "$", title="ALE",
#                     savefig=os.path.join(dirpath, "ale_N_{}_feat_{}_bins_{}.pdf".format(N, feat, 100)))

#     # dale with fixed bins, no err
#     dale = pythia.DALE(data=x, model=f, model_jac=dfdx, axis_limits=axis_limits)
#     binning = pythia.binning_methods.DynamicProgramming(max_nof_bins=20, min_points_per_bin=10, discount=0.5)
#     dale.fit([feat], binning_method=binning)
#     dale.plot(feature=feat,
#               confidence_interval="std",
#               title="RHALE",
#               ground_truth=ale_gt,
#               savefig=os.path.join(dirpath, "rhale_N_{}_feat_{}.pdf".format(N, feat)),
#               violin=True)


np.random.seed(seed=21)
axis_limits = np.array([[-1, 1], [-1, 1], [-1, 1]]).T
N = 100
x = generate_samples(N)
dirpath = os.path.realpath(os.path.join(uai_code_pardir, "concept_figure_0"))

# DALE
for feat in [0, 1, 2]: #, 1, 2]:

    xx = np.linspace(-1, 1, 100)

    # simple ale
    nof_bins = 20
    dale = pythia.DALE(data=x, model=f, model_jac=dfdx, axis_limits=axis_limits)
    binning = pythia.binning_methods.Fixed(nof_bins=nof_bins)
    dale.fit([feat], binning_method=binning)
    yy = dale.eval(feature=feat, x=xx, uncertainty=False)
    simple_ale_plot(xx, yy, xlabel="$x_" + str(feat + 1) + "$", title="ALE",
                    savefig=os.path.join(dirpath, "ale_N_{}_feat_{}_bins_{}.pdf".format(N, feat, 100)))

    # ALE with fixed bins and error bars
    dale = pythia.DALE(data=x, model=f, model_jac=dfdx, axis_limits=axis_limits)
    binning = pythia.binning_methods.Fixed(nof_bins=nof_bins)
    dale.fit([feat], binning_method=binning)
    dale.plot(feature=feat,
              confidence_interval="std",
              title="ALE with heterogeneity",
              ground_truth=ale_gt,
              savefig=os.path.join(dirpath, "ale_with_heter_N_{}_bins_{}_feat_{}.pdf".format(N, nof_bins, feat)),
              violin=True)

    # dale with fixed bins, no err
    dale = pythia.DALE(data=x, model=f, model_jac=dfdx, axis_limits=axis_limits)
    binning = pythia.binning_methods.DynamicProgramming(max_nof_bins=20, min_points_per_bin=10, discount=0.5)
    dale.fit([feat], binning_method=binning)
    dale.plot(feature=feat,
              confidence_interval="std",
              title="RHALE",
              ground_truth=ale_gt,
              savefig=os.path.join(dirpath, "rhale_N_{}_feat_{}.pdf".format(N, feat)),
              violin=True)
