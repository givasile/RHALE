import sys
import os
import matplotlib.pyplot as plt

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

savefig = True
np.random.seed(21)
path = uai_code_pardir

# gen dist
gen_dist = dist.Correlated_3D_1(D=3, x1_min=0, x1_max=1, x2_sigma=0, x3_sigma=0.5)
X = gen_dist.generate(N=500)
axis_limits = gen_dist.axis_limits
model = models.Example3(a1=2, a2=.5, a=0)

# RHALE
dale = pythia.DALE(data=X,
                   model=model.predict,
                   model_jac=model.jacobian,
                   axis_limits=axis_limits)
binning = pythia.binning_methods.DynamicProgramming(max_nof_bins=20, min_points_per_bin=10)
dale.fit(features="all", binning_method=binning)
for feat in range(3):
    if savefig:
        pathname = os.path.join(path, "example_2", "dale_feat_" + str(feat) + ".pdf")
        dale.plot(feature=feat, confidence_interval="std", savefig=pathname, violin=True)
    else:
        dale.plot(feature=feat, confidence_interval="std", violin=True)

# PDP with ICE
pdp_ice = pythia.pdp.PDPwithICE(data=X,
                                model=model.predict,
                                axis_limits=axis_limits)
pdp_ice.fit("all", normalize="zero_start")
for feat in range(3):
    if savefig:
        pathname = os.path.join(path, "example_2", "pdp_ice_feat_" + str(feat) + ".pdf")
        pdp_ice.plot(feature=feat, normalized=True, nof_points=300, savefig=pathname)
    else:
        pdp_ice.plot(feature=feat, normalized=True, nof_points=300)


# ICE std
pdp = pythia.PDP(data=X, model=model.predict, axis_limits=axis_limits)
pdp.fit("all", normalize="zero_start")
xx = np.linspace(0, 1, 500)
for feat in range(3):
    y, std, stderr = pdp.eval(x=xx, feature=feat, uncertainty=True)
    plt.figure()
    plt.title("ICE standard deviation for feature " + str(feat))
    plt.plot(xx, std, "g", label="std")
    plt.xlabel("x_" + str(feat))
    plt.ylabel("standard deviation")
    if savefig:
        curpath = os.path.join(path, "example_2", "ice_std_" + str(feat) + ".pdf")
        plt.savefig(curpath, bbox_inches="tight")
    plt.show(block=False)

# dICE std
dpdp = pythia.pdp.dPDP(data=X, model=model.predict, model_jac=model.jacobian, axis_limits=axis_limits)
xx = np.linspace(0, 1, 500)
for feat in range(3):
    y, std, stderr = dpdp.eval(x=xx, feature=feat, uncertainty=True)
    plt.figure()
    plt.title("d-ICE standard deviation for feature " + str(feat))
    plt.plot(xx, std, "g", label="std")
    plt.xlabel("x_" + str(feat))
    plt.ylabel("standard deviation")
    if savefig:
        curpath = os.path.join(path, "example_2", "dice_std_" + str(feat) + ".pdf")
        plt.savefig(curpath, bbox_inches="tight")
    plt.show(block=False)

