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
np.random.seed(23)
path = uai_code_pardir

# gen dist
gen_dist = dist.Correlated_3D_1(D=3, x1_min=0, x1_max=1, x2_sigma=0, x3_sigma=0.5)
X = gen_dist.generate(N=500)
axis_limits = gen_dist.axis_limits
model = models.Example3(a1=1, a2=1, a=1)


# RHALE
dale = pythia.DALE(data=X,
                   model=model.predict,
                   model_jac=model.jacobian,
                   axis_limits=axis_limits)
binning = pythia.binning_methods.DynamicProgramming(max_nof_bins=20, min_points_per_bin=10)
dale.fit(features="all", binning_method=binning)
for feat in range(3):
    if savefig:
        pathname = os.path.join(path, "example_3", "dale_feat_" + str(feat) + ".pdf")
        dale.plot(feature=feat, confidence_interval="std", savefig=pathname, violin=True)
    else:
        dale.plot(feature=feat, confidence_interval="std", violin=True)

# PDP with ICE
pdp_ice = pythia.pdp.PDPwithICE(data=X,
                                model=model.predict,
                                axis_limits=axis_limits)
# pdp_ice.fit("all")
for feat in range(3):
    if savefig:
        pathname = os.path.join(path, "example_3", "pdp_ice_feat_" + str(feat) + ".pdf")
        pdp_ice.plot(feature=feat, normalized=True, nof_points=300, savefig=pathname)
    else:
        pdp_ice.plot(feature=feat, normalized=True, nof_points=300)


# ICE std
pdp = pythia.PDP(data=X, model=model.predict, axis_limits=axis_limits)
# pdp.fit("all")
xx = np.linspace(0, 1, 500)
for feat in range(3):
    y, std, stderr = pdp.eval(x=xx, feature=feat, uncertainty=True)
    plt.figure()
    plt.title("ICE standard deviation for feature " + str(feat))
    plt.plot(xx, std, "g", label="std")
    plt.xlabel("x_" + str(feat))
    plt.ylabel("standard deviation")
    if savefig:
        curpath = os.path.join(path, "example_3", "ice_std_" + str(feat) + ".pdf")
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
        curpath = os.path.join(path, "example_3", "dice_std_" + str(feat) + ".pdf")
        plt.savefig(curpath, bbox_inches="tight")
    plt.show(block=False)

#
#
# dale = fe.DALE(data=X,
#                model=model.predict,
#                model_jac=model.jacobian,
#                axis_limits=axis_limits)
#
# alg_params = {"bin_method" : "dp", "max_nof_bins" : 20, "min_points_per_bin": 10}
# dale.fit(features="all", alg_params=alg_params)
# y, var, stderr = dale.eval(x=np.linspace(axis_limits[0,0], axis_limits[1,0], 100),
#                            s=0,
#                            uncertainty=True)
# if savefig:
#     pathname = os.path.join(path, "example_3", "dale_feat_0.pdf")
#     dale.plot(s=0, error="std", savefig=pathname)
#     pathname = os.path.join(path, "example_3", "dale_feat_1.pdf")
#     dale.plot(s=1, error="std", savefig=pathname)
#     pathname = os.path.join(path, "example_3", "dale_feat_2.pdf")
#     dale.plot(s=2, error="std", savefig=pathname)
# else:
#     dale.plot(s=0, error="std")
#     dale.plot(s=1, error="std")
#     dale.plot(s=2, error="std")
#
# # PDP with ICE
# pdp_ice = fe.PDPwithICE(data=X,
#                         model=model.predict,
#                         axis_limits=axis_limits)
# if savefig:
#     pathname = os.path.join(path, "example_3", "pdp_ice_feat_0.pdf")
#     pdp_ice.plot(s=0, normalized=True, nof_points=300, savefig=pathname)
#     pathname = os.path.join(path, "example_3", "pdp_ice_feat_1.pdf")
#     pdp_ice.plot(s=1, normalized=True, nof_points=300, savefig=pathname)
#     pathname = os.path.join(path, "example_3", "pdp_ice_feat_2.pdf")
#     pdp_ice.plot(s=2, normalized=True, nof_points=300, savefig=pathname)
# else:
#     pdp_ice.plot(s=0, normalized=True, nof_points=300)
#     pdp_ice.plot(s=1, normalized=True, nof_points=300)
#     pdp_ice.plot(s=2, normalized=True, nof_points=300)
