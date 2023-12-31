import matplotlib.pyplot as plt

from dictionaries import *
from kepler_fitfunctions import *

# options to run code
mcmc_run = False

nburn = 500 # put back run burnin in mcmc function
nprod = 500

pars = kd_combo_priors[:, 0]
priors = kd_combo_priors

ss=True

# kepler_lc = lc(pars, kepler, SuperSample=True)

# kepler_ss = PhaseFold(kepler, pars, new_dict=True, SuperSample = True)
# kepler_no_ss = PhaseFold(kepler, pars, new_dict=True, SuperSample = False)
#
# plt.plot(kepler_ss["phase_time"], kepler_ss["lc_php"], color="red", linestyle=":")
# plt.plot(kepler_no_ss["phase_time"], kepler_no_ss["lc_php"], color="green", linestyle=":")
#
# plt.xlim(-0.03, 0.03)
# plt.scatter(kepler_ss["phasefold_results"], kepler["flux"], s=1, alpha=0.5)
# plt.plot(kepler_ss["phasefold_results"], kepler_ss["lc_php"])
# plt.plot(kepler_no_ss["phase_time"], kepler_no_ss["lc_php"])
#
# plt.show()
#sys.exit()

# can I try fitting the first 5 transits
# pars[0] = 2457346.32942 - 0.01
# pars[1] = 0.723672336 + 0.002
# lc_r003 = lc(pars, kepler)
# plt.plot(kepler["time"], lc_r003, color="red")
# plt.scatter(kepler["time"], kepler["flux"], s=1)
# #plt.xlim(kepler["time"][0], kepler["time"][1000])
# plt.show()
# sys.exit()

# fig, axes = plt.subplots(3, figsize=(5, 10))
# pars[2] = 0.03
# lc_r003 = lc(pars, kepler)
# lnp_r003 = np.sum(stats.norm.logpdf(kepler["flux"] - lc_r003, loc=0, scale=kepler["error"]))
# std_r003 = np.std(kepler["flux"] - lc_r003)
# print("RpRs = 0.03: lnp:", lnp_r003)
# ph_r003 = PhaseFold(kepler, pars, new_dict=True)
# axes[0].plot(ph_r003["phase_time"], ph_r003["lc_php"], color="red")
# axes[0].scatter(ph_r003["phasefold_results"], kepler["flux"], s=1)
# axes[0].set_title("RpRs = 0.03")
#
# pars[2] = 0.09
# lc_r009 = lc(pars, kepler)
# lnp_r009 = np.sum(stats.norm.logpdf(kepler["flux"] - lc_r009, loc=0, scale=kepler["error"]))
# std_r009 = np.std(kepler["flux"] - lc_r009)
# print("RpRs = 0.09: lnp:", lnp_r009)
# ph_r009 = PhaseFold(kepler, pars, new_dict=True)
# axes[1].plot(ph_r009["phase_time"], ph_r009["lc_php"], color="red")
# axes[1].scatter(ph_r009["phasefold_results"], kepler["flux"], s=1)
# axes[1].set_title("RpRs = 0.09")
#
# pars[2] = 0.15
# lc_r015 = lc(pars, kepler)
# lnp_r015 = np.sum(stats.norm.logpdf(kepler["flux"] - lc_r015, loc=0, scale=kepler["error"]))
# std_r015 = np.std(kepler["flux"] - lc_r015)
# print("RpRs = 0.15: lnp:", lnp_r015)
# ph_r015 = PhaseFold(kepler, pars, new_dict=True)
# axes[2].plot(ph_r015["phase_time"], ph_r015["lc_php"], color="red")
# axes[2].scatter(ph_r015["phasefold_results"], kepler["flux"], s=1)
# axes[2].set_title("RpRs = 0.15")
#
# print("-------------------------------------------")
# print("RpRs = 0.03: std:", std_r003)
# print("RpRs = 0.09: std:", std_r009)
# print("RpRs = 0.15: std:", std_r015)
#
# plt.tight_layout()
# plt.show()

# Visualization of likelihood surface========================
#keys = ["t0", "log Period", "RpRs", "log ars", "cosi", "esinw", "ecosw"]
# LNP landscape code
# nsigma = 10
#
# axis_dict = {
#     k: np.linspace(pars[i] - priors[i][1]*nsigma, pars[i] + priors[i][1]*nsigma, 100)
#     for i, k in enumerate(keys)}
#
# lnp_dict = {
#     a[0]: [lnp(list(pars[:i]) + [x] + list(pars[i:]), priors, kepler) for x in a[1]]
#     for i, a in enumerate(axis_dict.items())
# }
#
# fig, axes = plt.subplots(7, figsize = (5, 30))
# for i, key in enumerate(keys):
#     axes[i].plot(axis_dict[key], lnp_dict[key])
#     axes[i].set_title(key)
# plt.tight_layout()
# plt.show()
# plt.savefig("lnp space, initial parameters")

# runs mcmc using MCMC function if run_mcmc1
if mcmc_run:
    kepler["error"] = kepler["error"] * 2.6 # scales error by TB value?
    flat_sample = run_mcmc(pars, priors, nburn, nprod, kepler, run="K1", plot_corner=True, SuperSample=ss)

    # extract median values, standard deviation from flat sample using flatsample_pars funciton
if mcmc_run == False:
    flat_sample = np.load("./Data/flat_sample_kepler.npy")

mcmc_pars, mcmc_err = flatsample_pars(flat_sample)

kepler_lc = lc(mcmc_pars, kepler, SuperSample=ss)
kepler_ph_mcmc = PhaseFold(kepler, mcmc_pars, new_dict=True, SuperSample=ss)

plt.plot(kepler_ph_mcmc["phase_time"], kepler_ph_mcmc["lc_php"], color="red")
plt.scatter(kepler_ph_mcmc["phasefold_results"], kepler["flux"], s=1)
plt.title("phasefold lc post-mcmc ")
plt.xlim(-0.02, 0.02)
plt.show()

resid = kepler["flux"] - kepler_lc
plt.scatter(kepler_ph_mcmc["phasefold_results"], resid, s=1)
plt.title("Kepler residuals bestfit")
plt.show()

print("STD kepler resid: ", np.std(resid))

# printing parameters + their errors
labels = ["T0", "log_period", "RpRsK", "log_ars", "cosi",
          "esinw", "ecosw", "depthk"]

for i, label in enumerate(labels):
    print(labels[i], ":", mcmc_pars[i], "+/-", mcmc_err[i])
