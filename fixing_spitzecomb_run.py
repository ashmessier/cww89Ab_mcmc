from fixing_dictionaries import *
from fixing_spitzercomb_functions import *

# options to run code
run_mcmc_1 = False

nburn = 500 # put back run burnin in mcmc function
nprod = 2000

pars = sd_combo_priors[:, 0]
priors = sd_combo_priors

# making dictionary of pars/priors
# pars_to_use = ["T0", "log_period", "RpRs1", "RpRs2", "RpRsK", "log_ars", "cosi",
#           "esinw", "ecosw", "slope1", "slope2"]
#
# priors_to_apply = ["T0", "log_period", "log_ars", "cosi",
#           "esinw", "ecosw"]
#
# pars = {}
# priors = {}
# for i, label in enumerate(pars_to_use):
#     pars[label] = pars_arr[i]
#     priors[label] = priors_arr[i]

# make complete model (LC, ramp, BLISS) + add to data dictionary
ch1_bliss_lc = lc(pars, ch1, ch=1)
ch1_ramp = ramp(pars[8], ch1)
ch1_bliss_model = ch1_bliss_lc * ch1_ramp
ch1_blissfunc = BLISSmodel(ch1, ch1_bliss_model, plotgrid=False, returnfunc=True)
ch1_blissflux = ch1_blissfunc.ev(ch1["xpos"], ch1["ypos"])
ch1["pre_mcmc_lc"] = ch1_bliss_model * ch1_blissflux

ch2_bliss_lc = lc(pars, ch2, ch=2)
ch2_ramp = ramp(pars[9], ch2)
ch2_bliss_model = ch1_bliss_lc * ch2_ramp
ch2_blissfunc = BLISSmodel(ch2, ch2_bliss_model, plotgrid=False, returnfunc=True)
ch2_blissflux = ch2_blissfunc.ev(ch2["xpos"], ch2["ypos"])
ch2["pre_mcmc_lc"] = ch2_bliss_model * ch2_blissflux

# calculating residuals of detrended data + add to data dictionary
ch1["pre_mcmc_residuals"] = ch1["flux"] / ch1["pre_mcmc_lc"]
ch2["pre_mcmc_residuals"] = ch2["flux"] / ch2["pre_mcmc_lc"]
# Scale errors on Spitzer data
    # Calculate standard deviation of residuals
ch1_resid_std = np.std(ch1["pre_mcmc_residuals"])
ch2_resid_std = np.std(ch2["pre_mcmc_residuals"])
    # Calculate mean of error
ch1_err_mean = np.mean(ch1["error_raw"])
ch2_err_mean = np.mean(ch2["error_raw"])
    # scale = std_residuals / mean_err
ch1_scale = ch1_resid_std / ch1_err_mean
ch2_scale = ch2_resid_std / ch2_err_mean

    # scaled error = error * scale
ch1["error"] = ch1["error_raw"] * ch1_scale
ch2["error"] = ch2["error_raw"] * ch2_scale

print("lnp pre mcmc1:", lnp(pars, priors, ch1, ch2))
# runs mcmc using MCMC function if run_mcmc1
if run_mcmc_1:
    # print("lnp errscale_mcmc1:", lnp(pars, priors, ch1, ch2, kepler))
    flat_sample1 = run_mcmc(pars, priors, nburn, nprod, ch1, ch2 ,plot_corner=True, run="spitzer_only")

if run_mcmc_1 == False:
    flat_sample1 = np.load("flat_sample_spitzer_only.npy")

# extract median values, standard deviation from flat sample using flatsample_pars funciton
mcmc_pars1, mcmc_errs = flatsample_pars(flat_sample1)

print("lnp pos mcmc1:", lnp(mcmc_pars1, priors, ch1, ch2))
# print("T0:", mcmc_pars1[0] ,"vs", pars[0])
# print("period", 10**mcmc_pars1[1], "vs", 10**pars[1])

# plt.scatter(ch2["time"], ch2["flux"], s=1)
# plt.plot(ch2["time"], lc(mcmc_pars1, ch2, ch=2))
# # plt.xlim(kepler["time"][0], kepler["time"][500])
# plt.show()
#
# plt.scatter(ch1["time"], ch1["flux"], s=1)
# plt.plot(ch1["time"], lc(mcmc_pars1, ch1, ch=1))
# # plt.xlim(kepler["time"][0], kepler["time"][500])
# plt.show()
#
# #plots raw kepler data with bestfit lightcurve to ensure period, t0 is accurate visually
# plt.scatter(kepler["time"], kepler["flux"], s=1)
# plt.plot(kepler["time"], lc(mcmc_pars1, kepler, ch="kepler"))
# plt.xlim(kepler["time"][0], kepler["time"][500])
# plt.show()

 # phasefolds kepler data using bestfit period, t0

# calculates bestfit model for all three datasets
ch1["final_mcmc_lc"] = lc(mcmc_pars1, ch1, ch=1)
ch1_ramp = ramp(mcmc_pars1[8], ch1)
ch1_bliss_model = ch1_bliss_lc * ch1_ramp
ch1_blissfunc = BLISSmodel(ch1, ch1_bliss_model, plotgrid=False, returnfunc=True)
ch1_blissflux = ch1_blissfunc.ev(ch1["xpos"], ch1["ypos"])
mcmc2_lc_all1 = ch1_bliss_model * ch1_blissflux

ch2["final_mcmc_lc"] = lc(mcmc_pars1, ch2, ch=2)
ch2_ramp = ramp(mcmc_pars1[9], ch2)
ch2_bliss_model = ch1_bliss_lc * ch2_ramp
ch2_blissfunc = BLISSmodel(ch2, ch2_bliss_model, plotgrid=False, returnfunc=True)
ch2_blissflux = ch2_blissfunc.ev(ch2["xpos"], ch2["ypos"])
mcmc2_lc_all2 = ch2_bliss_model * ch2_blissflux


# for SPITZER data:
    # Divide out BLISSflux and ramp from raw flux
ch1["mcmc2_detrended_flux"] = ch1["flux"] / ch1_ramp / ch1_blissflux
ch2["mcmc2_detrended_flux"] = ch2["flux"] / ch2_ramp / ch2_blissflux

# plots final mcmc lc over detrended data, phasefolded data?
plt.figure(figsize=(10, 5))

plt.subplot(121)
plt.title("ch 1 post-mcmc2 lc")
plt.scatter(ch1["time"], ch1["mcmc2_detrended_flux"], s=1)
plt.plot(ch1["time"], ch1["final_mcmc_lc"], color="red")

plt.subplot(122)
plt.title("ch 2 post-mcmc2 lc")
plt.scatter(ch2["time"], ch2["mcmc2_detrended_flux"], s=1)
plt.plot(ch2["time"], ch2["final_mcmc_lc"], color="red")

plt.show()

# prints transit depths + errors
indices = [10, 11]
labels = ["T0", "log_period", "RpRs1", "RpRs2", "log_ars", "cosi",
          "esinw", "ecosw", "slope1", "slope2", "depth1", "depth2"]

for i in indices:
    print(labels[i], ":", mcmc_pars1[i], "+/-", mcmc_errs[i])

# save MCMC_2 detrended lightcurves, blissfluxes, final parameters, data to npz file for use in plotting code

"""TO DO

Fix fitting somehow so it centers on right T0 
Figure out how to go from phasefold -> regular LC (or a work-around)
Email Thomas tomorrow with updates (hopefully good!)"""

for i, label in enumerate(labels):
    print(label)
    print("pars:", priors[i])
    print("mcmc:", mcmc_pars1[i])