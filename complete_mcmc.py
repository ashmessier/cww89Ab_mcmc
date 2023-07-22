from fixing_dictionaries import *
from fixing_fitfunctions import *

# options to run code
run_mcmc_1 = True
#run_mcmc_2 = False

nburn = 500 # put back run burnin in mcmc function
nprod = 3000

pars = pars_lastrun
priors = combo_priors

ss=True # SuperSample

# make complete model (LC, ramp, BLISS) + add to data dictionary
ch1_bliss_lc = lc(pars, ch1, ch=1)
ch1_ramp = ramp(pars[9], ch1)
ch1_bliss_model = ch1_bliss_lc * ch1_ramp
ch1_blissfunc = BLISSmodel(ch1, ch1_bliss_model, plotgrid=False, returnfunc=True)
ch1_blissflux = ch1_blissfunc.ev(ch1["xpos"], ch1["ypos"])
ch1["pre_mcmc_lc"] = ch1_bliss_model * ch1_blissflux

ch2_bliss_lc = lc(pars, ch2, ch=2)
ch2_ramp = ramp(pars[10], ch2)
ch2_bliss_model = ch2_bliss_lc * ch2_ramp
ch2_blissfunc = BLISSmodel(ch2, ch2_bliss_model, plotgrid=False, returnfunc=True)
ch2_blissflux = ch2_blissfunc.ev(ch2["xpos"], ch2["ypos"])
ch2["pre_mcmc_lc"] = ch2_bliss_model * ch2_blissflux

kepler["pre_mcmc_lc"] = lc(pars, kepler, ch="kepler", SuperSample = ss)

# calculating residuals of detrended data + add to data dictionary
ch1["pre_mcmc_residuals"] = ch1["flux"] / ch1["pre_mcmc_lc"]
ch2["pre_mcmc_residuals"] = ch2["flux"] / ch2["pre_mcmc_lc"]
kepler["pre_mcmc_residuals"] = kepler["flux"] / kepler["pre_mcmc_lc"]

# plt.scatter(ch1["time"], ch1["flux"]/ch1_blissflux/ch1_ramp)
# plt.plot(ch1["time"], ch1_bliss_lc, color="red")
# plt.show()
# plt.scatter(ch2["time"], ch2["flux"]/ch2_blissflux/ch2_ramp)
# plt.plot(ch2["time"], ch2_bliss_lc, color="red")
# plt.show()

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

# runs mcmc using MCMC function if run_mcmc1
if run_mcmc_1:
    kepler["error"] = kepler["error"] * 2.6
    # print("lnp errscale_mcmc1:", lnp(pars, priors, ch1, ch2, kepler))
    flat_sample1 = run_mcmc(pars, priors, nburn, nprod, ch1, ch2, kepler ,plot_corner=True, SuperSample = ss, run=1)

if run_mcmc_1 == False:
    flat_sample1 = np.load("flat_sample_1.npy")

# extract median values, standard deviation from flat sample using flatsample_pars funciton
mcmc_pars1, mcmc_errs1 = flatsample_pars(flat_sample1)

print("lnp pos mcmc1:", lnp(mcmc_pars1, priors, ch1, ch2, kepler, SuperSample = ss))

ch1["final_mcmc_lc"] = lc(mcmc_pars1, ch1, ch=1)
ch1_ramp = ramp(mcmc_pars1[9], ch1)
ch1_bliss_model = ch1_bliss_lc * ch1_ramp
ch1_blissfunc = BLISSmodel(ch1, ch1_bliss_model, plotgrid=False, returnfunc=True)
ch1_blissflux = ch1_blissfunc.ev(ch1["xpos"], ch1["ypos"])
mcmc2_lc_all1 = ch1_bliss_model * ch1_blissflux

ch2["final_mcmc_lc"] = lc(mcmc_pars1, ch2, ch=2)
ch2_ramp = ramp(mcmc_pars1[10], ch2)
ch2_bliss_model = ch2_bliss_lc * ch2_ramp
ch2_blissfunc = BLISSmodel(ch2, ch2_bliss_model, plotgrid=False, returnfunc=True)
ch2_blissflux = ch2_blissfunc.ev(ch2["xpos"], ch2["ypos"])
mcmc2_lc_all2 = ch2_bliss_model * ch2_blissflux

kepler["final_mcmc_lc"] = lc(mcmc_pars1, kepler, ch="kepler", SuperSample = ss)

# for SPITZER data:
    # Divide out BLISSflux and ramp from raw flux
ch1["mcmc_detrended_flux"] = ch1["flux"] / ch1_ramp / ch1_blissflux
ch2["mcmc_detrended_flux"] = ch2["flux"] / ch2_ramp / ch2_blissflux

# plots final mcmc lc over detrended data, phasefolded data?
plt.figure(figsize=(15, 5))

plt.subplot(131)
plt.title("ch 1 post-mcmc lc")
plt.scatter(ch1["time"], ch1["mcmc_detrended_flux"], s=1)
plt.plot(ch1["time"], ch1["final_mcmc_lc"], color="red")

plt.subplot(132)
plt.title("ch 2 post-mcmc lc")
plt.scatter(ch2["time"], ch2["mcmc_detrended_flux"], s=1)
plt.plot(ch2["time"], ch2["final_mcmc_lc"], color="red")

plt.subplot(133)
plt.title("kepler post-mcmc lc")
plt.scatter(kepler["time"], kepler["flux"], s=1)
plt.plot(kepler["time"], kepler["final_mcmc_lc"], color="red")

plt.show()

# prints transit depths + errors
labels = ["T0", "log_period", "RpRs1", "RpRs2", "RpRsK", "log_ars", "cosi",
          "esinw", "ecosw", "slope1", "slope2", "depth1", "depth2", "depthk"]

for i, label in enumerate(labels):
    print(label, ":", mcmc_pars1[i], "+/-", mcmc_errs1[i])

np.savez("final_params_working", pars = mcmc_pars1, errors = mcmc_errs1)

np.savez("ch1_data", time=ch1["time"], flux=ch1["mcmc_detrended_flux"], err=ch1["error"], mcmc_lc=ch1["final_mcmc_lc"],
         residuals=ch1["mcmc_detrended_flux"] - ch1["final_mcmc_lc"], x=ch1["xpos"], y=ch1["ypos"], blissflux=ch1_blissflux)

np.savez("ch2_data", time=ch2["time"], flux=ch2["mcmc_detrended_flux"], err=ch2["error"], mcmc_lc=ch2["final_mcmc_lc"],
         residuals=ch2["mcmc_detrended_flux"] - ch2["final_mcmc_lc"], x=ch2["xpos"], y=ch2["ypos"], blissflux=ch2_blissflux)
np.savez("kepler_data", time=kepler["time"], flux=kepler["flux"], err=kepler["error"], mcmc_lc=kepler["final_mcmc_lc"],
         residuals=kepler["flux"] - kepler["final_mcmc_lc"])