from ss_dictionaries import *
from ss_fitfunctions import *

nburn = 1000 # put back run burnin in mcmc function
nprod = 2000

pars = pars_lastrun
priors = combo_priors

ss=True # SuperSample for Kepler data - deals with strange ingress/egress effects
ss_f_range = [5, 6, 7]

depths = []
errs = []
ss_fs = []

for ss_f in ss_f_range:
    print("current ss_f:", ss_f)
    # make complete model (LC, ramp, BLISS) + add to data dictionary for Spitzer channels
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

    # runs mcmc using MCMC function if run_mcmc1
    kepler["error"] = kepler["error"] * 2.6 # scales error on kepler function by value allowed by calculations
    flat_sample1 = run_mcmc(pars, priors, nburn, nprod, ch1, ch2, kepler ,plot_corner=True, SuperSample = ss, ss_f=ss_f) # runs MCMC
    #
    # labels = ["T0", "log_period", "RpRs1", "RpRs2", "RpRsK", "log_ars", "cosi",
    #           "esinw", "ecosw", "slope1", "slope2"]
    # fig = corner.corner(flat_sample1, labels=labels, show_titles=True)
    # plt.tight_layout()
    # plt.show()
    # plt.savefig("cornerplot.pdf", dpi=300, bbox_inches="tight")

    # fig, axes = plt.subplots(len(pars), figsize=(10, 17), sharex=True)
    # labels = ["T0", "log_period", "RpRs1", "RpRs2", "RpRsK", "log_ars", "cosi",
    #           "esinw", "ecosw", "slope1", "slope2"]
    #
    # for i, n in enumerate(np.arange(0,  len(pars))):
    #     ax = axes[i]
    #     for j in range(len(pars * 2)):
    #         ax.plot(sample[:, j, n], alpha=0.3)
    #     # ax.plot(sample[:, 1, n], alpha=0.3, color="red")
    #     # ax.plot(sample[:, 2, n], alpha=0.3, color="orange")
    #     # ax.plot(sample[:, 3, n], alpha=0.3, color="blue")
    #     # ax.plot(sample[:, 4, n], alpha=0.3, color="purple")
    #     ax.set_xlim(0, len(sample))
    #     ax.set_ylabel(labels[i])
    #     ax.yaxis.set_label_coords(-0.1, 0.5)
    #
    # axes[-1].set_xlabel("step number")
    # plt.savefig("walker_position_plots.pdf", dpi=300, bbox_inches="tight")
    # plt.show()
    #
    # likelihoods = sample.get_log_prob()
    # plt.title("Log likelihoods")
    # plt.plot(likelihoods)
    # plt.savefig("log_likelihood_plots.pdf", dpi=300, bbox_inches="tight")
    # plt.show()

    # extract median values, standard deviation from flat sample using flatsample_pars funciton
    mcmc_pars1, mcmc_errs1 = flatsample_pars(flat_sample1)

    # creates final, detrended lcs from MCMC pars
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

    kepler["final_mcmc_lc"] = lc(mcmc_pars1, kepler, ch="kepler", SuperSample=ss)
    kepler_ph = PhaseFold(kepler, mcmc_pars1, new_dict=True, SuperSample=ss, ch="kepler")

    # for SPITZER data:
    # Divide out BLISSflux and ramp from raw flux
    ch1["mcmc_detrended_flux"] = ch1["flux"] / ch1_ramp / ch1_blissflux
    ch2["mcmc_detrended_flux"] = ch2["flux"] / ch2_ramp / ch2_blissflux

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
    plt.scatter(kepler_ph["phasefold_results"], kepler["flux"], s=1)
    plt.xlim(-0.02, 0.02)
    plt.plot(kepler_ph["phase_time"], kepler_ph["lc_php"], color="red")

    plt.show()

    print("run", ss_f, ":", mcmc_pars1[13], mcmc_errs1[13])

    depths.append(mcmc_pars1[13])
    errs.append(mcmc_errs1[13])
    ss_fs.append(ss_f)

    np.savez("depths_errs_ss", depths=depths, errs=errs, ss_fs=ss_fs)

# np.savez("ch1_data", time=ch1["time"], flux=ch1["mcmc_detrended_flux"], err=ch1["error"], mcmc_lc=ch1["final_mcmc_lc"],
#          residuals=ch1["mcmc_detrended_flux"] - ch1["final_mcmc_lc"], x=ch1["xpos"], y=ch1["ypos"], blissflux=ch1_blissflux)
#
# np.savez("ch2_data", time=ch2["time"], flux=ch2["mcmc_detrended_flux"], err=ch2["error"], mcmc_lc=ch2["final_mcmc_lc"],
#          residuals=ch2["mcmc_detrended_flux"] - ch2["final_mcmc_lc"], x=ch2["xpos"], y=ch2["ypos"], blissflux=ch2_blissflux)
# np.savez("kepler_data", time=kepler["time"], flux=kepler["flux"], err=kepler["error"], mcmc_lc=kepler["final_mcmc_lc"],
#          residuals=kepler["flux"] - kepler["final_mcmc_lc"])