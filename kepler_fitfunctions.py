import numpy as np
import batman
import matplotlib.pyplot as plt
from scipy.interpolate import griddata, RectBivariateSpline
from scipy.stats import binned_statistic
import scipy.stats as stats
from dictionaries import *
import sys
from multiprocessing import Pool, cpu_count
import emcee, corner

def BintoSpec(wave1, wave2, model):
    nbins = wave1.shape[0]
    deltas = np.ediff1d(wave1)
    binnedmodel = np.zeros(nbins)
    for i in range(nbins):
        if i==0: wavestart = wave1[i] - deltas[i]/2.
        else: wavestart = wave1[i] - deltas[i-1]/2.
        if i==nbins: wavestop = wave1[i] + deltas[i-1]/2.
        else: wavestop = wave1[i] + deltas[i-1]/2.
        waverange = np.where((wave2>wavestart)&(wave2<wavestop))
        binnedmodel[i] = np.average(model[waverange])
    return binnedmodel

# LC function that takes data dictionaries
def lc(pars_values, data, SuperSample=False):
    if not isinstance(data, dict): # checks to see if the data imput is a dictionary
        time = data # if not, uses the array assigned to time instead of dictionary
    else:
        time = data["time"]

    params = batman.TransitParams()  # object to store transit parameters
    params.t0 = pars_values[0]  # time of inferior conjunction
    params.per = 10 ** pars_values[1]  # orbital period
    params.rp = pars_values[2]  # planet radius (in units of stellar radii)
    params.a = 10 ** pars_values[3]  # semi-major axis (in units of stellar radii)
    params.inc = np.arccos(np.fabs(pars_values[4])) * (180 / np.pi)  # orbital inclination (in degrees)
    params.ecc = np.sqrt(pars_values[6] ** 2 + pars_values[5] ** 2)  # eccentricity
    params.w = np.arctan2(pars_values[5], pars_values[6]) * (180 / np.pi)  # longitude of periastron (in degrees)
    params.limb_dark = "quadratic"  # limb darkening model
    params.u = [0.39456, 0.26712]

    if SuperSample:
        SuperSampleTime = np.arange(time[0], time[-1], 0.004) # 0.004 = 6 minute candence (opposed to the 30 expected)
        m = batman.TransitModel(params, SuperSampleTime)
        SuperSampleFlux = m.light_curve(params)
        model = BintoSpec(time, SuperSampleTime, SuperSampleFlux)
    else:
        m = batman.TransitModel(params, time)
        model = m.light_curve(params)

    if 0 in model:
        print("model=0 in lc function")

    return model

# converts T0 and period to phase time
def PhaseTimes(time, t0, period, t0center=True):
    if t0center:
        phase = np.mod((time-t0)-period/2.,period) / period
    else:
        phase = np.mod((time-t0),period) / period
    return phase - 0.5

# phasefolds time, lightcurve + adds to dictionaries
def PhaseFold(data, pars, residuals=False, binned=False, new_dict = False, SuperSample = False, ch=""):
    time=data["time"]
    flux=data["flux"]
    if binned == True:
        time = data["binned_time"]
        flux = data["binned_flux"]

    t0 = pars[0]
    period = 10**pars[1]
    phasefold_pars = PhaseTimes(time, t0, period)
    time_pars = np.linspace(-0.5, 0.5, len(flux))
    php_times_pars = time_pars * period + t0
    lc_php_pars = lc(pars, php_times_pars, SuperSample=SuperSample)

    if residuals:
        residual_flux = flux - lc_php_pars
        data["residuals_php"] = residual_flux
    if new_dict:
        data_phasefold = {}
        data_phasefold["lc_php"] = lc_php_pars # phasefolded lightcurve
        data_phasefold["phasefold_results"] = phasefold_pars # results from PhaseFold function
        data_phasefold["lc_times"] = php_times_pars # these used to make lightcurve
        data_phasefold["phase_time"] = time_pars # time to plot
        return data_phasefold

    # updates old dictionary
    else:
        data["lc_php"] = lc_php_pars
        data["phasefold_results"] = phasefold_pars
        data["lc_times"] = php_times_pars
        data["phase_time"] = time_pars

# fix LNP function
def lnp(pars, priors, kepler, SuperSample=False):
    scale = 1

    models = {}
    models["kepler"] = lc(pars, kepler, SuperSample=SuperSample)

    # Calculate the log-likelihood
    log_prob_data = 0.0
    log_prob_data += np.sum(stats.norm.logpdf(kepler["flux"] - models["kepler"], loc=0, scale=kepler["error"]))

    # Calculate the log-likelihood of prior values
    log_prob_prior = 0.0
    log_prob_prior += stats.norm.logpdf(pars[0], loc=priors[0][0], scale=priors[0][1]/scale) # t0
    log_prob_prior += stats.norm.logpdf(pars[1], loc=priors[1][0], scale=priors[1][1]/scale) # log period
  #  log_prob_prior += stats.norm.logpdf(pars[2], loc=priors[2][0], scale=priors[2][1]/scale) # RpRsk
  #  log_prob_prior += stats.norm.logpdf(pars[3], loc=priors[3][0], scale=priors[3][1]/scale) # log_ars
    #log_prob_prior += stats.norm.logpdf(pars[4], loc=priors[4][0], scale=priors[4][1]/scale) # cosi
    log_prob_prior += stats.norm.logpdf(pars[5], loc=priors[5][0], scale=priors[5][1]/scale) # esinw
    log_prob_prior += stats.norm.logpdf(pars[6], loc=priors[6][0], scale=priors[6][1]/scale) # ecosw
   # log_prob_prior += stats.norm.logpdf(pars[7], loc=priors[7][0], scale=priors[7][1]/scale) # U1
  #  log_prob_prior += stats.norm.logpdf(pars[8], loc=priors[8][0], scale=priors[8][1]/scale) # U2

    # Combine log-likelihoods

    log_likelihood_value = log_prob_data + log_prob_prior

    return log_likelihood_value

# Makes walker array, runs mcmc
def run_mcmc(pars, priors, nburn, nprod, kepler, plot_corner = False, SuperSample=False, run="k1", plot_walkers=False):
    ndim = len(pars)
    nwalkers = 2*ndim

    pos = np.empty((nwalkers, ndim))
    pos_errscale = 1
    for i, par in enumerate(pars):
        pos[:, i] = np.random.normal(par, priors[i][1]/pos_errscale, nwalkers) # priors errscale used to be /10

    with Pool(processes=cpu_count()) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnp,
                                        args=(priors, kepler, SuperSample), pool=pool)
        pos, _, _ = sampler.run_mcmc(pos, nburn, progress=True)  # runs mcmc on burnin values
        # fig, axes = plt.subplots(len(pos), figsize=(5, 20))
        # plt.suptitle("Post burnin walker positions")
        # for i, par in enumerate(pos):
        #     kepler_ph = PhaseFold(kepler, par, new_dict=True)
        #     # curve = lc(par, kepler, ch="kepler")
        #     axes[i].plot(kepler_ph["phase_time"], kepler_ph["lc_php"], color="red")
        #     axes[i].scatter(kepler_ph["phasefold_results"], kepler["flux"], s=1)
        #     axes[i].set_title("walker array:" + str(i))
        #     axes[i].set_xlim(-0.025, 0.025)
        # plt.tight_layout()
        # plt.show()
        sampler.reset()
        pos, _, _ = sampler.run_mcmc(pos, nprod, progress=True)  # runs from positions of burnin values

    # creates flat_sample array for corner plot generation
    flat_sample = sampler.get_chain(discard=0, thin=1, flat=True)  # flattens list of samples

    np.save(f"flat_sample_{run}", flat_sample)  # saves flatsamples array

    if plot_corner:
        labels = ["T0", "log_period", "RpRsK", "log_ars", "cosi",
                  "esinw", "ecosw"]
        fig = corner.corner(flat_sample, labels=labels, show_titles=True)
        plt.tight_layout()
        plt.show()

    if plot_walkers:
        samples = sampler.get_chain()
        incides = [0]
        labels = ["RpRs_K"]

        plt.plot(samples[:, :, 2], "k", alpha=0.3)
        plt.xlim(0, len(samples))
        plt.ylabel(labels)
        #ax.yaxis.set_label_coords(-0.1, 0.5)
        plt.show()

    return flat_sample

# gets parameters from flatsample
def flatsample_pars(flat_sample):
    # values from mcmc fitting
    T0 = flat_sample[:, 0]
    log_period = flat_sample[:, 1]
    RpRsk = flat_sample[:, 2]
    log_a = flat_sample[:, 3]
    cosi = flat_sample[:, 4]
    esinw = flat_sample[:, 5]
    ecosw = flat_sample[:, 6]
   # u1 = flat_sample[:, 7]
   # u2 = flat_sample[:, 8]
    depthk = RpRsk ** 2 * 1000000
    np.column_stack((flat_sample, depthk))

    # initializes lists to iterate over to find median values, STDs from flat_sample -> get best fit parameters from mcmc + error
    value_list = [T0, log_period, RpRsk, log_a, cosi, esinw, ecosw, depthk]
    mcmc_pars = []
    mcmc_stds = []

    # iterates over lists to calculate median value, std
    for i, parameter in enumerate(value_list):
        median_parameter = np.median(parameter)
        mcmc_pars.append(median_parameter)

        sig1 = np.percentile(parameter, 16)  # one sigma away from median on left
        sig2 = np.percentile(parameter, 84)  # "", right
        diff_sig1 = median_parameter - sig1  # std on left
        diff_sig2 = sig2 - median_parameter  # "", right
        avg_std = (diff_sig1 + diff_sig2) / 2  # average spread (accounts for small skew?)
        mcmc_stds.append(avg_std)

    return mcmc_pars, mcmc_stds

# FOR PLOTTING
def bin_data(data, pars, ch="", dict_append = False):
    nbins = round(len(data["time"]) / 10)
    binned_data = binned_statistic(data["time"], data["flux"], bins=nbins) # bins time, flux
    binned_BJD_data = binned_statistic(data["BJD_decimal"], data["residuals"], bins=nbins) # bins time, flux
    binned_flux = binned_data[0]
    binned_time = binned_data[1]
    binned_time = binned_time[1:]
    binned_BJD = binned_BJD_data[1]
    binned_BJD = binned_BJD[1:]
    binned_residuals = binned_BJD_data[0]

    binned_error = binned_statistic(data["time"], data["err"], bins=nbins) # bins time, error
    binned_err = binned_error[0]

    _, binned_model = lc(pars, binned_time, ch) # lc model for binned data

    if dict_append == False: return binned_time, binned_flux, binned_err, binned_model, binned_BJD
    else:
        data["binned_time"] = binned_time
        data["binned_flux"] = binned_flux
        data["binned_err"] = binned_err
        data["binned_lc"] = binned_model
        data["binned_BJD"] = binned_BJD
        data["binned_residuals"] = binned_residuals

