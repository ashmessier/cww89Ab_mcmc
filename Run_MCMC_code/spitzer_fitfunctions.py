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
from pathlib import Path
import os

data_dir = Path("Data")
data_dir.mkdir(parents=True, exist_ok=True)

def ramp(slope, data):
    return  1 + slope * (data["time"] - np.median(data["time"]))

# LC function that takes data dictionaries
def lc(pars, data, ch=""):
    if not isinstance(data, dict): # checks to see if the data imput is a dictionary
        time = data # if not, uses the array assigned to time instead of dictionary
    else:
        time = data["time"]

    params = batman.TransitParams()  # object to store transit parameters
    params.t0 = pars[0]  # time of inferior conjunction
    params.per = 10 ** pars[1]  # orbital period
    # params.rp = pars[2]  # planet radius (in units of stellar radii)
    params.a = 10 ** pars[4]  # semi-major axis (in units of stellar radii)
    params.inc = np.arccos(np.fabs(pars[5])) * (180 / np.pi)  # orbital inclination (in degrees)
    params.ecc = np.sqrt(pars[7] ** 2 + pars[6] ** 2)  # eccentricity
    params.w = np.arctan2(pars[6], pars[7]) * (180 / np.pi)  # longitude of periastron (in degrees)
    params.limb_dark = "quadratic"  # limb darkening model

    if ch == 1:
        params.u = [0.095, 0.133]  # limb darkening coefficients [u1, u2] # 2 for quadratic
        params.rp = pars[2]

    elif ch == 2:
        params.u = [0.09, 0.098]
        params.rp = pars[3]

    m = batman.TransitModel(params, time)  # initializes model
    model = m.light_curve(params)

    if 0 in model:
        print("model=0 in lc function")

    return model

# BLISS function for pixel mapping reasons
def BLISSmodel(data, modelforBLISS, plotgrid=False, returnfunc=True):
    flux = data["flux"]
    xpos = data["xpos"]
    ypos = data["ypos"]
    BLISSparams = data["BLISSparams"]

    blissgrid = np.zeros((BLISSparams['nx'], BLISSparams['ny']))
    for i in range(BLISSparams['nx']):
        for j in range(BLISSparams['ny']):
            try:
                idxs = BLISSparams['knotsgrid'][i, j][1:].astype(int)
                blissgrid[i, j] = np.mean(flux[idxs] / modelforBLISS[idxs])
            except TypeError:
             blissgrid[i, j] = -99.

    goodidx = np.where(blissgrid > -1.)

    goodlocs = np.column_stack((BLISSparams['knotsx'][goodidx[0]], BLISSparams['knotsy'][goodidx[1]]))
    goodblissList = []
    for i in range(goodlocs.shape[0]):
        goodblissList.append(blissgrid[goodidx[0][i], goodidx[1][i]])
    goodbliss = np.array(goodblissList)

    griddedbliss = griddata(goodlocs, goodbliss, (BLISSparams['grid_x_y'][0],BLISSparams['grid_x_y'][1]), method='nearest')
    try:
        blissfunc = RectBivariateSpline(BLISSparams['knotsx'], BLISSparams['knotsy'], griddedbliss, s=0, kx=2, ky=2)
    except ValueError:
        return -99 # in case it fails
    blissflux = blissfunc.ev(xpos, ypos)

    if plotgrid:
        plt.plot(xpos, ypos, '.k', ms=1)
        plt.savefig('blissgridXY.png', dpi=120)
        plt.clf()
        plt.close()

        imshowextent = [BLISSparams['xmin'],BLISSparams['xmax'],BLISSparams['ymin'],BLISSparams['ymax']]
        plt.imshow(np.flipud(griddedbliss.transpose()), cmap='seismic', interpolation='none', extent=imshowextent, aspect=0.5, vmin=0.97, vmax=1.03)
        cbar = plt.colorbar()
        cbar.set_label('Sensitivity', rotation=270, labelpad=20, fontsize=16)
        plt.xlabel('X-pixel', fontsize=16)
        plt.ylabel('Y-pixel', fontsize=16)
        plt.savefig('blissgrid.png', dpi=120)
        plt.clf()
        plt.close()

        stepsize = 0.001
        gridforplot = np.mgrid[BLISSparams['xmin']:BLISSparams['xmax']:stepsize, BLISSparams['ymin']:BLISSparams['ymax']:stepsize]
        flatx = gridforplot[0].flatten()
        flaty = gridforplot[1].flatten()
        testflux = blissfunc.ev(flatx, flaty)
        knotsxstandin = np.arange(BLISSparams['xmin'], BLISSparams['xmax'], stepsize)
        knotsystandin = np.arange(BLISSparams['ymin'], BLISSparams['ymax'], stepsize)
        plotflux = testflux.reshape((knotsxstandin.shape[0],knotsystandin.shape[0]))
        plt.imshow(np.flipud(plotflux.transpose()), cmap='gray', interpolation='none', extent=imshowextent, aspect=0.5, vmin=0.985, vmax=1.015)
        #plt.imshow(numpy.flipud(plotflux.transpose()), cmap='gray', interpolation='none', extent=imshowextent, aspect=0.5)
        # cbar = plt.colorbar()
        # cbar.set_label('BLISS residuals', rotation=270, labelpad=20)
        plt.savefig('blissfunc.png', dpi=120)
        plt.clf()
        plt.close()

    if returnfunc: return blissfunc
    return blissflux

# converts T0 and period to phase time
def PhaseTimes(time, t0, period, t0center=True):
    if t0center:
        phase = np.mod((time-t0)-period/2.,period) / period
    else:
        phase = np.mod((time-t0),period) / period
    return phase - 0.5

# phasefolds time, lightcurve + adds to dictionaries
def PhaseFold(data, pars, residuals=False, binned=False, new_dict = False, ch=""):
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
    lc_php_pars = lc(pars, php_times_pars, ch=ch)

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

# deleted make model, the ramp
def make_model(pars, data, ch=""):
    if ch == 1:
        ramp = 1 + pars[8] * (data["time"] - np.median(data["time"]))  # 1+slope * median time
    elif ch == 2:
        ramp = 1 + pars[9] * (data["time"] - np.median(data["time"]))  # 1+slope * median time

    lightcurve = lc(pars, data, ch)
    modelforBLISS = lightcurve * ramp  # accounts for non-system related linear trends
    blissfunc = BLISSmodel(data, modelforBLISS, plotgrid=False, returnfunc=True)
    blissflux = blissfunc.ev(data["xpos"], data["ypos"])
    model = lightcurve * ramp * blissflux
    return model

# fix LNP function
def lnp(pars, priors, ch1, ch2):
    scale = 1

    models = {}
    models["ch1"] = make_model(pars, ch1, ch=1)
    models["ch2"] = make_model(pars, ch2, ch=2)

    # Calculate the log-likelihood
    log_prob_data = 0.0
    log_prob_data += np.sum(stats.norm.logpdf(ch1["flux"] - models["ch1"], loc=0, scale=ch1["error"]))
    log_prob_data += np.sum(stats.norm.logpdf(ch2["flux"] - models["ch2"], loc=0, scale=ch2["error"]))

    # Calculate the log-likelihood of prior values
    log_prob_prior = 0.0
    log_prob_prior += stats.norm.logpdf(pars[0], loc=priors[0][0], scale=priors[0][1]/scale) # t0
    log_prob_prior += stats.norm.logpdf(pars[1], loc=priors[1][0], scale=priors[1][1]/scale) # log period
    #log_prob_prior += stats.norm.logpdf(pars[2], loc=priors[2][0], scale=priors[2][1]/scale) # RPRs1
   # log_prob_prior += stats.norm.logpdf(pars[3], loc=priors[3][0], scale=priors[3][1]/scale) # Rprs2
    log_prob_prior += stats.norm.logpdf(pars[4], loc=priors[4][0], scale=priors[4][1]/scale) # olog_ars
    log_prob_prior += stats.norm.logpdf(pars[5], loc=priors[5][0], scale=priors[5][1]/scale) # cosi
    log_prob_prior += stats.norm.logpdf(pars[6], loc=priors[6][0], scale=priors[6][1]/scale) # esinw
    log_prob_prior += stats.norm.logpdf(pars[7], loc=priors[7][0], scale=priors[7][1]/scale) # ecos1
    #log_prob_prior += stats.norm.logpdf(pars[8], loc=priors[8][0], scale=priors[8][1]/scale) # slope1
    #log_prob_prior += stats.norm.logpdf(pars[9], loc=priors[9][0], scale=priors[9][1]/scale) # slope2
    # Combine log-likelihoods

    log_likelihood_value = log_prob_data + log_prob_prior

    return log_likelihood_value

# Makes walker array, runs mcmc
def run_mcmc(pars, priors, nburn, nprod, ch1, ch2, plot_corner = False, run="spitzer_only"):
    ndim = len(pars)
    nwalkers = 2*ndim

    pos = np.empty((nwalkers, ndim))
    for i, par in enumerate(pars):
        pos[:, i] = np.random.normal(par, priors[i][1]/10, nwalkers)

    with Pool(processes=cpu_count()) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnp,
                                        args=(priors, ch1, ch2), pool=pool)
        pos, _, _ = sampler.run_mcmc(pos, nburn, progress=True)  # runs mcmc on burnin values
        sampler.reset()
        pos, _, _ = sampler.run_mcmc(pos, nprod, progress=True)  # runs from positions of burnin values

    # creates flat_sample array for corner plot generation
    flat_sample = sampler.get_chain(discard=0, thin=1, flat=True)  # flattens list of samples

    flatsample_spitzer = os.path.join(data_dir, "flat_sample_spitzer")
    np.save(flatsample_spitzer, flat_sample)  # saves flatsamples array

    if plot_corner:
        labels = ["T0", "log_period", "RpRs1", "RpRs2", "log_ars", "cosi",
                  "esinw", "ecosw", "slope1", "slope2"]
        fig = corner.corner(flat_sample, labels=labels, show_titles=True)
        plt.tight_layout()
        plt.show()

    return flat_sample

# gets parameters from flatsample
def flatsample_pars(flat_sample):
    # values from mcmc fitting
    T0 = flat_sample[:, 0]
    log_period = flat_sample[:, 1]
    RpRs1 = flat_sample[:, 2]
    RpRs2 = flat_sample[:, 3]
    log_a = flat_sample[:, 4]
    cosi = flat_sample[:, 5]
    esinw = flat_sample[:, 6]
    ecosw = flat_sample[:, 7]
    slope1 = flat_sample[:, 8]
    slope2 = flat_sample[:, 9]
    depth1 = RpRs1 ** 2 * 1000000  # in ppm
    depth2 = RpRs2 ** 2 * 1000000  # in ppm
    np.column_stack((flat_sample, depth1))  # adds on transit depth to flat sample
    np.column_stack((flat_sample, depth2))  # adds on transit depth to flat sample

    # initializes lists to iterate over to find median values, STDs from flat_sample -> get best fit parameters from mcmc + error
    value_list = [T0, log_period, RpRs1, RpRs2, log_a, cosi, esinw, ecosw, slope1, slope2, depth1, depth2]
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

