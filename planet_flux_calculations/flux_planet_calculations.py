import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.integrate import trapezoid
import math
import sys

from astropy.modeling.models import BlackBody
from astropy import units as u

# IMPORTS =============================================================================================
# loads in BT-settl stellar atmosphere models # in angstroms
spec_B_raw = np.loadtxt("lte038.0-4.5-0.0a+0.0.BT-Settl.spec.7.dat.txt")
spec_A_raw = np.loadtxt("lte058.0-4.5-0.0a+0.0.BT-Settl.spec.7.dat.txt")

# loads in transmission profiles
kep_transmission_raw = np.loadtxt("Kepler_Kepler.K.dat")
spitz_ch1_raw = np.loadtxt("Spitzer_IRAC.I1.dat")
spitz_ch2_raw = np.loadtxt("Spitzer_IRAC.I2.dat")

mcmc_output = np.load("../final_params.npz")
pars = mcmc_output["pars"]
errs = mcmc_output["errors"]
RpRs_kepler = [pars[4], errs[4]]
depth_ch1 = [pars[11], errs[11]]
depth_ch2 = [pars[12], errs[12]]
depth_kepler = [pars[13], errs[13]]

# FUNCTION FOR CALCULATING FLUX OVER A WAVELENGTH
def trapz(lam, flux):
    """
    :param lam: units of AA -> converts to cm
    :param flux array: units of ergs/cm3/s
    :return: surface brightness in ergs/cm2/s
    """
    integral = 0
    lam = lam.to(u.cm)
    flux = flux.to(u.erg / u.cm**3 /u.s)

    n = len(lam)
    for i in range(1, n):
        h = lam[i] - lam[i-1]
        integral += (flux[i] + flux[i-1]) * h / 2

    return integral

def find_lum_per_filter(filter, spectrum):
    """
    :param filter: array of filter transmission profile information in the form ((wavelength (A)), (transmission))
    :param spectrum: Stellar spectrum model array in the form ((wavelength(A)), (flux (erg/cm2/s/A)))
    :return: integrated flux per wavelength of filter in ergs/cm^2/s
    """
    # indices in star wavelengtg array where wavelength is equal to the smallest / largest wavelength in the star spectrum array
    i_min = (np.abs(spectrum[:, 0] - np.min(filter[:, 0]))).argmin()
    i_max = (np.abs(spectrum[:, 0] - np.max(filter[:, 0]))).argmin()

    # shortens star spectrum arrays at desired indices
    spec = spectrum[i_min:i_max]

    # Creates interpolation function to transpose x axis from star data onto spitzer transmission spectrum
    interp_func = interp1d(filter[:, 0], filter[:, 1])

    # calculates new x axis, uses interpolation function to find new y axis
    wavelength = spec[:, 0] * u.AA # wavelength from star array, adds correct units
    flux = spec[:, 1] * (u.erg / u.cm**2 / u.s / u.AA) # grabs flux from array, adds correct units
    transmission = interp_func(wavelength) # calculates new kepler transmission array values of same length of transmission x values
    scaled_flux = transmission * flux
    # plt.plot(spec[:, 0], spec[:, 1])
    # plt.plot(spec[:, 0],scaled_flux)
    #
    # plt.show()

    Sp = trapz(wavelength, scaled_flux) # surface brightness of object
    print("Surface brightness:", Sp)
    return Sp

Sp_ch2_starA = find_lum_per_filter(spitz_ch2_raw, spec_A_raw)

def Planck(lam, T):
    # takes lam in angstroms
    h = 6.62617e-27 * u.erg * u.s
    c = 2.99792458e10 * u.cm/u.s
    k_b = 1.3807e-16 * u.cm**2 * u.g / u.s**2 / u.K
    lam = lam.to(u.cm)

    t1 = (2 * h * c**2) /(lam**5)
    te = (math.e)**((h*c)/(k_b*T*lam))
    eq = t1 * (1/(te - 1))

    return eq * np.pi

def find_bb_temp(Sp, filter):
    temp_guess_min = int(input("temp_min:"))
    temp_guess_max = int(input("temp_max:"))
    temp_guesses = np.arange(temp_guess_min, temp_guess_max, 10)

    values = [] # initialzies list for flux per temperature calculations
    lam = filter[:, 0] * u.AA # converts wavelength array from filter transmission to angstroms
    transmission = filter[:, 1] # assigns transmission array

    for T in temp_guesses: # iterates over all temperature guesses
        flux = Planck(lam, T*u.K) # calculates array of planck function values over wavelength range of filter
        integrand = flux * transmission # scales blackbody flux by transmission of filter
        term = trapz(lam, integrand) # integrates scaled flux over wavelength range
        values.append(term.value) # adds values wit stripped units to array

    temp_index = np.abs(values - Sp.value).argmin()
    temp = temp_guesses[temp_index]

    print("Bestfit Temperature:", temp * u.K)
    return temp*u.K

best_temp = find_bb_temp(Sp_ch2_starA, spec_A_raw)


# Testing Planck function, notes
# temp = 5800*u.K
# wav = np.linspace(1000, 60000, len(spec_A_raw)) * u.AA
# BB = Planck(wav, temp).to(u.erg / u.cm**2 / u.s / u.AA)
#
# #plt.plot(wav, spec_A_raw[:, 1]/BB, color="red", linestyle = "--")
# plt.plot(wav, BB)
# plt.plot(spec_A_raw[:, 0], spec_A_raw[:, 1], alpha=0.5, linewidth=0.05)
# plt.xlim(0, 50000)
# #plt.ylim(0, 50000)
#
# plt.show()

# if run minimization for surface brightness give me close to 5800 K
# once gives temp, make plot again ^^
# in 4.5, try figure out what temperature you get for 5800K stellar model (close to 5800)
# whateger it gives me, take best temperatire again + calcualte corresponding BB and make plot of division
# may find better temperature
# if works, examine why kepler transit depth error
# plot walkers from chain for depth + see what that looks like
# calculate surface brightness for a blackbody

# make walker plot JUST with Kepler
# see what depth chains look like
# why





