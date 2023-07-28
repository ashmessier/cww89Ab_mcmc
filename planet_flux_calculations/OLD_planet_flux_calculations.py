import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid
import math
import sys

from astropy.modeling.models import BlackBody
from astropy import units as u

Kepler = True
Johnson_V = True
Johnson_B = True

# loads in BT-settl stellar atmosphere models # in angstroms
spec_B_raw = np.loadtxt("lte038.0-4.5-0.0a+0.0.BT-Settl.spec.7.dat.txt")
spec_A_raw = np.loadtxt("lte058.0-4.5-0.0a+0.0.BT-Settl.spec.7.dat.txt")

# loads in transmission profiles
kep_transmission_raw = np.loadtxt("Kepler_Kepler.K.dat")
johnson_V_raw = np.loadtxt("GCPD_Johnson.V.dat")
johnson_B_raw = np.loadtxt("GCPD_Johnson.B.dat")

# loads in depths and errors from MCMC fitting
mcmc_output = np.load("../final_params_working.npz")
pars = mcmc_output["pars"]
errs = mcmc_output["errors"]
# Printing parameters/errs in LaTeX format for writeup
#
per = 10**pars[1]
ars = 10**pars[5]
inc = np.arccos(np.fabs(pars[6])) * (180 / np.pi)
e = np.sqrt(pars[8] ** 2 + pars[7] ** 2)
w = np.arctan2(pars[7], pars[8]) * (180 / np.pi)

sig_per = errs[1]/(np.log(10))
sig_ars = errs[5]/(np.log(10))
sig_inc = (pars[6])/(1-(pars[6]**2))
sig_e = np.sqrt((pars[7]/(e) * errs[7])**2 + (pars[8]/(e) * errs[8])**2)
import math
def calculate_error_on_x(y, z, delta_y, delta_z):
    # Calculate partial derivatives
    partial_x_partial_y = (1 / (1 + (y / z) ** 2)) * (1 / z)
    partial_x_partial_z = (1 / (1 + (y / z) ** 2)) * (-y / z ** 2)
    # Calculate error on x using error propagation formula
    delta_x = math.sqrt((partial_x_partial_y * delta_y) ** 2 + (partial_x_partial_z * delta_z) ** 2)
    return delta_x

# Example usage
y = pars[7]
z = pars[8]
delta_y = errs[7]
delta_z = errs[8]
sig_w = calculate_error_on_x(y, z, delta_y, delta_z)
#
pars_mod = [per, ars, inc, e, w]
errs_mod = [sig_per, sig_ars, sig_inc, sig_e, sig_w]
# pars_mod = [2457346.329, 5.292569,0.09321,0.09321, 12.62, 88.53, 0.1929, 345.9, 0, 0]
# errs_mod = [0.00011, 0.000026, 0.00046, 0.00046, 0.15, 0.7, 0.0019, 1, 0.0001, 0.0001]

for i, par in enumerate(pars_mod):
    err = errs_mod[i]
    print(f"${round(par, 5)} \pm {round(err, 8)}$")

sys.exit()

#makes variables with parameters, errors from mcmc_output for use in calculation

RpRs_kepler = [pars[4], errs[4]]
depth_ch1 = [pars[11], errs[11]]
depth_ch2 = [pars[12], errs[12]]
depth_kepler = [pars[13], errs[13]]

# COMBINING KEPLER TRANSMISSION PROFILE WITH MODEL ATMOSPHERE AND INTEGRATING TO FIND TOTAL FLUX
if Kepler:
    # KEPLER FILTER ---------------------------------------------------
    # First step in making stellar spectra arrays the same length / wavelength range as Kepler transmission filter
    # finds indices to slice star flux arrays along by finding closest value in array to desired slice wavelength
    A_3500 = (np.abs(spec_A_raw[:, 0] - 3500)).argmin()
    A_9500 = (np.abs(spec_A_raw[:, 0] - 9500)).argmin()
    B_3500 = (np.abs(spec_B_raw[:, 0] - 3500)).argmin()
    B_9500 = (np.abs(spec_B_raw[:, 0] - 9500)).argmin()

    # shortens star spectrum arrays at desired indices
    spec_A = spec_A_raw[A_3500:A_9500]
    spec_B = spec_B_raw[B_3500:B_9500]

    # shortens kepler transmission spectrum to same wavelength regime
    # Kepler needs to be a little larger than star flux arrays so interpolation works
    kep_3500 = (np.abs(kep_transmission_raw - 3490)).argmin() # finds indices to slice transmission array on
    kep_9500 = (np.abs(kep_transmission_raw - 9510)).argmin()
    kepler_transmission = kep_transmission_raw[kep_3500:kep_9500]

    # Creates interpolation function to transpose x axis from star data onto kepler transmission spectrum
    interp_func = interp1d(kepler_transmission[:, 0], kepler_transmission[:, 1])

    # A-------------------------------------------------------------------
    # calculates new x axis, uses interpolation function to find new y axis
    wavelength_A = spec_A[:, 0] # wavelength from star a array
    flux_A = spec_A[:, 1] # grabs flux from array
    transmission_A = interp_func(wavelength_A) # calculates new kepler transmission array values of same length of transmission x values
    scaled_flux_A = transmission_A * flux_A
    # plt.plot(wavelength_A, transmission_A * flux_A.max(), zorder=3, color="black", label="Kepler transmission profile")
    # plt.plot(wavelength_A, flux_A, linewidth=1, label="raw flux")
    # plt.plot(wavelength_A, scaled_flux_A, linewidth=1, label="transmission scaled flux")
    # plt.show()

    # B --------------------------------------------------------
    wavelength_B = spec_B[:, 0]
    flux_B = spec_B[:, 1] # grabs flux from array
    transmission_B = interp_func(wavelength_B) # calculates new flux values off of transmission x values
    scaled_flux_B = transmission_B * flux_B
    # plt.plot(wavelength_B, transmission_B * flux_B.max(), zorder=3, color="black", label="Kepler transmission profile")
    # plt.plot(wavelength_B, flux_B, linewidth=1, label="raw flux")
    # plt.plot(wavelength_B, scaled_flux_B, linewidth=1, label="transmission scaled flux")
    # plt.show()

    #Integrate to find total fluxes??? of stars as seen by Kepler bc kepler curve
    F_A = trapezoid(scaled_flux_A, x=wavelength_A)
    F_B = trapezoid(scaled_flux_B, x=wavelength_B)

# JOHNSON V , B FILTER for CHECKING =====================================================================================================
if Johnson_V:
    # First step in making stellar spectra arrays the same length / wavelength range as Kepler transmission filter
    # finds indices to slice star flux arrays along by finding closest value in array to desired slice wavelength
    A_4710 = (np.abs(spec_A_raw[:, 0] - 4720)).argmin()
    A_7680= (np.abs(spec_A_raw[:, 0] - 7675)).argmin()
    B_4710= (np.abs(spec_B_raw[:, 0] - 4720)).argmin()
    B_7680 = (np.abs(spec_B_raw[:, 0] - 7675)).argmin()

    # shortens star spectrum arrays at desired indices
    specA_V = spec_A_raw[A_4710:A_7680]
    specB_V = spec_B_raw[B_4710:B_7680]

    # shortens transmission spectrum to same wavelength regime
    # Kepler needs to be a little larger than star flux arrays so interpolation works
    john_4710 = (np.abs(johnson_V_raw - 4710 + 20)).argmin() # finds indices to slice transmission array on
    john_7680 = (np.abs(johnson_V_raw - 7680 - 20)).argmin()

    johnV_transmission = johnson_V_raw[john_4710:john_7680]

    # Creates interpolation function to transpose x axis from star data onto kepler transmission spectrum
    interp_func_jV = interp1d(johnV_transmission[:, 0], johnV_transmission[:, 1])

    # A-------------------------------------------------------------------
    # calculates new x axis, uses interpolation function to find new y axis
    wavelengthA_V = specA_V[:, 0] # wavelength from star a array
    fluxA_V= specA_V[:, 1] # grabs flux from array
    transmissionA_V = interp_func_jV(wavelengthA_V) # calculates new kepler transmission array values of same length of transmission x values
    scaled_fluxA_V = transmissionA_V * fluxA_V

    # B --------------------------------------------------------
    wavelengthB_V = specB_V[:, 0]
    fluxB_V = specB_V[:, 1] # grabs flux from array
    transmissionB_V = interp_func_jV(wavelengthB_V) # calculates new flux values off of transmission x values
    scaled_fluxB_V = transmissionB_V * fluxB_V

    #Integrate to find total fluxes??? of stars as seen through filter of choice
    FA_V = trapezoid(scaled_fluxA_V, x=wavelengthA_V)
    FB_V = trapezoid(scaled_fluxB_V, x=wavelengthB_V)

if Johnson_B:
    # First step in making stellar spectra arrays the same length / wavelength range as Kepler transmission filter
    # finds indices to slice star flux arrays along by finding closest value in array to desired slice wavelength
    A_3570 = (np.abs(spec_A_raw[:, 0] - 3570)).argmin()
    A_5440 = (np.abs(spec_A_raw[:, 0] - 5440)).argmin()
    B_3570 = (np.abs(spec_B_raw[:, 0] - 3570)).argmin()
    B_5440 = (np.abs(spec_B_raw[:, 0] - 5440)).argmin()

    # shortens star spectrum arrays at desired indices
    specA_B = spec_A_raw[A_3570:A_5440]
    specB_B = spec_B_raw[B_3570:B_5440]

    # shortens kepler transmission spectrum to same wavelength regime
    # Kepler needs to be a little larger than star flux arrays so interpolation works
    john_3570 = (np.abs(johnson_B_raw - 3570 + 20)).argmin()  # finds indices to slice transmission array on
    john_5440 = (np.abs(johnson_B_raw - 5440 - 20)).argmin()

    johnB_transmission = johnson_B_raw[john_3570:john_5440]

    # Creates interpolation function to transpose x axis from star data onto kepler transmission spectrum
    interp_func_jB = interp1d(johnB_transmission[:, 0], johnB_transmission[:, 1])

    # A-------------------------------------------------------------------
    # calculates new x axis, uses interpolation function to find new y axis
    wavelengthA_B = specA_B[:, 0]  # wavelength from star a array
    fluxA_B = specA_B[:, 1]  # grabs flux from array
    transmissionA_B = interp_func_jB(wavelengthA_B)  # calculates new kepler transmission array values of same length of transmission x values
    scaled_fluxA_B = transmissionA_B * fluxA_B

    # B --------------------------------------------------------
    wavelengthB_B = specB_B[:, 0]
    fluxB_B = specB_B[:, 1]  # grabs flux from array
    transmissionB_B = interp_func_jB(wavelengthB_B)  # calculates new flux values off of transmission x values
    scaled_fluxB_B = transmissionB_B * fluxB_B

    # Integrate to find total fluxes??? of stars as seen by Kepler bc kepler curve
    FA_B = trapezoid(scaled_fluxA_B, x=wavelengthA_B)
    FB_B= trapezoid(scaled_fluxB_B, x=wavelengthB_B)

# zero point magnitude callibration values for Johnson B, V
B = 6.4e-9 # in ergs/cm-2s-1A-1
B_eff = 992.11
V = 3.75e-9
V_eff = 870.34

conv = 3e-5

f_B_lam = conv * B * B_eff
f_V_lam = conv * V * V_eff

mBV_A = -2.5 * np.log10(FA_B/FA_V) + 2.5 * np.log10(f_B_lam/f_V_lam)
mBV_B = -2.5 * np.log10(FB_B/FB_V) + 2.5 * np.log10(f_B_lam/f_V_lam)

#print(mBV_A, mBV_B) # this is working

#print(kpV_A - 1, kpV_B- 1)
# Kp - v = 2.5log(Fkp/Fv) -> = -0.06
# Kp - v (m0 star) = -0.43

# if right = 0.62 for primary star for B - V
# 1.42 for secondary

# Units after integration: ergs/cm^2/s

# values for star radii from Thomas
to_cm = 6.95700e10 # radius of the sun in cm
R_A = 1.01 * to_cm# radius of A
sig_R_A = 0.04 *to_cm# uncertainty on radius
R_B = 0.52 * to_cm # raidus of B
sig_R_B = 0.06 * to_cm# uncertainty of radius of B

Rp = (RpRs_kepler[0]) * R_A  # Assume Kepler radius is the true radius of brown dwarf, convert from RpRs -> Rp using R_A # Rp in cm

F_sys = (R_A**2)*F_A + (R_B**2)*F_B # flux of system # now in units of ergs/s
sig_RA = 2 * sig_R_A * R_A # uncertainty on first flux term
sig_RB = 2 * sig_R_B * R_B # uncertainty on second flux term
sig_sys = F_A * sig_RA + F_B * sig_RB # uncertainty on system, total

# take stellar model and figure out what surface brightness I get : now takign model and miltiplying by filter, but do this with spitzer 4.5, get surface brightness
# do Sa = integral (spectral model) * (F transmission for spitzer 4.5) d(lam)
# using Sa, calculate T using temperature loop for planet.

# use astropy function for units and/or blackbody function
# astropy blackbody function gives ergs/cm2/s/A/str (may have to multiply by 4pi or something to get the steradians to go away
# )

def find_Sp(depth_lam, RpRs_kepler, depth_k):
    Rp, sig_rp = RpRs_kepler  #radius kepler in cm
    del_lam, sig_lam = depth_lam # depth other channel in ppm
    del_k, sig_k = depth_k # depth kepler in ppm

    # plugging values into formula
    Fn = F_sys * (del_k/del_lam - 1) # solve for nighttime flux

    # uncertainty on value:
    # first, find error on del_k/del_lam:
    sig_delk_dellam = (del_k/del_lam) * np.sqrt((sig_k/del_k)**2 + (sig_lam/del_lam)**2)
    sig_Fn = (F_sys * (del_k/del_lam - 1)) * np.sqrt((sig_delk_dellam/(del_k/del_lam))**2 + (sig_sys/F_sys)**2)

    # Fn = Rp**2 * Fp(lam)
    # Fp(lam) = Fn / (Rp**2)
    Sp = Fn / (Rp**2) # calculate planet luminosity

    sig_Rp2 = sig_rp * 2 * Rp # error on planet radius squared
    sig_Sp = Sp * np.sqrt((sig_Fn/Fn)**2 + (sig_Rp2/Rp)**2) # calculate error on flux from planet

    return Sp, sig_Sp

# uses function to find flux from planet (luminosity??)
Sp_ch1 = find_Sp(depth_ch1, RpRs_kepler, depth_kepler)
Sp_ch2 = find_Sp(depth_ch2, RpRs_kepler, depth_kepler)

# Loads in Spitzer transmission filters
spitz_ch1_raw = np.loadtxt("Spitzer_IRAC.I1.dat")
spitz_ch2_raw = np.loadtxt("Spitzer_IRAC.I2.dat")

# wavelengths
lam_ch1 = spitz_ch1_raw[:, 0]
lam_ch2 = spitz_ch2_raw[:, 0]

# transmission data
spitz_tr_ch1 = spitz_ch1_raw[:, 1]
spitz_tr_ch2 = spitz_ch2_raw[:, 1]

# we know: Sp = integral(planck function (lam, T) * spitzer filter)
    # constants for Planck function in cgs units
h = 6.62617e-27
c = 2.99792458e10
k_b =1.3807e-16
A_to_cm = 1e-8

# defines Planck function for input of lambda, T
def Planck(lam, T):
    t1 = (2 * h * c**2)/((lam*A_to_cm)**5)
    te = (math.e)**((h*c)/(k_b*T*(lam*A_to_cm))) # converts angstroms to cm
    eq = t1 * (1/(te - 1))
    return eq

# T: guess range for object temperature
potential_temps = np.arange(1000, 7000, 20)

fluxes_ch1 = []
fluxes_ch2 = []
fluxes_ch2_A = []

for T in potential_temps: # iterates over potential temperature values in a guess range

    # calculates array of Planck function values for a blackbody of temperature T at a wavelength range defined by the filter
    # bb = BlackBody(temperature=T)
    # planck_ch1 = bb(lam_ch1)
    # planck_ch2 = bb(lam_ch2)

    planck_ch1 = Planck(lam_ch1, T)
    planck_ch2 = Planck(lam_ch2, T)

    # scales Planck function by filter transmission profile
    integrand_ch1 = planck_ch1 * spitz_tr_ch1
    integrand_ch2 = planck_ch2 * spitz_tr_ch2

    # calculates integral of filter * planck function to find expected flux through filter
    value_ch1 = trapezoid(integrand_ch1, x=lam_ch1)
    value_ch2 = trapezoid(integrand_ch2, x=lam_ch2)

    # adds flux value to list
    fluxes_ch1.append(value_ch1)
    fluxes_ch2.append(value_ch2)

# finds index of best temperature fit per flux by finding minimum value of array when subtracting observed value from calculated flux.
int_1 = np.abs(fluxes_ch1 - Sp_ch1[0]).argmin()
int_2 = np.abs(fluxes_ch2 - Sp_ch2[0]).argmin()

# uses best index to find best temperature fit
T_ch1 = potential_temps[int_1]
T_ch2  = potential_temps[int_2]

print("T: ch 1:", T_ch1)
print("T: ch 2:", T_ch2)

#--------------------------------------------------------------------------------------------------------------------
# TRIAL calcualting temperature of star A in ch2 to verify methods of temperature calcualtion
    # first calcualte integral of star spectrum over spitzer ch2 bandpass Sa_A_ch2
A_min = (np.abs(spec_A_raw[:, 0] - 37224.9-10)).argmin()
A_max = (np.abs(spec_A_raw[:, 0] - 52219.8+10)).argmin()

# shortens star spectrum arrays at desired indices
spec_A = spec_A_raw[A_min:A_max]

# Creates interpolation function to transpose x axis from star data onto spitzer transmission spectrum
interp_func = interp1d(spitz_ch2_raw[:, 0], spitz_ch2_raw[:, 1])

# calculates new x axis, uses interpolation function to find new y axis
wavelength_A = spec_A[:, 0]  # wavelength from star a array
flux_A = spec_A[:, 1]  # grabs flux from array
transmission_A = interp_func(wavelength_A)  # calculates new kepler transmission array values of same length of transmission x values
scaled_flux_A = transmission_A * flux_A

F_A = trapezoid(scaled_flux_A, x=wavelength_A)
print(F_A)

    # then use calculated value as Sa_A_ch2 = integral(planck function(lam, T) * transmission) over lam to find T

# T: guess range for object temperature
potential_temps = np.arange(1000, 7000, 20)

fluxes_ch2_A = []

for T in potential_temps: # iterates over potential temperature values in a guess range

    # calculates array of Planck function values for a blackbody of temperature T at a wavelength range defined by the filter
    planck_ch2 = Planck(lam_ch2, T)

    # scales Planck function by filter transmission profile
    integrand_ch2 = planck_ch2 * spitz_tr_ch2

    # calculates integral of filter * planck function to find expected flux through filter
    value_ch2 = trapezoid(integrand_ch2, x=lam_ch2)

    # adds flux value to list
    fluxes_ch2_A.append(value_ch2)

# finds index of best temperature fit per flux by finding minimum value of array when subtracting observed value from calculated flux.
int_2 = (np.abs(fluxes_ch2_A - F_A)).argmin()

# uses best index to find best temperature fit
T_ch2  = potential_temps[int_2]

print("T: ch 2:", T_ch2)




