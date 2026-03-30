import numpy as np
import math

def get_kw_at_temp(temp_celsius):
    """
    Calculates the autoionization constant of water (Kw) at a specific temperature.
    Uses a standard thermodynamic equation (T in Kelvin).
    """
    temp_k = temp_celsius + 273.15
    ln_kw = -5839.42 / temp_k - 22.4773 * math.log(temp_k) + 61.2062
    kw = math.exp(ln_kw)
    return kw

def get_pkw_at_temp(temp_celsius):
    """Returns pKw (-log10 Kw) for a given Celsius temperature."""
    kw = get_kw_at_temp(temp_celsius)
    return -math.log10(kw)

def calculate_ph_from_concentration(molarity, temp_celsius):
    """
    Calculates pH from NaOH concentration (mol/L).
    """
    if molarity <= 0:
        raise ValueError("Concentration must be greater than 0")
        
    oh_concentration = molarity
    poh = -math.log10(oh_concentration)
    pkw = get_pkw_at_temp(temp_celsius)
    ph = pkw - poh
    return ph

def calc_pH_from_wt_ratio_T(wt_ratio, phi_l_s, molewt_NaOH, temp_celsius):
    """
    Calculates pH from NaOH weight ratio (g/g).
    """
    molarity = calc_conc_from_wt_ratio(wt_ratio, phi_l_s, molewt_NaOH)
    return calculate_ph_from_concentration(molarity, temp_celsius)


def calculate_concentration_from_ph(ph, temp_celsius):
    """
    Calculates NaOH concentration (mol/L) from pH.
    """
    pkw = get_pkw_at_temp(temp_celsius)
    poh = pkw - ph
    oh_concentration = 10 ** (-poh) 
    return oh_concentration

def calc_wt_ratio_from_ph_T(pH, phi_l_s, molewt_NaOH, temp_celsius):
    molarity = calculate_concentration_from_ph(pH, temp_celsius)
    wt_ratio = calc_wt_ratio_from_conc(molarity, phi_l_s, molewt_NaOH)
    return wt_ratio


def calc_pH(NaOH_conc):
    # NaOH_conc is in mol/L
    try:
        clipped = np.clip(np.array(NaOH_conc), 1e-10, None) # avoid log of zero or negative
        pH = 14 + np.log10(clipped)
    except ValueError as e:
        print(f"Error calculating pH: {e} with NaOH_conc={NaOH_conc}")
        pH = None
    return pH

    
def calc_wt_ratio_from_pH(pH, phi_l_s, molewt_NaOH):
    NaOH_conc = 10**(-14 + pH) # mol/L
    wt_ratio = NaOH_conc * molewt_NaOH * phi_l_s # g/g
    return wt_ratio

def calc_wt_ratio_from_conc(NaOH_conc, phi_l_s, molewt_NaOH):
    wt_ratio = NaOH_conc * molewt_NaOH * phi_l_s # g/g
    return wt_ratio

def calc_conc_from_wt_ratio(wt_ratio, phi_l_s, molewt_NaOH):
    NaOH_conc = wt_ratio / (molewt_NaOH * phi_l_s) # mol/L
    return NaOH_conc



    