import numpy as np

def arrhenius(A, Ea, T):
    R = 8.314 # J/(mol*K)
    return A * np.exp(-Ea/(R*T))

def rates(X: list, c_naoh:float, A: list, Ea: list, T: float, b:list):
    X_lig, X_ace= X   
    A_lig, A_ace= A
    Ea_lig, Ea_ace= Ea
    b_lig, b_ace = b
    T_K = T + 273.15 # convert to K
    r_lig = -arrhenius(A_lig, Ea_lig, T_K) * c_naoh * X_lig
    r_ace = -arrhenius(A_ace, Ea_ace, T_K) * c_naoh * X_ace
    r_naoh = b_lig * r_lig + b_ace * r_ace 
    return r_lig, r_ace, r_naoh

