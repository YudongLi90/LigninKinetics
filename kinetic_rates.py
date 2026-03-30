import numpy as np

def arrhenius(A, Ea, T):
    R = 8.314 # J/(mol*K)
    return A * np.exp(-Ea/(R*T))

def rates(X: list, c_naoh:float, A: list, Ea: list, T: float, b:list):
    X_lig, X_ace, X_xyl, X_C = X   
    A_lig, A_ace, A_xyl, A_C = A
    Ea_lig, Ea_ace, Ea_xyl, Ea_C = Ea
    b_lig, b_ace, b_xyl, b_C = b
    T_K = T + 273.15 # convert to K
    r_lig = -arrhenius(A_lig, Ea_lig, T_K) * c_naoh * X_lig
    r_ace = -arrhenius(A_ace, Ea_ace, T_K) * c_naoh * X_ace
    r_xyl = -arrhenius(A_xyl, Ea_xyl, T_K) * c_naoh * X_xyl
    r_C = -arrhenius(A_C, Ea_C, T_K) * c_naoh * X_C
    r_naoh = b_lig * r_lig + b_ace * r_ace + b_xyl * r_xyl + b_C * r_C
    return r_lig, r_ace, r_xyl, r_C, r_naoh

