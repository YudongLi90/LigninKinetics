import traceback
import numpy as np

from constants import *
from kinetic_rates import arrhenius
from utils import calc_pH

from scikits.odes.odeint import odeint

import os
from importlib import reload 
import logging
reload(logging)

logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.INFO, datefmt='%I:%M:%S')


class Deacetylation:
    def __init__(self, debug=False, silent=True):    
        self._w = None # mol/L for naoh, mass fraction g/g for Lignin, Acetyl, Xylan, Cellulose, 
        self._A = None
        self._Ea = None
        self._b = None
        self._n = None
        self._T = None
        self._orders = None
        self._duration = 60*60*2 # 2 hours in seconds
        self.molewt_NaOH = 40.0 # g/mol
        self.phi_l_s = 20/600 # L/g, liquid to solid ratio (20L liquor, 600g biomass)
        self.silent = silent
        if not self.silent:
            self.logdir = os.path.join(os.getcwd(), 'logs')
            if not os.path.exists(self.logdir):
                os.makedirs(self.logdir)
            
            self.logger = logging.getLogger('Deacetylation')
            if debug:
                self.logger.setLevel(logging.DEBUG)
            
            if not self.logger.handlers:
                file_path = os.path.join(self.logdir, 'deacetylation.log')
                file_handler = logging.FileHandler(file_path, mode='a') # mode='a' appends
                formatter = logging.Formatter('%(asctime)s %(levelname)s:%(message)s', datefmt='%I:%M:%S')
                file_handler.setFormatter(formatter)
                self.logger.addHandler(file_handler)


    def first_order_rate(self, w, rates):
        r_lig = -arrhenius(self._A[Lignin], self._Ea[Lignin], self._T) * w[NaOH] * w[Lignin]
        r_ace = -arrhenius(self._A[Acetyl], self._Ea[Acetyl], self._T) * w[NaOH] * w[Acetyl]
        r_xyl = -arrhenius(self._A[Xylan], self._Ea[Xylan], self._T) * w[NaOH] * w[Xylan]
        r_C = -arrhenius(self._A[Glucan], self._Ea[Glucan], self._T) * w[NaOH] * w[Glucan]
        r_naoh = self._b[Lignin] * r_lig + self._b[Acetyl] * r_ace + self._b[Xylan] * r_xyl + self._b[Glucan] * r_C
        # convert r_naoh from g/g/s to mol/L/s
        r_naoh = r_naoh / self.phi_l_s / self.molewt_NaOH if w[NaOH] > 0 else 0
        
        rates[Lignin] = r_lig
        rates[Acetyl] = r_ace
        rates[Xylan] = r_xyl
        rates[Glucan] = r_C
        rates[NaOH] = r_naoh

    def kinetic_rate(self, w, rates):
        naoh_conc = max(w[NaOH], 1e-10) # avoid zero
        lig_conc = max(w[Lignin], 0)
        ace_conc = max(w[Acetyl], 0)
        xyl_conc = max(w[Xylan], 0)
        glu_conc = max(w[Glucan], 0)

        r_lig = -arrhenius(self._A[Lignin], self._Ea[Lignin], self._T) * naoh_conc**self._n[Lignin] * lig_conc
        r_ace = -arrhenius(self._A[Acetyl], self._Ea[Acetyl], self._T) * naoh_conc**self._n[Acetyl] * ace_conc
        r_xyl = -arrhenius(self._A[Xylan], self._Ea[Xylan], self._T) * naoh_conc**self._n[Xylan] * xyl_conc
        r_C = -arrhenius(self._A[Glucan], self._Ea[Glucan], self._T) * naoh_conc**self._n[Glucan] * glu_conc
        r_naoh = self._b[Lignin] * r_lig + self._b[Acetyl] * r_ace + self._b[Xylan] * r_xyl + self._b[Glucan] * r_C
        # convert r_naoh from g/g/s to mol/L/s
        r_naoh = r_naoh / self.phi_l_s / self.molewt_NaOH if w[NaOH] > 0 else 0
        
        rates[Lignin] = r_lig
        rates[Acetyl] = r_ace
        rates[Xylan] = r_xyl
        rates[Glucan] = r_C
        rates[NaOH] = r_naoh


    
    def run_deacetylation(self):
        # assert all properties are set
        assert self._w is not None, "Initial concentrations (w) are not set."
        assert self._A is not None, "Pre-exponential factors (A) are not set."
        assert self._Ea is not None, "Activation energies (Ea) are not set."
        assert self._b is not None, "Stoichiometric coefficients (b) are not set."
        assert self._n is not None, "Reaction orders (n) are not set."
        assert self._T is not None, "Temperature (T) is not set."
        self._w[NaOH] -= self.titration_amount / self.phi_l_s / self.molewt_NaOH # subtract titration amount from initial NaOH concentration (after converting from g/g to mol/L)

        def rhs(t, w, rates):
            self.kinetic_rate(w, rates)
            if not self.silent:
                self.logger.debug(f"Time: {t}, Rates: {rates}, Concentrations: {w}")
        
        extra_options = {'old_api': False, 'rtol': 1e-6, 'atol': 1e-12, 'max_steps': 500000}
        try:
            solution = odeint(rhs, np.linspace(0, self._duration, int(self._duration/5)+1), self._w, method='bdf', **extra_options)
        except Exception as e:
            if not self.silent:
                self.logger.error(f"ODE solver failed: temperature {self._T-273.15} NaOH loading {self.NaOH_loading}, initial concentrations {self._w}, A {self._A}, Ea {self._Ea}, b {self._b}. Error: {e}")
                self.logger.error(traceback.format_exc())
            raise e

        return solution.values.t, solution.values.y
    
    
    def set_experimental_data(self, datadict):
        self.datadict = datadict
        self.l_time = datadict['time']*60 # convert minutes to seconds
        self.l_lig_yield = datadict['Lignin_yield']
        self.l_ace_yield = datadict['Acetyl_yield']
        self.l_xyl_yield = datadict['Xylan_yield']
        self.l_glu_yield = datadict['Glucan_yield']
        self.l_pH = datadict['pH'] 
        self.l_NaOH_yield = datadict['NaOH_yield']
        self.NaOH_loading = datadict['NaOH_loading']
        w = np.zeros(5)
        w[NaOH] = datadict['NaOH_loading'] * 1e-3
        w[NaOH] = w[NaOH] / self.phi_l_s / self.molewt_NaOH # convert g/g biomass of NaOH to mol/L
        w[Lignin] = datadict["Lignin"]*0.01 
        w[Xylan] = datadict["Xylan"]*0.01
        w[Glucan] = datadict["Glucan"]*0.01
        w[Acetyl] = datadict["Acetyl"]*0.01
        self._w = w
        self._T = datadict["temperature"] + 273.15
        if not self.silent:
            self.logger.debug(f"Experimental data set: {datadict}")
    
    def set_exp_conditons(self, exp_conditon_dict):
        self.NaOH_loading = exp_conditon_dict['NaOH_loading']
        w = np.zeros(5)
        w[NaOH] = exp_conditon_dict['NaOH_loading'] * 1e-3
        w[NaOH] = w[NaOH] / self.phi_l_s / self.molewt_NaOH # convert g/g biomass of NaOH to mol/L
        w[Lignin] = exp_conditon_dict["Lignin"]*0.01 
        w[Xylan] = exp_conditon_dict["Xylan"]*0.01
        w[Glucan] = exp_conditon_dict["Glucan"]*0.01
        w[Acetyl] = exp_conditon_dict["Acetyl"]*0.01
        self._w = w
        self._T = exp_conditon_dict["temperature"] + 273.15
        self._duration = exp_conditon_dict.get("duration", self._duration) # use default duration if not provided

    def get_yield_pred_all(self):
        t, y = self.run_deacetylation()
        self.pred = {
            'time': t,
            'Lignin': y[:, Lignin],
            'Acetyl': y[:, Acetyl],
            'Xylan': y[:, Xylan],
            'Glucan': y[:, Glucan],
            'NaOH': y[:, NaOH], # in mol/L
            'pH': calc_pH(y[:, NaOH])
        }

        self.pred["NaOH_yield"] = (self.NaOH_loading/1000 - y[:, NaOH] * self.phi_l_s * self.molewt_NaOH)/(self.NaOH_loading/1000)
        # initialize exp_pred with empty lists 
        exp_pred = {k: [] for k in self.pred.keys()}

        
        exp_pred['time'].append(self._duration)
        for key in ['Lignin', 'Acetyl', 'Xylan', 'Glucan', 'NaOH']:
            try:
                exp_pred[key].append(self.pred[key][-1].item())
            except IndexError as e:
                if not self.silent:
                    self.logger.error(f"key {key}, yield Error: {e}")
                raise e
        exp_pred['pH'].append(calc_pH(y[-1, NaOH].item()))

        exp_pred['NaOH_yield'] = 1 - np.array(exp_pred['NaOH']) * self.phi_l_s * self.molewt_NaOH/(self.NaOH_loading/1000)
        exp_pred['Glucan_yield'] = 1 - np.array(exp_pred['Glucan'])/self._w[Glucan]
        exp_pred['Xylan_yield'] = 1 - np.array(exp_pred['Xylan'])/self._w[Xylan]
        exp_pred['Lignin_yield'] = 1 - np.array(exp_pred['Lignin'])/self._w[Lignin]
        exp_pred['Acetyl_yield'] = 1 - np.array(exp_pred['Acetyl'])/self._w[Acetyl]
        return exp_pred
    
    def get_yield_at_timelst(self, time_lst):
        t, y = self.run_deacetylation()
        pred = {
            'time': t,
            'Lignin': y[:, Lignin],
            'Acetyl': y[:, Acetyl],
            'Xylan': y[:, Xylan],
            'Glucan': y[:, Glucan],
            'NaOH': y[:, NaOH], # in mol/L
            'pH': calc_pH(y[:, NaOH])
        }

        pred["NaOH_yield"] = (self.NaOH_loading/1000 - y[:, NaOH] * self.phi_l_s * self.molewt_NaOH)/(self.NaOH_loading/1000)
        exp_pred = {k: [] for k in pred.keys()}

        for time in time_lst:
            if time > self._duration:
                if not self.silent:
                    self.logger.error(f"Requested time {time} exceeds simulation duration {self._duration}. Returning NaN.")
                raise ValueError(f"Requested time {time} exceeds simulation duration {self._duration}.")
            else:
                exp_pred['time'].append(time)
                for key in ['Lignin', 'Acetyl', 'Xylan', 'Glucan', 'NaOH']:
                    try:
                        exp_pred[key].append(pred[key][t==time].item())
                    except IndexError as e:
                        if not self.silent:
                            self.logger.error(f"Time {time},  key {key}, yield Error: {e}")
                        raise e
                exp_pred['pH'].append(calc_pH(y[t==time, NaOH].item()))

        exp_pred['NaOH_yield'] = 1 - np.array(exp_pred['NaOH']) * self.phi_l_s * self.molewt_NaOH/(self.NaOH_loading/1000)
        exp_pred['Glucan_yield'] = 1 - np.array(exp_pred['Glucan'])/self._w[Glucan]
        exp_pred['Xylan_yield'] = 1 - np.array(exp_pred['Xylan'])/self._w[Xylan]
        exp_pred['Lignin_yield'] = 1 - np.array(exp_pred['Lignin'])/self._w[Lignin]
        exp_pred['Acetyl_yield'] = 1 - np.array(exp_pred['Acetyl'])/self._w[Acetyl]
        return exp_pred

    def get_prediction(self):
        t, y = self.run_deacetylation()
        self.pred = {
            'time': t,
            'Lignin': y[:, Lignin],
            'Acetyl': y[:, Acetyl],
            'Xylan': y[:, Xylan],
            'Glucan': y[:, Glucan],
            'NaOH': y[:, NaOH], # in mol/L
            'pH': calc_pH(y[:, NaOH])
        }

        self.pred["NaOH_yield"] = (self.NaOH_loading/1000 - y[:, NaOH] * self.phi_l_s * self.molewt_NaOH)/(self.NaOH_loading/1000)
        # initialize exp_pred with empty lists 
        exp_pred = {k: [] for k in self.pred.keys()}

        for time in self.l_time:
            if time > self._duration:
                if not self.silent:
                    self.logger.error(f"Requested time {time} exceeds simulation duration {self._duration}. Returning NaN.")
                raise ValueError(f"Requested time {time} exceeds simulation duration {self._duration}.")
            else:
                    exp_pred['time'].append(time)
                    for key in ['Lignin', 'Acetyl', 'Xylan', 'Glucan', 'NaOH']:
                        try:
                            # exp_pred[key].append(self.pred[key][t==time].item())
                            interp_val = np.interp(time, t, self.pred[key])
                            exp_pred[key].append(interp_val)
                        except IndexError as e:
                            if not self.silent:
                                self.logger.error(f"Time {time},  key {key}, yield Error: {e}")
                            raise e
                    # exp_pred['pH'].append(calc_pH(y[t==time, NaOH].item()))
                    interp_naoh = np.interp(time, t, self.pred['NaOH'])
                    exp_pred['pH'].append(calc_pH(interp_naoh))


        exp_pred['NaOH_yield'] = 1 - np.array(exp_pred['NaOH']) * self.phi_l_s * self.molewt_NaOH/(self.NaOH_loading/1000)
        exp_pred['Glucan_yield'] = 1 - np.array(exp_pred['Glucan'])/self._w[Glucan]
        exp_pred['Xylan_yield'] = 1 - np.array(exp_pred['Xylan'])/self._w[Xylan]
        exp_pred['Lignin_yield'] = 1 - np.array(exp_pred['Lignin'])/self._w[Lignin]
        exp_pred['Acetyl_yield'] = 1 - np.array(exp_pred['Acetyl'])/self._w[Acetyl]

        exp = {
            'time': self.l_time,
            'Lignin_yield': self.l_lig_yield,
            'Acetyl_yield': self.l_ace_yield,
            'Xylan_yield': self.l_xyl_yield,
            'Glucan_yield': self.l_glu_yield,
            'pH': self.l_pH,
            'NaOH_yield': self.l_NaOH_yield
        }
        # for key in exp_pred.keys():
        #     exp_pred[key] = np.array(exp_pred[key])
        return exp, exp_pred 

        
    def set_parameters(self, A, Ea, b=np.array([0,0.2,0.93,0,0]), n=np.ones(5), titration_amount=0):
        """_summary_

        Args:
            A (_type_): _description_
            Ea (_type_): _description_
            b (_type_): _description_
        """
        # self._w = w
        # # convert g/g biomass of NaOH to mol/L
        # self._w[NaOH] = w[NaOH] / self.phi_l_s / self.molewt_NaOH
        self._A = A
        self._Ea = Ea
        self._b = b
        self._n = n
        self.titration_amount = titration_amount # in g/g biomass: amount of NaOH added during titration, converted to mol/L in the run_deacetylation method
        if not self.silent:
            self.logger.debug(f"Kinetic parameters set: A={A}, Ea={Ea}, b={b}, n={n}, titration_amount={titration_amount}")

    def set_parameters_from_dict(self, params):
        A = np.zeros(5)
        Ea = np.zeros(5)

        A[Lignin] = params['A_lig']
        A[Acetyl] = params['A_ace']
        Ea[Lignin] = params['Ea_lig']*1000 # convert from kJ/mol to J/mol
        Ea[Acetyl] = params['Ea_ace']*1000 # convert from kJ/mol to J/mol

        b = np.zeros(5)
        b[Lignin] = params['b_lig']
        b[Acetyl] = 0.93 # theoretical acetyl to NaOH

        n = np.ones(5)
        n[Lignin] = params['n_lig']
        n[Acetyl] = params['n_ace']

        self.set_parameters(A, Ea, b, n)