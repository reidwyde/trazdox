class growth_model(object):
    def __init__(self, times, T0):
        self._times = times
        self._T0 = T0
        
        
    def dTdt(self, *argv):
        return 0
    def dDdt(self, *argv):
        return 0
    def dHdt(self, *argv):
        return 0
    def dOdt(self, *argv):
        return 0
    
    def rk_T(self, *argv):
        return 0
    def rk_D(self, *argv):
        return 0
    def rk_H(self, *argv):
        return 0
    def rk_O(self, *argv):
        return 0

    def get_param(self, param_name, n=10):
        return pm.summary(self.trace)['mean'][param_name]
    
    def print_params(self):
        print(self.param_list)
        
"""
growth model class must be modified to reflect a new differential equation
list of things that must be changed:

groups
simulate
dTdt, dDdt, etc...
rk_T, rk_D, etc...
in backwards method: param_estimates
th_forward_model
"""    

from get_tumor_db import get_tumor_db
from parse_tumor_db import parse_tumor_db
import numpy as np
import treatment_model as tm
import plot_data
import rk_utils
import pymc3 as pm
import theano
import theano.tensor as tt
from theano.compile.ops import as_op

ts, Ts, sigmas = parse_tumor_db(get_tumor_db())

class growth_model_1(growth_model):
    def __init__(self):
        
        tumor_size_db = get_tumor_db()
        ts, Ts, sigmas = parse_tumor_db(tumor_size_db)

        super().__init__(ts, np.array(Ts[:,0]).reshape(-1,))
        self.sim_times = np.linspace(7,70,100*(70-7+1)) # original time indexing started at day 7
        
        self.groups, self._times = [0,1,2,3,4,5], ts
        self.Sds_fit, self.Shs_fit = tm.get_Sd_impulse(ts), tm.get_Sh_impulse(ts)
        self.Sds_sim, self.Shs_sim = tm.get_Sd_impulse(self.sim_times), tm.get_Sh_impulse(self.sim_times)
        self.Sds, self.Shs = self.Sds_fit, self.Shs_fit 

        #lower and upper estimates
        self.param_estimates = {
            'r':[0.0001,0.1],
            'lambda_h': [0.0001, 10],
            'lambda_d': [0.0001, 10],
            'lambda_hd': [0.0001, 10],
            'tau_d': [0.0001, 0.5],
            'tau_h': [0.0001, 0.3],
            'lambda_dh': [0.0001,10]
        }
        
        self.param_list = list(self.param_estimates.keys())
            
        plot_data.plot_combined_treatment(ts, self.groups, self.Sds, self.Shs)
        
        
    def _simulate(self, fit_params, times):
        #self.sim_T, self.sim_D, self.sim_H, self.sim_O = self.rungeKutta_all_params(times, \
        #                    self._T0, self.dTdt, self.dDdt, self.dHdt, self.dOdt, self.Sds, self.Shs, fit_params)
        self.sim_T, self.sim_D, self.sim_H, self.sim_O = self.rungeKutta_all_params(times, fit_params)
        return self.sim_T
    """
    def simulate(self, r, lambda_h, lambda_d, lambda_hd, tau_d, tau_h, lambda_dh, times=None):
        if times is None: times = self._times        
        return self._simulate([r, lambda_h, lambda_d, lambda_hd, tau_d, tau_h, lambda_dh], times)
    """
    def simulate(self, fit_params, times=None):
        if times is None: times = self._times        
        return self._simulate(fit_params, times)
    
    
    #don't take the respective state variable from state_vec here, since runge kutta relies on 
    #the respective state variable being iterated forward
    def dDdt(self, D, state_vec, fit_params):
        #r, lambda_h, lambda_d, lambda_hd, tau_d, tau_h, lambda_dh = [x for x in params]
        tau_D = fit_params[4]
        return -tau_D*D
    
    def dHdt(self, H, state_vec, fit_params):
        #r, lambda_h, lambda_d, lambda_hd, tau_d, tau_h, lambda_dh = [x for x in params]
        tau_H = fit_params[5]
        return -tau_H*H
    
    def dOdt(self, O, state_vec, fit_params):
        #r, lambda_h, lambda_d, lambda_hd, tau_d, tau_h, lambda_dh = [x for x in params]
        return 0
    
    def dTdt(self, T, state_vec, fit_params):
        #r, lambda_h, lambda_d, lambda_hd, tau_d, tau_h, lambda_dh = [x for x in params]
        r = fit_params[0]
        lambda_h = fit_params[1]
        lambda_d = fit_params[2]
        lambda_hd = fit_params[3]
        D = state_vec[0]
        H = state_vec[1]
        return (r - lambda_h*H - lambda_d*D - lambda_hd * H * D)*T
    
    def rk_T(self, h, state_vec, fit_params):
        T = state_vec[3]
        return rk_utils.rk_X(h, T, state_vec, fit_params, self.dTdt)
    
    def rk_D(self, h, state_vec, fit_params):
        D = state_vec[0]
        return rk_utils.rk_X(h, D, state_vec, fit_params, self.dDdt)
    
    def rk_H(self, h, state_vec, fit_params):
        H = state_vec[1]
        return rk_utils.rk_X(h, H, state_vec, fit_params, self.dHdt)
    
    def rk_O(self, h, state_vec, fit_params):
        O = state_vec[2]
        return rk_utils.rk_X(h, O, state_vec, fit_params, self.dOdt)
    
    
    def rungeKutta_all_params(self, times, fit_params):
        _, _, _, _, _, _, lambda_dh = [x for x in fit_params]
        time_len = len(times.ravel())
        ret_T = np.zeros((self._T0.shape[0], time_len))
        ret_D = np.zeros(ret_T.shape)
        ret_H = np.zeros(ret_T.shape)
        ret_O = np.zeros(ret_T.shape)
        T = self._T0
        D = ret_D[:,0]
        H = ret_H[:,0]
        O = ret_O[:,0]
        ret_T[:, 0] = T
        for i in range(1, time_len):
            h = times[i] - times[i-1]
            Sd = self.Sds[:,i-1]
            Sh = self.Shs[:,i-1]

            state_vec = [D, H, O, T]

            D_new = rk_utils.rk_var_update(D, self.rk_D(h, state_vec, fit_params)) + Sd
            H_new = rk_utils.rk_var_update(H, self.rk_H(h, state_vec, fit_params)) + Sh*np.exp(-lambda_dh*D)
            O_new = rk_utils.rk_var_update(O, self.rk_O(h, state_vec, fit_params))
            T_new = rk_utils.rk_var_update(T, self.rk_T(h, state_vec, fit_params))

            D = D_new
            H = H_new
            O = O_new         
            T = T_new 

            ret_D[:,i] = D
            ret_H[:,i] = H
            ret_O[:,i] = O
            ret_T[:,i] = T

        return ret_T, ret_D, ret_H, ret_O

    def backward(self):
        T_obs = Ts
        #sigmas_obs = np.ones(T_obs.shape)
        sigmas_obs = np.repeat(sigmas[0:,0:1], T_obs.shape[1], axis=1)
        
        with pm.Model() as model:
            #form priors
            prior_distributions = [pm.Uniform(list(self.param_estimates.keys())[ii], lower=self.param_estimates[self.param_list[ii]][0], upper=self.param_estimates[self.param_list[ii]][1]) for ii in range(len(self.param_list))]
            
            prior_distributions = tt.as_tensor_variable(prior_distributions)
            
            @as_op(itypes=[tt.dvector], otypes=[tt.dmatrix]) 
            def th_forward_model(prior_distributions):
                th_states = self.simulate(prior_distributions)
                return th_states
            
            forward = th_forward_model(prior_distributions)
            
            
            T = pm.Normal('T', mu=forward, sigma=sigmas_obs, observed=T_obs)

            # Initial points for each of the chains
            np.random.seed(100)
            n_chains = 5
            
            
            #random start positions for MCMC
            startsmc = []
            for _ in range(n_chains):
                startsmc_dict = {}
                for param_name in self.param_list:
                    startsmc_dict[param_name] = np.random.uniform(self.param_estimates[param_name][0],self.param_estimates[param_name][1])
                startsmc.append(startsmc_dict)
            
            num_samples = 80 # Within each chain, operations are sequential and cannot be parallelized.
            # if the number of samples drops too low, then the model will throw a negative minors in cholesky factorization
            num_tune = int(num_samples/5)
            #step = pm.SMC()
            
            step = pm.Metropolis()
            self.trace = pm.sample(num_samples, step=step, tune=num_tune, chains = n_chains, cores=1, start=startsmc)
            """
            self.trace = pm.sample_smc(2000)
            """
            
            pm.traceplot(self.trace)
            #print('log marginal likelihood: ' + str(np.log(model.marginal_likelihood)))
            #pm.compare(self.trace, ic='WAIC')
            #pm.dic(self.trace, model)
            pooled_waic = pm.waic(self.trace, model)
            print(pooled_waic)  
            model.name = 'model1'
            #pm.compare(self.trace)
            self.df_comp_WAIC = pm.compare({model: self.trace})
