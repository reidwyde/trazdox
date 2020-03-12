#todo: fix get param

import arviz as az
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
    
    def get_param(self, param_name, n=10):
        return pm.summary(self.trace)['mean'][param_name]
    
    def print_params(self):
        print(self.param_list)
        
    def _simulate(self, fit_params, times):
        self.sim_T, self.sim_D, self.sim_H, self.sim_O = self.rungeKutta_all_params(times, fit_params)
        return self.sim_T
   
    def simulate(self, fit_params, times=None):
        if times is None: times = self._times        
        return self._simulate(fit_params, times)
        
        
    #for numerical stability within RK, we need to do the application of the drugs after having calculated the rk_var_update
    #this means that the blocking variable lambda_dh needs to be involved in this method. If lambda_dh is not the last param 
    #in fit params for a given model, this method needs to be overwritten in the child model class
    def rungeKutta_all_params(self, times, fit_params):
        lambda_dh = fit_params[-1]
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
            
            #random start positions for MCMC
            startsmc = []
            for _ in range(self.num_chains):
                startsmc_dict = {}
                for param_name in self.param_list:
                    startsmc_dict[param_name] = np.random.uniform(self.param_estimates[param_name][0],self.param_estimates[param_name][1])
                startsmc.append(startsmc_dict)
            
            # Within each chain, operations are sequential and cannot be parallelized.
            # if the number of samples drops too low, then the model will throw a negative minors in cholesky factorization
            num_tune = int(self.num_samples/2)
            
            #step = pm.SMC()
            step = pm.Metropolis()
            prior = pm.sample_prior_predictive()
            self.trace = pm.sample(self.num_samples, step=step, tune=num_tune, chains = self.num_chains, cores=1, start=startsmc)
            pm.traceplot(self.trace)
            
            posterior_predictive = pm.sample_posterior_predictive(self.trace)
            
            data = az.from_pymc3(trace=self.trace, prior = prior, posterior_predictive = posterior_predictive)
            
            az.plot_posterior(data,round_to=4, credible_interval=0.95);
            
            
            #doesn't work
            #print('log marginal likelihood: ' + str(np.log(model.marginal_likelihood)))
            #pm.compare(self.trace, ic='WAIC')
            #pm.dic(self.trace, model)
            #pm.compare(self.trace)
            
            
            #turned off
            #pooled_waic = pm.waic(self.trace, model)
            #print(pooled_waic)  
            #model.name = 'model1'
            #self.df_comp_WAIC = pm.compare({model: self.trace})

        return
    
    def sim_graph_model(self, sim_params):
        self.Sds = self.Sds_sim
        self.Shs = self.Shs_sim
        sim_T = self.simulate(sim_params, self.sim_times)
        plot_data.plot_sims_vs_actual(self.groups, ts, Ts, sigmas, self.sim_times, sim_T)
        plot_data.plot_D(self.sim_times, self.sim_D)
        plot_data.plot_H(self.sim_times, self.sim_H)
        plot_data.plot_O(self.sim_times, self.sim_O)

        for ii in range(len(sim_params)):
            print(self.param_list[ii] + ' = ' + str(sim_params[ii]))

        return
    
    #todo: make it so fit_sim calls sim
    def fit_sim_graph_model(self):
        self.backward()
        self.Sds = self.Sds_sim
        self.Shs = self.Shs_sim
        sim_T = self.simulate([self.get_param(x) for x in self.param_list], self.sim_times)
        plot_data.plot_sims_vs_actual(self.groups, ts, Ts, sigmas, self.sim_times, sim_T)
        plot_data.plot_D(self.sim_times, self.sim_D)
        plot_data.plot_H(self.sim_times, self.sim_H)
        plot_data.plot_O(self.sim_times, self.sim_O)

        for ii in range(len(self.param_list)):
            print(self.param_list[ii] + ' = ' + str(self.get_param(ii)))

        return 
        
        
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
            'lambda_h': [0.0001, 0.1],
            'lambda_hd': [0.0001, 0.2],
            'tau_d': [0, 0.5],
            'tau_h': [0.0001, 0.3],
            'lambda_dh': [0.0001,40]
        }
        
        self.param_list = list(self.param_estimates.keys())
        
        self.num_samples = 2000
        self.num_chains = 40
            
        plot_data.plot_combined_treatment(ts, self.groups, self.Sds, self.Shs)
        
    
    #don't take the respective state variable from state_vec here, since runge kutta relies on 
    #the respective state variable being iterated forward
    def dDdt(self, D, state_vec, fit_params):
        #r, lambda_h, lambda_hd, tau_d, tau_h, lambda_dh = [x for x in params]
        tau_D = fit_params[3]
        return -tau_D*D
    
    def dHdt(self, H, state_vec, fit_params):
        #r, lambda_h, lambda_hd, tau_d, tau_h, lambda_dh = [x for x in params]
        tau_H = fit_params[4]
        return -tau_H*H
    
    def dOdt(self, O, state_vec, fit_params):
        #r, lambda_h, lambda_hd, tau_d, tau_h, lambda_dh = [x for x in params]
        return 0
    
    def dTdt(self, T, state_vec, fit_params):
        #r, lambda_h, lambda_hd, tau_d, tau_h, lambda_dh = [x for x in params]
        r = fit_params[0]
        lambda_h = fit_params[1]
        lambda_hd = fit_params[2]
        D = state_vec[0]
        H = state_vec[1]
        return (r - lambda_h*H - lambda_hd * H * D)*T
    
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
    
     
class growth_model_2(growth_model):
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
            'lambda_h': [0, 0.1],
            'lambda_ho': [0, 1],
            'lambda_od': [0, 1],
            'tau_o': [0, 0.5],
            'tau_d': [0, 0.5],
            'tau_h': [0, 0.3],
            'lambda_dh': [0,40]
        }
        
        self.param_list = list(self.param_estimates.keys())
        
        self.num_samples = 1000
        self.num_chains = 40
            
        plot_data.plot_combined_treatment(ts, self.groups, self.Sds, self.Shs)
        
        
    #don't take the respective state variable from state_vec here, since runge kutta relies on 
    #the respective state variable being iterated forward
    def dDdt(self, D, state_vec, fit_params):
        #r, lambda_h, lambda_ho, lambda_od, tau_o, tau_d, tau_h, lambda_dh = [x for x in params]
        tau_d = fit_params[5]
        return -tau_d*D
    
    def dHdt(self, H, state_vec, fit_params):
        #r, lambda_h, lambda_ho, lambda_od, tau_o, tau_d, tau_h, lambda_dh = [x for x in params]
        tau_h = fit_params[6]
        return -tau_h*H
    
    def dOdt(self, O, state_vec, fit_params):
        #r, lambda_h, lambda_ho, lambda_od, tau_o, tau_d, tau_h, lambda_dh = [x for x in params]
        D = state_vec[0]
        lambda_od = fit_params[3]
        tau_o = fit_params[4]
        return lambda_od*D - tau_o*O
    
    def dTdt(self, T, state_vec, fit_params):
        #r, lambda_h, lambda_ho, lambda_od, tau_o, tau_d, tau_h, lambda_dh = [x for x in params]
        r = fit_params[0]
        lambda_h = fit_params[1]
        lambda_ho = fit_params[2]
        H = state_vec[1]
        O = state_vec[2]
        return (r - lambda_h*H - lambda_ho * H * O)*T
    
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
    
    
    
class growth_model_2b(growth_model):
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
            'lambda_h': [0.0001, 0.1],
            'lambda_o': [0.0001, 0.1],
            'lambda_od': [0.0001, 0.1],
            'tau_o': [0.0001, 0.5],
            'lambda_oh': [0.0001, 0.1],
            'tau_d': [0.0001, 0.5],
            'tau_h': [0.0001, 0.3],
            'lambda_dh': [0.0001,10]
        }
        
        self.param_list = list(self.param_estimates.keys())
        
        self.num_samples = 1000
        self.num_chains = 40
            
        plot_data.plot_combined_treatment(ts, self.groups, self.Sds, self.Shs)
        
        
    #don't take the respective state variable from state_vec here, since runge kutta relies on 
    #the respective state variable being iterated forward
    def dDdt(self, D, state_vec, fit_params):
        #r, lambda_h, lambda_o, lambda_od, tau_o, lambda_oh, tau_d, tau_h, lambda_dh = [x for x in params]
        tau_d = fit_params[6]
        return -tau_d*D
    
    def dHdt(self, H, state_vec, fit_params):
        #r, lambda_h, lambda_o, lambda_od, tau_o, lambda_oh, tau_d, tau_h, lambda_dh = [x for x in params]
        tau_h = fit_params[7]
        return -tau_h*H
    
    def dOdt(self, O, state_vec, fit_params):
        #r, lambda_h, lambda_o, lambda_od, tau_o, lambda_oh, tau_d, tau_h, lambda_dh = [x for x in params]
        D = state_vec[0]
        H = state_vec[1]
        lambda_od = fit_params[3]
        tau_o = fit_params[4]
        lambda_oh = fit_params[5]
        return lambda_od*D - tau_o*O*np.exp(-lambda_oh*H)
    
    def dTdt(self, T, state_vec, fit_params):
        #r, lambda_h, lambda_o, lambda_od, tau_o, lambda_oh, tau_d, tau_h, lambda_dh = [x for x in params]
        r = fit_params[0]
        lambda_h = fit_params[1]
        lambda_o = fit_params[2]
        H = state_vec[1]
        O = state_vec[2]
        return (r - lambda_h*H - lambda_o * O)*T
    

    
class growth_model_3(growth_model):
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
            'lambda_h': [0.0001, 0.1],
            'lambda_o': [0, 1],
            'lambda_odh': [0, 1],
            'tau_o': [0, 0.5],
            'tau_d': [0, 0.5],
            'tau_h': [0, 0.3],
            'lambda_dh': [0,40]
        }
        
        self.param_list = list(self.param_estimates.keys())
        
        self.num_samples = 1000
        self.num_chains = 40
        
        plot_data.plot_combined_treatment(ts, self.groups, self.Sds, self.Shs)
        
        
    #don't take the respective state variable from state_vec here, since runge kutta relies on 
    #the respective state variable being iterated forward
    def dDdt(self, D, state_vec, fit_params):
        #r, lambda_h, lambda_o, lambda_odh, tau_o, tau_d, tau_h, lambda_dh = [x for x in params]
        tau_d = fit_params[5]
        return -tau_d*D
    
    def dHdt(self, H, state_vec, fit_params):
        #r, lambda_h, lambda_o, lambda_odh, tau_o, tau_d, tau_h, lambda_dh = [x for x in params]
        tau_h = fit_params[6]
        return -tau_h*H
    
    def dOdt(self, O, state_vec, fit_params):
        #r, lambda_h, lambda_o, lambda_odh, tau_o, tau_d, tau_h, lambda_dh = [x for x in params]
        D = state_vec[0]
        H = state_vec[1]
        lambda_odh = fit_params[3]
        tau_o = fit_params[4]
        return lambda_odh*D*H - tau_o*O
    
    def dTdt(self, T, state_vec, fit_params):
        #r, lambda_h, lambda_o, lambda_odh, tau_o, tau_d, tau_h, lambda_dh = [x for x in params]
        r = fit_params[0]
        lambda_h = fit_params[1]
        lambda_o = fit_params[2]
        H = state_vec[1]
        O = state_vec[2]
        return (r - lambda_h*H - lambda_o * O)*T
    

class growth_model_4(growth_model):
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
            'lambda_o': [0, 0.2],
            'lambda_oh': [0, 0.5],
            'lambda_odh': [0, 1],
            'tau_o': [0, 0.5],
            'tau_d': [0, 0.1],
            'tau_h': [0, 1],
            'lambda_dh': [0, 40]
        }
        
        self.param_list = list(self.param_estimates.keys())
        
        self.num_samples = 1000
        self.num_chains = 20
        
        plot_data.plot_combined_treatment(ts, self.groups, self.Sds, self.Shs)
        
        
    #don't take the respective state variable from state_vec here, since runge kutta relies on 
    #the respective state variable being iterated forward
    def dDdt(self, D, state_vec, fit_params):
        #r, lambda_o, lambda_oh, lambda_odh, tau_o, tau_d, tau_h, lambda_dh = [x for x in params]
        tau_d = fit_params[5]
        return -tau_d*D
    
    def dHdt(self, H, state_vec, fit_params):
        #r, lambda_o, lambda_oh, lambda_odh, tau_o, tau_d, tau_h, lambda_dh = [x for x in params]
        tau_h = fit_params[6]
        return -tau_h*H
    
    def dOdt(self, O, state_vec, fit_params):
        #r, lambda_o, lambda_oh, lambda_odh, tau_o, tau_d, tau_h, lambda_dh = [x for x in params]
        D = state_vec[0]
        H = state_vec[1]
        lambda_oh = fit_params[2]
        lambda_odh = fit_params[3]
        tau_o = fit_params[4]
        return lambda_oh*H + lambda_odh*D*H - tau_o*O
    
    def dTdt(self, T, state_vec, fit_params):
        #r, lambda_o, lambda_oh, lambda_odh, tau_o, tau_d, tau_h, lambda_dh = [x for x in params]
        r = fit_params[0]
        lambda_o = fit_params[1]
        O = state_vec[2]
        return (r - lambda_o * O)*T
    
    
    