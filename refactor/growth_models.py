import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pymc3 as pm
from pymc3.ode import DifferentialEquation
from scipy.integrate import solve_ivp, odeint
import arviz as az
from pymc3 import traceplot
from pymc3.distributions import Interpolated
from scipy import stats
import matplotlib as mpl
import time
import theano.tensor as tt
from theano.compile.ops import as_op
from collections import OrderedDict 


import warnings
warnings.filterwarnings("ignore")


# 
# models 2,3,4
# time-it methods using a decorator
# cull imports
# 
# mybinder (online git repos for notebooks)
# generate requirements.txt



path_to_data = "filtered_data_csv.csv"
data = pd.read_csv(path_to_data)
data.head()


data_array = data.to_numpy()
days = data_array[:,0]
tumor_sizes = data_array[:,1::2].T
tumor_sigmas = data_array[:,2::2].T
T0 = tumor_sizes[:,0]

print(tumor_sizes.shape)
print(T0.shape)




plt.figure(figsize=(16,10))
for ii in range(6):
    plt.subplot(2,3,ii+1)
    plt.ylim(0,2500)
    plt.scatter(days, tumor_sizes[ii,:])
    plt.errorbar(days, tumor_sizes[ii,:], tumor_sigmas[ii,:])
    plt.title("group {}".format(ii))
#plt.show()
plt.savefig("tumor_sizes.jpg")
plt.close("all")



class growth_model(object):
    def simulate(self, fit_params, times=days):      
        return self.rhs(times, fit_params)
        
    def rhs(self, times, fit_params):
        return np.zeros((6,))
    
    def get_param(self, param_name):
        return pm.summary(self.trace)['mean'][param_name]
    
        
    def backward(self):
        with pm.Model() as model:
            #form priors
            #prior_distributions = [pm.Uniform(list(self.param_estimates.keys())[ii], lower=self.param_estimates[param][0], upper=self.param_estimates[param][1]) for ii, param in enumerate(self.param_list)]
            assert isinstance(self.param_estimates, OrderedDict)
            
            prior_distributions = [pm.Uniform(param_name, param_est[0], param_est[1]) for param_name, param_est in self.param_estimates.items()]
            
            prior_distributions = tt.as_tensor_variable(prior_distributions)
            
            @as_op(itypes=[tt.dvector], otypes=[tt.dmatrix]) 
            def th_forward_model(prior_distributions):
                th_states = self.simulate(prior_distributions)
                return th_states
            
            forward = th_forward_model(prior_distributions)
            
            
            T = pm.Normal('T', mu=forward, sigma=tumor_sigmas, observed=tumor_sizes)
            
            
            num_tune = self.num_samples//5
            
            #step = pm.SMC()
            step = pm.Metropolis()
            #prior = pm.sample_prior_predictive()
            self.trace = pm.sample(self.num_samples, step=step, tune=num_tune, chains = self.num_chains, cores=self.num_cores)
            #trace = pm.sample_smc(kernel="ABC")
            
            pm.traceplot(self.trace)
            plt.savefig("traceplot.jpg")
            plt.close("all")
            
            #posterior_predictive = pm.sample_posterior_predictive(self.trace)
            
            #data = az.from_pymc3(trace=self.trace, prior = prior, posterior_predictive = posterior_predictive)
            #data = az.from_pymc3(trace=self.trace, posterior_predictive = posterior_predictive)
            #az.plot_posterior(data, round_to=4, credible_interval=0.95);
            #plt.savefig("posterior.jpg")
            #plt.close("all")
            

            """
            t2 = time.time()
            print("sampling posteriors")
            posterior_predictive = pm.sample_posterior_predictive(self.trace)
            print("plotting posterior samples")
            #az.plot_trace(posterior_predictive)
            traceplot(posterior_predictive)
            plt.savefig("posterior_samples.jpg")
            plt.close("all")
            t3 = time.time()
            print(t3-t2, "seconds")
            """


            """
            print("generating posterior distributions")
            #posterior_distribution = az.from_pymc3(trace=self.trace, prior=prior, posterior_predictive=posterior_predictive)
            posterior_distribution = az.from_pymc3(trace=self.trace, posterior_predictive=posterior_predictive)
            print("plotting posterior distribution")
            az.plot_posterior(posterior_distribution)#, show=True)
            #traceplot(posterior_distribution)
            plt.savefig("posterior_dist.jpg")
            plt.close("all")
            plt.close("all")
            t4 = time.time()
            print(t4-t3, "seconds")
            """
 
            
            sim_times = days
            sim_params = []
            for param_name in self.param_estimates.keys():
                print(param_name + ' = ' + str(self.get_param(param_name)))
                sim_params.append(self.get_param(param_name))
                
            
            #visualize drug concentrations
            sim_y = self.rhs(sim_times, sim_params)

            #self.set_sim_DH(sim_params)
            plt.figure(figsize=(16,10))
            for ii in range(6):
                plt.subplot(2,3,ii+1)
                plt.xlim(7,70)
                plt.plot(gm1.sim_D['t'], gm1.sim_D['y'][ii,:])
                plt.plot(gm1.sim_H['t'], gm1.sim_H['y'][ii,:])
                plt.legend(["doxorubicin","herceptin"])
                plt.title("group {}".format(ii))
            #plt.show()
            plt.savefig("drug_concentrations.jpg")
            plt.close("all")
 
            
            #visualize tumor growth
            #sim_y = self.rhs(sim_times, sim_params)
            plt.figure(figsize=(16,10))
            for ii in range(6):
                plt.subplot(2,3,ii+1)
                plt.xlim(7,70)
                plt.ylim(0,2500)
                plt.scatter(days, tumor_sizes[ii,:])
                plt.errorbar(days, tumor_sizes[ii,:], tumor_sigmas[ii,:])
                plt.plot(sim_times, sim_y[ii,:])
                plt.title("group {}".format(ii))
    
            #plt.show()
            plt.savefig("rhs.jpg")
            plt.close("all")

            


# ## Growth Model 1
# 
# 
# $$ \frac{d \phi_t}{dt} = (r - \lambda_h \phi_h   - \lambda_{hd} \phi_h \phi_d ) \phi_t (1 - \frac{ \phi_t }{K}) $$
# 
# $$ \frac{d \phi_d}{dt} = - \tau_d \phi_d + \delta (t - \eta_d) $$
# 
# $$ \frac{d \phi_h}{dt} = - \tau_h \phi_h + \delta (t - \eta_h) e^{-\lambda_{dh} \phi_d} $$
# 
# r, lambda_h, lambda_hd, tau_d, tau_h, lambda_dh




eps=1

class growth_model_1(growth_model):
    def __init__(self):
        super().__init__()
        
        """
        self.param_estimates = {
            'r':[0,1],
            'lambda_h': [0,1],
            'lambda_hd': [0,1],
            'tau_d': [0,1],
            'tau_h': [0, 1],
            'lambda_dh': [0,1],
        }
        """
        self.param_estimates = OrderedDict()
        self.param_estimates['r'] = [0,1]
        self.param_estimates['lambda_h'] = [0,1]
        self.param_estimates['lambda_hd'] = [0,1]
        self.param_estimates['tau_d'] = [0,1]
        self.param_estimates['tau_h'] = [0, 1]
        self.param_estimates['lambda_dh'] = [0,20]
        self.param_estimates['K'] = [0,1e5]
        
        
        self.dose_time_D = np.array([[-1,-1],
                                     [-1,39],
                                     [-1,-1],
                                     [-1,35],
                                     [-1,39],
                                     [35,38]])
        
        self.dose_time_H = np.array([[-1,-1],
                                     [-1,-1],
                                     [35,38],
                                     [36,39],
                                     [35,38],
                                     [35,38]])
    
        
        #self.param_list = list(self.param_estimates.keys())
        
        self.num_samples = 30 #300
        self.num_chains = 1 #48
        self.num_cores = 1 #5
        self.t_span = (7,70)
        
    
    def rhs(self, times, fit_params):
        #print("rhs")
        self.sim_D = solve_ivp(self.dDdt, self.t_span, [0]*6, method='Radau', t_eval=np.arange(self.t_span[0], self.t_span[1], eps), max_step=eps, args=fit_params)
        self.sim_H = solve_ivp(self.dHdt, self.t_span, [0]*6, method='Radau', t_eval=np.arange(self.t_span[0], self.t_span[1], eps), max_step=eps, args=fit_params)
        sim_T = solve_ivp(self.dTdt, self.t_span, T0, method='Radau', t_eval=times, max_step=eps, args=fit_params)
        return sim_T['y']
        
        
    def dTdt(self, t, y, r, lambda_h, lambda_hd, tau_d, tau_h, lambda_dh, K):
        D = self.get_y_of_t(t, self.sim_D['t'], self.sim_D['y'])
        H = self.get_y_of_t(t, self.sim_H['t'], self.sim_H['y'])

        return y*(1-y/K)*(r - lambda_h*H - lambda_hd*H*D)
    
    def dDdt(self, t, y, r, lambda_h, lambda_hd, tau_d, tau_h, lambda_dh, K):
        def activation(t):
            #return (np.abs(self.dose_time_D[:,0]-t) <= eps) + (np.abs(self.dose_time_D[:,1]-t) <= eps)
            first_dose = np.logical_and(t >= self.dose_time_D[:,0], t < (self.dose_time_D[:,0] + 1))
            second_dose = np.logical_and(t >= self.dose_time_D[:,1], t < (self.dose_time_D[:,1] + 1))
            return np.logical_or(first_dose, second_dose)
    
        return -tau_d*y + activation(t)
    
    def dHdt(self, t, y, r, lambda_h, lambda_hd, tau_d, tau_h, lambda_dh, K):
        def activation(t):
            #return (np.abs(self.dose_time_H[:,0]-t) <= eps) + (np.abs(self.dose_time_H[:,1]-t) <= eps)
            first_dose = np.logical_and(t >= self.dose_time_H[:,0], t < (self.dose_time_H[:,0] + 1))
            second_dose = np.logical_and(t >= self.dose_time_H[:,1], t < (self.dose_time_H[:,1] + 1))
            return np.logical_or(first_dose, second_dose)
    
        D = self.get_y_of_t(t-1, self.sim_D['t'], self.sim_D['y'])
        
        return -tau_h*y + activation(t)*np.exp(D*(-lambda_dh))
    
    
    def get_y_of_t(self, t0, t, y):
     
        idxs = np.logical_and(t>=t0, t<t0+1)
        if np.sum(idxs) == 0:
            #print("returning zeros")
            return np.zeros((6,))
        ret = y[:,idxs]
        if len(ret.shape)>1:
            return ret[:,0]
        return ret
        
        
        
    


# In[7]:


gm1 = growth_model_1()
print(gm1.dose_time_D)
print(gm1.dose_time_H)
"""
r = 6.19e-2
lambda_h = 2.89e-2
lambda_hd = 7.03e-2
tau_d = 1.72e-3
tau_h = 4.17e-3
lambda_dh = 9.78
K =
"""
r = 6.880681e-02
lambda_h = 4.224530e-02
lambda_hd = 9.572470e-02
tau_d = 9.874700e-03
tau_h = 1.908911e-02
lambda_dh = 3.028982e+01
K = 4.421688e+03




sim_params = np.array([r, lambda_h, lambda_hd, tau_d, tau_h, lambda_dh, K])
#gm1.set_sim_DH(sim_params)
gm1.rhs(None, sim_params)

print(gm1.sim_D['y'].shape)
print(gm1.sim_H['y'].shape)
print(gm1.sim_D['t'].shape)
print(gm1.sim_H['t'].shape)
dense_times = gm1.sim_D['t']

plt.figure(figsize=(16,10))
for ii in range(6):
    plt.subplot(2,3,ii+1)
    plt.xlim(7,70)
    plt.plot(gm1.sim_D['t'], gm1.sim_D['y'][ii,:])
    plt.plot(gm1.sim_H['t'], gm1.sim_H['y'][ii,:])
    plt.legend(["doxorubicin","herceptin"])
    plt.title("group {}".format(ii))
#plt.show()
plt.savefig("drug_concentrations_fixed.jpg")
plt.close("all")



#sim_times = dense_times
sim_times = days
sim_y = gm1.rhs(sim_times, sim_params)
plt.figure(figsize=(16,10))
for ii in range(6):
    plt.subplot(2,3,ii+1)
    plt.xlim(7,70)
    plt.ylim(0,2500)
    plt.scatter(days, tumor_sizes[ii,:])
    plt.errorbar(days, tumor_sizes[ii,:], tumor_sigmas[ii,:])
    plt.plot(sim_times, sim_y[ii,:])
    plt.title("group {}".format(ii))
#plt.show()
plt.savefig("rhs_fixed.jpg")
plt.close("all")




growth_model_1().backward()

