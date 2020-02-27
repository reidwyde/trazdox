"""
def rk_var_update(var, ks):
    k1, k2, k3, k4 = [x for x in ks]
    var = var + (1.0 / 6.0)*(k1 + 2 * k2 + 2 * k3 + k4)
    var[var<0] = 0
    return var

def rk_X(h, t0, X, state_vec, fit_params, dXdt):
    #state vector holds drugs and latent variables, 
    #state vector is a list that holds any combination of the state variables (T, D, H, O)
    #X is the variable to simulate using RK
    #dXdt is the differential equation value
    #fit_params are the parameters being fitted in this system
    k1 = h * dXdt(t0, X, state_vec, fit_params) 
    k2 = h * dXdt(t0+0.5*h, X + 0.5 * k1, state_vec, fit_params) 
    k3 = h * dXdt(t0+0.5*h, X + 0.5 * k2, state_vec, fit_params) 
    k4 = h * dXdt(t0+h, X + k3, state_vec, fit_params)   
    return k1, k2, k3, k4

"""

def rk_var_update(var, ks):
        k1, k2, k3, k4 = [x for x in ks]
        var = var + (1.0 / 6.0)*(k1 + 2 * k2 + 2 * k3 + k4)
        var[var<0] = 0
        return var

def rk_X(h, X, state_vec, fit_params, dXdt):
    #state vector holds drugs and latent variables, 
    #state vector is a list that holds any combination of the state variables (T, D, H, O)
    #X is the variable to simulate using RK
    #dXdt is the differential equation value
    #fit_params are the parameters being fitted in this system
    k1 = h * dXdt(X, state_vec, fit_params) 
    k2 = h * dXdt(X + 0.5 * k1, state_vec, fit_params) 
    k3 = h * dXdt(X + 0.5 * k2, state_vec, fit_params) 
    k4 = h * dXdt(X + k3, state_vec, fit_params)   
    return k1, k2, k3, k4   


"""
each class defines a rk_{} method for each state variable in the system
the rk_{} method must:
    unpack the state vector for the right state variable being simulated
    call rk_X with the correct differential equation

Example:
def rk_T(self, h, t0, state_vec, fit_params):
    T = state_vec[0]
    return rk_X(h, t0, T, state_vec, fit_params, self.dTdt)

"""

