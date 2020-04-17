import numpy as np

def get_log_likelihood(Ts, sim_T_ts, sigmas):
    result = np.zeros((Ts.shape[0],1))
    N = Ts.shape[1]

    for i in range(result.shape[0]):
        group_Ts = Ts[i,:].ravel()
        group_sim_T_ts = sim_T_ts[i,:].ravel()
        sigma = sigmas[i,0]
        first_term = -N / 2 * np.log(2 * np.pi * sigma**2)
        second_term = -1 / (2*sigma**2) * sum((group_Ts-group_sim_T_ts)**2)
        result[i,0] = first_term + second_term
    return sum(result.ravel())
