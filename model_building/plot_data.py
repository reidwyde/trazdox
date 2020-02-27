import matplotlib.pyplot as plt

def plot_tumor_data(ts, Ts):
    fig, axs = plt.subplots(2, 3, figsize=(15,8))
    for group in range(6):
        axs[int(group/3), int(group%3)].title.set_text('Group ' + str(group))
        axs[int(group/3), int(group%3)].scatter(ts, Ts[group])
    plt.show()
    
def plot_sim_by_group(sim_times, T_sim, group):
    plt.figure(figsize=[4,2.5])
    plt.plot(sim_times, T_sim)
    plt.title('Group ' + str(group))
    plt.xlabel('Day')
    plt.ylabel('Size')
    plt.show()
    
def plot_sim(sim_times, T_sim):
    plt.figure(figsize=[16,10])
    for group in range(6):
        plt.subplot(2,3,group+1)
        plt.plot(sim_times, T_sim[group,:])
        plt.title('Group ' + str(group))
        plt.xlabel('Day')
        plt.ylabel('amt')
    plt.show()
    
def plot_sims_vs_actual(groups, ts, Ts, sigmas, sim_times, T_sim):
    plt.figure(figsize=[16,10])
    for ii in range(len(groups)):
        group = groups[ii]
        #plt.subplot(2,3,group+1)
        plt.figure()
        plt.scatter(ts, Ts[group,:])
        plt.errorbar(ts, Ts[group,:], sigmas[group,:],fmt='.', capsize=2)
        plt.plot(sim_times, T_sim[ii,:])
        plt.title('Group ' + str(group))
        plt.xlabel('Day')
        plt.ylabel('Size')
    plt.show()
    
def plot_short(times, var, title):
    plt.figure()
    plt.plot(times.ravel(), var.ravel())
    plt.title(title)
    plt.show()
        
def stem_short(times, var, title):
    plt.figure()
    plt.stem(times.ravel(), var.ravel())
    plt.title(title)
    plt.show()
    

#plotting treatment
def plot_combined_treatment(ts, groups, Sds, Shs):
    plt.figure(figsize=[16,10])
    for ii in range(len(groups)):
        group = groups[ii]
        plt.subplot(2,3,ii+1)
        plt.stem(ts, Sds[ii,:], 'b', markerfmt='bo', label='delta_D', use_line_collection=True)
        plt.stem(ts, Shs[ii,:], 'g', markerfmt='go', label='delta_H', use_line_collection=True)
        plt.title('Group ' + str(group) + ' Treatment')
        plt.legend()
    plt.show()
    
    
def graph_Sds(groups, times, Sds):
    plt.figure(figsize=[16,10])
    for ii in range(len(self.groups)):
        group = groups[ii]
        plt.subplot(2,3,ii+1)
        plt.stem(times, Sds[ii, :])
        plt.title('Group ' + str(group) + ' Sd')
        plt.xlabel('Day')
        plt.ylabel('Size')
        plt.show()
        
def graph_Shs(groups, times, Shs):
    plt.figure(figsize=[16,10])
    for ii in range(len(self.groups)):
        group = groups[ii]
        plt.subplot(2,3,ii+1)
        plt.stem(times, Shs[ii, :])
        plt.title('Group ' + str(group) + ' Sh')
        plt.xlabel('Day')
        plt.ylabel('Size')
        plt.show()
        
        

# plotting state variables   
def plot_D(sim_times, sim_D, groups=[0,1,2,3,4,5]):
    plt.figure(figsize=[16,10])
    for ii in range(len(groups)):
        group = groups[ii]
        plt.subplot(2,3,ii+1)
        plt.plot(sim_times, sim_D[ii,:])
        plt.title('D for Group ' + str(group))
        plt.xlabel('Day')
        plt.ylabel('amt')
    plt.show()

def plot_H(sim_times, sim_H, groups=[0,1,2,3,4,5]):
    plt.figure(figsize=[16,10])
    for ii in range(len(groups)):
        group = groups[ii]
        plt.subplot(2,3,ii+1)
        plt.plot(sim_times, sim_H[ii,:])
        plt.title('H for Group ' + str(group))
        plt.xlabel('Day')
        plt.ylabel('amt')
    plt.show()

def plot_O(sim_times, sim_O, groups=[0,1,2,3,4,5]):
    plt.figure(figsize=[16,10])
    for ii in range(len(groups)):
        group = groups[ii]
        plt.subplot(2,3,ii+1)
        plt.plot(sim_times, sim_O[ii,:])
        plt.title('O for Group ' + str(group))
        plt.xlabel('Day')
        plt.ylabel('amt')
    plt.show()

