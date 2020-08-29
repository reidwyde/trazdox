# Data Driven Growth Models for Combination Therapy for HER2+ Breast Cancer

## Traversing differential equations parameter space using Markov Chain Monte Carlo

#### Background: Chemotherapy (Doxorubicin) and immunotherapy (Trastuzumab/Herceptin) were given to adult female rats previously injected with HER2+ breast cancer [ ]. Hypotheses on Trastuzumab's synergistic efficacy include an effect of improved vascular regularization within tumors, which improves chemotherapy delivery and efficacy [ ]. This project investigates growth models for HER2+ cancer in the presence of different combinations of Doxorubicin and Trastuzumab. The goal is to deliver a data validated growth model that informs optimal treatment using these therapies, as well as characterize the tumoral system, including the relationship between tumor size, natural growth rate, drug clearing rate, and magnitude of drug effect.



### Section 1: Model introduction and motivation



#### System 0A,B,C

Single equation family

##### model evolutions
 - exponential -> logistic -> allee, with multiple permutations
 - time delay
 - asymmetrical drug interaction; constriction on the window of action vs increasing clearing rate vs blocking entrance to the system
 - explicit choice to ignore carrying capacity; if tumor reaches biophysical carrying capacity from size, subject is likely already dead
 
 These considerations motivated characterizing the system by a family of coupled differential equations, as was done in [sorace et al]

#### System 1

$$ \frac{dT}{dt} = (r - \lambda_H H - (\lambda_D + \lambda_HD H) \;D) \; T $$

$$ \frac{dD}{dt} = - \tau_D D $$

$$ \frac{dH}{dt} = - \tau_H H $$

$$ H(t) = H(t) + \delta_H e^{-\lambda_{DH}}{D(t)} $$

$$ D(t) = D(t) + \delta_D $$


With:


$ r $ The natural growth rate

$ \lambda_H $ The effect of Herceptin on the tumor (likely through immune signalling) 

$ \lambda_{OD} $ The effect of Doxorubicin on the number of ROS

$ \lambda_{ODH} $ The synergistic effect of Herceptin's NRF2 suppression and Doxorubicin's production of ROS

$ \lambda_{DH} $ Herceptin's suppression of NRF2 on Doxorubicin clearing rate 

$$
\;
\;
$$


$ \tau_O $ The natural clearing rate of ROS

$ \tau_H $ The natural clearing rate of Herceptin

$ \tau_D $ The natural clearing rate of Doxorubcin

$$
\;
\;
$$


$ \delta_H $ The Kronecker delta representing Herceptin drug delivery on the specified treatment days

$ \delta_D $ The Kronecker delta representing Doxorubicin drug delivery on the specified treatment days




#### System 1B

$$ \frac{dT}{dt} = ( r - \delta_D D - \delta_{DH} D H - \delta_H H ) T  $$


$$ \frac{dH}{dt} = S_H - \tau_H H - \tau_{DH} D H $$


$$ \frac{dD}{dt} = S_D - \tau_D D
\;\;\;\;
or
\;\;\;\;
\frac{dD}{dt} = S_D - \tau_D D + \tau_{HD} H 
$$



#### System 2

We consider the pharmacodynamics and cell signalling pathway that doxorubicin and herceptin affect. Doxorubicin has been known to increase radically oxidizing species (ROS, or 'free radicals'), e.g. quinones or ketones, and herceptin is known to supress NRF2, which is a signalling molecule that curtails the effects and amount of free radicals in the cell.  


We will consider effective free radical concentration as a latent variable, with tumor size being directly effected by the free radical concentration, and the free radical concentration being effected by both doxorubicin and herceptin. We will also consider that herceptin has immune signalling properties independent of the free radical production mechanism, and can thus affect tumor size independently of doxorubicin. We choose $O$ for the radically oxidizing species variable to avoid confusion with tumor growth rate $r$.


We also consider that doxorubicin has a destructive effect on the receptors that herceptin binds to. This prevents herceptin from signaling on the system.



$$ \frac{dT}{dt} = (r - \delta_{HO} H O - \delta_H H ) T $$


$$ \frac{dO}{dt} = \delta_{DO} D - \tau_O O $$


$$ \frac{dH}{dt} = S_H - \tau_H H - \tau_{DH} D H $$


$$ 
\frac{dD}{dt} = S_D - \tau_D D
\;\;\;\;
or 
\;\;\;\;
\frac{dD}{dt} = S_D - \tau_D D + \tau_{HD} H 
$$



#### system 3A


$$ \frac{dT}{dt} = (r - \delta_o O - \delta_H H ) T $$


$$ \frac{dO}{dt} = \delta_{DO} D - \tau_O O + \delta_{dho} O D H $$


$$ \frac{dH}{dt} = S_H - \tau_H H - \tau_{DH} D H $$


$$ \frac{dD}{dt} = S_D - \tau_D D
\;\;\;\;
or 
\;\;\;\;
\frac{dD}{dt} = S_D - \tau_D D + \tau_{HD} H 
$$


#### system 3B
Doxorubicin burns the trk family receptors, and thus prevents herceptin from entering or affecting the cell



$$ \frac{dT}{dt} = (r - \lambda_O O - \lambda_H H ) T $$


$$ \frac{dO}{dt} = \lambda_{ODH} D H - \tau_O O $$


$$ \frac{dH}{dt} = \delta_He^{-\lambda_{HD}D} - \tau_H H $$


$$ \frac{dD}{dt} = \delta_D - \tau_D D$$

With:


$ r $ The natural growth rate

$ \lambda_O $ The effect of ROS on the tumor 

$ \lambda_H $ The effect of Herceptin on the tumor (likely through immune signalling) 

$ \lambda_{OD} $ The effect of Doxorubicin on the number of ROS

$ \lambda_{ODH} $ The synergistic effect of Herceptin's NRF2 suppression and Doxorubicin's production of ROS

$ \lambda_{DH} $ Herceptin's suppression of NRF2 on Doxorubicin clearing rate 

$$
\;
\;
$$


$ \tau_O $ The natural clearing rate of ROS

$ \tau_H $ The natural clearing rate of Herceptin

$ \tau_D $ The natural clearing rate of Doxorubcin

$$
\;
\;
$$


$ \delta_H $ The Kronecker delta representing Herceptin drug delivery on the specified treatment days

$ \delta_D $ The Kronecker delta representing Doxorubicin drug delivery on the specified treatment days



#### system 4

Doxorubicin burns the trk family receptors, and thus prevents herceptin from entering or affecting the cell
$$ \frac{dT}{dt} = (r - \lambda_O O) T $$


$$ \frac{dO}{dt} = \lambda_{OH} H + \lambda_{ODH} D H - \tau_O O $$


$$ \frac{dH}{dt} = \delta_He^{-\lambda_{HD}D} - \tau_H H $$


$$ \frac{dD}{dt} = \delta_D - \tau_D D $$




With:


$ r $ The natural growth rate

$ \lambda_O $ The effect of ROS on the tumor 

$ \lambda_H $ The effect of Herceptin on the tumor (likely through immune signalling) 

$ \lambda_{OD} $ The effect of Doxorubicin on the number of ROS

$ \lambda_{ODH} $ The synergistic effect of Herceptin's NRF2 suppression and Doxorubicin's production of ROS

$ \lambda_{DH} $ Herceptin's suppression of NRF2 on Doxorubicin clearing rate 

$$
\;
\;
$$


$ \tau_O $ The natural clearing rate of ROS

$ \tau_H $ The natural clearing rate of Herceptin

$ \tau_D $ The natural clearing rate of Doxorubcin

$$
\;
\;
$$


$ \delta_H $ The Kronecker delta representing Herceptin drug delivery on the specified treatment days

$ \delta_D $ The Kronecker delta representing Doxorubicin drug delivery on the specified treatment days



### Section 2: Model Selection

Information criterion describes the effective behavior of the model, weighted by the number of parameters the model relies on. All other things being equal, a model with equal descriptive power but fewer parameters is preferred. 

Qualitatively, we want the model selected to 'fit' the data. 


### Section 3: Optimal Control for optimal treatment paradigm

#### Introducing optimal control
Given a system of differential equations that describe the tumoral system and different drug effects, it is possible to solve the optimal control problem. The output of the optimal control problem is a drug delivery schedule, and simulated tumor size and drug concentration for each time step. 


A hamiltonian is formulated from the system of differential equations. 

The optimal control is then a protocol for traversal through the state space of the model. 


#### Defining limits of the system

The state space of the model is governed by several boundary values including:
- The limit of instantaneous drug delivery (the patient's physical capabilities to handle chemotherapy and immunotherapy day by day0
- The limit of total drug delivery (supply of such drugs is limited by cost and manufacturable quantity and availability)
- The limit of instantaneous tumor size (how dangerous the presence of the tumor is for reasons of throwing a blood clot, contributing to necrosis, straining the system) 
- The limit of final tumor size at the end of treatment (governing whether the treatment aims to fully eliminate the tumor, or to reduce it to such a size that it does not present a large risk to the patient)


#### Defining the cost functional

The cost functional is described: 

J = w1*x1 + w2*x2 + ... + int(... wN-1 * xN-1 + wN-2 * xN-2) dt


Different choices for each of these boundary values, as well as the regularizing weights will determine how the optimal control model navigates the state space. The relative magnitude of the weights represents the importance of each consideration. Weights are also required to be chosen such that the optimal control model reaches convergence. 

#### Calculating the optimal control

The model converges by a forward - backward simulation, run iteratively for several thousand iterations. 

The model was converted from continuous domain to discrete domain by setting time step equal to 1 day. This is to make the optimal treatment performable on a hypothetical real world patient that would need to leave the treatment center at regular intervals. 


#### Other considerations
Linear vs quadratic cost behavior. 

Relative tradeoff

Model simiulation limitations (start date for each drug, impediments under a highly varying system)

### Section 4: Interpreting results

After completing the optimal control problem, we compare the simulated results of optimal treatment to the experimental, and find that our best model is able to outperform the given treatment schedule. 



### Section 5: Conclusion and Future work

We propose a method for model selection, 

A proposed model for drug-drug interactive tumoral action (the exponential block action)

A framework for drug-drug interactive optimal control and cost functional


Future work:
More data, more models, extension to other forms of cancer and other combination treatments
