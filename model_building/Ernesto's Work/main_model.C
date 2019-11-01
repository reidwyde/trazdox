#include <iostream>
#include <stdio.h>
#include <fstream>
#include <string.h>
#include <sstream>
#include <math.h>
#include <vector>

#include "main_model.h"

using namespace std;

void RungeKutta(std::vector<double>& QoI,double time,const double time_step,vector<double> Parameters,std::vector<double> tras_time,std::vector<double> doxo_time,const unsigned int model_n);
void fmodel(std::vector<double> qoi,std::vector<double>& auxiliar,double rk_time,vector<double> Parameters,std::vector<double> tras_time,std::vector<double> doxo_time,double time,const unsigned int model_n);

void main_code(std::vector<double> Parameters,
  	           std::vector<double> time_vec,
	           const double ic_tumor,
	           std::vector<double>& tumor_solution,
	           std::vector<double> tras_time,
	           std::vector<double> doxo_time,
	           const unsigned int model_n){
  /////----- Read argumments -----/////
  const double time_step              = 0.01;
  const unsigned int N_time           = time_vec.size()-1;
  unsigned int n_timesteps            = time_vec[N_time]/time_step;
  const unsigned int number_equations = 1;
  double curr_time                    = time_vec[0];
  unsigned int t_step                 = time_vec[0]/time_step;
  /*
  cout << endl;
  cout << "Initial time:     " << time_vec[0] << endl;
  cout << "Final time:       " << time_vec[N_time] << endl;
  cout << "Time to run:      " << time_vec[N_time]-time_vec[0] << endl;
  cout << "Timestep:         " << time_step << endl;
  cout << "Final timestep:   " << n_timesteps << endl;
  cout << "Initial timestep: " << t_step << endl;
  cout << endl;
  */
  if(model_n==1){
    for(unsigned int i=0; i<=N_time; i++){
      double tumor = (ic_tumor/exp(Parameters[0]*time_vec[0]))*exp(Parameters[0]*time_vec[i]);
      tumor_solution.push_back(tumor);
    }
  }
  else if(model_n==2){
    for(unsigned int i=0; i<=N_time; i++){
      double Cl=(ic_tumor*Parameters[1]/(Parameters[1]-ic_tumor))*(1.0/exp(Parameters[0]*time_vec[0]));
      double tumor = Cl*exp(Parameters[0]*time_vec[i])/(1.0+Cl*exp(Parameters[0]*time_vec[i])/Parameters[1]);
      tumor_solution.push_back(tumor);
    }
  }
  else if(model_n==3){
    for(unsigned int i=0; i<=N_time; i++){
      double Ca=(ic_tumor-Parameters[2])/exp(Parameters[0]*time_vec[0]);
      double tumor = Ca*exp(Parameters[0]*time_vec[i])+Parameters[2];
      if(tumor<0.0) tumor = 0.0;
      tumor_solution.push_back(tumor);
    }
  }
  else if(model_n==4){
    bool neg = false;
    for(unsigned int i=0; i<=N_time; i++){
      double C1=Parameters[1]/(Parameters[1]-Parameters[2]);
      double C2=((Parameters[2]-ic_tumor)/(ic_tumor-Parameters[1]))*(1.0/exp(Parameters[0]*time_vec[0]/C1));
      double tumor = (Parameters[1]*C2*exp(Parameters[0]*time_vec[i]/C1)+Parameters[2])/(1.0+C2*exp(Parameters[0]*time_vec[i]/C1));
      if(tumor<0.0){tumor = 0.0; neg = true;}
      if(neg) tumor = 0.0;
      tumor_solution.push_back(tumor);
    }
  }
  else{
    std::vector<unsigned int> saved_times(N_time+1);
    for(unsigned int i=0; i<=N_time; i++)
      saved_times[i] = time_vec[i]/time_step;
    std::vector<double> QoI(number_equations);
    QoI[0] = ic_tumor;
    tumor_solution.push_back(QoI[0]);
    do{
      RungeKutta(QoI,curr_time,time_step,Parameters,tras_time,doxo_time,model_n);
      t_step++;
      curr_time = t_step*time_step;
      for(unsigned int i=1; i<=N_time; i++){
        if(t_step==saved_times[i]){
          tumor_solution.push_back(QoI[0]);
          break;
        }
      }
    }while(t_step<n_timesteps);
  }
}

void RungeKutta(std::vector<double>& QoI,double time,const double time_step,vector<double> Parameters,std::vector<double> tras_time,std::vector<double> doxo_time,const unsigned int model_n){

  const unsigned int N = QoI.size();
  std::vector<double> k(4*N,0.0);
  std::vector<double> qoi(N,0.0);
  std::vector<double> auxiliar(N,0.0);
  double rk_time;
  
  for(unsigned int i = 0; i < N; i++){qoi[i] = QoI[i];}
  rk_time = time;
  fmodel(qoi,auxiliar,rk_time,Parameters,tras_time,doxo_time,time,model_n);
  for(unsigned int i = 0; i < N; i++){k[i*4+0] = time_step*auxiliar[i];}
  
  for(unsigned int i = 0; i < N; i++){qoi[i] = QoI[i]+(k[i*4+0]/2.);}
  rk_time = time+0.5*time_step;
  fmodel(qoi,auxiliar,rk_time,Parameters,tras_time,doxo_time,time,model_n);
  for(unsigned int i = 0; i < N; i++){k[i*4+1] = time_step*auxiliar[i];}
 
  for(unsigned int i = 0; i < N; i++){qoi[i] = QoI[i]+(k[i*4+1]/2.);}
  rk_time = time+0.5*time_step;
  fmodel(qoi,auxiliar,rk_time,Parameters,tras_time,doxo_time,time,model_n);
  for(unsigned int i = 0; i < N; i++){k[i*4+2] = time_step*auxiliar[i];}

  for(unsigned int i = 0; i < N; i++){qoi[i] = QoI[i]+k[i*4+2];}
  rk_time = time+time_step;
  fmodel(qoi,auxiliar,rk_time,Parameters,tras_time,doxo_time,time,model_n);
  for(unsigned int i = 0; i < N; i++){k[i*4+3] = time_step*auxiliar[i];}
  
  for(unsigned int i = 0; i < N; i++){QoI[i] = QoI[i]+(1./6.)*(k[i*4+0]+2*(k[i*4+1]+k[i*4+2])+k[i*4+3]);}
  if(QoI[0]<=0.0)
    QoI[0] = 0.0;
}

void fmodel(std::vector<double> qoi,std::vector<double>& auxiliar,double rk_time,vector<double> Parameters,std::vector<double> tras_time,std::vector<double> doxo_time,double time,const unsigned int model_n){
  if(qoi[0]<=0.0)
    auxiliar[0] = qoi[0];
  else{
    double tumor_growth = Parameters[0];
    double carrying_cap = Parameters[1];
    double critical_vol = Parameters[2];
    double delta_tras   = 0.0;
    double tras_decay   = Parameters[4];
    double delta_doxo   = 0.0;
    double doxo_decay   = Parameters[6];
    double delta_trdo   = Parameters[7];
    double dotr_decay   = Parameters[8];
    unsigned int N_tras = tras_time.size();
    unsigned int N_doxo = doxo_time.size();
    if(rk_time>=tras_time[0])
      delta_tras = Parameters[3];
    if(rk_time>=doxo_time[0])
      delta_doxo = Parameters[5];
    //============================== Trastuzumab ==============================
    double treat_tras = 0.0;
    double drug_tras = 0.0;
    for(unsigned int i=0; i<N_tras; i++)
      if(time>=tras_time[i] && tras_time[0]!=0){
	    treat_tras += delta_tras*exp(-tras_decay*(rk_time-tras_time[i]));
	    drug_tras += exp(-tras_decay*(rk_time-tras_time[i]));
	  }
    //============================== Doxorubicin ==============================
    double treat_doxo = 0.0;
    double drug_doxo = 0.0;
    for(unsigned int i=0; i<N_doxo; i++)
      if(time>=doxo_time[i] && doxo_time[0]!=0){
	    treat_doxo += delta_doxo*exp(-doxo_decay*(rk_time-doxo_time[i]));
	    drug_doxo += exp(-doxo_decay*(rk_time-doxo_time[i]));
	  }
   //============================== Trastuzumab +Doxorubicin ================== HERE REID! and case 5 below! The code is not clean (there is some trash when computing doxo that is not being used - tmp_tras)
    double doxo = 0.0;
    double tras = 0.0;
    for(unsigned int i=0; i<N_tras; i++){
      double tmp_doxo = 0.0;
      if(time>=tras_time[i] && tras_time[0]>0){
        for(unsigned int j=0; j<N_doxo; j++){
          if(tras_time[i]>doxo_time[j] && doxo_time[0]>0)
	        tmp_doxo += exp(-doxo_decay*(rk_time-doxo_time[j]));
	    }
	    tras += exp(-(tras_decay+dotr_decay*tmp_doxo)*(rk_time-tras_time[i]));
	  }
	}
    for(unsigned int i=0; i<N_doxo; i++){
      double tmp_tras = 0.0;
      if(time>=doxo_time[i] && doxo_time[0]>0){
        for(unsigned int j=0; j<N_tras; j++){
          if(doxo_time[i]>tras_time[j] && tras_time[0]>0)
	        tmp_tras += exp(-tras_decay*(rk_time-tras_time[j]));
	    }
	    doxo += (delta_doxo+delta_trdo*tras)*exp(-doxo_decay*(rk_time-doxo_time[i]));
	  }
	}
	tras=tras*delta_tras;
	//cout << "T(tras,doxo) = T(" << treat_tras << "," << treat_doxo << "), TrasDoxo = (" << tras << "," << doxo << ")" << endl;
    //============================== Tumor Cells ==============================
    switch(model_n){
      //case 1: auxiliar[0]=(tumor_growth-treat_tras-treat_doxo)*qoi[0];
      //case 2: auxiliar[0]=tumor_growth*qoi[0]*(1.0-qoi[0]/carrying_cap);
      //case 3: auxiliar[0]=tumor_growth*qoi[0]-tumor_growth*critical_vol;
      //case 4: auxiliar[0]=tumor_growth*qoi[0]*(1.0-qoi[0]/carrying_cap)-tumor_growth*critical_vol*(1.0-qoi[0]/carrying_cap);
      //auxiliar[0]=(tumor_growth-treat_tras)*qoi[0]-(tumor_growth-treat_tras)*critical_vol*(1.0+treat_doxo);
      //auxiliar[0]=(tumor_growth-treat_doxo)*qoi[0]-(tumor_growth-treat_doxo)*critical_vol*(1.0+treat_tras);
      //auxiliar[0]=(tumor_growth-treat_tras)*qoi[0]-tumor_growth*critical_vol*(1.0+treat_doxo);
      //auxiliar[0]=(tumor_growth-treat_doxo)*qoi[0]-tumor_growth*critical_vol*(1.0+treat_tras);
      case 5: auxiliar[0]=tumor_growth*qoi[0]*(1.0-qoi[0]/carrying_cap)-(tras+doxo)*qoi[0];
              break;
      case 6: auxiliar[0]=(tumor_growth-treat_tras-treat_doxo)*qoi[0]-(tumor_growth-treat_tras-treat_doxo)*critical_vol;
              break;
      case 7: auxiliar[0]=tumor_growth*qoi[0]-tumor_growth*critical_vol*(1.0+treat_doxo+treat_tras);
              break;
      case 8: auxiliar[0]=(tumor_growth-treat_tras-treat_doxo)*qoi[0]-tumor_growth*critical_vol;
              break;
      case 9: auxiliar[0]=(tumor_growth-treat_tras-doxo)*qoi[0];
              break;
      default: cout << "Invalid model number" << endl;
              break;
    }
  }
}
