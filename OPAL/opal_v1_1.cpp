#include <iostream>
using namespace std;



/* OPAL c++ code version 1.1 by Reid Wyde
 * Last update 7.26.2019
 *
 OPAL Algorithm source:
 *Selection, calibration, and validation of models of tumor growth, E. A. B. F. Lima et al.
 Copyright 2016 World Scientific Publishing Company
 *
 *
 */


// CPP level 0
/*
int main() 
{
    cout << "Hello, World!";
    return 0;
}
*/



//CPP level 1
int main(int argc, char* argv[])
{
    std::cout << "Program Name: " << std::endl;
    std::cout << argv[0] << std::endl;
    std::cout << "Number of arguments" << std::endl;
    std::cout << argc << std::endl; 
    std::cout << "Arguments passed: " << std::endl;
    for (int ii = 0; ii<argc; ii++){

    	std::cout << argv[ii] << std::endl; 

    }

    return 0;
}

//Ernesto
//Ernesto will provide the forward model, it runs for each group, it returns the results for each model


//Reid
//come up with a set of models
//We can start going through literature and find some models that we think are important
//or, Reid starts getting familiar with QUESO, it's on github
//Wrt QUESO, there is a PDF, QUESO user's manual, there is an example makefile
//See if you can install queso, if not, we can try to use other things. 
//QUESo uses markov chain monte carlo, maybe find another implementation of markov chain monte carlo
//we can use other methods. like LM



//Top Level TODO
//Make the github repo and share it with Ernesto
//Get rid of the * notation
//decide on log_file structure
//add log code
//fix return statements





//TODO
int start(){

   //identify a set of all possible models
   // return M = {P_1(theta_1), ... , P_m(theta_m)}
   return 0;
}

//TODO
int sens_analysis(/* M */ ){

   //eliminate models with parameters to which the model output is insensitive 
   //retun M_bar = {P_1_bar(theta_1_bar), ... , P_l_bar(theta_l_bar)}
   return 0;
}

//TODO
int occam(/* M_bar */){
    //Choose model(s) in the lowest occam category
    //return M* = {P_1*(theta_1*), ... , P_k*(theta_k*)} 
    return 0;
}

//TODO
int calibrate( /*int M* */ ){
    //Calibrate all models in M*
    //return M*_calibrated
    return 0;
}

//TODO
int plausibility(/* M*_calibrated */ ){
   //Compute plausibilities and identify most plausible model P_j*
   //return P_j*
    return 0;
}

//TODO
int validation(/*  P_j* */){
    //Submit P_j* to validation test
    //return is_valid
    return 0;
}


//TODO
int predict_QoI(/* P_j* */ ){
    //use validated params to predict QoI
    //return QoI
    return 0;
}

//TODO
int identify_new_models (){
    //return a new set of possible models 
    return 0;
}

/*

M_occam_ordered
	occam_category[]


occam_category
	int occam_order
	Model[] models


Model
	double[] theta
	@model_method
	double plausibility

Evidence
	double evidence




//TODO 
QoI opal(){
	This one uses loops
	init_QoI, init_log_file 
	Model[] M = start();
	Model[] M_bar = sens_analysis(M);
	M_occam_ordered M_star = order_occam(M_bar); // we now have the sets in increasing order of occam category
	for (occam_category category : M_star){
		for(int ii = 0; ii< category.models.size(); ii++){
			category.models[ii] = calibrate(categories.models[ii]);
		}
		
		Evidence evidence = compute_evidence(category);
		
		double plausibility = 0;
		int plausibility_index = -1;
		for(int ii = 0; ii< category.models.size(); ii++){
			this_plaus =  compute_plausibility(evidence, category.models[ii]);
			category.models[ii].plausibility = this_plaus
			if (this_plaus) > plausibility){
				plausibility = this_plaus;
				plaus_index = ii;
			}
		}
		best_model = category.models[plaus_index];
		is_valid = validate(best_model);
		if(is_valid){
			return QoI(best_model);
		}
	}

}





*/


//Good job me





