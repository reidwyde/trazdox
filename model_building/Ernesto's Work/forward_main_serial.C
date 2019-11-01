// Running the main function as a function to test for vegf_prod values.
#include "general_libraries.h"
#include "main_model.h"

int main(){
  
  std::vector<double> Parameters(9, 0);
  GetPot input_file("parameters.in");
  Parameters[0] = input_file("par_r",1.0);
  Parameters[1] = input_file("par_k",1.0);
  Parameters[2] = input_file("par_a",1.0);
  string file_name_1 = input_file("file_1","data.dat");
  string file_name_2 = input_file("file_2","data.dat");
  string file_name_3 = input_file("file_3","data.dat");
  string file_name_4 = input_file("file_4","data.dat");
  string file_name_5 = input_file("file_5","data.dat");
  string file_name_6 = input_file("file_6","data.dat");
  std::vector<double> tras_time(2,0);
  std::vector<double> doxo_time(1,0);
  std::ifstream read;
  std::string line;
  std::vector<double> time_vec;
  double ic_tumor;
  std::vector<double> tumor_solution;
  ofstream out_file;
  GetPot input_model("model.in");
  const unsigned int model_n = input_model("model_n",1);
  bool verbose=false;
  //--------------------------------------------------
  // Reading Tumor time
  //--------------------------------------------------
  read.open(file_name_1);
  if(!read.is_open())
    cout << "Error opening data file." << endl;
  while(std::getline(read, line)){
    double time;
    std::istringstream iss (line);
    iss >> time;
    time_vec.push_back(time);
  }
  read.close();
  for(unsigned int i=0; i<time_vec.size(); i++)
    cout << time_vec[i] << " ";
  cout << "<-- Tumor Time" << endl;
  //vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
  // Group 01
  //^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  read.open(file_name_1);
  if(!read.is_open())
    cout << "Error opening data file." << endl;
  std::getline(read, line);
  {
    double time;
    std::istringstream iss (line);
    iss >> time;
    iss >> ic_tumor;
  }
  read.close();
  main_code(Parameters,time_vec,ic_tumor,tumor_solution,tras_time,doxo_time,model_n);
  out_file.precision(16);
  out_file.open("tumor_evolution_1.txt");
  for(unsigned int i=0; i<tumor_solution.size(); i++)
    out_file << time_vec[i] << " " << tumor_solution[i] << endl;
  if(verbose){
    for(unsigned int i=0; i<tumor_solution.size(); i++){
      cout << time_vec[i] << " " << tumor_solution[i] << endl;
    }
  }
  out_file.close();
  //vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
  // Group 03
  //^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  Parameters[3] = input_file("par_dt",1.0);
  Parameters[4] = input_file("par_tt",1.0);
  tras_time[0] = 35.0;
  tras_time[1] = 38.0;
  read.open(file_name_3);
  if(!read.is_open())
    cout << "Error opening data file." << endl;
  std::getline(read, line);
  {
    double time;
    std::istringstream iss (line);
    iss >> time;
    iss >> ic_tumor;
  }
  read.close();
  tumor_solution.clear();
  main_code(Parameters,time_vec,ic_tumor,tumor_solution,tras_time,doxo_time,model_n);
  out_file.precision(16);
  out_file.open("tumor_evolution_3.txt");
  for(unsigned int i=0; i<tumor_solution.size(); i++)
    out_file << time_vec[i] << " " << tumor_solution[i] << endl;
  if(verbose){
  cout << endl;
  for(unsigned int i=0; i<tumor_solution.size(); i++){
    double exact_sol = 0;
    double treat = 0;
    for(unsigned int tt=0; tt<tras_time.size(); tt++)
      if(time_vec[i]>tras_time[tt])
        treat += Parameters[2]*(exp(Parameters[3]*(tras_time[tt]-time_vec[i]))-1.0)/Parameters[3];
    exact_sol = ((ic_tumor-Parameters[1])/exp(Parameters[0]*time_vec[0]))*exp(Parameters[0]*time_vec[i]+treat)+Parameters[1];
    cout << time_vec[i] << " " << tumor_solution[i] << " " << exact_sol << endl;
  }
  }
  out_file.close();
  //vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
  // Group 02
  //^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  Parameters[3] = 0.0;
  Parameters[4] = 0.0;
  Parameters[5] = input_file("par_dd",1.0);
  Parameters[6] = input_file("par_td",1.0);
  doxo_time[0] = 39.0;
  tras_time[0] = 0.0;
  tras_time[1] = 0.0;
  read.open(file_name_2);
  if(!read.is_open())
    cout << "Error opening data file." << endl;
  std::getline(read, line);
  {
    double time;
    std::istringstream iss (line);
    iss >> time;
    iss >> ic_tumor;
  }
  read.close();
  tumor_solution.clear();
  main_code(Parameters,time_vec,ic_tumor,tumor_solution,tras_time,doxo_time,model_n);
  out_file.precision(16);
  out_file.open("tumor_evolution_2.txt");
  for(unsigned int i=0; i<tumor_solution.size(); i++)
    out_file << time_vec[i] << " " << tumor_solution[i] << endl;
  if(verbose){
  cout << endl;
  for(unsigned int i=0; i<tumor_solution.size(); i++){
    double exact_sol = 0;
    double treat = 0;
    for(unsigned int tt=0; tt<doxo_time.size(); tt++)
      if(time_vec[i]>doxo_time[tt])
        treat += Parameters[4]*(exp(Parameters[5]*(doxo_time[tt]-time_vec[i]))-1.0)/Parameters[5];
    exact_sol = ((ic_tumor-Parameters[1])/exp(Parameters[0]*time_vec[0]))*exp(Parameters[0]*time_vec[i]+treat)+Parameters[1];
    cout << time_vec[i] << " " << tumor_solution[i] << " " << exact_sol << endl;
  }
  }
  out_file.close();
  //vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
  // Group 04
  //^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  Parameters[3] = input_file("par_dt",1.0);
  Parameters[4] = input_file("par_tt",1.0);
  tras_time[0] = 36.0;
  tras_time[1] = 39.0;
  doxo_time[0] = 35.0;
  Parameters[7] = input_file("par_del_td",1.0);
  Parameters[8] = input_file("par_tau_dt",1.0);
  read.open(file_name_4);
  if(!read.is_open())
    cout << "Error opening data file." << endl;
  std::getline(read, line);
  {
    double time;
    std::istringstream iss (line);
    iss >> time;
    iss >> ic_tumor;
  }
  read.close();
  tumor_solution.clear();
  main_code(Parameters,time_vec,ic_tumor,tumor_solution,tras_time,doxo_time,model_n);
  out_file.precision(16);
  out_file.open("tumor_evolution_4.txt");
  for(unsigned int i=0; i<tumor_solution.size(); i++)
    out_file << time_vec[i] << " " << tumor_solution[i] << endl;
  if(verbose){
  cout << endl;
  for(unsigned int i=0; i<tumor_solution.size(); i++){
    double exact_sol = 0;
    double treat = 0;
    for(unsigned int tt=0; tt<doxo_time.size(); tt++)
      if(time_vec[i]>doxo_time[tt])
        treat += Parameters[4]*(exp(Parameters[5]*(doxo_time[tt]-time_vec[i]))-1.0)/Parameters[5];
    for(unsigned int tt=0; tt<tras_time.size(); tt++)
      if(time_vec[i]>tras_time[tt])
        treat += Parameters[2]*(exp(Parameters[3]*(tras_time[tt]-time_vec[i]))-1.0)/Parameters[3];
    exact_sol = ((ic_tumor-Parameters[1])/exp(Parameters[0]*time_vec[0]))*exp(Parameters[0]*time_vec[i]+treat)+Parameters[1];
    cout << time_vec[i] << " " << tumor_solution[i] << " " << exact_sol << endl;
  }
  }
  out_file.close();
  //vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
  // Group 05
  //^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  tras_time[0] = 35.0;
  tras_time[1] = 38.0;
  doxo_time[0] = 39.0;
  read.open(file_name_5);
  if(!read.is_open())
    cout << "Error opening data file." << endl;
  std::getline(read, line);
  {
    double time;
    std::istringstream iss (line);
    iss >> time;
    iss >> ic_tumor;
  }
  read.close();
  tumor_solution.clear();
  main_code(Parameters,time_vec,ic_tumor,tumor_solution,tras_time,doxo_time,model_n);
  out_file.precision(16);
  out_file.open("tumor_evolution_5.txt");
  for(unsigned int i=0; i<tumor_solution.size(); i++)
    out_file << time_vec[i] << " " << tumor_solution[i] << endl;
  if(verbose){
  cout << endl;
  for(unsigned int i=0; i<tumor_solution.size(); i++){
    double exact_sol = 0;
    double treat = 0;
    for(unsigned int tt=0; tt<doxo_time.size(); tt++)
      if(time_vec[i]>doxo_time[tt])
        treat += Parameters[4]*(exp(Parameters[5]*(doxo_time[tt]-time_vec[i]))-1.0)/Parameters[5];
    for(unsigned int tt=0; tt<tras_time.size(); tt++)
      if(time_vec[i]>tras_time[tt])
        treat += Parameters[2]*(exp(Parameters[3]*(tras_time[tt]-time_vec[i]))-1.0)/Parameters[3];
    exact_sol = ((ic_tumor-Parameters[1])/exp(Parameters[0]*time_vec[0]))*exp(Parameters[0]*time_vec[i]+treat)+Parameters[1];
    cout << time_vec[i] << " " << tumor_solution[i] << " " << exact_sol << endl;
  }
  }
  out_file.close();
  //vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
  // Group 06
  //^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  tras_time[0] = 35.0;
  tras_time[1] = 38.0;
  doxo_time[0] = tras_time[0];
  doxo_time.push_back(tras_time[1]);
  read.open(file_name_6);
  if(!read.is_open())
    cout << "Error opening data file." << endl;
  std::getline(read, line);
  {
    double time;
    std::istringstream iss (line);
    iss >> time;
    iss >> ic_tumor;
  }
  read.close();
  tumor_solution.clear();
  main_code(Parameters,time_vec,ic_tumor,tumor_solution,tras_time,doxo_time,model_n);
  out_file.precision(16);
  out_file.open("tumor_evolution_6.txt");
  for(unsigned int i=0; i<tumor_solution.size(); i++)
    out_file << time_vec[i] << " " << tumor_solution[i] << endl;
  if(verbose){
  cout << endl;
  for(unsigned int i=0; i<tumor_solution.size(); i++){
    double exact_sol = 0;
    double treat = 0;
    for(unsigned int tt=0; tt<doxo_time.size(); tt++)
      if(time_vec[i]>doxo_time[tt])
        treat += Parameters[4]*(exp(Parameters[5]*(doxo_time[tt]-time_vec[i]))-1.0)/Parameters[5];
    for(unsigned int tt=0; tt<tras_time.size(); tt++)
      if(time_vec[i]>tras_time[tt])
        treat += Parameters[2]*(exp(Parameters[3]*(tras_time[tt]-time_vec[i]))-1.0)/Parameters[3];
    exact_sol = ((ic_tumor-Parameters[1])/exp(Parameters[0]*time_vec[0]))*exp(Parameters[0]*time_vec[i]+treat)+Parameters[1];
    cout << time_vec[i] << " " << tumor_solution[i] << " " << exact_sol << endl;
  }
  }
  out_file.close();
  return 0; 
}
