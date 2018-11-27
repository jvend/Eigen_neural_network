#include <iostream>
#include <cmath>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <Eigen/Dense>
#include "../neural_layer.h"
#include "../neural_network.h"
#include "../load_save.h"

int main()
{

//Check label import
std::vector <int > label_test = import("test_import_int.txt");
for(int i=0; i<label_test.size(); i++)
 { std::cout << label_test[i] << std::endl; }
std::cout << std::endl;

//Check vector import
std::vector<Eigen::VectorXd > vec_test = import("test_import_vec.txt", 5);
for(int i=0; i<vec_test.size(); i++)
 { std::cout << vec_test[i].transpose() << std::endl; }
std::cout << std::endl;

//Check matrix import
Eigen::MatrixXd mat_test = import("test_import_vec.txt", 3,5);
std::cout << mat_test << std::endl;




 return 0;
}
