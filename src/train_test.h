#ifndef __IMPORT_H
#define __IMPORT_H
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <chrono>
#include <random>
#include <Eigen/Dense>

template<typename type>
void Shuffle(std::vector<type>& vector)
 {
   std::random_device rd;
   std::mt19937 g(rd());
   std::shuffle(vector.begin(), vector.end(), g);
 }

void Train_Network(Neural_Network_Base & network, std::vector<std::vector<Eigen::VectorXd > > & data)
 {
  int num_label = data[0][1].size();
  std::vector<int> label_counter(num_label, 0);
  for(int j=0; j<network.epochs; j++) {
    // During first epoch count number of labels in each category
    if(j==0) { 
      for(int i=0; i<data.size(); i++)
       { int expected = -1;
         data[i][1].maxCoeff(&expected);
         label_counter[expected]++;
       }
     }

    Shuffle(data); // for stochastic grad descent
    for(int i=0; i<data.size(); i++)
     { network.Backprop(data[i][0],data[i][1]); }
    std::cout << "Epoch: " << j << std::endl;
    
    std::vector<int> out_label_counter(num_label, 0);

    int num_correct = 0;
    double cost     = 0;
    for(int i=0; i<data.size(); i++)
     {
       Eigen::VectorXd outvec = network.Run(data[i][0]);
       cost += network.Cost(data[i][1],outvec);
       int expected = -1; int actual = -1;
       data[i][1].maxCoeff(&expected);
       outvec.maxCoeff(&actual);
       if(actual == expected) { num_correct++; } 
       for(int label_val=0; label_val<num_label; label_val++)
        { if (actual == label_val){out_label_counter[label_val]++;} }
     }
    double cost_reg = 0; //Regularization penalty
    for(int i=0; i<=network.hidden_layer_num; i++)
     { cost_reg += 0.5*network.lambda/network.run_size*network.Layer[i].weight_matrix.squaredNorm(); }
    cost += cost_reg;
 
    std::cout << "   Cost             = " << cost << std::endl;
    std::cout << "   Pred  = ";
    for(int label_val=0; label_val<num_label; label_val++)
     {std::cout << out_label_counter[label_val] << "/" << label_counter[label_val]; if(label_val!=num_label-1){ std::cout << ", ";} }
    std::cout << std::endl;

    double fraction_correct = num_correct*1.0/data.size();
    std::cout << "   Number correct   = " << num_correct << "/" << data.size() << " = " << fraction_correct << std::endl;
   }
 }

void Test_Network(Neural_Network_Base & network, std::vector<std::vector<Eigen::VectorXd > > & data)
 {
   std::cout << std::endl;
   std::cout << "Testing Network" << std::endl;
   int num_label = data[0][1].size();
   std::vector<int> label_counter(num_label, 0);
   for(int i=0; i<data.size(); i++)
    { int expected = -1;
      data[i][1].maxCoeff(&expected);
      label_counter[expected]++;
    }
 
   std::vector<int> correct_label_counter(num_label, 0);

   double cost = 0;
     for(int i=0; i<data.size(); i++)
      {
        Eigen::VectorXd outvec = network.Run(data[i][0]);
        cost += network.Cost(data[i][1],outvec);
        int expected = -1; int actual = -1;
        data[i][1].maxCoeff(&expected);
        outvec.maxCoeff(&actual);
        for(int label_val=0; label_val<num_label; label_val++)
         { if (actual == label_val && actual == expected){correct_label_counter[label_val]++;} }
      }

     int num_correct = 0;
     for(int label_val=0; label_val<num_label; label_val++){ num_correct += correct_label_counter[label_val];}

     double cost_reg = 0;
     for(int i=0; i<=network.hidden_layer_num; i++)
      { cost_reg += 0.5*network.lambda/network.run_size*network.Layer[i].weight_matrix.squaredNorm(); }
     cost += cost_reg;

     std::cout << "   Cost             = " << cost << std::endl;
     double fraction_correct = num_correct*1.0/data.size();
     std::cout << "   Number correct   = " << num_correct << "/" << data.size() << " = " << fraction_correct << std::endl;
     for(int label_val=0; label_val<num_label; label_val++)
      { double fraction_correct = 1.0*correct_label_counter[label_val]/label_counter[label_val];
        std::cout << "   Number correct in class " << label_val << ": " << correct_label_counter[label_val] << "/" << label_counter[label_val] << " = " << fraction_correct << std::endl;  } 
 }

#endif
