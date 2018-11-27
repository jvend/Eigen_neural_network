#ifndef __NEURAL_LAYER_H
#define __NEURAL_LAYER_H

#include <iostream>
#include <cmath>
#include <Eigen/Dense>
#include <chrono>
#include <random>

class Neural_Layer
 {
   public:
   //Make a constructor that automatically creates and initializes weight_matrix
   //Initialize with appropriate Gaussian
   int inputs; int outputs;
   Eigen::VectorXd bias;
   Eigen::VectorXd bias_cor;
   Eigen::MatrixXd weight_matrix;
   Eigen::MatrixXd weight_matrix_cor;
   Neural_Layer(int inputs_tmp, int outputs_tmp);
   double sigmoid(double weighted_input, double cvar);
   double sigmoidp(double weighted_input); // sigmoid derivative
   Eigen::VectorXd softmax(Eigen::VectorXd weighted_input, double cvar);
   Eigen::VectorXd out_unact;
   Eigen::VectorXd out;
   bool softmax_flag = false;
   Eigen::VectorXd Run(Eigen::VectorXd& input_vec);
   Eigen::VectorXd delta;
   void Delta(const Eigen::VectorXd& delta_lp1, const Eigen::MatrixXd& weight_matrix_lp1);
 };

double inline Neural_Layer::
sigmoid(double weighted_input, double cvar = 1.0)
   {return 1.0/(1.0+std::exp(-cvar*weighted_input));}

double inline Neural_Layer::
sigmoidp(double weighted_input)
   {return sigmoid(weighted_input)*(1.0-sigmoid(weighted_input));}

Eigen::VectorXd inline Neural_Layer::
softmax(Eigen::VectorXd weighted_input, double cvar = 1.0)
   {
    Eigen::VectorXd output = Eigen::VectorXd::Zero(weighted_input.size());
    double tot = 0;
    for(int i=0; i<weighted_input.size(); i++)
     {tot += std::exp(cvar*weighted_input(i));}
    for(int i=0; i<weighted_input.size(); i++)
     {output(i) = std::exp(cvar*weighted_input(i))/tot;}
    return output;
   }

inline Neural_Layer::
Neural_Layer(int inputs_tmp, int outputs_tmp)
   :
   inputs(inputs_tmp),
   outputs(outputs_tmp),
   bias(Eigen::VectorXd::Zero(outputs)),
   bias_cor(Eigen::VectorXd::Zero(outputs)),
   weight_matrix(Eigen::MatrixXd::Zero(outputs,inputs)),
   weight_matrix_cor(Eigen::MatrixXd::Zero(outputs,inputs)),
   out(Eigen::VectorXd::Zero(outputs)),
   out_unact(Eigen::VectorXd::Zero(outputs)),
   delta(Eigen::VectorXd::Zero(outputs))
   {
     //Initialize weights with appropriate gaussian distribution
     //Construct a random generator engine from a time-based seed:
     unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
     std::default_random_engine generator (seed);
   
     //distribution(mean,std_dev)
     std::normal_distribution<double> distribution (0.0,1.0/std::sqrt(inputs));
     std::normal_distribution<double> distribution2 (0.0,1.0);

     for (int i=0; i<outputs; i++) {
     for (int j=0; j<inputs;  j++) {
      weight_matrix(i,j) = distribution(generator);
      }}
     
     for (int i=0; i<outputs; i++)
      {bias(i) = distribution2(generator);}
   }

Eigen::VectorXd inline Neural_Layer::
Run(Eigen::VectorXd& input_vec)
   {
     out_unact = weight_matrix*input_vec + bias;
     if(softmax_flag==false)
      {
       for(int i=0; i<outputs; i++)
        {out(i)=sigmoid(out_unact(i));}
      }
     else
      { out = softmax(out_unact); }

   return out;
   }

void inline Neural_Layer::
Delta(const Eigen::VectorXd& delta_lp1, const Eigen::MatrixXd& weight_matrix_lp1)
   {
     delta = weight_matrix_lp1.transpose()*delta_lp1;
     for(int i=0; i<outputs; i++)
      {delta(i)*=sigmoidp(out_unact(i));}
   }

#endif
