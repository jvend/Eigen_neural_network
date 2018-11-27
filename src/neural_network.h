#ifndef __NEURAL_NETWORK_H
#define __NEURAL_NETWORK_H

#include "neural_layer.h"
#include <iostream>

//Eventually want to derive classification and regression networks from base class
class Neural_Network_Base
 {
   public:
    int init_input_num;
    int final_output_num;
    int hidden_layer_num;
    int neurons_per_layer;
    std::vector<Neural_Layer> Layer;

    int run_size;
    int batch_size;
    int epochs;
    double descent_rate;
    double lambda; //regularization parameter
    bool softmax_flag;

    Neural_Network_Base(int init_input_num, int final_output_num, int hidden_layer_num, int neurons_per_layer, double descent_rate, double lambda, int batch_size, int epochs, int run_size, bool softmax_flag);
    Eigen::VectorXd out_unact;
    Eigen::VectorXd out;
    Eigen::VectorXd Run(Eigen::VectorXd& input_vec);
    double Cost(Eigen::VectorXd& expected, Eigen::VectorXd& actual);
    void Backprop(Eigen::VectorXd& input_vec, Eigen::VectorXd& expected);

   private:
    int batch_ct;
    double cutoff;
 };

inline Neural_Network_Base::
  Neural_Network_Base(int init_input_num, int final_output_num, int hidden_layer_num, int neurons_per_layer, double descent_rate, double lambda, int batch_size, int epochs, int run_size, bool softmax_flag)
  :
  init_input_num(init_input_num),
  final_output_num(final_output_num),
  hidden_layer_num(hidden_layer_num),
  neurons_per_layer(neurons_per_layer),
  descent_rate(descent_rate),
  lambda(lambda),
  batch_size(batch_size),
  epochs(epochs),
  run_size(run_size),
  softmax_flag(softmax_flag),
  batch_ct(0),
  cutoff(1e-10),
  out(Eigen::VectorXd::Zero(final_output_num))
  //out_unact(Eigen::VectorXd::Zero(final_output_num))
  {
    for(int i=0; i<=hidden_layer_num; i++)
     {
       if(i==0)
         {
          Neural_Layer layer(init_input_num,neurons_per_layer);
          Layer.push_back(layer);
         }
       else if(i==hidden_layer_num)
         {
          Neural_Layer layer(neurons_per_layer,final_output_num);
          if(softmax_flag==true){layer.softmax_flag = true;}
          Layer.push_back(layer);
         }
       else //Implement hidden layers
         {
          Neural_Layer layer(neurons_per_layer,neurons_per_layer);
          Layer.push_back(layer);
         }   
     } 
  }

double inline Neural_Network_Base::
Cost(Eigen::VectorXd& expected, Eigen::VectorXd& actual)
  {
    //double cost = 0.5*actual.dot(expected); //Quadratic Cost
    double cost = 0.0; //Cross-Entropy Cost
    if(softmax_flag==false)
     {
      for (int i=0; i<actual.size(); i++)
       {
         cost += -1.0/run_size*( expected(i)*( (std::isinf(log(actual(i))) && std::abs(expected(i)) < cutoff) ? 0 : log(actual(i)) ) \
              + (1-expected(i))*( (std::isinf(log(1-actual(i))) && std::abs(expected(i)-1) < cutoff) ? 0 : log(1-actual(i)) ) ); 
       }
     }
    else
     { 
      for (int i=0; i<actual.size(); i++)
       {
         cost += -1.0/run_size*( expected(i)*( (std::isinf(log(actual(i))) && std::abs(expected(i)) < cutoff) ? 0 : log(actual(i)) ) ); 
       }
     }

    return cost;
  }

//Feedforward
Eigen::VectorXd inline Neural_Network_Base::
Run(Eigen::VectorXd& input_vec)
  {
    Eigen::VectorXd init_vec = input_vec;
    for(int i=0; i<=hidden_layer_num; i++)
     { 
      init_vec = Layer[i].Run(init_vec);
     }
    out = init_vec; 
    return out;
  }

//Backpropagation routine w/ stochastic gradient descent
void inline Neural_Network_Base::
Backprop(Eigen::VectorXd& input_vec, Eigen::VectorXd& expected)
  {
    //Compute error for each layer
    Layer[hidden_layer_num].delta = Run(input_vec) - expected;
    Eigen::VectorXd delta_l = Layer[hidden_layer_num].delta;
    for(int i=hidden_layer_num-1; i>=0; i--)
     {
      Layer[i].Delta(delta_l,Layer[i+1].weight_matrix);
      delta_l = Layer[i].delta;
     } 

    //Compute change to weights and biases
    if(batch_ct == 0)
     { for(int i=0; i<=hidden_layer_num; i++)
        { Layer[i].weight_matrix_cor = Eigen::MatrixXd::Zero(Layer[i].outputs,Layer[i].inputs);
          Layer[i].bias_cor          = Eigen::VectorXd::Zero(Layer[i].outputs); }
     }

    for(int i=0; i<=hidden_layer_num; i++)
     {
      if(i==0)
       { 
         for(int j=0; j<init_input_num; j++)
          { 
            Eigen::VectorXd weight_cor_colj = -descent_rate/batch_size*Layer[i].delta*input_vec(j);
            Layer[i].weight_matrix_cor.col(j) += weight_cor_colj; 
          }
       }
      else
       { 
         for(int j=0; j<Layer[i-1].outputs; j++)
          {
            Eigen::VectorXd weight_cor_colj = -descent_rate/batch_size*Layer[i].delta*Layer[i-1].out[j];
            Layer[i].weight_matrix_cor.col(j) += weight_cor_colj;
          }
       }

      Layer[i].bias_cor += -descent_rate/batch_size*Layer[i].delta;

     }

    batch_ct++;
    if(batch_ct >= batch_size)
     {
       for(int i=0; i<=hidden_layer_num; i++)
        {
          Layer[i].weight_matrix *= (1.0-descent_rate*lambda/run_size);
          Layer[i].weight_matrix += Layer[i].weight_matrix_cor;
          Layer[i].bias          += Layer[i].bias_cor;
        }
      batch_ct = 0;
     }

  }

#endif
