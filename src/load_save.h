#ifndef __LOAD_SAVE_H
#define __LOAD_SAVE_H
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <Eigen/Dense>

std::vector <int > import(std::string filename)
{
  std::vector <int > myvec;
  std::ifstream myfile (filename);
  std::string line;
  while(getline(myfile, line))
    {
      std::stringstream ss(line);
      std::string d1;
      if(ss >> d1)
       {
         int val1 = stoi(d1);
         myvec.push_back(val1);
       }
    }
  myfile.close();
  return myvec;
}

std::vector <Eigen::VectorXd > import(std::string filename, int vec_dim)
{
  std::vector <Eigen::VectorXd > myvec;
  Eigen::VectorXd image = Eigen::VectorXd::Zero(vec_dim);
  std::ifstream myfile (filename);
  std::string line;
  while(getline(myfile, line))
    {
      std::stringstream ss(line);
      std::string d1;
      int count = 0;
      while(ss >> d1)
       {
         count++;
         if(count > vec_dim){std::cout << "Error: import vector dimension mismatch" << std::endl; std::exit(1);}
         double val1 = stod(d1);
         image(count-1) = val1;
         if(count == vec_dim)
          {myvec.push_back(image);}
       }
    }
  myfile.close();
  return myvec;
}

Eigen::MatrixXd import(std::string filename, int row_dim, int col_dim)
{
  Eigen::MatrixXd image = Eigen::MatrixXd::Zero(row_dim,col_dim);
  std::ifstream myfile (filename);
  std::string line;
  int count_row = 0;
  while(getline(myfile, line))
    {
      count_row++;
      if(count_row > row_dim){std::cout << "Error: import matrix row dimension mismatch" << std::endl; std::exit(1);}
      std::stringstream ss(line);
      std::string d1;
      int count_col = 0;
      while(ss >> d1)
       {
         count_col++;
         if(count_col > col_dim){std::cout << "Error: import matrix column dimension mismatch" << std::endl; std::exit(1);}
         double val1 = stod(d1);
         image(count_row-1,count_col-1) = val1;
       }  
    }  
  myfile.close();
  return image;
} 

std::vector<std::vector<std::vector<Eigen::VectorXd > > > import_labeled_data(std::string images_path, std::string labels_path, int image_size, int class_num, double split_fraction = -1)
{
 //split_fraction(if specified) allows us to divide training data into training data and cross-validation data
 std::vector<Eigen::VectorXd > images = import(images_path, image_size); 
 std::vector<int> labels = import(labels_path);
 if( images.size() != labels.size() ){ std::cout << "Error: the number of labels does not equal the number of images" << std::endl; std::exit(1); }

 int split_size;
 if (split_fraction < -0.5) { split_size = labels.size();}
 else {split_size = split_fraction*labels.size();}

 std::vector<std::vector<std::vector<Eigen::VectorXd > > > data_holder;
 std::vector<std::vector<Eigen::VectorXd > > data1;
 std::vector<std::vector<Eigen::VectorXd > > data2;

 for(int i=0; i<split_size; i++)
  { 
    std::vector<Eigen::VectorXd > row;
    Eigen::VectorXd label = Eigen::VectorXd::Zero(class_num);
    label(labels[i]) = 1.0;
    row.push_back(images[i]); row.push_back(label);
    data1.push_back(row);
  }
  data_holder.push_back(data1);

 for(int i=split_size; i<labels.size(); i++)
  { 
    std::vector<Eigen::VectorXd > row;
    Eigen::VectorXd label = Eigen::VectorXd::Zero(class_num);
    label(labels[i]) = 1.0;
    row.push_back(images[i]); row.push_back(label);
    data2.push_back(row);
  }
  if(split_size != labels.size()){ data_holder.push_back(data2);}
  
 return data_holder;
}

//Display MNIST digits in ascii
void Display_digit(std::vector<std::vector<Eigen::VectorXd > > & data, int image_num)
{
  for(int j=0; j<784; j++){
   if(data[image_num][0](j)<0.2)
    {std::cout << " ";}
   if(data[image_num][0](j)>=0.2 && data[image_num][0](j)<0.7)
    {std::cout << "x";}
   if(data[image_num][0](j)>=0.7)
    {std::cout << "X";}
   if( (j+1)%28 == 0 )
    {std::cout << std::endl;}
   }
  std::cout << std::endl;
}

void Save(Neural_Network_Base & network, std::string file_base_name, int hidden_layer_num)
 {
   for(int i=0; i<hidden_layer_num+1; i++)
    {
      std::ofstream weights_ofstream ("./trained/" + file_base_name + "_weights" + std::to_string(i) + ".txt");
      weights_ofstream << network.Layer[i].weight_matrix << std::endl;
      weights_ofstream.close();
      
      std::ofstream bias ("./trained/" + file_base_name + "_bias" + std::to_string(i) + ".txt");
      bias << network.Layer[i].bias.transpose() << std::endl;
      bias.close();
    }
 }

void Load(Neural_Network_Base & network, std::string file_base_name, int hidden_layer_num)
 {
   for(int i=0; i<hidden_layer_num+1; i++)
    {
     std::vector<Eigen::VectorXd > bias_container = import("./trained/" + file_base_name + "_bias" + std::to_string(i) + ".txt",network.Layer[i].outputs);
     Eigen::VectorXd bias_i = bias_container[0];
     Eigen::MatrixXd weights_i = import("./trained/" + file_base_name + "_weights" + std::to_string(i) + ".txt",network.Layer[i].outputs,network.Layer[i].inputs);
     if( weights_i.rows() != network.Layer[i].weight_matrix.rows() || weights_i.cols() != network.Layer[i].weight_matrix.cols() || bias_i.size() != network.Layer[i].bias.size() )
      {std::cout << "Error: size mismatch between loaded NN and initial" << std::endl; std::exit(1); }
     network.Layer[i].weight_matrix = weights_i;
     network.Layer[i].bias          = bias_i;
    } 
 }

#endif
