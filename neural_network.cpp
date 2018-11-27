#include <iostream>
#include <vector>
#include <sstream>
#include <fstream>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include "./src/neural_layer.h"
#include "./src/neural_network.h"
#include "./src/load_save.h"
#include "./src/train_test.h"

int main(int argc, char* argv[])
{

int image_size = 784;   // Length of image vectors
int class_num  = 10;    // Number of possible classes/labels
double training_fraction = 0.8; //fraction of data used for training vs cross-validation

// Training/testing data location
std::string data_path("./data/");
std::string train_images("MNIST_training_images.txt");
std::string train_labels("MNIST_training_labels.txt");
std::string test_images("MNIST_testing_images_sample.txt");
std::string test_labels("MNIST_testing_labels_sample.txt");

// NN Flags
bool train_flag     = false;
bool test_flag      = true;
bool save_NN        = false;          
bool load_prev_NN   = true;  
std::string file_base_name("MNIST");   // Used when saving NN weights and biases

// Create the network and define hyper-parameters
int input_num         = image_size;
int output_num        = class_num;
int hidden_layer_num  = 1;
int neurons_per_layer = 50;
double descent_rate   = 0.3;           
double lambda         = 0.1;           // L2 regularization parameter
int run_size          = 1;             // Number of images used during train/test run (defaulted here to 1 but set below by dataset sizes)
int batch_size        = 10;
int epochs            = 30;
bool soft_max_flag    = false;

Neural_Network_Base network(input_num, output_num, hidden_layer_num, neurons_per_layer, descent_rate, lambda, batch_size, epochs, run_size, soft_max_flag);

//////Train the network, test on labeled data, and save
if( train_flag == true )
{
    // Import the training data and divide into training/cross-validation sets
    // Below, the vector dimensions of import_labeled_data return refer to [train/cross-val set][image/label num][image/label val]
    std::vector<std::vector<std::vector<Eigen::VectorXd > > > training_data = import_labeled_data(data_path + train_images, data_path + train_labels, image_size, class_num, training_fraction);
    
    int training_size = training_data[0].size();
    int cross_val_size = training_data[1].size();

    // Train the network
    network.run_size = training_size;
    Train_Network(network,training_data[0]);

    // Testing for cross-validation and hyperparameter tuning
    network.run_size = cross_val_size;
    Test_Network(network,training_data[1]);

    // Save the trained network
    if(save_NN == true) 
     { Save(network,file_base_name,hidden_layer_num); }
}

//Load a previously trained network and test
if( test_flag == true )
{
    // Import the testing data
    std::vector<std::vector<std::vector<Eigen::VectorXd > > > testing_data = import_labeled_data(data_path + test_images, data_path + test_labels, image_size, class_num);
    int testing_size = testing_data[0].size();

    //Uncomment the below line to see the only "incorrectly" determined digit in our sample test set. The dataset says it's a 9 but it doesn't look like that to human eyes either.
    //Display_digit(testing_data[0],80);

    // Load a previously trained network (care needs to be taken here to ensure that the imported NN has the same layer sizes as that specified above)
    if(load_prev_NN == true)
     { Load(network,file_base_name,hidden_layer_num); }

    // Apply network to test data
    network.run_size = testing_size;
    Test_Network(network,testing_data[0]);
}

return 0;
}
