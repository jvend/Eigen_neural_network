# Eigen_neural_network

Eigen_neural_network is a header-only feed-forward fully-connected neural network implemented with Eigen. 
Neural network weights and biases can be saved and reloaded.A more sophisticated version would use automatic 
differention and be gpu-accelerated. These capabilities are offered by TensorFlow but this implementation was 
sufficient for our purposes at its time of creation. A version of this was used in our paper 
(Phys. Rev. Lett. 120, 257204 (2018)).

The only dependency for this code is the C++ linear algebra library Eigen. Since Eigen is header-only, 
it is very user-friendly. It is also fast. As a test we've trained a single-layer network on MNIST. We 
acheive a test error of less than 4% on the test dataset. We've provided a few example digits in 
./data/MNIST_image_sample.txt with labels MNIST_label_sample.txt. These can be visualized with the 
Display_digit function in load_save.h. The whole dataset can be found at http://yann.lecun.com/exb/mnist/

I've provided a makefile for compiling. The Eigen directory location needs to be changed but that should be it.
