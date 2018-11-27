EIGEN_DIR      = /Users/jvenderley/Documents/programs/eigen-eigen-26667be4f70b

#Choose compiler and flags
CPP_COM       = g++
CPP_FLAGS     = -m64 -std=c++11 -O3

#Rules ------------------

%.o: %.cpp
	$(CPP_COM) -c $(CPP_FLAGS) \
   -I$(EIGEN_DIR) \
   -o $@ $<

#Targets -----------------

build: neural_network

all: neural_network 

neural_network: neural_network.o 
	$(CPP_COM) $(CPP_FLAGS) neural_network.o -o neural_network \
   -I$(EIGEN_DIR) \


clean:
	rm -fr *.o  neural_network
