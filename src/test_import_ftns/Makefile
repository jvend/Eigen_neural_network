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

build: test_load_save

all: test_load_save 

test_load_save: test_load_save.o 
	$(CPP_COM) $(CPP_FLAGS) test_load_save.o -o test_load_save \
   -I$(EIGEN_DIR) \


clean:
	rm -fr *.o  test_load_save
