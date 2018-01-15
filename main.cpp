#include<iostream>
#include<iomanip>
#include<cstdlib>
#include<cmath>
#include<random>
#include<chrono>
#include "armadillo"
#include "neuralnetwork.cpp"

int main(int argc, char* argv[]){
	int i, num_layers;

	if(argc>1){
		num_layers = atoi(argv[1]);
	}
	else{
		std::cout << "Error. Enter number_of_layers on same line." << std::endl;
		exit(1);
	}

	arma::vec 	sizes = arma::zeros<arma::vec>(num_layers),
				a = arma::zeros<arma::vec>(2),
				b = arma::zeros<arma::vec>(4),
				c = b;

	for(i=0;i<num_layers;i++){
		sizes(i) = i+2;
	}
	for(i=0;i<2;i++){
		a(i) = 1.0;
	}
	for(i=0;i<4;i++){
		c(i) = 0.0;
	}

	neuralnetwork 	simple(sizes);
	NN_core			error(sizes);
	simple.print();

	a.print();
	b = simple.feedforward(a);
	std::cout << std::endl;
	b.print();
	std::cout << std::endl;
	c.print();
	std::cout << std::endl;
	simple.backpropagation(a,c,error);
	error.print();

	return 0;
}
