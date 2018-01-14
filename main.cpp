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
				a = arma::zeros<arma::vec>(2);

	for(i=0;i<num_layers;i++){
		sizes(i) = i+2;
	}
	for(i=0;i<2;i++){
		a(i) = 1.0;
	}

	neuralnetwork simple(sizes);
	simple.print();

	a.print();
	a = simple.feedforward(a);
	std::cout << std::endl;
	a.print();

	return 0;
}
