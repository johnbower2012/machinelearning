#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include<iostream>
#include<iomanip>
#include<cstdlib>
#include<cmath>
#include<random>
#include<chrono>
#include "armadillo"
#include "NN_core.cpp"

inline double sigmoid(double x);
arma::vec sigmoid_vec(arma::vec x);
arma::vec sigmoid_derivative_vec(arma::vec x);

arma::vec hadamard_product(arma::vec a, arma::vec b);

class neuralnetwork{
	public:
		NN_core core;

		neuralnetwork(arma::vec);

		arma::vec feedforward(arma::vec);
		arma::vec cost_derivative(arma::vec,arma::vec);
		void backpropagation(arma::vec, arma::vec, NN_core&);
		void stochastic_gradient_descent(arma::vec,double,arma::mat);
		

		void print();
};

#endif
