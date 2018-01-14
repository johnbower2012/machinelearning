#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include<iostream>
#include<iomanip>
#include<cstdlib>
#include<cmath>
#include<random>
#include<chrono>
#include "armadillo"

inline double sigmoid(double x);
arma::vec sigmoid_vec(arma::vec x);
arma::vec sigmoid_deriv_vec(arma::vec x);

arma::vec hadamard_product(arma::vec a, arma::vec b);

class neuralnetwork{
	public:
		int num_layers;

		arma::vec 	sizes_vec;
		arma::vec*	bias_vecs;
		arma::mat*	weights_mats;

		neuralnetwork(arma::vec);

		arma::vec feedforward(arma::vec);

		void print();
};

#endif
