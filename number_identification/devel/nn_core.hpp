#ifndef NN_CORE_H
#define NN_CORE_H

#include<iostream>
#include<iomanip>
#include<cstdlib>
#include "armadillo"

class nn_core{
	public:
		int num_layers;
	
		arma::vec 	sizes_vec;
		arma::vec*	bias_vecs; 
		arma::mat*	weights_mats;

		nn_core();
		nn_core(arma::vec);
		~nn_core();

		void copy(nn_core&);
		void random_normal(double,double);
		void print();
};

#endif
