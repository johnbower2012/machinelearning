#ifndef NN_CORE_H
#define NN_CORE_H

#include<iostream>
#include<iomanip>
#include<cstdlib>
#include "armadillo"

class NN_core{
	public:
		int num_layers;
	
		arma::vec 	sizes_vec;
		arma::vec*	bias_vecs; 
		arma::mat*	weights_mats;

		NN_core();
		NN_core(arma::vec);
		~NN_core();

		void copy(NN_core&);
		void random();
		void print();
};

#endif
