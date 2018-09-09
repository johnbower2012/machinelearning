#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include<iostream>
#include<iomanip>
#include<cstdlib>
#include<cmath>
#include<random>
#include<chrono>
#include "armadillo"
#include "nn_core.cpp"

inline double sigmoid(double x);
arma::vec sigmoid_vec(arma::vec x);
arma::vec sigmoid_derivative_vec(arma::vec x);

arma::vec hadamard_product(arma::vec a, arma::vec b);

class nn_data{
public:
  int num_length,
    num_data,
    num_outcomes;
  
  arma::mat 	vector_mat,
    index_mat;
  
  nn_data();
  nn_data(arma::mat, arma::vec, int);
  ~nn_data();

  arma::vec vectorize(int);
  void print();
};

class neuralnetwork{
public:
  nn_core core;
  nn_data data;
  
  neuralnetwork(arma::vec,arma::mat,arma::vec,int);

  arma::vec feedforward(arma::vec);
  arma::vec cost_derivative(arma::vec,arma::vec);
  void backpropagation(arma::vec, arma::vec, nn_core&);
  void stochastic_gradient_descent(int,double,int);
  void update_mini_batch(arma::mat,double);
		
  void print();
};


#endif
