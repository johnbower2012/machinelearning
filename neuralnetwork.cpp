#include<iostream>
#include<iomanip>
#include<cstdlib>
#include<cmath>
#include<random>
#include<chrono>
#include "armadillo"

inline double sigmoid(double x){
	return 1.0/(1.0+exp(-x));
}
arma::vec sigmoid_vec(arma::vec x);

class neuralnetwork{
	public:
		int num_layers;

		arma::vec 	sizes_vec;
		arma::vec* 	bias_vecs;
		arma::mat* weights_mats;

		neuralnetwork(arma::vec);

		arma::vec feedforward(arma::vec);

		void print();
};

/************************
 *						*
 * BEGIN MAIN FUNCTION	*
 *						*
 ************************/

int main(int argc, char* argv[]){
	int i, j, k, num_layers,
		l1, l2, l3;

	if(argc>1){
		num_layers = atoi(argv[1]);
	}
	else{
		std::cout << "Error. Enter number_of_layers on same line." << std::endl;
		exit(1);
	}

	arma::vec 	sizes = arma::zeros<arma::vec>(num_layers),
				a = arma::zeros<arma::vec>(2),
				a_prime = arma::zeros<arma::vec>(1+num_layers);

	for(i=0;i<num_layers;i++){
		sizes(i) = i+2;
	}
	for(i=0;i<2;i++){
		a(i) = 1.0;
	}

	neuralnetwork simple(sizes);
	simple.print();
	a_prime = simple.feedforward(a);

	a_prime.print();

	return 0;
}

/************************
 *						*
 * END MAIN FUNCTION	*
 *						*
 ************************/

neuralnetwork::neuralnetwork (arma::vec input_vec){
	int i, j, k,
		ith_size, iplus1th_size;
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::default_random_engine generator(seed);
	std::normal_distribution<double> dist(0,1.0);
	double random;

	sizes_vec = input_vec;
	num_layers = sizes_vec.n_elem;

	bias_vecs = new arma::vec[num_layers-1];
	weights_mats = new arma::mat[num_layers-1];

	for(i=0;i<num_layers-1;i++){
		ith_size = sizes_vec(i);
		iplus1th_size = sizes_vec(i+1);
		bias_vecs[i] = arma::zeros<arma::vec>(iplus1th_size);
		weights_mats[i] = arma::zeros<arma::mat>(iplus1th_size,ith_size);
		for(j=0;j<iplus1th_size;j++){
			random = dist(generator);
			bias_vecs[i](j) = random;
			for(k=0;k<ith_size;k++){
				random = dist(generator);
				weights_mats[i](j,k) = random;
			}
		}
	}
}
arma::vec neuralnetwork::feedforward(arma::vec a){
	int i;
	for(i=0;i<num_layers-1;i++){
		a = sigmoid_vec(weights_mats[i]*a + bias_vecs[i]);
	}
	return a;
/*
	//FOR USE IF YOU WISH TO CHECK THE PROGRESS

	arma::vec* a_prime = new arma::vec[num_layers];
	for(i=0;i<num_layers;i++){
		a_prime[i] = arma::zeros<arma::vec>(sizes_vec(i));
	}

	a_prime[0] = a;
	for(i=0;i<num_layers-1;i++){
		a_prime[i+1] = sigmoid_vec(weights_mats[i]*a_prime[i] + bias_vecs[i]);
	}
	return a_prime[num_layers-1];
*/
}
void neuralnetwork::print(){
	int i, j, k,
		l1, l2;

	std::cout << "num_layers:" << std::setw(10) << num_layers << std::endl;
	std::cout << "size_vec:  ";
	for(i=0;i<num_layers;i++){
		std::cout << std::setw(10) << sizes_vec(i);
	}
	std::cout << std::endl << std::endl;
	std::cout << "bias vectors:" << std::endl;
	for(i=0;i<num_layers-1;i++){
		l1 = sizes_vec(i);
		for(j=0;j<l1;j++){
			std::cout << std::setw(10) << bias_vecs[i](j);
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;

	std::cout << "weights_mats:" << std::endl;
	for(i=0;i<num_layers-1;i++){
		l1 = sizes_vec(i);
		l2 = sizes_vec(i+1);
		std::cout << i << " --> " << i+1 << ":" << std::endl;
		for(j=0;j<l2;j++){
			for(k=0;k<l1;k++){
				std::cout << std::setw(10) << weights_mats[i](j,k);
			}
			std::cout << std::endl;
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
}


arma::vec sigmoid_vec(arma::vec x){
	int i, N = x.n_elem;
	arma::vec value = arma::zeros<arma::vec>(N);
	for(i=0;i<N;i++){
		value(i) = sigmoid(x(i));
	}
	return value;
}









