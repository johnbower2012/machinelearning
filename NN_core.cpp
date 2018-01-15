#include "NN_core.hpp"

NN_core::NN_core(){
	num_layers = 0;
	bias_vecs = nullptr;
	weights_mats = nullptr;
}
NN_core::NN_core(arma::vec input_vec){
	int i, ith_size, iplus1th_size;

	sizes_vec = input_vec;
	num_layers = sizes_vec.n_elem;

	bias_vecs = new arma::vec[num_layers-1];
	weights_mats = new arma::mat[num_layers-1];

	for(i=0;i<num_layers-1;i++){
		ith_size = sizes_vec(i);
		iplus1th_size = sizes_vec(i+1);

		bias_vecs[i] = arma::zeros<arma::vec>(iplus1th_size);
		weights_mats[i] = arma::zeros<arma::mat>(iplus1th_size,ith_size);
	}
}
NN_core::~NN_core(){
	num_layers = 0;
	delete[] bias_vecs;
	delete[] weights_mats;
}

void NN_core::copy(NN_core& copy){
	int i, j, k,
		l1, l2;
	num_layers = copy.num_layers;

	for(i=0;i<num_layers;i++){
		sizes_vec(i) = copy.sizes_vec(i);
	}
	for(i=0;i<num_layers-1;i++){
		l1 = sizes_vec(i);
		l2 = sizes_vec(i+1);
		for(j=0;j<l2;j++){
			bias_vecs[i](j)	= copy.bias_vecs[i](j);
			for(k=0;k<l1;k++){
				weights_mats[i](j,k) = copy.weights_mats[i](j,k);
			}
		}
	}
}
void NN_core::random_normal(double mean, double variance){
	int i, j, k,
		ith_size, iplus1th_size;
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::default_random_engine generator(seed);
	std::normal_distribution<double> dist(mean,sqrt(variance));
	double random;

	for(i=0;i<num_layers-1;i++){
		ith_size = sizes_vec(i);
		iplus1th_size = sizes_vec(i+1);
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
void NN_core::print(){
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
		l1 = sizes_vec(i+1);
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




