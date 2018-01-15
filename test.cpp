#include<iostream>
#include<iomanip>
#include<cstdlib>
#include "armadillo"
#include "NN_core.cpp"

int main(int argc, char* argv[]){
	int i, N=3;

	arma::vec sizes_vec = arma::zeros<arma::vec>(N);
	for(i=0;i<N;i++){
		sizes_vec(i) = 5 - i;
	}
	NN_core delta1(sizes_vec),
			delta2(sizes_vec);
	std::cout << "11111111111111111111111111111111" << std::endl;
	delta1.print();
	std::cout << "22222222222222222222222222222222" << std::endl;
	delta2.print();
	delta2.random_normal(0.0,1.0);
	std::cout << "22222222222222222222222222222222" << std::endl;	
	delta2.print();
	delta1.copy(delta2);
	std::cout << "11111111111111111111111111111111" << std::endl;
	delta1.print();
	std::cout << "11111111111111111111111111111111" << std::endl;

	return 0;

}





