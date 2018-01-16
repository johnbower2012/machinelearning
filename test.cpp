#include<iostream>
#include<iomanip>
#include<cstdlib>
#include "armadillo"

int main(int argc, char* argv[]){
	int i, j, a, b;
	a = atoi(argv[1]);
	b = atoi(argv[2]);

	arma::mat test = arma::zeros<arma::mat>(a,b);

	for(i=0;i<a;i++){
		for(j=0;j<b;j++){
			test(i,j) = i - j;
		}
	}

	test.print();
	test = arma::shuffle(test);
	test.print();

	return 0;

}





