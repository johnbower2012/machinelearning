#include<iostream>
#include<iomanip>
#include<cstdlib>
#include<cmath>
#include<random>
#include<chrono>
#include "armadillo"
#include "neuralnetwork.cpp"

arma::vec dectobin(int a, int digits);
int winlossdraw(arma::vec binary, int digits);
int outtonumber(arma::vec out);

int main(int argc, char* argv[]){
	int i, j, num_layers,
		epochs, mini_batch_size,
		edge, ttt, total,
		wld, outcomes;
	double eta;

	if(argc>5){
		edge = atoi(argv[1]);
		num_layers = atoi(argv[2]);
		epochs = atoi(argv[3]);
		mini_batch_size = atoi(argv[4]);
		eta = atof(argv[5]);

		ttt = edge*edge;
		total = pow(2,ttt);
		outcomes=4;
	}
	else{
		std::cout << "Error. Enter 'edge number_of_layers epochs mbs eta' on same line." << std::endl;
		exit(1);
	}

	arma::vec 	sizes = arma::zeros<arma::vec>(num_layers),
				indices = arma::zeros<arma::vec>(total),
				binary = arma::zeros<arma::vec>(total),
				out = arma::zeros<arma::vec>(outcomes),
				random = arma::zeros<arma::vec>(mini_batch_size);
	arma::mat	data = arma::zeros<arma::mat>(ttt,total);

	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::default_random_engine generator(seed);
	std::uniform_int_distribution<int> dist(0,total-1);
	int ran,k;
	double ratio;
	for(i=0;i<mini_batch_size;i++){
		ran = dist(generator);
		random(i) = ran;
	}

	sizes(0) = ttt;
	sizes(num_layers-1) = 4;
	for(i=1;i<num_layers-1;i++){
		std::cout << "layer" << i+1 << ":";
		std::cin >> k;
		sizes(i) = k;
	}

	for(i=0;i<total;i++){
		binary = dectobin(i,ttt);
		for(j=0;j<ttt;j++){
			data(j,i) = binary(j);
		}
		wld = winlossdraw(binary,ttt);
		indices(i) = wld;
	}
	neuralnetwork ticktacktoe(sizes,data,indices,outcomes);
	ratio=0.0;
	for(i=0;i<mini_batch_size;i++){
		ran = random(i);
		for(j=0;j<ttt;j++){
			binary(j) = data(j,ran);
		}
		out = ticktacktoe.feedforward(binary);
		if((indices(ran)-outtonumber(out))==0){
			ratio++;
		}
	}
	ratio /= (double) mini_batch_size;
	std::cout << ratio << std::endl;
	std::cout << "************************" << std::endl;

	ticktacktoe.stochastic_gradient_descent(epochs,eta,mini_batch_size);
	
	ratio=0.0;
	for(i=0;i<mini_batch_size;i++){
		ran = random(i);
		for(j=0;j<ttt;j++){
			binary(j) = data(j,ran);
		}
		out = ticktacktoe.feedforward(binary);
		if((indices(ran)-outtonumber(out))==0){
			ratio++;
		}
	}
	ratio /= (double) mini_batch_size;

	std::cout << ratio << std::endl;
	return 0;
}

arma::vec dectobin(int a, int digits){
	int i, rem,
		dec = a;
	arma::vec binary = arma::zeros<arma::vec>(digits);
	for(i=0;i<digits;i++){
		if(dec==0){
			break;
		}
		rem = dec%2;
		binary(i) = rem;
		dec /= 2;
	}
	return binary;
}
int winlossdraw(arma::vec binary, int digits){
	int i, j, N = sqrt(digits),
		sum=0,one=0,zero=0;	
	for(i=0;i<N;i++){
		sum = 0;
		for(j=0;j<N;j++){
			sum += binary(N*i+j);
		}
		if(sum==0){
			zero++;
		}
		else if(sum==N){
			one++;
		}

		
		sum = 0;
		for(j=0;j<N;j++){
			sum += binary(i+N*j);
		}
		if(sum==0){
			zero++;
		}
		else if(sum==N){
			one++;
		}
	}

	sum=0;
	for(i=0;i<N;i++){
		sum += binary(N*i+1);
	}
	if(sum==0){
		zero++;
	}
	else if(sum==N){
		one++;
	}
	sum=0;
	for(i=0;i<N;i++){
		sum += binary((N-1)*(i+1));
	}
	if(sum==0){
		zero++;
	}
	else if(sum==N){
		one++;
	}

	if(zero==0){
		if(one==0){
			return 2;
		}
		else{
			return 1;
		}
	}
	else if(one==0){
		return 0;
	}
	else{
		return 3;
	}
}
int outtonumber(arma::vec out){
	int i, N = out.n_elem, index=-1;
	double max=-100;
	for(i=0;i<N;i++){
		if(max<out(i)){
			max = out(i);
			index = i;
		}
	}
	return index;
}

