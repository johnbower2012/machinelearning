#include "neuralnetwork.hpp"

neuralnetwork::neuralnetwork (arma::vec sizes_vec, arma::mat vectors, arma::vec indices, int outcomes) : core(sizes_vec), data(vectors, indices, outcomes){
	core.random_normal(0,1.0);
}
arma::vec neuralnetwork::feedforward(arma::vec a){
  int i;
  for(i=0;i<core.num_layers-1;i++){
    a = sigmoid_vec(core.weights_mats[i]*a + core.bias_vecs[i]);
  }
  return a;
}
arma::vec neuralnetwork::cost_derivative(arma::vec a, arma::vec y){
  return (a - y);
}
void neuralnetwork::backpropagation(arma::vec input_vec, arma::vec output_vec, nn_core& derivatives){
  int i, j, k,
    ith_size, iplus1th_size,
    num_layers = core.num_layers;
  double delta_j;

  arma::vec*	zs_vecs = new arma::vec[num_layers-1],
    *activations_vecs = new arma::vec[num_layers],
    *deltas_vecs = new arma::vec[num_layers-1];
  
  activations_vecs[0] = arma::zeros<arma::vec>(core.sizes_vec(0));
  for(i=0;i<num_layers-1;i++){
    zs_vecs[i] = arma::zeros<arma::vec>(core.sizes_vec(i));
    activations_vecs[i] = arma::zeros<arma::vec>(core.sizes_vec(i+1));
    deltas_vecs[i] = arma::zeros<arma::vec>(core.sizes_vec(i));
  }

  activations_vecs[0] = input_vec;
  for(i=0;i<num_layers-1;i++){
    zs_vecs[i] = core.weights_mats[i]*activations_vecs[i] + core.bias_vecs[i];
    activations_vecs[i+1] = sigmoid_vec(zs_vecs[i]);
  }

  deltas_vecs[num_layers-2] = hadamard_product(cost_derivative(activations_vecs[num_layers-1],output_vec),sigmoid_derivative_vec(zs_vecs[num_layers-2]));
  for(i=num_layers-3;i>-1;i--){
    deltas_vecs[i] = hadamard_product(core.weights_mats[i+1].t()*deltas_vecs[i+1],sigmoid_derivative_vec(zs_vecs[i]));
  }

  for(i=0;i<num_layers-1;i++){
    ith_size = core.sizes_vec(i);
    iplus1th_size = core.sizes_vec(i+1);
    for(j=0;j<iplus1th_size;j++){
      delta_j = deltas_vecs[i](j);
      derivatives.bias_vecs[i](j) = delta_j;
      for(k=0;k<ith_size;k++){
	derivatives.weights_mats[i](j,k) = delta_j*activations_vecs[i](k);
      }
    }
  }
}
void neuralnetwork::stochastic_gradient_descent(int epochs, double eta, int mini_batch_size){
  int i, j, k,
    mini_batch, min, max;
  arma::mat 	indices,
    all_indices = arma::zeros<arma::mat>(data.num_data,2);

  for(i=0;i<data.num_data;i++){
    all_indices(i,0) = data.index_mat(i,0);
    all_indices(i,1) = data.index_mat(i,1);
  }
  all_indices = arma::shuffle(all_indices);

  if(mini_batch_size<data.num_data){
    mini_batch = data.num_data/mini_batch_size+1;
    indices = arma::zeros<arma::mat>(mini_batch_size,2);
  }
  else{
    mini_batch = 1;
    indices = arma::zeros<arma::mat>(data.num_data,2);
  }

  for(i=0;i<epochs;i++){
    for(j=0;j<mini_batch;j++){
      min = j*mini_batch_size;
      max = min + mini_batch_size;
      if(max > data.num_data){
	max = data.num_data;
	mini_batch_size = max - min;
	indices = arma::zeros<arma::mat>(mini_batch_size,2);
      }
      
      for(k=0;k<mini_batch_size;k++){
	indices(k,0) = all_indices(min+k,0);
	indices(k,1) = all_indices(min+k,1);
      }
      update_mini_batch(indices, eta);
    }
  }
  

  printf("Epoch%d: ",i);
}
void neuralnetwork::update_mini_batch(arma::mat index_mat, double eta){
  int i, j, k, l,
    selection,
    mini_batch_size = index_mat.n_rows;
  nn_core derivatives(core.sizes_vec);
  arma::vec 	input_vec = arma::zeros<arma::vec>(core.sizes_vec(0)),
    output_vec = arma::zeros<arma::vec>(core.sizes_vec(core.num_layers-1));
  
  for(i=0;i<mini_batch_size;i++){
    for(j=0;j<core.sizes_vec(0);j++){
      selection = index_mat(i,0);
      input_vec(j) = data.vector_mat(j,selection);
    }
    selection = index_mat(i,1);
    output_vec = data.vectorize(selection);
    
    backpropagation(input_vec, output_vec, derivatives);
    for(j=0;j<core.num_layers-1;j++){
      for(k=0;k<core.sizes_vec(j+1);k++){
	core.bias_vecs[j](k) -= eta*derivatives.bias_vecs[j](k);
	for(l=0;l<core.sizes_vec(j);l++){
	  core.weights_mats[j](k,l) -= eta*derivatives.weights_mats[j](k,l);
	}
      }
    }
  }
}
void neuralnetwork::print(){
	int i, j, k,
		l1, l2;

	std::cout << "num_layers:" << std::setw(10) << core.num_layers << std::endl;
	std::cout << "size_vec:  ";
	for(i=0;i<core.num_layers;i++){
		std::cout << std::setw(10) << core.sizes_vec(i);
	}
	std::cout << std::endl << std::endl;
	std::cout << "bias vectors:" << std::endl;
	for(i=0;i<core.num_layers-1;i++){
		l1 = core.sizes_vec(i+1);
		for(j=0;j<l1;j++){
			std::cout << std::setw(10) << core.bias_vecs[i](j);
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;

	std::cout << "weights_mats:" << std::endl;
	for(i=0;i<core.num_layers-1;i++){
		l1 = core.sizes_vec(i);
		l2 = core.sizes_vec(i+1);
		std::cout << i << " --> " << i+1 << ":" << std::endl;
		for(j=0;j<l2;j++){
			for(k=0;k<l1;k++){
				std::cout << std::setw(10) << core.weights_mats[i](j,k);
			}
			std::cout << std::endl;
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
}

nn_data::nn_data(){
}
nn_data::nn_data(arma::mat vectors, arma::vec indices, int outcomes){
	int i, j;

	num_length = vectors.n_rows;
	num_data = vectors.n_cols;
	num_outcomes = outcomes;

	vector_mat = arma::zeros<arma::mat>(num_length,num_data);
	for(i=0;i<num_length;i++){
		for(j=0;j<num_data;j++){
			vector_mat(i,j) = vectors(i,j);
		}
	}
	index_mat = arma::zeros<arma::mat>(num_data,2);
	for(i=0;i<num_data;i++){
		index_mat(i,0) = i;
		index_mat(i,1) = indices(i);
	}		
}
nn_data::~nn_data(){
}
arma::vec nn_data::vectorize(int a){
	arma::vec vector_form = arma::zeros<arma::vec>(num_outcomes);
	vector_form(a) = 1.0;
	return vector_form;
}
void nn_data::print(){
	int i, j;
	for(i=0;i<num_length;i++){
		for(j=0;j<num_data;j++){
			std::cout << std::setw(15) << vector_mat(i,j);
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
	for(i=0;i<num_data;i++){
		std::cout << std::setw(15) << index_mat(i,1);
	}
	std::cout << std::endl;
}


inline double sigmoid(double x){
	return 1.0/(1.0+exp(-x));
}
arma::vec sigmoid_vec(arma::vec x){
	int i, N = x.n_elem;
	arma::vec value = arma::zeros<arma::vec>(N);
	for(i=0;i<N;i++){
		value(i) = sigmoid(x(i));
	}
	return value;
}
arma::vec sigmoid_derivative_vec(arma::vec x){
	int i, N = x.n_elem;
	arma::vec 	sigma = arma::zeros<arma::vec>(N),
				ones = sigma;
	for(i=0;i<N;i++){
		ones(i) = 1.0;
	}
	sigma = sigmoid_vec(x);
	return hadamard_product(sigma, ones-sigma);
}

arma::vec hadamard_product(arma::vec a, arma::vec b){
	int i, N;
	N = a.n_elem;
	arma::vec c = arma::zeros<arma::vec>(N);
	for(i=0;i<N;i++){
		c(i) = a(i)*b(i);
	}
	return c;
}







