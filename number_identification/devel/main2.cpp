#include<iostream>
#include<iomanip>
#include<cstdlib>
#include<cmath>
#include<random>
#include<chrono>
#include "armadillo"
#include "devel/neuralnetwork.cpp"
#include "read_mnist.cpp"

arma::vec dectobin(int a, int digits);
int winlossdraw(arma::vec binary, int digits);
int outtonumber(arma::vec out);

int main(int argc, char* argv[]){
  int i, j, num_layers=3,
    epochs=1, mini_batch_size=10,
    edge=28, data_of_an_image=edge*edge,
    outcomes=10;
  double eta=3.0;
  int number_of_train_images=60000,
    number_of_test_images=10000;

	
 arma::vec sizes = arma::zeros<arma::vec>(num_layers),
   indices = arma::zeros<arma::vec>(number_of_train_images),
   out = arma::zeros<arma::vec>(outcomes),
   random = arma::zeros<arma::vec>(mini_batch_size),
   test = arma::zeros<arma::vec>(data_of_an_image);
 arma::mat data = arma::zeros<arma::mat>(number_of_train_images,data_of_an_image);


 vector<vector <double> > data_10k;
 vector<vector <double> > data_60k;
 vector<double> label_10k;
 vector<double> label_60k;
 string pathway = "data/";

 string data_60k_name = pathway+"train-images.idx3-ubyte";
 string label_60k_name = pathway+"train-labels.idx1-ubyte";
 ReadMNISTData(data_60k_name.c_str(),number_of_train_images,data_of_an_image,data_60k);
 ReadMNISTLabel(label_60k_name.c_str(),number_of_train_images,label_60k);

 string data_10k_name = pathway+"t10k-images.idx3-ubyte";
 string label_10k_name = pathway+"t10k-labels.idx1-ubyte";
 ReadMNISTData(data_10k_name.c_str(),number_of_test_images,data_of_an_image,data_10k);
 ReadMNISTLabel(label_10k_name.c_str(),number_of_test_images,label_10k);

 double ratio;

 sizes(0) = data_of_an_image;
 sizes(1) = 30;
 sizes(2) = outcomes;

 for(i=0;i<number_of_train_images;i++){
   indices(i) = label_60k[i];
   for(j=0;j<sizes(0);j++){
     data(i,j) = data_60k[i][j];
   }
 }

 neuralnetwork mnist(sizes,data,indices,outcomes);

 mnist.stochastic_gradient_descent(epochs,eta,mini_batch_size);
	
 ratio=0.0;
 indices.set_size(number_of_test_images);
 data.set_size(number_of_test_images,data_of_an_image);
 for(i=0;i<number_of_test_images;i++){
   indices(i) = label_10k[i];
   for(j=0;j<sizes(0);j++){
     data(i,j) = data_10k[i][j];
   }
 }

 for(i=0;i<number_of_test_images-9990;i++){
   test = data.row(i).t();
   out = mnist.feedforward(test);
   out.print("------");
   if((label_10k[i]-outtonumber(out))==0){
     ratio++;
   }
 }
 ratio /= (double) mini_batch_size;

 std::cout << ratio << std::endl;
 return 0;
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

