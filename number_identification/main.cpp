#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include "lib/mnist.cpp"
#include "lib/neuralnetwork.cpp"
      

int main(int argc, char* argv[]){
  int number_of_images;
  int data_of_an_image;
  std::string pathway = "data/";
  std::string data_name, label_name;
  MNISTData train, test;

  number_of_images=60000;
  data_of_an_image=28*28;
  data_name = pathway+"train-images.idx3-ubyte";
  label_name = pathway+"train-labels.idx1-ubyte";
  train = ReadMNIST(data_name.c_str(),label_name.c_str(),number_of_images,data_of_an_image);

  number_of_images=10000;
  data_of_an_image=28*28;
  data_name = pathway+"t10k-images.idx3-ubyte";
  label_name = pathway+"t10k-labels.idx1-ubyte";
  test = ReadMNIST(data_name.c_str(),label_name.c_str(),number_of_images,data_of_an_image);

  PrintMNIST(test.data,test.labels,0,5);

  /*TESTING::********
  int number_of_layers=3;
  std::vector<int> layers(number_of_layers);
  layers[0] = 1;
  layers[1] = 1;
  layers[2] = 3;
  NeuralNetwork net(layers);
  for(int i=0;i<number_of_layers-1;i++){
    for(int j=0;j<layers[i+1];j++){
      net.biases[i][j] = i + j;
      for(int k=0;k<layers[i];k++){
	net.weights[i][j][k] = -j+k+1;
      }
    }
  }

  Print_Vector(net.layers);
  for(int i=0;i<number_of_layers-1;i++){
    printf("bias%d\n",i+1);
    Print_Vector(net.biases[i]);
    printf("weights%d\n",i+1);
    for(int j=0;j<layers[i+1];j++){
      Print_Vector(net.weights[i][j]);
    }
    printf("\n");
  }
  std::vector<double> feed(1),fed;
  feed[0]=3.0;
  fed = net.FeedForward(feed);
  Print_Vector(fed);
  */


  return 0;
}
