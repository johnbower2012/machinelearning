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
  std::vector<int> layers(3);

  layers[0] = 784; layers[1] = 30; layers[2] = 10;
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
  /*
  //PrintMNIST(test.data,test.labels,0,5);
  CNeuralNetwork net(layers);
  unsigned int seed=1;
  double mean=0.0,stddev=1.0;
  net.Randomize(seed,mean,stddev);
  int epochs=30;
  int mbs=10;
  double eta=0;
  net.SGD(train,epochs,mbs,eta,test);
  */

  //TESTING::********
  int number_of_layers=3;
  layers[0] = 10;
  layers[1] = 5;
  layers[2] = 5;
  CNeuralNetwork net(layers);
  net.Randomize(1,0,1);
  MNISTData core;
  core.data.resize(10000,std::vector<double>(5));
  core.labels.resize(10000);
  for(int i=0;i<10000;i++){
    for(int j=0;j<10;j++){
      core.data[i][j] = (i+j/2)%5;
    }
    core.labels[i] = (i%5);
  }
  net.SGD(core,5,10,0.1,core);

  /*  
  Print_Vector(net.layers);
  for(int i=0;i<number_of_layers-1;i++){
    printf("bias%d\n",i+1);
    Print_Vector(net.core.biases[i]);
    printf("weights%d\n",i+1);
    for(int j=0;j<layers[i+1];j++){
      Print_Vector(net.core.weights[i][j]);
    }
    printf("\n");
  }
  std::vector<double> feed(1),fed;
  feed[0]=3.0;
  fed = net.FeedForward(feed);
  printf("output:\n");
  Print_Vector(fed);
  feed[0]=9.0;
  fed = net.FeedForward(feed);
  printf("output:\n");
  Print_Vector(fed);
  */

  return 0;
}

