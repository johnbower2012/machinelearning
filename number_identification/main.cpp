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
  //std::vector<int> layers(3);
  //layers[0] = 784; layers[1] = 30; layers[2] = 10;
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
  /*
  CNeuralNetwork net(layers);
  SNNCore delta = net.BackPropogation(test,0);
  */
  //*TESTING::********
  int number_of_layers=3;
  std::vector<int> layers(number_of_layers);
  layers[0] = 1;
  layers[1] = 2;
  layers[2] = 3;
  CNeuralNetwork net(layers);
  for(int i=0;i<number_of_layers-1;i++){
    for(int j=0;j<layers[i+1];j++){
      net.core.biases[i][j] = i + j;
      for(int k=0;k<layers[i];k++){
	net.core.weights[i][j][k] = -j+k+1;
      }
    }
  }

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
  feed[0]=3.0;
  for(int i=0;i<number_of_layers-1;i++){
    printf("AL%d:\n",i);
    fed = net.AdvanceLayer(feed,i);
    Print_Vector(fed);
    feed = Sigmoid(fed);
    printf("SM%d:\n",i);
    Print_Vector(feed);      
  }
  
  MNISTData net_feed;
  net_feed.data = std::vector<std::vector<double> > (5,std::vector<double>(1));
  net_feed.labels = std::vector<double> (5);
  for(int i=0;i<5;i++){
    for(int j=0;j<1;j++){
      net_feed.data[i][j] = 3.0;
    }
    net_feed.labels[i] = 1;
  }
  net.BackPropogation(net_feed,0);
  
  //  */


  return 0;
}

