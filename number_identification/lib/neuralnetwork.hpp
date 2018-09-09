#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>

struct SNNCore{
  std::vector<double> *biases;
  std::vector<std::vector<double> > *weights;
};

class CNeuralNetwork{
public:
  std::vector<int> layers;
  SNNCore core;

  CNeuralNetwork();
  CNeuralNetwork(const std::vector<int> &Layers);
  ~CNeuralNetwork();

  std::vector<double> FeedForward(std::vector<double> Feed);
  std::vector<double> FeedForward(std::vector<double> Feed, int layer);
  std::vector<double> CostDerivative(std::vector<double> Activations, std::vector<double> Labels);
  SNNCore BackPropogation(MNISTData Mnist, int datum);
  void SGD(MNISTData Train, int Epochs, int MiniBatchSize, double Eta, MNISTData Test);
  
};  

inline double Sigmoid(double z);
std::vector<double> Sigmoid(std::vector<double> Z);
std::vector<double> SigmoidDerivative(std::vector<double> Z);
std::vector<double> HadamardProduct(const std::vector<double> *a, const std::vector<double> *b);
std::vector<double> LabelToVector(int Label);
template<class T>
void Print_Vector(const std::vector<T> &vector);

#endif