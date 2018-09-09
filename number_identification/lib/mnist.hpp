#ifndef MNIST_H
#define MNIST_H

#include<iomanip>
#include<vector>
#include<fstream>

struct MNISTData{
  std::vector<std::vector<double> > data;
  std::vector<double> labels;
};

int ReverseInt (int i);
void ReadMNISTData(const char* FileName, int NumberOfImages, int DataOfAnImage,std::vector<std::vector<double> > &arr);
void ReadMNISTLabel(const char* FileName, int NumberOfImages,std::vector<double> &arr);
MNISTData ReadMNIST(const char* DataName, const char* LabelName, int NumberOfImages, int DataOfAnImage);
void PrintMNIST(std::vector<std::vector<double> > &data, std::vector<double> &label, int start, int end);

#endif
