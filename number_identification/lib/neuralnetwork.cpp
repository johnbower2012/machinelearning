#include "neuralnetwork.hpp"

CNeuralNetwork::CNeuralNetwork(const std::vector<int> &Layers){
  this->layers = Layers;
  int number_of_layers = Layers.size();
  this->core.biases = new std::vector<double>[number_of_layers-1];
  this->core.weights = new std::vector<std::vector<double> >[number_of_layers-1];
  for(int i=0;i<number_of_layers-1;i++){
    this->core.weights[i] = std::vector<std::vector<double> >(Layers[i+1],std::vector<double>(Layers[i]));
    this->core.biases[i] = std::vector<double> (Layers[i+1]);
  }
}
CNeuralNetwork::~CNeuralNetwork(){
  delete[] core.weights;
  delete[] core.biases;
}
std::vector<double> CNeuralNetwork::FeedForward(std::vector<double> Feed){
  int number_of_layers = layers.size();
  std::vector<double> temp_next,temp_now=Feed;
  for(int layer=0;layer<number_of_layers-1;layer++){
    int nodes_next=layers[layer+1];
    temp_next.resize(nodes_next,0.0);
    for(int node_next=0;node_next<nodes_next;node_next++){
      int nodes_now=layers[layer];
      for(int node_now=0;node_now<nodes_now;node_now++){
	temp_next[node_next] = this->core.weights[layer][node_next][node_now]*temp_now[node_now];
      }
      temp_next[node_next] += this->core.biases[layer][node_next];
    }
    temp_now = Sigmoid(temp_next);
  }
  return temp_now;
}
std::vector<double> CNeuralNetwork::AdvanceLayer(std::vector<double> Feed, int Layer){
  std::vector<double> temp_next,temp_now=Feed;
  int nodes_next = this->layers[Layer+1];
  temp_next.resize(nodes_next,0.0);
  for(int node_next=0;node_next<nodes_next;node_next++){
    int nodes_now = this->layers[Layer];
    for(int node_now=0;node_now<nodes_now;node_now++){
      temp_next[node_next] = this->core.weights[Layer][node_next][node_now]*temp_now[node_now];
    }
    temp_next[node_next] += this->core.biases[Layer][node_next];
  }
  return temp_next;
}
std::vector<double> CNeuralNetwork::CostDerivative(std::vector<double> Activations, std::vector<double> Labels){
  std::vector<double> CD = Activations;
  for(int i=0;i<CD.size();i++){
    CD[i] -= Labels[i];
  }
  return CD;
}
SNNCore CNeuralNetwork::BackPropogation(MNISTData Mnist,int datum){
  int numberOfData = Mnist.labels.size(), nodes_next, nodes_now;
  std::vector<double> costder, fedforward, sigmaprime, label;
  std::vector<double> *z, *activations, *error;
  SNNCore delta;
  int numberOfLayers = this->layers.size();
  //dynamic memory allocation
  z = new std::vector<double>[numberOfLayers];
  activations = new std::vector<double>[numberOfLayers];
  error = new std::vector<double>[numberOfLayers-1];
  delta.biases = new std::vector<double>[numberOfLayers-1];
  delta.weights = new std::vector<std::vector<double> >[numberOfLayers-1];
  //assign memory && calculate activations
  z[0] = Mnist.data[datum];
  activations[0] = Mnist.data[datum];
  for(int layer=0;layer<numberOfLayers-1;layer++){
    z[layer+1] = AdvanceLayer(activations[layer],layer);
    activations[layer+1] = Sigmoid(z[layer+1]);
    Print_Vector(z[layer+1]);
    Print_Vector(activations[layer+1]);
    delta.biases[layer] = std::vector<double> (this->layers[layer+1]);
    delta.weights[layer] = std::vector<std::vector<double> > (this->layers[layer+1],std::vector<double>(this->layers[layer]));
  }
  //calculate error for final layer
  label = LabelToVector(Mnist.labels[datum]);
  costder = CostDerivative(activations[numberOfLayers-1],label);
  sigmaprime = SigmoidDerivative(z[numberOfLayers-1]);
  Print_Vector(sigmaprime);
  error[numberOfLayers-2] = HadamardProduct(costder,sigmaprime);
  printf("error%d:",numberOfLayers-2);
  Print_Vector(error[numberOfLayers-2]);
  //calculate error for each layer, final-1, final-2,...,2
  //the indices are displaced by one counter for weights & error relative to z&a since w&e run from layer 2-L while z&a run from 1-L
  for(int layer=numberOfLayers-3;layer>-1;layer--){
    sigmaprime = SigmoidDerivative(z[layer+1]);
    nodes_now = this->layers[layer+1];
    nodes_next = this->layers[layer+2];
    costder.resize(nodes_now);
    for(int node_now=0;node_now<nodes_now;node_now++){
      costder[node_now] = 0.0;
    }
    Print_Vector(costder);
    for(int node_now=0;node_now<nodes_now;node_now++){
      for(int node_next=0;node_next<nodes_next;node_next++){
	printf("%f ",core.weights[layer+1][node_next][node_now]);
	costder[node_now] += this->core.weights[layer+1][node_next][node_now]*error[layer+1][node_next];
      }
      printf("\n");
    }
    printf("costder%d:",layer);
    Print_Vector(costder);
    error[layer] = HadamardProduct(costder,sigmaprime);
    printf("error%d:",layer);
    Print_Vector(error[layer]);
  }
  //calculate gradient of cost function
  for(int layer=1;layer<numberOfLayers;layer++){
    delta.biases[layer-1] = error[layer-1];
    nodes_now = this->layers[layer-1];
    nodes_next = this->layers[layer];
    for(int node_now=0;node_now<nodes_now;node_now++){
      for(int node_next=0;node_next<nodes_next;node_next++){
	delta.weights[layer-1][node_next][node_now] = activations[layer][node_now]*error[layer-1][node_next];
      }
    }
  }
  //delete dynamic memory
  delete[] activations;
  delete[] error;

  //return gradient of cost function
  return delta;
}
void CNeuralNetwork::SGD(MNISTData Train, int Epochs, int MiniBatchSize, double Eta, MNISTData Test){
  
}

/***********
   SIGMOID FUNCTIONS
 ***********/

inline double Sigmoid(double z){
  return 1.0/(1.0+exp(-z));
}
std::vector<double> Sigmoid(std::vector<double> Z){
  int size = Z.size();
  std::vector<double> sigmoid(size);
  for(int i=0;i<size;i++){
    sigmoid[i] = 1.0/(1.0+exp(-Z[i]));
  }
  return sigmoid;
}
std::vector<double> SigmoidDerivative(std::vector<double> Z){
  int size = Z.size();
  double expon,expon1;
  std::vector<double> sigmoidder(size);
  for(int i=0;i<size;i++){
    expon = exp(-Z[i]);
    expon1 = expon+1;
    expon1 *= expon1;
    sigmoidder[i] = expon/expon1;
  }
  return sigmoidder;
}
std::vector<double> HadamardProduct(const std::vector<double> &a, const std::vector<double> &b){
  int size = a.size();
  std::vector<double> c(size);
  for(int i=0;i<size;i++){
    c[i] = (a[i])*(b[i]);
  }
  return c;
}
std::vector<double> LabelToVector(int Label){
  std::vector<double> vector(10,0.0);
  vector[Label] = 1.0;
  return vector;
}
template<class T>
void Print_Vector(const std::vector<T> &vector){
  int size = vector.size();
  for(int i=0;i<size;i++){
    printf("%f ",(double)vector[i]);
  }
  printf("\n");
}
