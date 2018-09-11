#include "neuralnetwork.hpp"

CNeuralNetwork::CNeuralNetwork(const std::vector<int> &Layers){
  this->layers = Layers;
  int numberOfLayers = Layers.size();
  this->core.biases.resize(numberOfLayers-1);
  this->core.weights.resize(numberOfLayers-1);
  for(int i=0;i<numberOfLayers-1;i++){
    this->core.biases[i] = std::vector<double> (Layers[i+1]);
    this->core.weights[i] = std::vector<std::vector<double> >(Layers[i+1],std::vector<double>(Layers[i]));
  }
}
CNeuralNetwork::~CNeuralNetwork(){
}
void CNeuralNetwork::Randomize(unsigned Seed, double Mean, double STDDev){
  std::default_random_engine generator(Seed);
  std::normal_distribution<double> normal(Mean,STDDev);

  int numberOfLayers = this->layers.size();
  int nodes_now,nodes_next;
  for(int layer=0;layer<numberOfLayers-1;layer++){
    this->core.weights[layer] = std::vector<std::vector<double> >(this->layers[layer+1],std::vector<double>(this->layers[layer]));
    this->core.biases[layer] = std::vector<double> (this->layers[layer+1]);
    nodes_now = this->layers[layer];
    nodes_next = this->layers[layer+1];
    for(int node_next=0;node_next<nodes_next;node_next++){
      this->core.biases[layer][node_next] = normal(generator);
      for(int node_now=0;node_now<nodes_now;node_now++){
	this->core.weights[layer][node_next][node_now] = normal(generator);
      }
    }
  }

}
std::vector<double> CNeuralNetwork::FeedForward(std::vector<double> Feed){
  int number_of_layers = this->layers.size();
  std::vector<double> temp_next,temp_now=Feed;
  for(int layer=0;layer<number_of_layers-1;layer++){
    int nodes_now=layers[layer];
    int nodes_next=layers[layer+1];
    temp_next.resize(nodes_next,0.0);
    for(int node_next=0;node_next<nodes_next;node_next++){
      for(int node_now=0;node_now<nodes_now;node_now++){
	temp_next[node_next] += this->core.weights[layer][node_next][node_now]*temp_now[node_now];
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
  std::vector<std::vector<double> > z, activations, error;
  SNNCore delta;
  int numberOfLayers = this->layers.size();
  //dynamic memory allocation
  z.resize(numberOfLayers);
  activations.resize(numberOfLayers);
  error.resize(numberOfLayers-1);
  delta.biases.resize(numberOfLayers-1);
  delta.weights.resize(numberOfLayers-1);
  //assign memory && calculate activations
  z[0] = Mnist.data[datum];
  activations[0] = Mnist.data[datum];
  for(int layer=0;layer<numberOfLayers-1;layer++){
    z[layer+1] = AdvanceLayer(activations[layer],layer);
    activations[layer+1] = Sigmoid(z[layer+1]);
    delta.biases[layer] = std::vector<double> (this->layers[layer+1]);
    delta.weights[layer] = std::vector<std::vector<double> > (this->layers[layer+1],std::vector<double>(this->layers[layer]));
  }
  //calculate error for final layer
  label = LabelToVector(Mnist.labels[datum]);
  costder = CostDerivative(activations[numberOfLayers-1],label);
  sigmaprime = SigmoidDerivative(z[numberOfLayers-1]);
  error[numberOfLayers-2] = HadamardProduct(costder,sigmaprime);
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
    for(int node_now=0;node_now<nodes_now;node_now++){
      for(int node_next=0;node_next<nodes_next;node_next++){
	costder[node_now] += this->core.weights[layer+1][node_next][node_now]*error[layer+1][node_next];
      }
    }
    error[layer] = HadamardProduct(costder,sigmaprime);
  }
  //calculate gradient of cost function
  //again, note displacement of layer index between w&e and a
  for(int layer=0;layer<numberOfLayers-1;layer++){
    nodes_now = this->layers[layer];
    nodes_next = this->layers[layer+1];
    for(int node_next=0;node_next<nodes_next;node_next++){
      delta.biases[layer][node_next] = error[layer][node_next];
      for(int node_now=0;node_now<nodes_now;node_now++){
	delta.weights[layer][node_next][node_now] = activations[layer][node_now]*error[layer][node_next];
      }
    }
  }

  //return gradient of cost function
  return delta;
}
void CNeuralNetwork::UpdateMiniBatch(MNISTData MiniBatch, double LearningRate){
  int minibatchsize = MiniBatch.labels.size();
  int numberOfLayers = this->layers.size();
  int nodes_next, nodes_now;
  SNNCore delta,temp;
  delta.biases.resize(numberOfLayers-1);
  delta.weights.resize(numberOfLayers-1);
  for(int i=0;i<numberOfLayers-1;i++){
    delta.weights[i] = std::vector<std::vector<double> >(this->layers[i+1],std::vector<double>(this->layers[i]));
    delta.biases[i] = std::vector<double> (this->layers[i+1]);
    for(int j=0;j<this->layers[i+1];j++){
      delta.biases[i][j] = 0.0;
      for(int k=0;k<this->layers[i];k++){
	delta.weights[i][j][k] = 0.0;
      }
    }
  }
  for(int mini=0;mini<minibatchsize;mini++){
    temp = this->BackPropogation(MiniBatch,mini);
    for(int layer=0;layer<numberOfLayers-1;layer++){
      nodes_now = this->layers[layer];
      nodes_next = this->layers[layer+1];
      for(int node_next=0;node_next<nodes_next;node_next++){
	delta.biases[layer][node_next] += temp.biases[layer][node_next];
	for(int node_now=0;node_now<nodes_now;node_now++){
	  delta.weights[layer][node_next][node_now] += temp.weights[layer][node_next][node_now];
	}
      }
    }
  }
  for(int layer=0;layer<numberOfLayers-1;layer++){
    nodes_now = this->layers[layer];
    nodes_next = this->layers[layer+1];
    for(int node_next=0;node_next<nodes_next;node_next++){
      this->core.biases[layer][node_next] -= LearningRate*temp.biases[layer][node_next];
      for(int node_now=0;node_now<nodes_now;node_now++){
	this->core.weights[layer][node_next][node_now] -= LearningRate*delta.weights[layer][node_next][node_now];
      }
    }
  }
}
void CNeuralNetwork::SGD(MNISTData Train, int Epochs, int MiniBatchSize, double Eta, MNISTData Test){
  int tests=Test.labels.size();
  int trains=Train.labels.size();
  int datalength=Train.data[0].size();
  int minibatches=trains/MiniBatchSize;
  int percent=0;
  std::clock_t start, end;
  MNISTData minibatch;
  int ratio;
  minibatch.data.resize(MiniBatchSize,std::vector<double>(datalength));
  minibatch.labels.resize(MiniBatchSize);
  std::vector<int>random;
  for(int i=0;i<trains;i++){
    random.push_back(i);
  }
  //run SGD for epoch in Epochs
  for(int epoch=0;epoch<Epochs;epoch++){
    //initial test
    if(tests>0){
      ratio=this->Evaluate(Test);
      printf("Epoch%d: %d / %d\n",epoch,ratio,tests);
    }
    //randomize minibatch order
    std::random_shuffle(random.begin(),random.end());
    //run minibatches
    start = clock();
    percent=0;
    for(int mb=0;mb<minibatches;mb++){
      for(int mbi=0;mbi<MiniBatchSize;mbi++){
	minibatch.data[mbi] = Train.data[mb*MiniBatchSize+mbi];
	minibatch.labels[mbi] = Train.labels[mb*MiniBatchSize+mbi];
      }
      this->UpdateMiniBatch(minibatch,Eta);
      if(mb%(minibatches/10)==0){
	printf("     %d%%\n",percent);
	percent+=10;
      }
    }
    end=clock();
    printf("Time: %f\n",(end-start)/(double)CLOCKS_PER_SEC);
  }
  //final test;
  if(tests>0){
    ratio=this->Evaluate(Test);
    printf("Epoch%d: %d / %d \n",Epochs,ratio,tests);
  }

}
int CNeuralNetwork::Evaluate(MNISTData Test){
  std::vector<double> output=this->FeedForward(Test.data[0]);
  int tests=Test.labels.size();
  int outs=output.size();
  int index,ratio=0;
  double max;

  for(int test=0;test<tests;test++){
    output = this->FeedForward(Test.data[test]);
    max=0; index=0;
    for(int out=0;out<outs;out++){
      if(max<output[out]){
	max=output[out];
	index=out;
      }
    }
    if(index==Test.labels[test]){
      ratio++;
    }
   }
  return ratio;
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
