#include "mnist.hpp"

int ReverseInt (int i)
{
  unsigned char ch1, ch2, ch3, ch4;
  ch1=i&255;
  ch2=(i>>8)&255;
  ch3=(i>>16)&255;
  ch4=(i>>24)&255;
  return((int)ch1<<24)+((int)ch2<<16)+((int)ch3<<8)+ch4;
}
void ReadMNISTData(const char* FileName, int NumberOfImages, int DataOfAnImage,std::vector<std::vector<double> > &arr)
{
  arr.resize(NumberOfImages,std::vector<double>(DataOfAnImage));
  std::ifstream file (FileName,std::ios::binary);
  if (file.is_open())
    {
      int magic_number=0;
      int number_of_images=0;
      int n_rows=0;
      int n_cols=0;
      file.read((char*)&magic_number,sizeof(magic_number));
      magic_number= ReverseInt(magic_number);
      file.read((char*)&number_of_images,sizeof(number_of_images));
      number_of_images= ReverseInt(number_of_images);
      file.read((char*)&n_rows,sizeof(n_rows));
      n_rows= ReverseInt(n_rows);
      file.read((char*)&n_cols,sizeof(n_cols));
      n_cols= ReverseInt(n_cols);
      for(int i=0;i<number_of_images;++i)
        {
	  for(int r=0;r<n_rows;++r)
            {
	      for(int c=0;c<n_cols;++c)
                {
		  unsigned char temp=0;
		  file.read((char*)&temp,sizeof(temp));
		  arr[i][(n_rows*r)+c]= (double)temp;
                }
            }
        }
    }
}
void ReadMNISTLabel(const char* FileName, int NumberOfImages,std::vector<double> &arr)
{
  arr.resize(NumberOfImages);
  std::ifstream file (FileName,std::ios::binary);
  if (file.is_open())
    {
      int magic_number=0;
      int number_of_images=0;
      file.read((char*)&magic_number,sizeof(magic_number));
      magic_number= ReverseInt(magic_number);
      file.read((char*)&number_of_images,sizeof(number_of_images));
      number_of_images= ReverseInt(number_of_images);
      for(int i=0;i<number_of_images;++i)
        {
	  unsigned char temp=0;
	  file.read((char*)&temp,sizeof(temp));
	  arr[i]= (double)temp;
	}
    }
}
MNISTData ReadMNIST(const char* DataName, const char* LabelName, int NumberOfImages, int DataOfAnImage)
{
  MNISTData mnist;
  mnist.data.resize(NumberOfImages,std::vector<double>(DataOfAnImage));
  mnist.labels.resize(NumberOfImages);
  std::ifstream datafile (DataName,std::ios::binary);
  if (datafile.is_open())
    {
      int magic_number=0;
      int number_of_images=0;
      int n_rows=0;
      int n_cols=0;
      datafile.read((char*)&magic_number,sizeof(magic_number));
      magic_number= ReverseInt(magic_number);
      datafile.read((char*)&number_of_images,sizeof(number_of_images));
      number_of_images= ReverseInt(number_of_images);
      datafile.read((char*)&n_rows,sizeof(n_rows));
      n_rows= ReverseInt(n_rows);
      datafile.read((char*)&n_cols,sizeof(n_cols));
      n_cols= ReverseInt(n_cols);
      for(int i=0;i<number_of_images;++i)
        {
	  for(int r=0;r<n_rows;++r)
            {
	      for(int c=0;c<n_cols;++c)
                {
		  unsigned char temp=0;
		  datafile.read((char*)&temp,sizeof(temp));
		  mnist.data[i][(n_rows*r)+c]= (double)temp;
                }
            }
        }
    }
  std::ifstream labelfile (LabelName,std::ios::binary);
  if (datafile.is_open())
    {
      int magic_number=0;
      int number_of_images=0;
      labelfile.read((char*)&magic_number,sizeof(magic_number));
      magic_number= ReverseInt(magic_number);
      labelfile.read((char*)&number_of_images,sizeof(number_of_images));
      number_of_images= ReverseInt(number_of_images);
      for(int i=0;i<number_of_images;++i)
        {
	  unsigned char temp=0;
	  labelfile.read((char*)&temp,sizeof(temp));
	  mnist.labels[i]= (double)temp;
	}
    }
  return mnist;
}

void PrintMNIST(std::vector<std::vector<double> > &data, std::vector<double> &label, int start, int finish)
{
  for(int i=start;i<finish;i++){
    printf("Label: %f\n",label[i]);
    for(int j=0;j<28;j++){
      for(int k=0;k<28;k++){
	int image=0;
	if(data[i][28*j+k]>128){image=1;}
	printf("%d",image);
      }
      printf("\n");
    }
    printf("\n");
  }
}
