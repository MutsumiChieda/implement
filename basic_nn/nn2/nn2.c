#include <stdio.h>
#include <math.h>

typedef struct newlon{
  double threshold;
  double weight[];
  newlon *next[];
}newlon;

double func(float x){
  return 1/(1+exp(-4.0*x));
}

int main(void){
  
  newlon n[5];
  double teacher[4];
  teacher = {0,1,1,0};
  
  return 0;
}
