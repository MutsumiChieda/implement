#include <stdio.h>
#include <math.h>

double func(float x){
  return 1/(1+exp(-4.0*x));
}

int main(void){

  double x1, x2;
  double a, b, c;
  double Wba,Wca,Wdb,Wdc,Web,Wec;
  double tha,thb,thc;
  
  Wba = 0.4;
  Wca = 0.3;
  Wdb = 0.5;
  Wdc = -0.3;
  Web = 0.1;
  Wec = 0.6;
  
  tha = -0.5;
  thb = 0.4;
  thc = -0.2;
  
  a=b=c=0.0;
  printf("x1:"); scanf("%lf",&x1);
  printf("x2:"); scanf("%lf",&x2);

  b = func(Wdb*x1 + Web*x2 + thb);
  c = func(Wdc*x1 + Wec*x2 + thc);
  a = func(Wba*b + Wca*c + tha);
  
  printf("b: %1.2lf\n", b);
  printf("c: %1.2lf\n", c);
  printf("a: %1.2lf\n", a);
  return 0;
}
