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
  
  Wab=  2.743674;
  Wac= -2.778689;
  Wbd= -1.310456;
  Wbe= -1.312891;
  Wcd= -1.691827;
  Wce= -1.704581;

  tha = -1.312300;
  thb = 1.956707;
  thc = 0.710779;
  
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
