#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define EPS 4.0
#define ETA 0.1
#define TIMES 1000
#define INIT_WEIGHT 0.3

double randNum(void)
{
  return ((double)rand()/RAND_MAX-0.5)*2.0*INIT_WEIGHT;
}

double func(double x){
  return 1/(1+exp(-1*EPS*x));
}

int main(void){
  srand((unsigned)time(NULL));
  double data[4][3] = {
    {0.0, 0.0, 0.0},
    {0.0, 1.0, 1.0},
    {1.0, 0.0, 1.0},
    {1.0, 1.0, 0.0}
  };
  double wbd, wbe, wcd, wce, wab, wac;
  double offb, offc, offa;
  double outd, oute, outb, outc, outa;
  double xb, xc, xa;
  double deltab, deltac, deltaa;
  int r;
  double error;
  double errorSum;
  int times;
  int seed;
  FILE *fp;

  fp = fopen("error.dat", "w");
  if (fp==NULL) {
    printf("can't open file.\n");
    exit(1);
  }
  
  seed = 0;
  srand(seed);
  
  wbd = randNum();
  wbe = randNum();
  wcd = randNum();
  wce = randNum();
  wab = randNum();
  wac = randNum();
  offb = randNum();
  offc = randNum();
  offa = randNum();
  
  for(times=0;times<TIMES; times++) {
    errorSum = 0.0;
    for(r=0; r<4; r++) {
      //Feed Forward
      
      // Input layer output
      outd = data[r][0];
      oute = data[r][1];
      
      // Hidden layer output
      xb = wbd*outd + wbe*oute + offb;
      outb = func(xb);

      xc = wcd*outd + wce*oute + offc;
      outc = func(xc);

      // Output layer output
      xa = wab*outb + wac*outc + offa;
      outa = func(xa);

      if(times==TIMES-1) {
        printf("[%d]=%.10f, (%f)\n", r, outa, data[r][2]);
      }

      //Back Propagation
      error = ((outa-data[r][2])*(outa-data[r][2]));
      errorSum += error;
      
      //Update Wab
      deltaa = (outa-data[r][2])*EPS*(1-outa)*outa;
      deltab = deltaa*wab*EPS*(1.0-outb)*outb;
      deltac = deltaa*wac*EPS*(1.0-outc)*outc;

      offa -= ETA*deltaa*1.0;
      offb -= ETA*deltab*1.0;
      offc -= ETA*deltac*1.0;

      wab -= ETA*deltaa*outb;
      wac -= ETA*deltaa*outc;
      wbd -= ETA*deltab*outd;
      wce -= ETA*deltac*oute;
      wbe -= ETA*deltab*oute;
      wcd -= ETA*deltac*outd;
    }
    printf("errorSum = %f\n", errorSum/4.0);
    fprintf(fp, "%f\n", errorSum/4.0);

  }

  fclose(fp);

  return 0;
}
