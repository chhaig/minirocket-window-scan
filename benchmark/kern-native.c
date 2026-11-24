// Match aeon: centre first, then taps 0..3 and 5..8; preserve float32 addition order.
// Single-thread C kernels for a like-for-like timing prototype (2026-11-23).
// native(): independent zero-padded crops, same algorithm as the aeon MiniRocket core.
// scan():   one pass over the recording; causal-context pooling; block-prefix counts (block=4).
#include <stdlib.h>
#include <string.h>
static void shifts(const float*x,int n,int d,float*Ca,float*Cg){ // Cg[j][t]=3*x[t+(j-4)d] (0 outside), Ca=-sum_j x[t+(j-4)d]
  memset(Ca,0,sizeof(float)*n);
  for(int jj=0;jj<9;jj++){int j=(jj==0?4:(jj<=4?jj-1:jj));float*g=Cg+(size_t)j*n;int sh=(j-4)*d;int lo=sh<0?-sh:0,hi=sh>0?n-sh:n;if(hi<lo)hi=lo;
    for(int t=0;t<lo&&t<n;t++)g[t]=0;for(int t=hi;t<n;t++)g[t]=0;
    for(int t=lo;t<hi;t++){float v=x[t+sh];g[t]=3*v;Ca[t]-=v;}}
}
void native(const float*X,int M,int W,const int*dil,const int*nfd,int nd,const float*bias,const int*idx,float*out,int F){
  float*Ca=malloc(sizeof(float)*W),*Cg=malloc(sizeof(float)*9*W),*C=malloc(sizeof(float)*W);
  for(int i=0;i<M;i++){const float*x=X+(size_t)i*W;float*o=out+(size_t)i*F;int fs=0;
    for(int di=0;di<nd;di++){int d=dil[di],p=4*d,nf=nfd[di];shifts(x,W,d,Ca,Cg);
      for(int k=0;k<84;k++){const float*a=Cg+(size_t)idx[3*k]*W,*b=Cg+(size_t)idx[3*k+1]*W,*c=Cg+(size_t)idx[3*k+2]*W;
        for(int t=0;t<W;t++)C[t]=Ca[t]+a[t]+b[t]+c[t];
        int lo=((di+k)%2==0)?0:p,hi=((di+k)%2==0)?W:W-p;
        for(int q=0;q<nf;q++){float bb=bias[fs+q];int cnt=0;for(int t=lo;t<hi;t++)cnt+=C[t]>bb;o[fs+q]=(float)cnt/(hi-lo);}
        fs+=nf;}}}
  free(Ca);free(Cg);free(C);
}
