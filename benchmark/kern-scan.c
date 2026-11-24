// Single-thread C kernels for a like-for-like timing prototype (2026-11-23).
// native(): independent zero-padded crops, same algorithm as the aeon MiniRocket core.
// scan():   one pass over the recording; causal-context pooling; block-prefix counts (block=4).
#include <stdlib.h>
#include <string.h>
static void shifts(const float*x,int n,int d,float*Ca,float*Cg){ // Cg[j][t]=3*x[t+(j-4)d] (0 outside), Ca=-sum_j x[t+(j-4)d]
  memset(Ca,0,sizeof(float)*n);
  for(int j=0;j<9;j++){float*g=Cg+(size_t)j*n;int sh=(j-4)*d;int lo=sh<0?-sh:0,hi=sh>0?n-sh:n;if(hi<lo)hi=lo;
    for(int t=0;t<lo&&t<n;t++)g[t]=0;for(int t=hi;t<n;t++)g[t]=0;
    for(int t=lo;t<hi;t++){float v=x[t+sh];g[t]=3*v;Ca[t]-=v;}}
}
// starts m*h, m<M; requires h%4==0, W%4==0. Padded-type features pool centres [s,s+W-4d); valid-type [s+4d,s+W-4d).
// Output is written feature-major into a scratch buffer and transposed in cache-sized tiles (included in timing).
void scan(const float*x,int N,int M,int h,int W,const int*dil,const int*nfd,int nd,const float*bias,const int*idx,float*out,int F){
  int nb=N/4;float*Ca=malloc(sizeof(float)*N),*Cg=malloc(sizeof(float)*9*(size_t)N),*C=malloc(sizeof(float)*(N+4));
  unsigned char*c4=malloc(nb+1);int*pre=malloc(sizeof(int)*(nb+2));float*T=malloc(sizeof(float)*(size_t)M*F);int fs=0,hb=h/4;
  for(int di=0;di<nd;di++){int d=dil[di],p=4*d,nf=nfd[di];shifts(x,N,d,Ca,Cg);
    for(int k=0;k<84;k++){const float*a=Cg+(size_t)idx[3*k]*N,*b=Cg+(size_t)idx[3*k+1]*N,*c=Cg+(size_t)idx[3*k+2]*N;
      for(int t=0;t<N;t++)C[t]=Ca[t]+a[t]+b[t]+c[t];
      int off=((di+k)%2==0)?0:p;float inv=1.0f/(W-p-off);int o1=off/4,o2=(W-p)/4;
      for(int q=0;q<nf;q++){float bb=bias[fs+q];
        for(int i=0;i<nb;i++){const float*z=C+4*i;c4[i]=(unsigned char)((z[0]>bb)+(z[1]>bb)+(z[2]>bb)+(z[3]>bb));}
        pre[0]=0;for(int i=0;i<nb;i++)pre[i+1]=pre[i]+c4[i];
        float*o=T+(size_t)(fs+q)*M;const int*p1=pre+o1,*p2=pre+o2;
        for(int m=0;m<M;m++)o[m]=(p2[m*hb]-p1[m*hb])*inv;}
      fs+=nf;}}
  for(int f0=0;f0<F;f0+=64)for(int m0=0;m0<M;m0+=64){int f1=f0+64<F?f0+64:F,m1=m0+64<M?m0+64:M;
    for(int m=m0;m<m1;m++)for(int f=f0;f<f1;f++)out[(size_t)m*F+f]=T[(size_t)f*M+m];}
  free(Ca);free(Cg);free(C);free(c4);free(pre);free(T);
}
