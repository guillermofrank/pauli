#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<math.h>
#include<time.h>

// simultation model

#define N         8000     // cantidad de particulas (64)
#define M         938.0    // masa de las particulas (938)
#define RHO       0.16     // densidad media en la celda (0.05)
#define TEMP      0.1      // temperatura inicial del sistema (4.0)
#define XPROTON   0.5      // fraction of protons (protons/total)

#define EMPTY     -1       // empty value -1 (no not change)
#define RCELL     5.5      // lado de las celdas cubicas (para listas de celdas ~18.5)
#define PANTAB    5000     // table length for Pandharipande
#define PAUTAB    5000     // table length for Pauli

// Pandharipande Medium model

#define VR        3088.118 
#define VA        2666.647
#define V0        373.118
#define MUR       1.7468
#define MUA       1.6000
#define MU0       1.5000

// Pauli model (Maruyama)

#define P0        120.0    // potential parameter (120)
#define Q0        1.644    // potential parameter (1.644)
#define D         207.0    // potential parameter (207)

// initial and cut-off distances

#define RI        0.001
#define PI        0.000
#define RCPAN     5.4      // cut-off for Pandharipande potential
#define RCRPAU    5.4      // cut-off for Pauli potential (~3*Q0)
#define RCPPAU    394.0    // cut-off for Pauli potential (~3*P0)

// simultation parameters

#define SEED      260572   // semilla inicial para rand()
#define SMAX      100      // cantidad de pasos de simulacion
#define SWRT      100      // pasos entre configuraciones guardadas
#define STER      100      // pasos entre magnitudes termodinamicas
#define TERM      100      // pasos de termalizacion
#define TEND      0.0      // temperatura final
#define TOLT      0.1      // tolerancia de temp para termalizar
#define TCTRL     0        // pasos de control de delta (0 = no controlar)
#define DELTAX    0.2      // tamaño del paso para x
#define DELTAP    0.2      // tamaño del paso para p
#define DELTAM    0.001    // tamaño de paso minimo
#define TOL       0.75     // tolerancia para la fraccion de delta_x y delta_p 
#define RATIO     0.1      // fraccion de delta_x y delta_p 

double  table_np[4*PANTAB],table_nn[4*PANTAB],table_pp[4*PANTAB],table_pauli[7*PAUTAB];

void    help();
double  randnum(int *semilla);
double  inicial(double *x,double *v,double *p,double *f,int *ptype,double rho,double temp,double xproton,int ndim);
int     read_atoms(char *myfile);
double  read_dim(char *myfile);
double  read_data(char *myfile,double *x,double *v,double *p,double *f,int *ptype,int atoms);
int     write_header(FILE *fp,double dim,int atoms,int timestep);
int     write_data(FILE *fp,double *x,double *v,double *p,double *f,int *ptype,int atoms);

int     buildlist(double *x,int *head,int *lscl,double rcell,int n,int nc);
int     computelist(double *x,double *v,double *p,double *f,double *e,int *head,int *lscl,int *ptype,double dim,int n,int nc);
void    build_pandha_table(double ri,double rf,int ntable);
void    build_pauli_table(double ri,double rf,double pi,double pf,int ntable);
void    get_table_pandha(double *data,int *ptype,double r2,int ii,int jj,int ntable);
void    get_table_pauli(double *data,int *ptype,double r2,double p2,char flag,int ii,int jj,int ntable);
int     tfmc(double *x,double *v,double *p,double *f,double delta_x,double delta_p,double tset,double dim,int n);
void    thermo(double *data,double *v,double *p,double *k,double *e,int n);
double  normaldist();
double  checkperformance1(double temp,double tend,double dim,int tmax,int nc,int n);

double  metropolis(double *x,double *p,double *pacc,int *head,int *lscl,int *ptype,double temp,double rcell,double delta_x,double delta_p,double dim,int n,int nc);

int main(int argc, char *argv[])
{
  char    option[20],myfile[40];
  int     i,n,ndim,seed,t,term,tmax,thsamp,tsamp,tctrl;
  int     nc,*head,*lscl,*ptype;
  double  rho,dim,rcell,temp,teff,tend,xproton;
  double  ekin,epair,etot,delta_x,delta_p,ratio;
  double  *x,*v,*p,*f,*k,*e,*data,pacc[2];
  clock_t c0,c1;
  //FILE    *fp1;

  c0=clock();
  
  n = N;
  rho = RHO;
  temp = TEMP;
  tend = TEND;
  seed = SEED; 
  term = TERM;
  tmax = SMAX;
  tsamp = SWRT;
  thsamp = STER;
  tctrl  = TCTRL;
  rcell= RCELL;
  xproton = XPROTON;
  ratio = RATIO;
  delta_x = DELTAX;
  delta_p = DELTAP;
  myfile[0]='\0';

  if (argc==2)
    {
      strcpy(option,argv[1]);
      if (!strcmp(option,"-h")) help();
      else if (!strcmp(option,"-help")) help();
           else
             {
               printf("\nsomething is wrong... not enough parameters!\n\n");
               exit(0);
             }
    }
  else
    { 
      for(i=1;i<argc;i=i+2)
        {
          strcpy(option,argv[i]);
          if (!strcmp(option,"-n") & (i+1<argc)) sscanf(argv[i+1],"%d",&n);
          if (!strcmp(option,"-x") & (i+1<argc)) sscanf(argv[i+1],"%lf",&xproton);
          if (!strcmp(option,"-rho") & (i+1<argc)) sscanf(argv[i+1],"%lf",&rho);
          if (!strcmp(option,"-temp") & (i+1<argc)) sscanf(argv[i+1],"%lf",&temp);
          if (!strcmp(option,"-tend") & (i+1<argc)) sscanf(argv[i+1],"%lf",&tend);
          if (!strcmp(option,"-delx") & (i+1<argc)) sscanf(argv[i+1],"%lf",&delta_x);
          if (!strcmp(option,"-delp") & (i+1<argc)) sscanf(argv[i+1],"%lf",&delta_p);
          if (!strcmp(option,"-term") & (i+1<argc)) sscanf(argv[i+1],"%d",&term);
          if (!strcmp(option,"-steps") & (i+1<argc)) sscanf(argv[i+1],"%d",&tmax);
          if (!strcmp(option,"-thsamp") & (i+1<argc)) sscanf(argv[i+1],"%d",&thsamp);
          if (!strcmp(option,"-tctrl") & (i+1<argc)) sscanf(argv[i+1],"%d",&tctrl);
          if (!strcmp(option,"-ctrl") & (i+1<argc)) sscanf(argv[i+1],"%lf",&ratio);
          if (!strcmp(option,"-tsamp") & (i+1<argc)) sscanf(argv[i+1],"%d",&tsamp);
          if (!strcmp(option,"-seed") & (i+1<argc)) sscanf(argv[i+1],"%d",&seed);
          if (!strcmp(option,"-initial") & (i+1<argc)) strcpy(myfile,argv[i+1]);
        }
    }

  srand(seed);

  if(myfile[0]!='\0')
    {
      n=read_atoms(myfile); printf("\nn = %d\n",n);
      dim=read_dim(myfile); printf("\ndim = %f\n",dim);
    
      if((double)n*dim) 
        {
          rho = (double)n/(dim*dim*dim);

          ptype = (int *)malloc(n*sizeof(int));
          data = (double *)malloc(10*sizeof(double));

          k = (double *)malloc(n*sizeof(double));
          e = (double *)malloc(n*sizeof(double));
          x = (double *)malloc(3*n*sizeof(double));
          v = (double *)malloc(3*n*sizeof(double));
          p = (double *)malloc(3*n*sizeof(double));
          f = (double *)malloc(3*n*sizeof(double));

          teff = read_data(myfile,x,v,p,f,ptype,n);
          term = 0;
          delta_x = DELTAM;
          delta_p = DELTAM;
        }
      else exit(0);
    }
  else
    {
      ndim=(int)round(cbrt(n));
      n=ndim*ndim*ndim;    //warning: this is total number of atoms

      ptype = (int *)malloc(n*sizeof(int));
      data = (double *)malloc(10*sizeof(double));  // 10 thermodynamical magnitudes (can be upscaled)

      k = (double *)malloc(n*sizeof(double));
      e = (double *)malloc(n*sizeof(double));
      x = (double *)malloc(3*n*sizeof(double));
      v = (double *)malloc(3*n*sizeof(double));
      p = (double *)malloc(3*n*sizeof(double));
      f = (double *)malloc(3*n*sizeof(double));

      dim=inicial(x,v,p,f,ptype,rho,temp,xproton,ndim); 
    }

  *(pacc+0) = 0.0;
  *(pacc+1) = 0.0;

  if (tend==0.0) tend = temp;

  if (!(nc=(int)floor(dim/RCELL))) nc++;

  rcell=checkperformance1(temp,tend,dim,tmax,nc,n);

  printf("\nbuilding lists\t\t\t[OK]\n");

  build_pandha_table(RI,RCPAN,PANTAB);
  build_pauli_table(RI,RCRPAU,PI,RCPPAU,PAUTAB);

  printf("building tables\t\t\t[OK]\n\n");


  head   = (int *)malloc(nc*nc*nc*sizeof(int));
  lscl   = (int *)malloc(n*sizeof(int));

  printf("step\ttemp_set\ttemp_eff\te_kinetic\te_potential\te_total \tpacc(x)\t\tpacc(p)\n\n");

  buildlist(x,head,lscl,rcell,n,nc);
  computelist(x,v,p,f,e,head,lscl,ptype,dim,n,nc);

  thermo(data,v,p,k,e,n);

  teff  = *(data+0);
  ekin  = *(data+1);
  epair = *(data+2);
  etot  = *(data+3);

  printf("%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n",t,temp,teff,ekin,epair,etot,*(pacc+0),*(pacc+1));
 
  t  = 1;
  tmax++;

  while (t<tmax)
    {
      *(pacc+0) = 0.0;
      *(pacc+1) = 0.0;

      for (i=0;i<n;i++) 
        {
          etot += metropolis(x,p,pacc,head,lscl,ptype,temp,rcell,delta_x,delta_p,dim,n,nc);
        }

      buildlist(x,head,lscl,rcell,n,nc);  

      if (tctrl>0 && t%tctrl==0) 
        {
          if (*(pacc+0) < 0.40) delta_x *= 1.0-ratio;
          if (*(pacc+0) > 0.60) delta_x *= 1.0+ratio;
          if (*(pacc+1) < 0.40) delta_p *= 1.0-ratio;
          if (*(pacc+1) > 0.60) delta_p *= 1.0+ratio; 
        }

      if(thsamp>0 && t%thsamp==0) 
        {
          computelist(x,v,p,f,e,head,lscl,ptype,dim,n,nc);

          thermo(data,v,p,k,e,n);
 
          teff  = *(data+0);
          ekin  = *(data+1);
          epair = *(data+2);
          //etot  = *(data+3);

          printf("%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n",t,temp,teff,ekin,epair,ekin+epair,*(pacc+0),*(pacc+1));
        }



      t++;
    }


  /*--------------------------------------------------------------------------------------------------
  if (term)
    {
      t=0;

      while (t<term)
        {
          
        }

      delta_x=controldelta(neighbor+0,delta_x,ratio);
      delta_p=controldelta(neighbor+1,delta_p,ratio); 
      thermo(data,v,p,k,e,n);
      teff  = *(data+0);

      if (fabs(temp-teff)>TOLT) printf("thermalisation\t\t\t[not completely thermalized. Reducing step...]\n");
      else printf("thermalisation\t\t\t[OK]\n");

      t=0;

      while (fabs(temp-teff)>TOLT && t<term)
        {
          tfmc(x,v,p,f,delta_x,delta_p,temp,dim,n);
          computelist(x,v,p,f,e,neighbor,head,lscl,ptype,dim,n,nc);
          buildlist(x,head,lscl,rcell,n,nc);
          thermo(data,v,p,k,e,n);

          teff  = *(data+0);
          ekin  = *(data+1);
          epair = *(data+2);
          etot  = *(data+3);
          dmin  = *(neighbor+0);
          pmin  = *(neighbor+1);

          t++;
        }


      if (fabs(temp-teff)>TOLT) printf("\t\t\t\t[temperature out of bound. I suggest to increase 'term']\n\n");
      else printf("\t\t\t\t[done]\n\n");
    }

  delta_x=controldelta(neighbor+0,delta_x,ratio);
  delta_p=controldelta(neighbor+1,delta_p,ratio); 

  if(tsamp>0) fp1=fopen("pandha_pauli_mc.lammpstrj","w");

  printf("step\ttemp_set\ttemp_eff\te_kinetic\te_potential\te_total \tdelx/dmin \tdelp/pmin\n\n");

  t=0;
  dt = (temp-tend)/(double)(tmax);

  while (t<tmax)
    {
      tfmc(x,v,p,f,delta_x,delta_p,temp,dim,n);
      computelist(x,v,p,f,e,neighbor,head,lscl,ptype,dim,n,nc);
      buildlist(x,head,lscl,rcell,n,nc);
      thermo(data,v,p,k,e,n);

      teff  = *(data+0);
      ekin  = *(data+1);
      epair = *(data+2);
      etot  = *(data+3);
      dmin  = *(neighbor+0);
      pmin  = *(neighbor+1);

      if(thsamp>0 && t%thsamp==0) 
        printf("%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n",t,temp,teff,ekin,epair,etot,delta_x/dmin,delta_p/pmin);

      if(tsamp>0 && t%tsamp==0)
        {
          write_header(fp1,dim,n,t);
          write_data(fp1,x,v,p,f,ptype,n);
        }

      if (tctrl>0 && t%tctrl==0) 
        {
          delta_x=controldelta(neighbor+0,delta_x,ratio);
          delta_p=controldelta(neighbor+1,delta_p,ratio);
        }

      t++;
      temp -= dt;
    }

  if(tsamp>0) fclose(fp1);
  
  ------------------------------------------------------------------------------*/

  free(data);
  free(ptype);

  free(k);
  free(e);
  free(x);
  free(v);
  free(p);
  free(f);

  free(head);
  free(lscl);

  c1=clock();

  printf("\nCPU time (sec.): %f\n\n",(float)(c1-c0)/CLOCKS_PER_SEC);

  return 1;
}

double inicial(double *x,double *v,double *p,double *f,int *ptype,double rho,double temp,double xproton,int ndim)
{
 int    h,i,j,k,m,n,np,nn,npup,npdn,nnup,nndn,nless,nplus;
 double dim,r,r0,pmx,pmy,pmz,pm2x,pm2y,pm2z,sum2,fs;
 
 n  = ndim*ndim*ndim;
 np = (int)(xproton*(double)n);
 nn = n-np;

 npup = 0;
 nnup = 0;
 npdn = 0;
 nndn = 0;

 //dim=round(cbrtl((double)n/rho));  no longer valid!!!
 dim=cbrtl((double)n/rho); 
 r=dim/(double)ndim;
 r0=r/2.0;
 
 pmx  = 0.0;
 pmy  = 0.0;
 pmz  = 0.0;
 pm2x = 0.0;
 pm2y = 0.0;
 pm2z = 0.0;
 sum2 = 0.0;
 
 h=0;
 for(i=0;i<ndim;i++)
    for(j=0;j<ndim;j++)
       for(k=0;k<ndim;k++)
         {
           m = 1+((i+j+k)%4);

           *(ptype+h) = m;

           if ((!(m%2)) && (np>0))  // set protons
             {
               if (m == 2) npup++;  // set protons up
               if (m == 4) npdn++;  // set protons down
               np--;
             }
           if ((m%2) && (nn>0))      // set neutrons
             {
               if (m == 1) nnup++;  // set neutrons up
               if (m == 3) nndn++;  // set neutrons down
               nn--;
             }

           *(v+3*h+0) = 0.0;
           *(v+3*h+1) = 0.0;
           *(v+3*h+2) = 0.0;

           *(f+3*h+0) = 0.0;
           *(f+3*h+1) = 0.0;
           *(f+3*h+2) = 0.0;

           *(x+3*h+0) = (double)i*r+r0;
           *(x+3*h+1) = (double)j*r+r0;
           *(x+3*h+2) = (double)k*r+r0;

           *(p+3*h+0) = normaldist();
           *(p+3*h+1) = normaldist();
           *(p+3*h+2) = normaldist();
                    
           pmx += (*(p+3*h+0));
           pmy += (*(p+3*h+1));
           pmz += (*(p+3*h+2));

           pm2x += (*(p+3*h+0))*(*(p+3*h+0));
           pm2y += (*(p+3*h+1))*(*(p+3*h+1));
           pm2z += (*(p+3*h+2))*(*(p+3*h+2));
           h++;
         }

  pmx = pmx/(double)h;
  pmy = pmy/(double)h;
  pmz = pmz/(double)h;

  pm2x = pm2x/(double)h;
  pm2y = pm2y/(double)h;
  pm2z = pm2z/(double)h;

  sum2 += (pm2x+pm2y+pm2z)/3.0;
  sum2 -= (pmx*pmx+pmy*pmy+pmz*pmz)/3.0;

  fs=sqrt(temp/(M*sum2));  // M*sum2 = current temperature (Boltzmann)

  for(i=0;i<h;i++)
    {
      *(p+3*i+0)=M*(*(p+3*i+0)-pmx)*fs;
      *(p+3*i+1)=M*(*(p+3*i+1)-pmy)*fs;
      *(p+3*i+2)=M*(*(p+3*i+2)-pmz)*fs;
    }

  printf("\n");
  printf("atoms         = %d\t\t",h);

  nless = (ndim-1)*(ndim-1)*(ndim-1);
  nplus = (ndim+1)*(ndim+1)*(ndim+1);
  if (ndim%2) printf("[warning: non-periodic arrangement, I suggest %d or %d atoms]\n",nless,nplus);
  else printf("\n");

  printf("protons  up   = %d\n",npup);
  printf("protons  down = %d\n",npdn);
  printf("neutrons up   = %d\n",nnup);
  printf("neutrons down = %d\n\n",nndn);
  printf("proton frac.  = %d/(%d+%d)\n\n",npup+npdn,npup+npdn,nnup+nndn);

  printf("density       = %f\n",(double)h/(dim*dim*dim));
  printf("box size      = %f\n",dim);
  printf("atom dist.    = %f\n",r);

  return dim;
}

double checkperformance1(double temp,double tend,double dim,int tmax,int nc,int n)
{
  int    ncells;
  double rcell;

  printf("\ninitial temp. = %f\n",temp);
  printf("final   temp. = %f\n",tend);
  printf("inc/dec temp. = %g\n",(temp-tend)/(double)tmax);

  ncells = nc*nc*nc;
  rcell  = dim/(double)nc;

  printf("\n");
  printf("cells         = %d\t\t\t",ncells);

  if (ncells<8) printf("[warning: too few cells. I suggest to resize the original cells]\n");
  else printf("\n"); 

  printf("cell size     = %f\t",rcell); 

  if (rcell!=RCELL) printf("[warning: cells resized (original %f)]\n",RCELL);
  else printf("\n"); 

  return rcell;
}


int buildlist(double *x,int *head,int *lscl,double rcell,int n,int nc)
{
  int i,c,cx,cy,cz,ncells;
  
  ncells = nc*nc*nc;

  for (c=0; c<ncells; c++) head[c] = EMPTY; // reset the headers, head 

  for (i=0; i<n; i++)                       // scan atoms to head, lscl 
    {
      cx = (int)(*(x+3*i+0)/rcell);
      cy = (int)(*(x+3*i+1)/rcell);
      cz = (int)(*(x+3*i+2)/rcell);
      c  = cx*nc*nc+cy*nc+cz;

      lscl[i] = head[c];                    // link to previous occupant (or EMPTY) 
      head[c] = i;                          // the last one goes to the header
    }

  return 1;
}


int computelist(double *x,double *v,double *p,double *f,double *e,int *head,int *lscl,int *ptype,double dim,int n,int nc)
{
  int    i,j,c,cx,cy,cz,cc,ccx,ccy,ccz;
  double dx,dy,dz,dpx,dpy,dpz,r,pr,r2,p2;
  double fx,fy,fz,fr,vx,vy,vz,vr,ep;
  double shift[3],ref[3];

  for (i=0; i<n; i++) 
    {
      *(e+i) = 0.0;
      *(f+3*i+0) = 0.0;
      *(f+3*i+1) = 0.0;
      *(f+3*i+2) = 0.0;
      *(v+3*i+0) = 0.0;
      *(v+3*i+1) = 0.0;
      *(v+3*i+2) = 0.0;
    }

  for (cx=0; cx<nc; cx++)                                  // scan inner cells 
    for (cy=0; cy<nc; cy++)
      for (cz=0; cz<nc; cz++) 
        {
          c = cx*nc*nc+cy*nc+cz;                           // calculate a scalar cell index
          
          for (ccx=cx-1; ccx<=cx+1; ccx++)                 // scan the neighbor cells (including itself)
            for (ccy=cy-1; ccy<=cy+1; ccy++)
              for (ccz=cz-1; ccz<=cz+1; ccz++) 
                {

                  // periodic boundary condition by shifting coordinates

                  *(shift+0) = 0.0;
                  *(shift+1) = 0.0;
                  *(shift+2) = 0.0;

                  if (ccx<0) *(shift+0) = -dim;
                  else if (ccx>=nc) *(shift+0) = dim;

                  if (ccy<0) *(shift+1) = -dim;
                  else if (ccy>=nc) *(shift+1) = dim;

                  if (ccz<0) *(shift+2) = -dim;
                  else if (ccz>=nc) *(shift+2) = dim;
                
                  cc = ((ccx+nc)%nc)*nc*nc+((ccy+nc)%nc)*nc+((ccz+nc)%nc);  
                  
                  i = head[c];                             // scan atom i in cell c
                  while (i != EMPTY) 
                    {
                      j   = head[cc];                      // scan atom j in cell cc

                      while (j != EMPTY) 
                        {
                          if (i < j)                       // avoid double counting and correct image positions 
                            {                            
                              dx = (*(x+3*i+0)) - (*(x+3*j+0)) - (*(shift+0));
                              dy = (*(x+3*i+1)) - (*(x+3*j+1)) - (*(shift+1));
                              dz = (*(x+3*i+2)) - (*(x+3*j+2)) - (*(shift+2));
                              r2 = dx*dx+dy*dy+dz*dz;
 
                              dpx = (*(p+3*i+0)) - (*(p+3*j+0));
                              dpy = (*(p+3*i+1)) - (*(p+3*j+1));
                              dpz = (*(p+3*i+2)) - (*(p+3*j+2));
                              p2  = dpx*dpx+dpy*dpy+dpz*dpz;

                              get_table_pandha(ref,ptype,r2,i,j,PANTAB); 

                              r  = *(ref+0); 
                              ep = *(ref+1); 
                              fr = *(ref+2);
                              
                              if (r > 0.0)
                                { 
                                  fx = dx*fr/r;
                                  fy = dy*fr/r;
                                  fz = dz*fr/r;

                                  *(e+i) += ep/(double)n;     
                                  
                                  *(f+3*i+0) += fx;
                                  *(f+3*i+1) += fy;
                                  *(f+3*i+2) += fz;
                                  *(f+3*j+0) -= fx;
                                  *(f+3*j+1) -= fy;
                                  *(f+3*j+2) -= fz;
                                }
                              
                              get_table_pauli(ref,ptype,r2,p2,'x',i,j,PAUTAB);
                                 
                              r  = *(ref+0);
                              ep = *(ref+1);
                              fr = *(ref+2);

                              if (r > 0.0)
                                {     
                                  // warning: no division by r is necessary!!!
                                  fx = dx*fr;
                                  fy = dy*fr;
                                  fz = dz*fr;

                                  *(e+i) += ep/(double)n;
                                  *(f+3*i+0) += fx;
                                  *(f+3*i+1) += fy;
                                  *(f+3*i+2) += fz;
                                  *(f+3*j+0) -= fx;
                                  *(f+3*j+1) -= fy;
                                  *(f+3*j+2) -= fz;
                                }

                              get_table_pauli(ref,ptype,r2,p2,'p',i,j,PAUTAB);

                              pr = *(ref+0);
                              ep = *(ref+1);
                              vr = *(ref+2);

                              if (pr > 0.0)
                                {            
                                  // warning: no division by r is necessary!!!
                                  // warning: the sign of vr follows the same rule as fr!!! 
                                  vx = dpx*vr;
                                  vy = dpy*vr;
                                  vz = dpz*vr;

                                  *(v+3*i+0) += vx;
                                  *(v+3*i+1) += vy;
                                  *(v+3*i+2) += vz; 
                                  *(v+3*j+0) -= vx;
                                  *(v+3*j+1) -= vy;
                                  *(v+3*j+2) -= vz; 
                                }
                            }
                          j = lscl[j];
                        }
                      i = lscl[i];
                    }
                }
      }

  return 1;
}

double metropolis(double *x,double *p,double *pacc,int *head,int *lscl,int *ptype,double temp,double rcell,double delta_x,double delta_p,double dim,int n,int nc)
{
  int    i,j,cx,cy,cz,cc,ccx,ccy,ccz;
  double delx,dely,delz,delpx,delpy,delpz;
  double px,py,pz,dx,dy,dz,dpx,dpy,dpz;
  double r2,p2,ec,ep,etot,prob;
  double pbc[3],shift[3],ref[3];

  ec   = 0.0;
  ep   = 0.0;
  etot = 0.0;

  i = rand()%n;

  cx = (int)(*(x+3*i+0)/rcell);
  cy = (int)(*(x+3*i+1)/rcell);
  cz = (int)(*(x+3*i+2)/rcell);
  //c  = cx*nc*nc+cy*nc+cz;

  px = *(p+3*i+0);
  py = *(p+3*i+1);
  pz = *(p+3*i+2);

  ec -= (px*px+py*py+pz*pz)/(2.0*M);

  delpx = delta_p*(2.0*((double)rand()/(double)RAND_MAX)-1.0);
  delpy = delta_p*(2.0*((double)rand()/(double)RAND_MAX)-1.0);
  delpz = delta_p*(2.0*((double)rand()/(double)RAND_MAX)-1.0);

  px += delpx;
  py += delpy;
  pz += delpz;

  ec += (px*px+py*py+pz*pz)/(2.0*M);

  for (ccx=cx-1; ccx<=cx+1; ccx++)                 // scan the neighbor cells (including itself)
    for (ccy=cy-1; ccy<=cy+1; ccy++)
      for (ccz=cz-1; ccz<=cz+1; ccz++) 
        {
          // periodic boundary condition by shifting coordinates

          *(shift+0) = 0.0;
          *(shift+1) = 0.0;
          *(shift+2) = 0.0;

          if (ccx<0) *(shift+0) = -dim;
          else if (ccx>=nc) *(shift+0) = dim;

          if (ccy<0) *(shift+1) = -dim;
          else if (ccy>=nc) *(shift+1) = dim;

          if (ccz<0) *(shift+2) = -dim;
          else if (ccz>=nc) *(shift+2) = dim;
                
          cc = ((ccx+nc)%nc)*nc*nc+((ccy+nc)%nc)*nc+((ccz+nc)%nc);  
                 
          j = head[cc];                        // scan atom j in cell cc

          while (j != EMPTY) 
             {
               if (i != j)                      // avoid double counting and correct image positions 
                 {                            
                   dx = (*(x+3*i+0)) - (*(x+3*j+0)) - (*(shift+0));
                   dy = (*(x+3*i+1)) - (*(x+3*j+1)) - (*(shift+1));
                   dz = (*(x+3*i+2)) - (*(x+3*j+2)) - (*(shift+2));
                   r2 = dx*dx+dy*dy+dz*dz;
                    
                   dpx = (*(p+3*i+0)) - (*(p+3*j+0));
                   dpy = (*(p+3*i+1)) - (*(p+3*j+1));
                   dpz = (*(p+3*i+2)) - (*(p+3*j+2));
                   p2  = dpx*dpx+dpy*dpy+dpz*dpz;
     
                   get_table_pauli(ref,ptype,r2,p2,'p',i,j,PAUTAB);
                   ep -= (*(ref+1)); 

                   dpx += delpx;
                   dpy += delpy;
                   dpz += delpz;
                   p2  = dpx*dpx+dpy*dpy+dpz*dpz;
     
                   get_table_pauli(ref,ptype,r2,p2,'p',i,j,PAUTAB);
                   ep += (*(ref+1)); 
                 }
               j = lscl[j];
             }
        }
   
  prob = (double)rand()/(double)RAND_MAX;

  if (prob < exp(-(ec+ep)/temp))  // accept
    {
      *(p+3*i+0) = px;
      *(p+3*i+1) = py;
      *(p+3*i+2) = pz;

      *(pacc+1) += 1.0/(double)n; 

      etot += ec+ep;
    }

  ep = 0.0;

  pbc[0] = 0.0;
  pbc[1] = 0.0;
  pbc[2] = 0.0;

  i = rand()%n;

  cx = (int)(*(x+3*i+0)/rcell);
  cy = (int)(*(x+3*i+1)/rcell);
  cz = (int)(*(x+3*i+2)/rcell);
  //c  = cx*nc*nc+cy*nc+cz;

  delx  = delta_x*(2.0*((double)rand()/(double)RAND_MAX)-1.0);
  dely  = delta_x*(2.0*((double)rand()/(double)RAND_MAX)-1.0);
  delz  = delta_x*(2.0*((double)rand()/(double)RAND_MAX)-1.0);

  if ((*(x+3*i+0) + delx) < 0.0) pbc[0] = dim;
  if ((*(x+3*i+0) + delx) > dim) pbc[0] =-dim;
  if ((*(x+3*i+1) + dely) < 0.0) pbc[1] = dim;
  if ((*(x+3*i+1) + dely) > dim) pbc[1] =-dim;
  if ((*(x+3*i+2) + delz) < 0.0) pbc[2] = dim;
  if ((*(x+3*i+2) + delz) > dim) pbc[2] =-dim;

  for (ccx=cx-1; ccx<=cx+1; ccx++)                 // scan the neighbor cells (including itself)
    for (ccy=cy-1; ccy<=cy+1; ccy++)
      for (ccz=cz-1; ccz<=cz+1; ccz++) 
        {
          // periodic boundary condition by shifting coordinates

          *(shift+0) = 0.0;
          *(shift+1) = 0.0;
          *(shift+2) = 0.0;

          if (ccx<0) *(shift+0) = -dim;
          else if (ccx>=nc) *(shift+0) = dim;

          if (ccy<0) *(shift+1) = -dim;
          else if (ccy>=nc) *(shift+1) = dim;

          if (ccz<0) *(shift+2) = -dim;
          else if (ccz>=nc) *(shift+2) = dim;
                
          cc = ((ccx+nc)%nc)*nc*nc+((ccy+nc)%nc)*nc+((ccz+nc)%nc);  
                 
          j = head[cc];                        // scan atom j in cell cc

          while (j != EMPTY) 
             {
               if (i != j)                       // avoid double counting and correct image positions 
                 {                            
                   dx = (*(x+3*i+0)) - (*(x+3*j+0)) - (*(shift+0));
                   dy = (*(x+3*i+1)) - (*(x+3*j+1)) - (*(shift+1));
                   dz = (*(x+3*i+2)) - (*(x+3*j+2)) - (*(shift+2));
                   r2 = dx*dx+dy*dy+dz*dz;
                          
                   dpx = (*(p+3*i+0)) - (*(p+3*j+0));
                   dpy = (*(p+3*i+1)) - (*(p+3*j+1));
                   dpz = (*(p+3*i+2)) - (*(p+3*j+2));
                   p2  = dpx*dpx+dpy*dpy+dpz*dpz;
     
                   get_table_pandha(ref,ptype,r2,i,j,PANTAB); 
                   ep -= (*(ref+1)); 

                   get_table_pauli(ref,ptype,r2,p2,'x',i,j,PAUTAB);
                   ep -= (*(ref+1));

                   dx += delx + pbc[0];
                   dy += dely + pbc[1];
                   dz += delz + pbc[2];
                   r2  = dx*dx+dy*dy+dz*dz;

                   get_table_pandha(ref,ptype,r2,i,j,PANTAB); 
                   ep += (*(ref+1)); 

                   get_table_pauli(ref,ptype,r2,p2,'x',i,j,PAUTAB);
                   ep += (*(ref+1));   
                 }
               j = lscl[j];
             } 
        }

  prob = (double)rand()/(double)RAND_MAX;

  if (prob < exp(-ep/temp))  // accept
    {
      *(x+3*i+0) += delx + pbc[0];
      *(x+3*i+1) += dely + pbc[1];
      *(x+3*i+2) += delz + pbc[2];
   
      *(pacc+0) += 1.0/(double)n;

      etot += ep;
    }

  return etot/(double)n;
}

void thermo(double *data,double *v,double *p,double *k,double *e,int n)
{
  int    i;
  double velx,vely,velz,vx,vy,vz,px,py,pz;
  double m2,ec,ep,tkin,temp,tvel;

  ec   = 0.0;
  ep   = 0.0;
  m2   = 2.0*M;
  tkin = 0.0;
  temp = 0.0;
  tvel = 0.0;


  for(i=0;i<n;i++) 
    {
      px = *(p+3*i+0);
      py = *(p+3*i+1);
      pz = *(p+3*i+2);

      velx = *(v+3*i+0);
      vely = *(v+3*i+1);
      velz = *(v+3*i+2);

      // warning: minus sign is correct!!!
    
      vx = px/M - velx;
      vy = py/M - vely;
      vz = pz/M - velz;

      *(k+i) = (px*px+py*py+pz*pz)/m2;

      ec   += (*(k+i));
      ep   += (*(e+i));
      temp += vx*px + vy*py + vz*pz;       // effective temperature
      tkin += (px*px + py*py + pz*pz)/M;   // kinetic   temperature
      tvel += M*(vx*vx + vy*vy + vz*vz);   // velocity  temperature

    }

  ec   = ec/(double)(n);
  temp = temp/(double)(3*n);
  tkin = tkin/(double)(3*n);
  tvel = tvel/(double)(3*n);

  *(data+0) = temp;
  *(data+1) = ec;
  *(data+2) = ep;
  *(data+3) = ec+ep;
  *(data+4) = tkin;
  *(data+5) = tvel;

  return;
}

void get_table_pandha(double *data,int *ptype,double r2,int ii,int jj,int ntable)
{
  int    i,itype,jtype;
  double dr,r,rmin,rmax;
  double frac,f1,f2,e1,e2;
  double *mytable;

  itype = *(ptype+ii);
  jtype = *(ptype+jj);

  if ((itype%2)+(jtype%2)==0) mytable=table_pp;
  else if ((itype%2)+(jtype%2)==1) mytable=table_np;
       else mytable=table_nn;

  *(data+0) = 0.0;
  *(data+1) = 0.0;
  *(data+2) = 0.0;

  // recall the format "r,r2,vnn,fnn" 

  rmin = *(mytable+0);
  rmax = *(mytable+4*(ntable-1)+0);
 
  r  = sqrt(r2);
  dr = (rmax-rmin)/(double)(ntable-1);

  if (r<rmax) 
    { 
      i = (int)((r-rmin)/dr);
      frac = (r-(*(mytable+4*i+0)))/dr;

      f1 = *(mytable+4*i+3); 
      f2 = *(mytable+4*(i+1)+3); 
      e1 = *(mytable+4*i+2); 
      e2 = *(mytable+4*(i+1)+2); 
 
      *(data+0) = r;
      *(data+1) = e1*(1.0-frac)+e2*frac;
      *(data+2) = f1*(1.0-frac)+f2*frac; 
    }
  
  return;
}

void get_table_pauli(double *data,int *ptype,double r2,double p2,char flag,int ii,int jj,int ntable)
{
  int    i,j,itype,jtype;
  double dr,r,rmin,rmax;
  double dp,p,pmin,pmax;
  double r02,p02,fracx,fracp;
  double f1,f2,fr,fp,er,ep,ecut;
 
  r02 = Q0*Q0;
  p02 = P0*P0;
  
  *(data+0) = 0.0;
  *(data+1) = 0.0;
  *(data+2) = 0.0;
 
  itype = *(ptype+ii);
  jtype = *(ptype+jj);

  if (itype==jtype)  
    {

      // recall the format "r,p,r2,p2,v,fr,fp,cutoff"
      // warning: D is consirered  to be unity here!!!
    
      rmin = *(table_pauli+0);
      rmax = *(table_pauli+7*(ntable-1)+0);

      r  = sqrt(r2);
      dr = (rmax-rmin)/(double)(ntable-1);
    
      pmin = *(table_pauli+1);
      pmax = *(table_pauli+7*(ntable-1)+1);

      p  = sqrt(p2);
      dp = (pmax-pmin)/(double)(ntable-1);
    
      if ((r<rmax) && (p<pmax))
        {
          i = (int)((r-rmin)/dr);
          j = (int)((p-pmin)/dp);
          
          fracx = (r-(*(table_pauli+7*i+0)))/dr;
          fracp = (p-(*(table_pauli+7*j+1)))/dp;
        
          // f1, f2 correspond to the exponentials only!!!

          f1 = *(table_pauli+7*i+4); 
          f2 = *(table_pauli+7*(i+1)+4); 
          er = f1*(1.0-fracx)+f2*fracx;
          fr = f1*(1.0-fracx)+f2*fracx;
        
          f1 = *(table_pauli+7*j+5); 
          f2 = *(table_pauli+7*(j+1)+5);
          ep = f1*(1.0-fracp)+f2*fracp;
          fp = f1*(1.0-fracp)+f2*fracp;
        
          ecut = *(table_pauli+7*i+6);
        
          if (flag=='x') 
            {
              *(data+0) = r;
              *(data+1) = D*(er*ep-ecut);
              *(data+2) = D*fr*fp/r02;
            }

          if (flag=='p') 
            {
              *(data+0) = p;
              *(data+1) = D*(er*ep-ecut);
              *(data+2) = D*fr*fp/p02;
            }
        }        
    }
  
  return;
}


int read_atoms(char *myfile)
{
  int  j,atoms,timestep;
  char trash1[40],trash2[40],trash3[40],trash4[40];
  float lim[6];
  FILE *fp;

  atoms=0;
  timestep=0;

  fp=fopen(myfile,"r");

  j=fscanf(fp,"%s %s\n",trash1,trash2);
  j=fscanf(fp,"%d\n",&timestep);
  j=fscanf(fp,"%s %s %s %s\n",trash1,trash2,trash3,trash4);
  j=fscanf(fp,"%d\n",&atoms);
  j=fscanf(fp,"%s %s %s",trash1,trash2,trash3);
  j=fscanf(fp,"%s %s %s\n",trash1,trash2,trash3);
  j=fscanf(fp,"%f %f\n",lim+0,lim+1);
  j=fscanf(fp,"%f %f\n",lim+2,lim+3);
  j=fscanf(fp,"%f %f\n",lim+4,lim+5);
  j=fscanf(fp,"%s %s %s %s",trash1,trash2,trash3,trash4);
  j=fscanf(fp,"%s %s %s",trash1,trash2,trash3);
  j=fscanf(fp,"%s %s %s",trash1,trash2,trash3);
  j=fscanf(fp,"%s %s %s",trash1,trash2,trash3);
  j=fscanf(fp,"%s %s %s\n",trash1,trash2,trash3);
  
  fclose(fp);
  
  if (!atoms) 
     {
       printf("\nSomething is wrong with the atoms!\n\n");
       exit(0);
     }

  j++; //to avoid warnings

  return atoms;
}

double read_dim(char *myfile)
{
  int  j,atoms,timestep;
  char trash1[40],trash2[40],trash3[40],trash4[40];
  float lim[6];
  double dim;
  FILE *fp;

  fp=fopen(myfile,"r");

  j=fscanf(fp,"%s %s\n",trash1,trash2);
  j=fscanf(fp,"%d\n",&timestep);
  j=fscanf(fp,"%s %s %s %s\n",trash1,trash2,trash3,trash4);
  j=fscanf(fp,"%d\n",&atoms);
  j=fscanf(fp,"%s %s %s",trash1,trash2,trash3);
  j=fscanf(fp,"%s %s %s\n",trash1,trash2,trash3);
  j=fscanf(fp,"%f %f\n",lim+0,lim+1);
  j=fscanf(fp,"%f %f\n",lim+2,lim+3);
  j=fscanf(fp,"%f %f\n",lim+4,lim+5);
  j=fscanf(fp,"%s %s %s %s",trash1,trash2,trash3,trash4);
  j=fscanf(fp,"%s %s %s",trash1,trash2,trash3);
  j=fscanf(fp,"%s %s %s",trash1,trash2,trash3);
  j=fscanf(fp,"%s %s %s",trash1,trash2,trash3);
  j=fscanf(fp,"%s %s %s\n",trash1,trash2,trash3);

  dim=(double)lim[1];  

  if (*(lim+0)) dim=0.0;
  if (*(lim+2)) dim=0.0;
  if (*(lim+4)) dim=0.0;

  if (*(lim+1)!=(*(lim+3))) dim=0.0;
  if (*(lim+1)!=(*(lim+5))) dim=0.0;
  if (*(lim+3)!=(*(lim+5))) dim=0.0;

  if(!dim) printf("\nSomething is wrong with the cell dimensions!\n\n");
  
  fclose(fp);

  j++;//to avoid warnings

  return dim;
}


double read_data(char *myfile,double *x,double *v,double *p,double *f,int *ptype,int atoms)
{
  int i,j,id,tid;
  char trash1[40],trash2[40],trash3[40],trash4[40];
  float rx,ry,rz,vx,vy,vz,px,py,pz,fx,fy,fz;
  double ec,t,vpx,vpy,vpz;
  FILE *fp;

  fp=fopen(myfile,"r");

  j=fscanf(fp,"%s %s\n",trash1,trash2);
  j=fscanf(fp,"%s\n",trash1);
  j=fscanf(fp,"%s %s %s %s\n",trash1,trash2,trash3,trash4);
  j=fscanf(fp,"%s\n",trash1);
  j=fscanf(fp,"%s %s %s",trash1,trash2,trash3);
  j=fscanf(fp,"%s %s %s\n",trash1,trash2,trash3);
  j=fscanf(fp,"%s %s\n",trash1,trash2);
  j=fscanf(fp,"%s %s\n",trash1,trash2);
  j=fscanf(fp,"%s %s\n",trash1,trash2);
  j=fscanf(fp,"%s %s %s %s",trash1,trash2,trash3,trash4);
  j=fscanf(fp,"%s %s %s",trash1,trash2,trash3);
  j=fscanf(fp,"%s %s %s",trash1,trash2,trash3);
  j=fscanf(fp,"%s %s %s",trash1,trash2,trash3);
  j=fscanf(fp,"%s %s %s\n",trash1,trash2,trash3);

  ec=0.0;

  for(i=0;i<atoms;i++)
   {
     j=fscanf(fp,"%d",&id);
     j=fscanf(fp,"%d",&tid);
     j=fscanf(fp,"%f %f %f",&rx,&ry,&rz);
     j=fscanf(fp,"%f %f %f",&vx,&vy,&vz);
     j=fscanf(fp,"%f %f %f",&px,&py,&pz);
     j=fscanf(fp,"%f %f %f\n",&fx,&fy,&fz);
     
     vpx=(double)vx*(double)px;
     vpy=(double)vy*(double)py;
     vpz=(double)vz*(double)pz;
     ec+=(vpx+vpy+vpz);

    id--;
    *(ptype+id)=tid;

    *(x+3*id+0)=(double)rx;
    *(x+3*id+1)=(double)ry;
    *(x+3*id+2)=(double)rz;

    // warning: minus to follows our own sign rules!!!
    
    *(v+3*id+0)=px/M-(double)vx;
    *(v+3*id+1)=py/M-(double)vy;
    *(v+3*id+2)=pz/M-(double)vz;

    *(p+3*id+0)=(double)px;
    *(p+3*id+1)=(double)py;
    *(p+3*id+2)=(double)pz;

    *(f+3*id+0)=(double)fx;
    *(f+3*id+1)=(double)fy;
    *(f+3*id+2)=(double)fz;
   }

   t=ec/(3.0*(double)atoms);

   fclose(fp);

   j++; //to avoid warnings

   return t;
}



int write_header(FILE *fp,double dim,int atoms,int timestep)
{
  fprintf(fp,"ITEM: TIMESTEP\n%d\n",timestep);
  fprintf(fp,"ITEM: NUMBER OF ATOMS\n%d\n",atoms);
  fprintf(fp,"ITEM: BOX BOUNDS pp pp pp\n");
  fprintf(fp,"%f %f\n",0.0,dim);
  fprintf(fp,"%f %f\n",0.0,dim);
  fprintf(fp,"%f %f\n",0.0,dim);
  fprintf(fp,"ITEM: ATOMS id type x y z vx vy vz px py pz fx fy fz\n");
  return 1;
}

int write_data(FILE *fp,double *x,double *v,double *p,double *f,int *ptype,int atoms)
{
  int    i;
  double vx,vy,vz;

  for(i=0;i<atoms;i++) 
    {
      vx = (*(p+3*i+0))/M-(*(v+3*i+0));
      vy = (*(p+3*i+1))/M-(*(v+3*i+1));
      vz = (*(p+3*i+2))/M-(*(v+3*i+2));

      fprintf(fp,"%d %d",i+1,*(ptype+i));
      fprintf(fp," %f %f %f",*(x+3*i+0),*(x+3*i+1),*(x+3*i+2));
      fprintf(fp," %f %f %f",vx,vy,vz);
      fprintf(fp," %f %f %f",*(p+3*i+0),*(p+3*i+1),*(p+3*i+2));
      fprintf(fp," %f %f %f",*(f+3*i+0),*(f+3*i+1),*(f+3*i+2));
      fprintf(fp,"\n");
    }

  return 1;
}

void build_pandha_table(double ri,double rf,int ntable)
{
  int     i;
  double  dr,r,r2,rinv,rcinv;
  double  exp_murr,exp_murcr,exp_muar,exp_muarc,exp_mu0r,exp_mu0rc;
  double  v1,v2,v3,vnp,vnn,vpp,fnp,fnn,fpp;

  dr = (rf-ri)/(double)(ntable-1);

  for(i=0 ; i<ntable ; i++)
    {
      r  =ri+(double)i*dr;
      r2 =r*r;
     
      rinv=1.0/r;
      rcinv=1.0/rf;

      exp_murr=1.0/exp(MUR*r);
      exp_murcr=1.0/exp(MUR*rf);
      exp_muar=1.0/exp(MUA*r);
      exp_muarc=1.0/exp(MUA*rf);
      exp_mu0r=1.0/exp(MU0*r);
      exp_mu0rc=1.0/exp(MU0*rf);

      v1=VR*rinv*exp_murr;
      v2=VA*rinv*exp_muar;
      v3=V0*rinv*exp_mu0r;

      vnp=v1-VR*rcinv*exp_murcr-v2+VA*rcinv*exp_muarc;
      vnn=v3-V0*rcinv*exp_mu0rc;
      vpp=vnn;

      fnp=(rinv+MUR)*v1-(rinv+MUA)*v2;
      fnn=(rinv+MU0)*v3;
      fpp=fnn;

      // store as "r,r2,vnn,fnn" (similar to Lammps format)

      *(table_np+4*i+0) = r;
      *(table_np+4*i+1) = r2;
      *(table_np+4*i+2) = vnp;
      *(table_np+4*i+3) = fnp;

      *(table_nn+4*i+0) = r;
      *(table_nn+4*i+1) = r2;
      *(table_nn+4*i+2) = vnn;
      *(table_nn+4*i+3) = fnn;
    
      *(table_pp+4*i+0) = r;
      *(table_pp+4*i+1) = r2;
      *(table_pp+4*i+2) = vpp;
      *(table_pp+4*i+3) = fpp;
    }

  return;
}

void build_pauli_table(double ri,double rf,double pi,double pf,int ntable)
{
  int    i;
  double q02,p02;
  double dr,dp,r,p,r2,p2,r2q2,p2p2,exprcr,exprcp,expr2,expp2;

  q02 = Q0*Q0;
  p02 = P0*P0;
  exprcr = exp(-0.5*(rf*rf)/q02);
  exprcp = exp(-0.5*(pf*pf)/p02);

  dr = (rf-ri)/(double)(ntable-1);
  dp = (pf-pi)/(double)(ntable-1);

  for(i=0 ; i<ntable ; i++)
    {
      r  = ri+(double)i*dr;
      p  = pi+(double)i*dp;
      r2 = r*r;
      p2 = p*p;

      r2q2=0.5*(r2)/q02;
      p2p2=0.5*(p2)/p02;

      expr2 = 1.0/exp(r2q2);
      expp2 = 1.0/exp(p2p2);

      // store as "r,p,r2,p2,fr,fp,cutoff" (extra columns with respect to Lammps format)
      // warning: D is consirered  to be unity here!!!
    
      *(table_pauli+7*i+0) = r;
      *(table_pauli+7*i+1) = p;
      *(table_pauli+7*i+2) = r2;
      *(table_pauli+7*i+3) = p2;
      *(table_pauli+7*i+4) = expr2;
      *(table_pauli+7*i+5) = expp2;
      *(table_pauli+7*i+6) = exprcr*exprcp;
    }

  return;
}

double normaldist() 
{
  double x,y,r2,c;

  r2=0.0;
 
  while(r2 == 0.0 || r2 > 1.0)
    {
      x  = 2.0*((double) rand() / (double)RAND_MAX) - 1.0;
      y  = 2.0*((double) rand() / (double)RAND_MAX) - 1.0;
      r2 = x*x + y*y;
    }

  c = sqrt(-2.0*log(r2)/r2);

  return x*c;
}


void help()
{
  printf("\nThis is a MD simulation program for the Pauli potential.\n\n");
  printf("-n       number of particles                (default 8000)\n");
  printf("-x       proton fraction                    (default 0.5)\n");
  printf("-rho     density                            (default 0.16)\n");
  printf("-temp    initial temperature                (default 0.1)\n");
  printf("-tend    ending temperature                 (default temp)\n");
  printf("-delx    monte carlo trial on coordinates x (default 0.2)\n");
  printf("-delp    monte carlo trial on momenta p     (default 0.2)\n");
  printf("-term    termalizacion steps                (default 100)\n");
  printf("-steps    monte carlo steps                 (default 100)\n");
  printf("-ctrl    ratio for delx and delp            (default 0.1)\n");
  printf("-tctrl   control period for delta           (default 0 = no control)\n");
  printf("-thsamp  sampling period for thermodynamics (default 100; 0 = no sampling)\n");
  printf("-tsamp   sampling period for configurations (default 100; 0 = no sampling)\n");
  printf("-seed    initial value for rand.\n");
  printf("-initial initial configuration (lammps format).\n");
  printf("\n");
  exit(0);
}

