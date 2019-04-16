/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include <math.h>
#include <stdlib.h>
#include "pair_pauli_cut.h"
#include "atom.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "comm.h"
#include "force.h"
#include "memory.h"
#include "error.h"
#include "update.h"
#include <string.h>

#define D0  10000.0
#define Q02 1.0
#define P02 1.0

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

PairPauliCut::PairPauliCut(LAMMPS *lmp) : Pair(lmp)
{
  single_enable = 0;
}

/* ---------------------------------------------------------------------- */

PairPauliCut::~PairPauliCut()
{
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);

    memory->destroy(cut_lj);
    memory->destroy(cut_coul);
    memory->destroy(epsilon);
    memory->destroy(sigma);
  }
}

/* ---------------------------------------------------------------------- */

void PairPauliCut::compute(int eflag, int vflag)
{
  int    i,j,ii,jj,inum,jnum,itype,jtype;
  double xtmp,ytmp,ztmp,pxtmp,pytmp,pztmp,delx,dely,delz,ecoul; // Guillermo Frank: temporary mu
  double delpx,delpy,delpz,rsq,psq,rinv,r2inv,p2inv,rcsq;       // Guillermo Frank: del mu, psq, p2inv, rcsq
  double forcecoul,fpair,vpair,factor_coul,factor_lj;           // Guillermo Frank: fpair, vpair
  int    *ilist,*jlist,*numneigh,**firstneigh;

  ecoul = 0.0;
  if (eflag || vflag) ev_setup(eflag,vflag);
  else evflag = vflag_fdotr = 0;

  double **x = atom->x;
  double **f = atom->f;
  double **v = atom->v;                     // Guillermo Frank: input velocity
  double *q = atom->q;
  double **mu = atom->mu;
  double *mass = atom->mass;                // Guillermo Frank: input mass
  int *type = atom->type;
  int nlocal = atom->nlocal;
  double *special_coul = force->special_coul;
  double *special_lj = force->special_lj;
  int newton_pair = force->newton_pair;

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  // loop over neighbors of my atoms

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    pxtmp = mu[i][0];                       // Guillermo Frank: assign mu_x
    pytmp = mu[i][1];                       // Guillermo Frank: assign mu_y
    pztmp = mu[i][2];                       // Guillermo Frank: assign mu_z

    itype = type[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];

    v[i][0] = mu[i][0]/mass[itype];         // Guillermo Frank: set vx=px/m
    v[i][1] = mu[i][1]/mass[itype];         // Guillermo Frank: set vy=py/m
    v[i][2] = mu[i][2]/mass[itype];         // Guillermo Frank: set vz=pz/m

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      factor_lj = special_lj[sbmask(j)];
      factor_coul = special_coul[sbmask(j)];
      j &= NEIGHMASK;

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      delpx = pxtmp - mu[j][0];                          // Guillermo Frank: pi-pj
      delpy = pytmp - mu[j][1];                          // Guillermo Frank: pi-pj
      delpz = pztmp - mu[j][2];                          // Guillermo Frank: pi-pj

      rsq = delx*delx + dely*dely + delz*delz;
      psq = delpx*delpx + delpy*delpy + delpz*delpz;     // Guillermo Frank: pqp
      jtype = type[j];

      rcsq = cut_lj[itype][jtype]*cut_lj[itype][jtype];  // Guillermo Frank: rcsq

      r2inv = 0.5*rsq/sigma[itype][jtype];               // Guillermo Frank: r^2/2q0^2
      p2inv = 0.5*psq/cut_coul[itype][jtype];            // Guillermo Frank: p^2/2p0^2

      if (2.0*(r2inv+p2inv) < rcsq) {

        forcecoul = epsilon[itype][jtype]/exp(r2inv+p2inv); // Guillermo Frank: D*esp(-r^2/2q0^2-p^2/2p0^2)
        fpair = 0.5*forcecoul/sigma[itype][jtype];          // Guillermo Frank: -grad_r(H)
        vpair = 0.5*forcecoul/cut_coul[itype][jtype];       // Guillermo Frank: p/m-grad_p(H)
 
        f[i][0] += delx*fpair;
        f[i][1] += dely*fpair; 
        f[i][2] += delz*fpair; 
        v[i][0] -= delx*vpair;
        v[i][1] -= dely*vpair;
        v[i][2] -= delz*vpair;

        if (newton_pair || j < nlocal) {
          f[j][0] -= delx*fpair;
          f[j][1] -= dely*fpair;
          f[j][2] -= delz*fpair;
        }

       ecoul = forcecoul-epsilon[itype][jtype]/exp(0.5*rcsq);

       ev_tally(i,j,nlocal,newton_pair,0.0,ecoul,fpair,delx,dely,delz);

      }
    }
  }

  //if (vflag_fdotr) virial_fdotr_compute();  NOT IMPLEMENTED YET!!!
}

/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

void PairPauliCut::allocate()
{
  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag,n+1,n+1,"pair:setflag");
  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++)
      setflag[i][j] = 0;

  memory->create(cutsq,n+1,n+1,"pair:cutsq");

  memory->create(cut_lj,n+1,n+1,"pair:cut_lj");
  memory->create(cut_coul,n+1,n+1,"pair:cut_coul");
  memory->create(epsilon,n+1,n+1,"pair:epsilon");
  memory->create(sigma,n+1,n+1,"pair:sigma");
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairPauliCut::settings(int narg, char **arg)
{
  if (narg < 1 || narg > 1)
    error->all(FLERR,"Incorrect args in pair_style command");

  if (strcmp(update->unit_style,"electron") == 0)
    error->all(FLERR,"Cannot (yet) use 'electron' units with dipoles");

  cut_lj_global = force->numeric(FLERR,arg[0]);

  cut_coul_global = P02;                     // Guillermo Frank: p0^2

  // reset cutoffs that have been explicitly set

  if (allocated) {
    int i,j;
    for (i = 1; i <= atom->ntypes; i++)
      for (j = i+1; j <= atom->ntypes; j++)
        if (setflag[i][j]) {
          epsilon[i][j]   = D0;          // Guillermo Frank: d0
          sigma[i][j]     = Q02;         // Guillermo Frank: q0^2
          cut_coul[i][j]  = P02;         // Guillermo Frank: p0^2
          cut_lj[i][j] = cut_lj_global;
        }
  }
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairPauliCut::coeff(int narg, char **arg)
{
  if (narg < 4 || narg > 6)
    error->all(FLERR,"Incorrect args for pair coefficients");
  if (!allocated) allocate();

  int ilo,ihi,jlo,jhi;
  force->bounds(FLERR,arg[0],atom->ntypes,ilo,ihi);
  force->bounds(FLERR,arg[1],atom->ntypes,jlo,jhi);

  /* pair_style itype jtype d0 q0 p0 rc */

  double d0 = D0;
  double q0 = Q02;
  double p0 = P02;
  double cut_one = cut_lj_global;

  if (narg > 2)  d0 = force->numeric(FLERR,arg[2]);
  if (narg > 3)  q0 = force->numeric(FLERR,arg[3]);
  if (narg > 4)  p0 = force->numeric(FLERR,arg[4]);
  if (narg == 6) cut_one = force->numeric(FLERR,arg[5]);
 
  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    for (int j = MAX(jlo,i); j <= jhi; j++) {
      epsilon[i][j]  = d0;                      // Guillermo Frank: d0
      sigma[i][j]    = q0*q0;                   // Guillermo Frank: q0^2
      cut_coul[i][j] = p0*p0;                   // Guillermo Frank: p0^2
      cut_lj[i][j]   = cut_one;                 // Guillermo Frank: rc
      setflag[i][j] = 1;
      count++;
    }
  }

  if (count == 0) error->all(FLERR,"Incorrect args for pair coefficients");
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

void PairPauliCut::init_style()
{
    if (!atom->q_flag || !atom->mu_flag)
    error->all(FLERR,"Pair dipole/cut requires atom attributes q, mu");

  neighbor->request(this,instance_me);
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairPauliCut::init_one(int i, int j)
{

  double d0 = D0;                    // Guillermo Frank
  double q0 = sqrt(Q02);             // Guillermo Frank
  double p0 = sqrt(P02);             // Guillermo Frank
  double cut_one = cut_lj_global;    // Guillermo Frank


  if (setflag[i][j] == 0) {

    epsilon[i][j]  = d0;             // Guillermo Frank: d0
    sigma[i][j]    = q0*q0;          // Guillermo Frank: q0^2
    cut_coul[i][j] = p0*p0;          // Guillermo Frank: p0^2
    cut_lj[i][j]   = cut_one;        // Guillermo Frank: rc
  }

  double cut = cut_lj[i][j];

  return cut;
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairPauliCut::write_restart(FILE *fp)
{
  write_restart_settings(fp);

  int i,j;
  for (i = 1; i <= atom->ntypes; i++)
    for (j = i; j <= atom->ntypes; j++) {
      fwrite(&setflag[i][j],sizeof(int),1,fp);
      if (setflag[i][j]) {
        fwrite(&epsilon[i][j],sizeof(double),1,fp);
        fwrite(&sigma[i][j],sizeof(double),1,fp);
        fwrite(&cut_lj[i][j],sizeof(double),1,fp);
        fwrite(&cut_coul[i][j],sizeof(double),1,fp);
      }
    }
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairPauliCut::read_restart(FILE *fp)
{
  read_restart_settings(fp);

  allocate();

  int i,j;
  int me = comm->me;
  for (i = 1; i <= atom->ntypes; i++)
    for (j = i; j <= atom->ntypes; j++) {
      if (me == 0) fread(&setflag[i][j],sizeof(int),1,fp);
      MPI_Bcast(&setflag[i][j],1,MPI_INT,0,world);
      if (setflag[i][j]) {
        if (me == 0) {
          fread(&epsilon[i][j],sizeof(double),1,fp);
          fread(&sigma[i][j],sizeof(double),1,fp);
          fread(&cut_lj[i][j],sizeof(double),1,fp);
          fread(&cut_coul[i][j],sizeof(double),1,fp);
        }
        MPI_Bcast(&epsilon[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&sigma[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&cut_lj[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&cut_coul[i][j],1,MPI_DOUBLE,0,world);
      }
    }
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairPauliCut::write_restart_settings(FILE *fp)
{
  fwrite(&cut_lj_global,sizeof(double),1,fp);
  fwrite(&cut_coul_global,sizeof(double),1,fp);
  fwrite(&offset_flag,sizeof(int),1,fp);
  fwrite(&mix_flag,sizeof(int),1,fp);
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairPauliCut::read_restart_settings(FILE *fp)
{
  if (comm->me == 0) {
    fread(&cut_lj_global,sizeof(double),1,fp);
    fread(&cut_coul_global,sizeof(double),1,fp);
    fread(&offset_flag,sizeof(int),1,fp);
    fread(&mix_flag,sizeof(int),1,fp);
  }
  MPI_Bcast(&cut_lj_global,1,MPI_DOUBLE,0,world);
  MPI_Bcast(&cut_coul_global,1,MPI_DOUBLE,0,world);
  MPI_Bcast(&offset_flag,1,MPI_INT,0,world);
  MPI_Bcast(&mix_flag,1,MPI_INT,0,world);
}
