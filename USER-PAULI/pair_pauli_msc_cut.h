/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS

PairStyle(pauli/msc/cut,PairPauliMscCut)

#else

#ifndef LMP_PAIR_PAULI_MSC_CUT_H
#define LMP_PAIR_PAULI_MSC_CUT_H

#include "pair.h"

namespace LAMMPS_NS {

class PairPauliMscCut : public Pair {
 public:
  PairPauliMscCut(class LAMMPS *);
  virtual ~PairPauliMscCut();
  virtual void compute(int, int);
  void settings(int, char **);
  void coeff(int, char **);
  void init_style();
  double init_one(int, int);
  void write_restart(FILE *);
  void read_restart(FILE *);
  void write_restart_settings(FILE *);
  void read_restart_settings(FILE *);

 protected:
  double cut_lj_global,cut_coul_global,cut_pandha_global;    // Guillermo Frank: cut_pandha_global
  double **epsilon,**sigma,**cut_lj,**cut_coul,**cut_pandha; // Guillermo Frank: typical cut_pandha = 5.4 fm
 
  void allocate();
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Incorrect args in pair_style command

Self-explanatory.

E: Cannot (yet) use 'electron' units with dipoles

This feature is not yet supported.

E: Incorrect args for pair coefficients

Self-explanatory.  Check the input script or data file.

E: Pair pauli/msc/cut requires atom attributes q, mu, torque

The atom style defined does not have these attributes.

*/
