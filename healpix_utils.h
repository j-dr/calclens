/* hacked apart healpix base2 functions 
 *  -converted int64 to long to work in C on 64 bit machines 
 *  -Matthew R Becker, Univ. of Chicago 2009
 */

/*
 *  This file is part of Healpix_cxx.
 *
 *  Healpix_cxx is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  Healpix_cxx is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with Healpix_cxx; if not, write to the Free Software
 *  Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
 *
 *  For more information about HEALPix, see http://healpix.jpl.nasa.gov
 */

/*
 *  Healpix_cxx is being developed at the Max-Planck-Institut fuer Astrophysik
 *  and financially supported by the Deutsches Zentrum fuer Luft- und Raumfahrt
 *  (DLR).
 */

/*! \file healpix_base2.h
 *  Copyright (C) 2003, 2004, 2005, 2006 Max-Planck-Society
 *  \author Martin Reinecke
 */

#ifdef MEMWATCH
#include "memwatch.h"
#endif

#ifdef USEMEMCHECK
#include <memcheck.h>
#endif

#ifdef DMALLOC
#include <dmalloc.h>
#endif

#ifndef HEALPIX_UTILS /* HEALPIX_UTILS */
#define HEALPIX_UTILS /* HEALPIX_UTILS */

/*! Functionality related to the HEALPix pixelisation. Supports resolutions up to
    N_side = 2^29. */
#define HEALPIX_UTILS_MAXORDER 29l

long order2nside(long order_);
long order2npix(long order_);
long npix2nside(long npix);
long nside2order(long nside);
long npix2order(long npix);
long nside2npix(long nside);

void nest2xyf(long pix, long *ix, long *iy, long *face_num, long order_);
long xyf2nest(long ix, long iy, long face_num, long order_);
void nest2ang(long pix, double *theta, double *phi, long order_);
void nest2vec(long pix, double *vec, long order_);
long ang2nest(double theta, double phi, long order_);
long vec2nest(double *vec, long order_);

void ring2xyf(long pix, long *ix, long *iy, long *face_num, long order_);
long xyf2ring(long ix, long iy, long face_num, long order_);
void ring2ang(long pix, double *theta, double *phi, long order_);
void ring2vec(long pix, double *vec, long order_);
long ang2ring(double theta, double phi, long order_);
long vec2ring(double *vec, long order_);

void getneighbors_nest(long pix, long *result, long order_);
void getneighbors_ring(long pix, long *result, long order_);
long nest2peano(long pix, long order_);
long peano2nest(long pix, long order_);
long nest2ring(long pix, long order_);
long ring2nest(long pix, long order_);

long ring_above(double z, long order_);
void get_interpol(double theta, double phi, long pix[4], double wgt[4], long order_);

void get_ring_info2(long ring, long *startpix, long *ringpix, double *costheta, double *sintheta, long *shifted, long order_);
long ring2ringnum(long ringind, long order_);

void tablefiller(void);

long isqrt(long i);
long ilog2(long i);
long imodulo(long v1, long v2);
long ifloor(double arg);

void vec2ang(double vec[3], double *theta, double *phi);
void ang2vec(double vec[3], double theta, double phi);
void vec2radec(double vec[3], double *ra, double *dec);
void ang2radec(double theta, double phi, double *ra, double *dec);
void radec2ang(double *theta, double *phi, double ra, double dec);

long alm2index(long l, long m);
void index2alm(long almindex, long *l, long *m);
long num_alms(long l);

void in_ring_realloc(long **listir, long *Nlistir, long Nextra);
void in_ring(long iz, double phi0, double dphi, long **listir, long *Nlistir, long order_);
void query_disc_inclusive_nest(double theta, double phi, double radius, long **listpix, long *Nlistpix, long order_);

//functions added by Matthew Becker, U of C 2011
int ring2triangle(long ring, long tri[4][3], long order);
long get_healpix_finitediff_info(double theta, double phi, long pixnest, double mabinv[2][2], double thetadiff[8], double phidiff[8], long nest[8], long order);

//interpolation for healpix using triangles
void get_interp_triangle(double theta, double phi, long ring[3], double wgt[3], long order);
int get_interp_polygon(double theta, double phi, long ringpix[8], double wgt[8], long order);

#endif /* HEALPIX_UTILS */
