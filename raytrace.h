//////////////////////////////////////////////////////////////////////
// header guard:
//////////////////////////////////////////////////////////////////////
#ifndef HEADER_GUARD_FOR_RAYTRACE_H
#define HEADER_GUARD_FOR_RAYTRACE_H

//////////////////////////////////////////////////////////////////////
// libraries:
//////////////////////////////////////////////////////////////////////
#include <fftw3.h>
#include <mpi.h>
#include <hdf5.h>

#include "profile.h"
#include "healpix_utils.h"
#include "healpix_shtrans.h"

#ifdef MEMWATCH
#include "memwatch.h"
#endif

#ifdef USEMEMCHECK
#include <memcheck.h>
#endif

#ifdef DMALLOC
#include <dmalloc.h>
#endif

//////////////////////////////////////////////////////////////////////
// other:
//////////////////////////////////////////////////////////////////////

#define RAYTRACEVERSION "CALCLENS v1.0"

/* Some debugging macros
   undef DEBUG for no debugging
   DEBUG_LEVEL = 0 is for basic debugging
   DEBUG_LEVEL = 1 is for messages printed by a single task but not as critical
   DEBUG_LEVEL = 2 or above is used for messages that every task will print

   define DEBUG_IO some output as follows
   1) activecells%d.dat - list of ring inds and nside vals for all active cells for each task (task # put into %d by printf)
   2) buffcells%d.dat - list of ring inds and nside vals for all buffer cells for each task (task # put into %d by printf)
   3) lists of inds from particle read: indget.dat are ring inds to read and indgetfile.dat are ring inds to read from file
*/

#ifdef NDEBUG
#undef DEBUG
#define DEBUG_LEVEL -1
#undef DEBUG_IO
#endif

#ifdef DEBUG
#ifndef DEBUG_LEVEL
#define DEBUG_LEVEL 0
#endif
#endif

#define MAX_FILENAME 1024

//tags for MPI send/recvs
#define TAG_NUMDATA_P2R       10
#define TAG_DATA_P2R_NORTH    11
#define TAG_DATA_P2R_SOUTH    12
#define TAG_NUMDATA_R2P       13
#define TAG_DATA_R2P          14
#define TAG_NUMNEST_R2P       15
#define TAG_NEST_R2P          16
#define TAG_NUMBUFF_R2P       17
#define TAG_BUFF_R2P          18

#define TAG_NUMNEST_GBR       19
#define TAG_NEST_GBR          20
#define TAG_NUMBUFF_GBR       21
#define TAG_BUFF_GBR          22

#define TAG_NUMBUFF_SOURCEGAL 23
#define TAG_BUFF_SOURCEGAL    24
#define TAG_RAYIO_TOTNUM      25
#define TAG_RAYIO_NUMCHUNK    26
#define TAG_RAYIO_CHUNKDATA   27 /* tags 27 - 37 are used in file_write_rays2fits in io.c */
#define TAG_GALSIO_TOTNUM     38
#define TAG_GALSIO_NUMCHUNK   39
#define TAG_GALSIO_CHUNKDATA  40 /* tags 40 - 46 are used in file_write_gals2fits in galsio.c */
#define TAG_NUMBUFF_PIO       47
#define TAG_BUFF_PIO          48
#define TAG_NUMPBUFF_PIO      49
#define TAG_PBUFF_PIO         50

#define TAG_NESTIND_GBR       51

#define TAG_NUMBUFF_LOADBAL   52
#define TAG_BUFFIND_LOADBAL   53
#define TAG_BUFF_LOADBAL      54

#define TAG_NUMBUFF_GALSDIST  55
#define TAG_BUFF_GALSDIST     56

#define TAG_DENS_NUM          57
#define TAG_DENS_RED          58

#define TAG_POTCELL_NUM       60
#define TAG_POTCELL_IDS       61
#define TAG_POTCELL_VALS      62

//constants
#define RHO_CRIT 2.77519737e11  /* Critial mass density in h^2 M_sun/Mpc^3 with H_{0} = h 100 km/s/Mpc*/
#define CSOL 299792.458         /* velocity of light in km/s */

/*
  MG patch and grid search options:

  GRIDSEARCH_RADIUS_ARCMIN - radius within which to search for galaxy images in arcminutes
  MIN_SMOOTH_TO_RAY_RATIO - minimum smoothing length is set to this factor times the ray grid size
  RAYBUFF_RADIUS_ARCMIN - radius used to get buffer rays ofr galaxy image search - should be maximum deflection code could ever see
  MGPATCH_SIZE_FAC - the size of the multigrid patch is set MGPATCH_SIZE_FAC*sqrt(4.0*M_PI/order2npix(rayTraceData.bundleOrder))
  NUM_MGPATCH_MIN - minimum number of cells to use for a MG patch on one side
  SMOOTHKERN_MGRESOLVE_FAC - # of resolution elements used to grid up particles for MG patches
  SMOOTHKERN_SHTRESOLVE_FAC - # of resolution elements used to grid up particles for SHT density
*/
#define GRIDSEARCH_RADIUS_ARCMIN   2.5
#define MIN_SMOOTH_TO_RAY_RATIO    0.5 //was 2
#define RAYBUFF_RADIUS_ARCMIN      10.0
#define MGPATCH_SIZE_FAC           4.0
#define NUM_MGPATCH_MIN            256
#define SMOOTHKERN_MGRESOLVE_FAC   3.0 //was 4
#define SMOOTHKERN_SHTRESOLVE_FAC  3.0 //was 4

/* macros for bit flags */
#define SETBITFLAG(x,b) ((x) |= (1 << (b)))
#define CLEARBITFLAG(x,b) ((x) &= (~(1 << (b))))
#define ISSETBITFLAG(x,b) ((x) & (1 << (b)))
#define PRIMARY_BUNDLECELL                       0    //primary domain decomp cells
#define PARTBUFF_BUNDLECELL                      1    //cells with buffer particles
#define MAPBUFF_BUNDLECELL                       2    //cells with map buffer cells - internal flag
#define RAYBUFF_BUNDLECELL                       3    //cells with buffer rays for grid search
#define FULLSKY_PARTDIST_PRIMARY_BUNDLECELL      4    //primary domain cells for sep. full sky density
#define FULLSKY_PARTDIST_MAPBUFF_BUNDLECELL      5    //map buffer cells for full sky density
#define NON_FULLSKY_PARTDIST_MAPBUFF_BUNDLECELL  6    //map buffer cells for usual running - not an internal flag
#define GRIDKAPPADENS_MAPBUFF_BUNDLECELL         7    //map buffer cells for gridding up particles in sep. kappa dens

typedef struct {
  //params in config file
  double WallTimeLimit;
  double WallTimeBetweenRestart;
  double OmegaM;
  double maxComvDistance;
  long NumLensPlanes;
  char LensPlanePath[MAX_FILENAME];
  char LensPlaneName[MAX_FILENAME];
  char LensPlaneType[MAX_FILENAME];
  char HEALPixLensPlaneMapPath[MAX_FILENAME];
  char HEALPixLensPlaneMapName[MAX_FILENAME];
  long HEALPixLensPlaneMapOrder;
  char OutputPath[MAX_FILENAME];
  char RayOutputName[MAX_FILENAME];
  long NumRayOutputFiles;
  long NumFilesIOInParallel;
  long bundleOrder;
  long rayOrder;
  double minRa;
  double maxRa;
  double minDec;
  double maxDec;
  double maxRayMemImbalance; /* controls max mem imbalance when trying to load balance CPU time for rays */
  char HEALPixRingWeightPath[MAX_FILENAME];
  char HEALPixWindowFunctionPath[MAX_FILENAME];
  long SHTOrder;
  double ComvSmoothingScale;
  double partMass;
  long NFFT;
  long MaxNFFT;
  char ThreeDPotSnapList[MAX_FILENAME];
  double LengthConvFact;

  /* for doing gals image search */
  char GalsFileList[MAX_FILENAME];
  char GalOutputName[MAX_FILENAME];
  long NumGalOutputFiles;

  /* for doing lensing maps */
  char MapRedshiftList[MAX_FILENAME];
  long CMBLensing;
  long MaxResMap;
  //internal params for code
  long Restart; /* set to index of ray plane to start with if you want to restart*/
  long CurrentPlaneNum;
  long CurrentMapNum;
  long poissonOrder;                  //order of SHT poisson solve step
  double galImageSearchRad;           //radius within which to search for galaxy images
  double galImageSearchRayBufferRad;  //radius within which to get rays from other processors to do grid search
  double partBuffRad;                 //radius within which to read parts and get buffer regions
  double minSL;
  double maxSL;
  double densfact;
  double backdens;
  double planeRadMinus1;
  double planeRad;
  double planeRadPlus1;
  long NumMGPatch;
  double minComvSmoothingScale;
  double maxComvSmoothingScale;
  double MGConvFact;
  long UseHEALPixLensPlaneMaps;
} RayTraceData;

// 64 bytes
typedef struct {
  long order;
  long nest;
  double n[3];
  long startRay;
  long over;
  long down;
} HEALPixTreeNode;

//only make one
typedef struct {
  long NumNodes;
  HEALPixTreeNode *nodes;
  long *links;
  double nodeArcSize[HEALPIX_UTILS_MAXORDER+1];
} HEALPixTreeData;

// 16 bytes
typedef struct {
  long ind;
  double cosrad;
} NNbrData;

//40 bytes
#define NFIELDS_LCPARTICLE ((hsize_t) 8)
typedef struct {
  long partid;
  float px;
  float py;
  float pz;
  float vx;
  float vy;
  float vz;
  float mass;
} LCParticle;

//36 bytes
typedef struct {
  float pos[3];
  float mass;
  float smoothingLength;
  float cosSmoothingLength;
  long nest;
  double r;
} Part;

//20 bytes
typedef struct {
  float pos[3];
  long index;
} SourceGal;

//56 bytes
typedef struct {
  double ra;
  double dec;
  double A00;
  double A01;
  double A10;
  double A11;
  long index;
} ImageGal;

typedef struct {
  long nest;
  double ra;
  double dec;
  double A00;
  double A01;
  double A10;
  double A11;
  double kappa;
} ImageCell;

// 176 bytes
typedef struct {
  long nest;
  double n[3];      //ray location
  double beta[3];   //ray direction
  double alpha[2];  //deflection angle mags in (theta,phi) dirs at ray loc n (lens pot first derivs)
  double A[4];      //inv mag mat at ray
  double Aprev[4];  //prev. inv mag mat at ray
  double U[4];      //lens pot second derivs at ray
  double phi;       //lens pot at ray
} HEALPixRay;

// 12 bytes
typedef struct {
  float val;
  long index;
} HEALPixMapCell;

// 52 bytes
typedef struct {
  long nest;
  unsigned int active;
  long Nparts;
  long firstPart;
  long Nrays;
  HEALPixRay *rays;
  long firstMapCell;
  double cpuTime;
} HEALPixBundleCell;

/* extern defs of global vars in globalvars.c */
extern const char *ProfileTagNames[];
extern RayTraceData rayTraceData;                        /* global struct with all vars from config file */
extern long NbundleCells;                                /* the number of bundle cells used for overall domain decomp */
extern HEALPixBundleCell *bundleCells;                   /* the vector of bundle cells used for domain decomp */
extern long *bundleCellsNest2RestrictedPeanoInd;         /* a global "hash" of bundle nest indexes that defines the domain covered by
							    rays in terms of a coniguous space-filling index set */
extern long *bundleCellsRestrictedPeanoInd2Nest;         /* inverse of bundleCellsNest2RestrictedPeanoInd */
extern long NrestrictedPeanoInd;                         /* number of restricted peano inds == total number of bundle cells with rays */
extern long *firstRestrictedPeanoIndTasks;               /* array which holds first index of section of RestrictedPeanoInds assigned to each MPI task
							    - this forms the domain decomp */
extern long *lastRestrictedPeanoIndTasks;                /* array which holds last index of section of RestrictedPeanoInds assigned to each MPI task
							    - this forms the domain decomp */
extern HEALPixRay *AllRaysGlobal;                       /* pointer to memory location of all rays */
extern long MaxNumAllRaysGlobal;                        /* maximum number of rays that can be stored */
extern long NumAllRaysGlobal;                           /* current number of rays stored */
extern int ThisTask;                                     /* this task's rank in MPI_COMM_WORLD */
extern int NTasks;                                       /* number of tasks in MPI_COMM_WORLD */
extern long NlensPlaneParts;                             /* number of particles in lens plane for this task */
extern Part *lensPlaneParts;                             /* vector of particles for this task */
extern SourceGal *SourceGalsGlobal;                      /* source gals for task */
extern long NumSourceGalsGlobal;                         /* # of gals in global vecs */
extern ImageGal *ImageGalsGlobal;                        /* images of source gals for task */
extern long NumImageGalsGlobal;                          /* # of gals in global vecs */

extern long NmapCells;
extern HEALPixMapCell *mapCells;
extern HEALPixMapCell *mapCellsGradTheta;
extern HEALPixMapCell *mapCellsGradPhi;
extern HEALPixMapCell *mapCellsGradThetaTheta;
extern HEALPixMapCell *mapCellsGradThetaPhi;
extern HEALPixMapCell *mapCellsGradPhiPhi;

/* in poissondrivers.c */
void fullsky_partdist_poissondriver(void);
void cutsky_partdist_poissondriver(void);

/* in mgpoissonsolve.c */
void mgpoissonsolve(double densfact, double backdens);

/* loadbalance.c */
void load_balance_tasks(void);
void getDomainDecompPerCPU(int report);

/* in shtpoissonsolve.c */
void do_healpix_sht_poisson_solve(double densfact, double backdens);

/* in partsmoothdens.c */
double spline_part_dens(double cosr, double sigma);
void get_smoothing_lengths(void);

/* in map_shuffle.c */
void healpixmap_ring2peano_shuffle(float **mapvec_in, HEALPixSHTPlan plan);
void healpixmap_peano2ring_shuffle(float *mapvec, HEALPixSHTPlan plan);

/* in rot_paratrans.c */
void generate_rotmat_axis_angle_countercw(double axis[3], double angle, double rotmat[3][3]);
void generate_rotmat_axis_angle_cw(double axis[3], double angle, double rotmat[3][3]);
void rot_vec_axis_angle_countercw(double vec[3], double rvec[3], double axis[3], double angle);
void rot_vec_axis_angle_cw(double vec[3], double rvec[3], double axis[3], double angle);
void paratrans_tangvec(double tvec[2], double vec[3], double rvec[3], double rtvec[2]);
void paratrans_tangtensor(double ttensor[2][2], double vec[3], double rvec[3], double rttensor[2][2]);
void rot_vec_axis_trigangle_countercw(double vec[3], double rvec[3], double axis[3], double cosangle, double sinangle);
void rot_ray_ang2radec(HEALPixRay *ray);
void rot_ray_radec2ang(HEALPixRay *ray);
void paratrans_ray_curr2obs(HEALPixRay *ray);
void paratrans_ray_obs2curr(HEALPixRay *ray);

/* in rayio.c */
void write_rays(void);

/* in partio.c */
/* in read_lensplanes_hdf5.c */
void readRayTracingPlaneAtPeanoInds(long planeNum, long HEALPixOrder, long *PeanoIndsToRead, long NumPeanoIndsToRead, Part **LCParts, long *NumLCParts);
void read_lcparts_at_planenum_all(long planeNum);
void read_lcparts_at_planenum_fullsky_partdist(long planeNum);
void read_lcparts_at_planenum(long planeNum);

/* in cosmocalc.h - distances - assumes flat lambda */
void init_cosmocalc(void);
double comvdist_integ_funct(double a, void *p);
double angdist(double a);
double comvdist(double z);
double angdistdiff(double amin, double amax);
double acomvdist(double dist);

/* in config.c */
void read_config(char *filename);

/* in raytrace.c */
void raytrace(void);

/* in healpix_fastdiscquery.c */
long query_disc_inclusive_nest_fast(double theta, double phi, double radius, long **listpix, long *NlistpixMax, long queryOrder);

/* in raytrace_utils.c */
void write_bundlecells2ascii(char fname_base[MAX_FILENAME]);
void mark_bundlecells(double mapbuffrad, int searchTag, int markTag);
void alloc_mapcells(int searchTag, int markTag);
void free_mapcells(void);
int test_vaccell_boundary(double ra, double dec, double radius);
int test_vaccell(double ra, double dec);
void alloc_rays(void);
void init_rays(void);
void destroy_rays(void);
void init_bundlecells(void);
void destroy_bundlecells(void);
void destroy_gals(void);
void destroy_parts(void);

/* in ioutils.c */
int strcmp_caseinsens(const char *s1, const char *s2);
void getPeanoIndsToReadFromFile(long HEALPixOrder, long *PeanoIndsToRead, long NumPeanoIndsToRead,
                                long FileHEALPixOrder, long **FilePeanoIndsToRead, long *NumFilePeanoIndsToRead);
long fnumlines(FILE *fp);
FILE *fopen_retry(const char *filename, const char *mode);

/* in rayprop.c */
void rayprop_sphere(double wp, double wpm1, double wpm2, long bundleCellInd);

/* in gridsearch.c */
void gridsearch(double wpm1, double wpm2);

/* in galsio.c */
void write_gals2fits(void);
void read_fits2gals(void);
void reorder_gals_nest(SourceGal *buffSgs, long NumBuffSgs);
int reorder_gals_for_tasks(long NumBuffGals, SourceGal *buffGals, int *sendCounts);
void reorder_gals_for_planes(void);

/* in nnbrs_healpixtree.c */
long nnbrsHEALPixTree(double n[3], double radius, double cmvRad, HEALPixRay *rays, HEALPixTreeData *td, NNbrData **NNbrs, long *maxNumNNbrs);
HEALPixTreeData *buildHEALPixTree(long Nrays, HEALPixRay *rays);
void destroyHEALPixTree(HEALPixTreeData *td);

/* in restart.c */
void read_restart(void);
void write_restart(void);
void write_rays(long mapnum);
void clean_gals_restart(void);

/* in fftpoissondriver.c */
void threedpot_poissondriver(void);

#ifdef PROPAGATE_TO_CMB_FROM_RESTART
/* in maputils.c */
void getNMaps(long* NMaps);
void getMapLensPlaneNums(int* lp_map, int NMaps);
void updateLensMap(HEALPixBundleCell *bundleCell, const long map_order,
                  long* map_pixel_sum_1, double *map_pixel_sum_A00, double *map_pixel_sum_A01, double *map_pixel_sum_A10,
                  double *map_pixel_sum_A11, double *map_pixel_sum_ra, double *map_pixel_sum_dec);

void MPI_ReduceLensMap(long *map_pixel_sum_1, double *map_pixel_sum_A00, double *map_pixel_sum_A01, double *map_pixel_sum_A10,
                       double *map_pixel_sum_A11, double *map_pixel_sum_ra, double *map_pixel_sum_dec,
                       const long map_n_pixels, const int root);

void writeFITSHEALPixLensMap(long *map_pixel_sum_1, double *map_pixel_sum_A00, double *map_pixel_sum_A01, double *map_pixel_sum_A10,
                  double *map_pixel_sum_A11, double *map_pixel_sum_ra, double *map_pixel_sum_dec,
                  const long map_n_pixels, const char *filename);

void writeSingleFITSHEALPixLensMap(const float *signal, long nside, const char *filename);

/* in propagate_to_cmb_from_restart.c */
double flat_LambdaCDM_line_of_sight_comoving_distance_for_redshift(const double z_, const double Omega_matter_);
void propagate_to_cmb_from_restart(void);
#endif /* defined PROPAGATE_TO_CMB_FROM_RESTART */


#endif /* HEADER_GUARD_FOR_RAYTRACE_H */
