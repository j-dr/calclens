# compile time options
#OPTS += -DBORNAPPRX #ray trace with born approximation
#OPTS += -DOUTPUTRAYDEFLECTIONS #output ray deflections
#OPTS += -DOUTPUTPHI #output lensing potential at ray position
OPTS += -DUSE_FITS_RAYOUT #set to use fits for writing rays
OPTS += -DUSE_FULLSKY_PARTDIST #set to tell the code to use a full sky particle distribution in the SHT step 
OPTS += -DSHTONLY #set to only use SHT for lensing
#OPTS += -DTHREEDPOT #define to use 3D potential to move rays

#testing options
#OPTS += -DNFWHALOTEST #define to write lensplanes and do test with an NFW halo - need POINTMASSTEST defined as well 
#OPTS += -DPOINTMASSTEST #define to write lensplanes and do a point mass test
#OPTS += -DKEEP_RAND_FRAC -DRAND_FRAC_TO_KEEP=0.015625 #define to keep a random fraction of particles

#!!! DO NOT CHANGE THESE UNLESS YOU ARE AN EXPERT !!!
#OPTS += -DDOUBLEFFTW #define to use double FFTW for 3D pot
#OPTS += -DNOBACKDENS #define to not subtract a background density from kappa grid - use for point mass test or NFW halo test
#OPTS += -DDEBUG_IO #define for some debugging I/O
#OPTS += -DDEBUG_IO_DD #output debug info for domain decomp
#OPTS += -DDEBUG -DDEBUG_LEVEL=2 #leave undefined for no debugging - 0,1, and 2 give progressively more output to stderr
#OPTS += -DTEST_CODE #define to run some basic test code
#OPTS += -DMEMWATCH -DMEMWATCH_STDIO #define to test for memory leaks, out of bounds, etc. for memory used in this code
#OPTS += -DUSEMEMCHECK #define to test for memory leaks, out of bounds, etc. for memory used in this code
#OPTS += -DDMALLOC -DDMALLOC_FUNC_CHECK #define to test for memory leaks, out of bounds, etc. for memory used in this code
#OPTS += -DDEF_GSL_IEEE_ENV #define the GSL IEEE environment variables - for debugging
OPTS += -DNGPSHTDENS #define to use NGP interp for SHT step
#OPTS += -DCICSHTDENS #define to use CIC interp for SHT step

#select your computer
COMP="orange"
#COMP="orion-gcc"
#COMP="midway"
#COMP="home"

################################
#edit/add to match your machine
#################################

#defaults if you need them
CC          =  mpicc
OPTIMIZE    =  -g -O0 #-Wall -wd981 #-wd1419 -wd810

ifeq ($(COMP),"home")
CC          =  mpicc
#EXTRACFLAGS =  -I/opt/local/include
#EXTRACLIB   =  -L/opt/local/lib
endif

ifeq ($(COMP),"orange")
CC          =  mpicc
OPTIMIZE    =  -g -O3 -Wall #-wd981 #-wd1419 -wd810
GSLI        =  -I$(SLAC_GSL_DIR)/include
GSLL        =  -L$(SLAC_GSL_DIR)/lib
FFTWI       =  -I$(MATTS_FFTW3_DIR)/include 
FFTWL       =  -L$(MATTS_FFTW3_DIR)/lib
HDF5I       =  -I$(SLAC_HDF5_DIR)/include 
HDF5L       =  -L$(SLAC_HDF5_DIR)/lib
FITSI       =  -I$(SLAC_CFITSIO_DIR)/include
FITSL       =  -L$(SLAC_CFITSIO_DIR)/lib
EXTRACFLAGS =
EXTRACLIB   =
endif

ifeq ($(COMP),"midway")
CC          =  mpicc
OPTIMIZE    =  -g -O3 #-Wall -wd981 #-wd1419 -wd810
EXTRACFLAGS =  -Wall -W -Wmissing-prototypes -Wstrict-prototypes -Wshadow -Wpointer-arith -Wcast-qual -Wcast-align 
	-Wwrite-strings -Wnested-externs -fshort-enums -fno-common -Dinline= #-Wconversion
EXTRACLIB   =  
endif

ifeq ($(COMP),"orion-gcc")
CC          =  mpicc
OPTIMIZE    =  -g -O3 -fno-math-errno #-Werror
GSLL        =  -lgsl -lgslcblas
FFTWL       =  -lfftw3 -lfftw3f
HDF5L       =  -lz -lhdf5_hl -lhdf5
FITSL       =  -lcfitsio
EXTRACFLAGS =  -I/home/beckermr/include #-Wall -W -Wmissing-prototypes -Wstrict-prototypes -Wconversion -Wshadow -Wpointer-arith -Wcast-qual -Wcast-align \
	-Wwrite-strings -Wnested-externs -fshort-enums -fno-common -Dinline=
EXTRACLIB   = -L/home/beckermr/lib
endif

#set it all up
ifeq (NFWHALOTEST,$(findstring NFWHALOTEST,$(OPTS)))
OPTS += -DNOBACKDENS -DPOINTMASSTEST
endif

ifeq (POINTMASSTEST,$(findstring POINTMASSTEST,$(OPTS)))
OPTS += -DNOBACKDENS
endif

ifneq (DOUBLEFFTW,$(findstring DOUBLEFFTW,$(OPTS)))
FFTWLIBS = -lfftw3f_mpi  
else
FFTWLIBS = -lfftw3_mpi -lfftw3
endif

CLINK=$(CC)
CFLAGS=$(OPTIMIZE) $(FFTWI) $(HDF5I) $(FITSI) $(GSLI) $(EXTRACFLAGS) $(OPTS)
CLIB=$(EXTRACLIB) $(FFTWL) $(HDF5L) $(FITSL) $(GSLL) -lgsl -lgslcblas $(FFTWLIBS) -lfftw3f -lz -lhdf5_hl -lhdf5  -lcfitsio -lm

ifeq (MEMWATCH,$(findstring MEMWATCH,$(CFLAGS)))
MEMWATCH=memwatch.o
endif

ifeq (USEMEMCHECK,$(findstring USEMEMCHECK,$(CFLAGS)))
CLIB += -lmemcheck
endif

ifeq (DMALLOC,$(findstring DMALLOC,$(CFLAGS)))
CLIB += -ldmalloc
endif

ifeq (TEST_CODE,$(findstring TEST_CODE,$(CFLAGS)))
TESTCODE=test_code.o
endif

OBJS = $(MEMWATCH) $(TESTCODE) raytrace.o raytrace_utils.o healpix_utils.o config.o profile.o globalvars.o cosmocalc.o healpix_fastdiscquery.o \
	read_lensplanes_hdf5.o rayio.o partio.o rayprop.o \
	galsio.o restart.o rot_paratrans.o nnbrs_healpixtree.o \
	healpix_plmgen.o healpix_shtrans.o shtpoissonsolve.o map_shuffle.o alm2map_transpose_mpi.o partsmoothdens.o \
	gridsearch.o loadbalance.o alm2allmaps_transpose_mpi.o map2alm_transpose_mpi.o mgpoissonsolve.o mgpoissonsolve_utils.o \
	poissondrivers.o fftpoissonsolve.o inthash.o ioutils.o lgadgetio.o fftpoissondriver.o \
	gridcellhash.o read_lensplanes_pixLC.o 

EXEC = raytrace
TEST = raytrace
all: $(EXEC) 
test: $(TEST)

OBJS1=$(OBJS) main.o
$(EXEC): $(OBJS1)
	$(CLINK) $(CFLAGS) -o $@ $(OBJS1) $(CLIB)

$(OBJS1): healpix_shtrans.h healpix_utils.h profile.h inthash.h fftpoissonsolve.h \
	raytrace.h mgpoissonsolve.h lgadgetio.h gridcellhash.h read_lensplanes_hdf5.h \
	read_lensplanes_pixLC.h \
	Makefile

.PHONY : clean
clean: 
	rm -f *.o

.PHONY : spotless
spotless: 
	rm -f *.o $(EXEC) $(TEST)

.PHONY : pristine
pristine:
	rm -f *.o $(EXEC) $(TEST) *~

