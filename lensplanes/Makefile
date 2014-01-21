# compile time options
#OPTS += -DNFWHALOTEST #define to write lensplanes and do test with an NFW halo - need POINTMASSTEST defined as well 
#OPTS += -DPOINTMASSTEST #define to write lensplanes and do a point mass test

#!!! DO NOT CHANGE THESE UNLESS YOU ARE AN EXPERT !!!
#OPTS += -DDEBUG_IO #define for some debugging I/O
#OPTS += -DDEBUG_IO_DD #output debug info for domain decomp
#OPTS += -DDEBUG -DDEBUG_LEVEL=0 #leave undefined for no debugging - 0,1, and 2 give progressively more output to stderr
#OPTS += -DTEST_CODE #define to run some basic test code
#OPTS += -DMEMWATCH -DMEMWATCH_STDIO #define to test for memory leaks, out of bounds, etc. for memory used in this code
#OPTS += -DUSEMEMCHECK #define to test for memory leaks, out of bounds, etc. for memory used in this code
#OPTS += -DDMALLOC -DDMALLOC_FUNC_CHECK #define to test for memory leaks, out of bounds, etc. for memory used in this code
#OPTS += -DDEF_GSL_IEEE_ENV #define the GSL IEEE environment variables - for debugging

#select your computer
#COMP="orange"
#COMP="orion-gcc"
#COMP="midway"
COMP="home"

################################
#edit/add to match your machine
#################################

#defaults if you need them
CC          =  mpicc
OPTIMIZE    =  -g -O3 #-Wall -wd981 #-wd1419 -wd810

ifeq ($(COMP),"home")
CC          =  openmpicc
EXTRACFLAGS = -I/opt/local/include
EXTRACLIB   = -L/opt/local/lib
endif

ifeq ($(COMP),"orange")
CC          =  mpicc
OPTIMIZE    =  -g -O3 #-Wall -wd981 #-wd1419 -wd810
GSLI        =  -I/afs/slac.stanford.edu/g/ki/software/gsl/1.15/include
GSLL        =  -L/afs/slac.stanford.edu/g/ki/software/gsl/1.15/lib
HDF5I       =  -I/afs/slac.stanford.edu/g/ki/software/hdf5/1.8.8/include 
HDF5L       =  -L/afs/slac.stanford.edu/g/ki/software/hdf5/1.8.8/lib
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
HDF5L       =  -lz -lhdf5_hl -lhdf5
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

CLINK=$(CC)
CFLAGS=$(OPTIMIZE) $(FFTWI) $(HDF5I) $(FITSI) $(GSLI) $(EXTRACFLAGS) $(OPTS)
CLIB=$(EXTRACLIB) $(FFTWL) $(HDF5L) $(FITSL) $(GSLL) -lgsl -lgslcblas -lz -lhdf5_hl -lhdf5 -lm

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

OBJS = $(MEMWATCH) $(TESTCODE) config.o ../cosmocalc.o globalvars.o \
	../healpix_utils.o ioutils.o lightconeio.o make_lensplanes_hdf5.o \
	make_lensplanes_pointmass_test.o makemaps.o profile.o \
	../read_lensplanes_hdf5.o

EXEC = makelensplanes
TEST = makelensplanes
all: $(EXEC) 
test: $(TEST)

OBJS1=$(OBJS) main.o
$(EXEC): $(OBJS1)
	$(CLINK) $(CFLAGS) -o $@ $(OBJS1) $(CLIB)

$(OBJS1): ../healpix_utils.h  profile.h  raytrace.h Makefile

.PHONY : clean
clean: 
	rm -f *.o

.PHONY : spotless
spotless: 
	rm -f *.o $(EXEC) $(TEST)

.PHONY : pristine
pristine:
	rm -f *.o $(EXEC) $(TEST) *~

