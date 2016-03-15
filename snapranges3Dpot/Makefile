# compile time options
#OPTS += -DBORNAPPRX #ray trace with born approximation
OPTS += -DOUTPUTRAYDEFLECTIONS #output ray deflections
OPTS += -DOUTPUTPHI #output lensing potential at ray position
OPTS += -DUSE_FITS_RAYOUT #set to use fits for writing rays
#OPTS += -DUSE_FULLSKY_PARTDIST #set to tell the code to use a full sky particle distribution in the SHT step 
#OPTS += -DSHTONLY #set to only use SHT for lensing
OPTS += -DTHREEDPOT #define to use 3D potential to move rays

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
CC          =  mpicc
#EXTRACFLAGS =  -I/opt/local/include
#EXTRACLIB   =  -L/opt/local/lib
endif

ifeq ($(COMP),"orange")
CC          =  mpicc
OPTIMIZE    =  -g -O3 #-Wall -wd981 #-wd1419 -wd810
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

OBJS = config.o globalvars.o cosmocalc.o lgadgetio.o ioutils.o

EXEC = snapranges3Dpot
all: $(EXEC) 

OBJS1=$(OBJS) main.o
$(EXEC): $(OBJS1)
	$(CLINK) $(CFLAGS) -o $@ $(OBJS1) $(CLIB)

$(OBJS1): raytrace.h ../lgadgetio.h Makefile

.PHONY : clean
clean: 
	rm -f *.o

.PHONY : spotless
spotless: 
	rm -f *.o $(EXEC) $(TEST)

.PHONY : pristine
pristine:
	rm -f *.o $(EXEC) $(TEST) *~

