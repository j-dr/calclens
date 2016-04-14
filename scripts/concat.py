#!/usr/bin/env python
import sys
import os
import glob
import fitsio
import itertools
import numpy as np
import yaml
from catwriter import CatWriter

# MPI
from mpi4py import MPI
comm = MPI.COMM_WORLD
ThisTask = comm.Get_rank()
NTasks = comm.Get_size()

# utils
def fnumlines(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

def read_yaml(fname):
    with open(fname) as fp:
        cfg = yaml.load(fp)
    return cfg

class CalclensConcat(object):
    """
    class containing scripts to concatenate and reorganize galaxy outputs from calclens
    
    The final output files will have the same names as the input ones except that they 
    will be prefixed by 'lensed_'.
    
    Parameters
    ----------
    conf: config dict (usually read from a yaml file, see below)
    
    Methods
    -------
    go: concat and reorg the files (no parameters)
    
    Example 1
    ---------
    
    # run in python
    conf = {#blah blah}
    cc = CalclensConcat(conf)
    cc.go()
    
    Example 2
    ---------
    
    It can be run from the command line like this
    
    python concat.oy concat.yaml
    
    Config File
    -----------
    A config file has the following keys 
        
    OutputPath: 'directory/for/outputs' # will be cleared so path should be unique
    InputPath: 'directory/of/inputs' # usually a path to outputs of calclens
    InputName: 'basename' # basename of galaxy outputs from calclens (the GalOutputName parameter)
    ConcatOutputName: 'basename' # basename of intermediate outputs from this script
    GalsFileList: 'path/to/gal/file/list' # same as parameter to calclens
    
    """

    def __init__(self,conf):
        self.conf = {}
        self.conf.update(conf)

        # get and broad cast info
        if ThisTask == 0:
            # clean up
            os.system('rm -rf %s' % self.conf['OutputPath'])
            os.system('mkdir -p '+self.conf['OutputPath'])

            # read in names of input gal catalogs
            galcatlist = []
            for line in open(self.conf['GalsFileList'],'r'):
                galcatlist.append(line.strip())

            # get number of lens planes and number of output files
            rname = os.path.join(self.conf['InputPath'],self.conf['InputName'])
            Npr = len(glob.glob(rname+'*.0000.fit')) # num lens planes
            Nfr = len(glob.glob(rname+'0000.*.fit')) # num output files
        else:
            galcatlist = None
            Npr = None
            Nfr = None
        galcatlist,Npr,Nfr = comm.bcast((galcatlist,Npr,Nfr),0)

        ################################
        comm.Barrier()
        ################################

        self.galcatlist = galcatlist
        self.num_lens_planes = Npr
        self.num_output_files = Nfr

    def go(self):
        self.map_to_input_files()
        self.concat(final=True)

    def concat(self,final=False):
        Npw = len(self.galcatlist)
        wpath = self.conf['OutputPath']
        wbase = self.conf['ConcatOutputName']

        ##########################
        comm.Barrier()
        ##########################

        if ThisTask == 0:
            print 'doing concat of files...'

        for fnum in range(Npw):
            if fnum%NTasks == ThisTask:
                snames = glob.glob(os.path.join(wpath,'*task*fnum%d.*.fit' % fnum))
                if len(snames) > 0:
                    try:
                        d = fitsio.read(os.path.join(wpath,wbase+'fnum%d.fit'%fnum))
                        d = list(d)
                    except:
                        d = []
                    for sname in snames:
                        di = fitsio.read(sname)
                        d.extend(list(di))
                    d = np.array(d,dtype=di.dtype.descr)

                    if final:
                        # sort by index
                        q = np.argsort(d['index'])
                        d = d[q]

                        # make new data dtype
                        dt = []
                        for item in d.dtype.descr:
                            dt.append(item)
                        dt.append(('id','i8'))

                        # build and fill
                        dw = np.zeros(len(d),dtype=dt)
                        for tag in d.dtype.names:
                            if tag in dw.dtype.names:
                                dw[tag] = d[tag]
                        del d

                        # get ID
                        gcat = fitsio.read(self.galcatlist[fnum],lower=True,columns=['id'])
                        dw['id'] = gcat['id'][dw['index']]
                        del gcat

                        pth,fname = os.path.split(self.galcatlist[fnum])
                        fitsio.write(os.path.join(self.conf['OutputPath'],'lensed_'+fname),dw,clobber=True)
                        del dw
                    else:
                        fitsio.write(os.path.join(wpath,wbase+'fnum%d.fit'%fnum),d,clobber=True)

                    for sname in snames:
                        os.remove(sname)

        ##########################
        comm.Barrier()
        ##########################

    def map_to_input_files(self):
        if ThisTask == 0:
            print 'mapping to input files:'

        # init writer
        writer = CatWriter(os.path.join(self.conf['OutputPath'],
                                        self.conf['ConcatOutputName']+'task%d_fnum%%d'%ThisTask))

        # do mapping
        # to undo index get fnum = gals.index mod Npw, inum = (gals.index - fnum)/Npw

        #vars to setup paths and files
        wpath = self.conf['OutputPath']
        wbase = self.conf['ConcatOutputName']
        rpath = self.conf['InputPath']
        rbase = self.conf['InputName']
        Npr = self.num_lens_planes
        Nfr = self.num_output_files
        Npw = len(self.galcatlist)
        
        for plane in range(0,Npr):
            if plane%NTasks == ThisTask:
                print "   ",ThisTask,":",plane+1,"of",Npr

                for sec in range(0,Nfr):
                    tail = '%04d.%04d.fit' % (plane,sec)
                    fread = os.path.join(rpath,rbase)+tail
                    if os.path.exists(fread):
                        try:
                            dset = fitsio.read(fread)
                        except IOError,e:
                            if e.message == "No extensions have data":
                                continue
                            else:
                                raise IOError("Weird problem reading FITS file '%s'!" % fread)
                    else:
                        continue

                    fnums = dset['index'] % Npw
                    finds = np.argsort(fnums)
                    dset = dset[finds]
                    fnums = fnums[finds]
                    dset['index'] = (dset['index'] - fnums)/Npw

                    for fnum, grp in itertools.groupby(range(len(dset)),lambda x: fnums[x]):
                        idx = [g for g in grp]
                        if False:
                            q, = np.where(fnums == fnum)
                            assert np.array_equal(q,idx)

                        wdset = dset[idx]
                        writer.add_data(fnum,wdset)

        writer.finalize_data()

if __name__ == '__main__':
    conf = read_yaml(sys.argv[1])
    
    cc = CalclensConcat(conf)
    cc.go()

