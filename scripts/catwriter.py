#!/usr/bin/env python
import fitsio
import numpy as np

class CatWriter(object):
    def __init__(self,bname,Nmax=200000):
        self.base_name = bname
        self.Nmax = Nmax

        self.data = {}
        self.ndata = {}
        self.dtypes = {}
        self.nwrit = {}

    def _init_dnum(self,dnum):
        self.nwrit[dnum] = 0
        self.dtypes[dnum] = None
        self.ndata[dnum] = 0
        self.data[dnum] = []

    def _write_data(self,dnum,finalize=False):
        Nchunks = self.ndata[dnum]/self.Nmax
        if finalize:
            if Nchunks*self.Nmax < self.ndata[dnum]:
                Nchunks += 1

        for chunk in range(Nchunks):
            start = 0
            stop = self.Nmax
            if stop > len(self.data[dnum]):
                stop = len(self.data[dnum])
            dw = self.data[dnum][start:stop]
            dw = np.array(dw,dtype=self.dtypes[dnum])
            wname = self.base_name % dnum
            wname += '.%d.fit' % self.nwrit[dnum]
            fitsio.write(wname,dw,clobber=True)
            self.nwrit[dnum] += 1

            if stop == len(self.data[dnum]):
                self.data[dnum] = []
            else:
                self.data[dnum] = self.data[dnum][stop:]
            self.ndata[dnum] = len(self.data[dnum])

    def add_data(self,dnum,data):
        if dnum not in self.data:
            self._init_dnum(dnum)
            self.dtypes[dnum] = data.dtype.descr

        assert self.dtypes[dnum] == data.dtype.descr
        self.data[dnum].extend(list(data))
        self.ndata[dnum] = len(self.data[dnum])

        if self.ndata[dnum] >= self.Nmax:
            self._write_data(dnum)

    def finalize_data(self):
        for dnum in self.data:
            if self.ndata[dnum] > 0:
                self._write_data(dnum,finalize=True)
                assert len(self.data[dnum]) == 0
