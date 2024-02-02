############################################################################
#
#  Program:         SuperLU
#
#  Module:          Makefile
#
#  Purpose:         Top-level Makefile
#
#  Creation date:   October 2, 1995
#
#  Modified:        February 4,  1997  Version 1.0
#		    November 15, 1997  Version 1.1
#		    September 1, 1999  Version 2.0
#		    October 15,  2003  Version 3.0
#		    August 1,    2008  Version 3.1
#
############################################################################

include make.inc
include conf.mk

all: lib

lib: blaslib superlulib

clean: cleanlib

install:
	cp SRC/*.a ./
	cp CBLAS/*.a ./
#	( cd INSTALL; $(MAKE) )
#	( cd INSTALL; cp lsame.c ../SRC/; \
#	  cp dlamch.c ../SRC/; cp slamch.c ../SRC/ )

blaslib:
	( cd CBLAS; $(MAKE) double)

superlulib:
	( cd SRC; $(MAKE) double)

tmglib:
	( cd TESTING/MATGEN; $(MAKE) )

matlabmex:
	( cd MATLAB; $(MAKE) )

testing:
	( cd TESTING ; $(MAKE) )

doc:
	doxygen Doxyfile

cleanlib:
	( cd SRC; $(MAKE) clean )
	( cd CBLAS; $(MAKE) clean )

cleantesting:
	( cd INSTALL; $(MAKE) clean )
	( cd TESTING; $(MAKE) clean )
	( cd MATLAB;  $(MAKE) clean )
	( cd EXAMPLE; $(MAKE) clean )
	( cd FORTRAN; $(MAKE) clean )
