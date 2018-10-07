PYTHON ?= python

all:
	( cd fastFM-core2 ; \
	  cmake -H. -B_lib -DCMAKE_BUILD_TYPE=Debug -DCMAKE_DEBUG_POSTFIX=d; \
	  cmake --build _lib; )
	( cd fastFM-core ; $(MAKE) lib )
	$(PYTHON) setup.py build_ext --inplace

.PHONY : clean
clean:
	( cd fastFM-core ; $(MAKE) clean )
	cd fastFM/
	rm -f *.so
	rm -rf build/
	rm -f fastFM/ffm.c
	rm -f fastFM/ffm2.cpp
