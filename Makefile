all:
	( cd fastFM-core ; $(MAKE) lib )
	python setup.py build_ext --inplace

.PHONY : clean
clean:
	( cd fastFM-core ; $(MAKE) clean )
	cd fastFM/
	rm -f *.so
	rm -rf build/
	rm -f fastFM/ffm.c
