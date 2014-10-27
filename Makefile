all:
	$(MAKE) -C fastFM-core/src/ lib
	python setup.py build_ext --inplace

.PHONY : clean
clean:
	cd fastFM/
	rm -f *.so
	rm -rf build/
	rm -f fastFM/ffm.c
