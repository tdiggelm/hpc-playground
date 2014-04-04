all: pi-parallel pi-single

pi-parallel: pi.c
	gcc -fopenmp -std=gnu99 $< -o $@

pi-single: pi.c
	gcc -std=gnu99 $< -o $@
	
.PHONY: clean
clean:
	rm -rf pi-parallel pi-single