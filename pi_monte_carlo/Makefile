pi: pi.c
ifeq ($(OPENMP), false)
	gcc -o pi pi.c -std=gnu99
else
	gcc -o pi pi.c -fopenmp -std=gnu99
endif
	
.PHONY: clean
clean:
	rm -rf pi