#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <omp.h>

// thread safe rand
static inline double my_rand(unsigned int *seed)
{
    return rand_r(seed) / (double)RAND_MAX;
}

double approx_pi(size_t n)
{
    double x;
    double y;
    int in = 0;
    #pragma omp parallel
    {
#if _OPENMP
        unsigned int myseed = omp_get_thread_num() + time(NULL);
#else
        unsigned int myseed = time(NULL);
#endif
        #pragma omp for private(x, y) reduction(+:in)
        for (int i = 0; i < n; i++) {
            x = my_rand(&myseed);
            y = my_rand(&myseed);
            if (((x*x)+(y*y)) <= 1.0)
                in++;
        }
    }
    return 4*(double)in/n;
}

int main(int argc, const char* argv[])
{
    size_t n = (argc == 1) ? 10000 : atol(argv[1]);
#if _OPENMP
    printf("calculating parallel (%ld iterations)...\n", n);
#else
    printf("calculating single (%ld iterations)...\n", n);
#endif
    printf("pi = %lf\n", approx_pi(n));
    return 0;
}