/* montecarlo.c -- estimacion de pi por Monte Carlo (acierto y fallo) con OpenMP.
 * Uso: ./montecarlo [N] [p]     (N muestras, p hilos; por defecto 1e8 y todos los nucleos)
 *
 * Generador: SplitMix64 de Steele et al. con "salto" directo.
 *   El estado k-esimo es  s_k = SEMILLA + k*GAMMA  (mod 2^64)  y el k-esimo
 *   numero aleatorio es  mezcla(s_k).  Como s_k es una funcion directa de k,
 *   se puede saltar a cualquier posicion sin recorrer la secuencia.
 *   La muestra i usa las posiciones 2i (x) y 2i+1 (y) de UNA sola secuencia
 *   maestra de periodo 2^64. Cada hilo recorre un tramo disjunto de esa
 *   secuencia, asi que no hay solapamiento, y el resultado no depende de p.
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#ifdef _OPENMP
#include <omp.h>
#else
#include <time.h>
#endif

#define SEMILLA 2225112ULL                 /* codigo del estudiante */
#ifndef PLAN
#define PLAN static
#endif
#define GAMMA   0x9E3779B97F4A7C15ULL      /* incremento de SplitMix64 */

/* Numero aleatorio uniforme en [0,1) en la posicion k de la secuencia maestra */
static inline double uniforme(uint64_t k)
{
    uint64_t z = SEMILLA + k * GAMMA;      /* salto directo al estado k */
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z = z ^ (z >> 31);
    return (double)(z >> 11) * (1.0 / 9007199254740992.0);   /* 53 bits / 2^53 */
}

static double reloj(void)
{
#ifdef _OPENMP
    return omp_get_wtime();
#else
    return (double)clock() / CLOCKS_PER_SEC;
#endif
}

int main(int argc, char *argv[])
{
    long N = (argc > 1) ? atol(argv[1]) : 100000000L;
#ifdef _OPENMP
    if (argc > 2) omp_set_num_threads(atoi(argv[2]));
    int p = omp_get_max_threads();
#else
    int p = 1;
#endif
    long dentro = 0;

    double t0 = reloj();

    /* Planificacion: static por defecto. Para II.5 se puede recompilar con
     * -DPLAN="dynamic,10000" (no usar schedule(runtime): libgomp usa dynamic,1). */
#ifdef _OPENMP
    #pragma omp parallel for reduction(+:dentro) schedule(PLAN)
#endif
    for (long i = 0; i < N; i++) {
        double x = uniforme(2 * (uint64_t)i);
        double y = uniforme(2 * (uint64_t)i + 1);
        if (x * x + y * y <= 1.0)
            dentro++;
    }

    double t = reloj() - t0;

    printf("N=%ld p=%d aciertos=%ld pi=%.6f tiempo=%.4f\n",
           N, p, dentro, 4.0 * dentro / (double)N, t);
    return 0;
}
