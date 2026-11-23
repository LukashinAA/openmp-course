/* Диагностика нагрузки и завершения редукции отдельно от основных замеров.
 * Запуск: OMP_NUM_THREADS=20 OMP_DYNAMIC=FALSE ./build/diagnostics 8 25 5
 * Вывод — CSV для static и dynamic,16, один прогрев перед каждой серией.
 */
#define main particles_main
#include "particles.c"
#undef main

static double control_tail(void)
{
    double start = 0.0;
    #pragma omp parallel default(none) shared(start)
    {
        #pragma omp barrier
        if (omp_get_thread_num() == 0) start = omp_get_wtime();
        #pragma omp barrier
    }
    return omp_get_wtime() - start;
}

static double diagnose(int n, int lj, const double *x, const double *y,
                       const double *z, const double *q,
                       double *fx, double *fy, double *fz,
                       double *times, unsigned long long *counts,
                       int *threads, double *tail)
{
    double energy = 0.0, start_tail = 0.0;
    for (int i = 0; i < n; i++) fx[i] = fy[i] = fz[i] = 0.0;

    #pragma omp parallel default(none) \
        shared(n, lj, x, y, z, q, times, counts, threads, start_tail) \
        reduction(+:energy, fx[0:n], fy[0:n], fz[0:n])
    {
        int id = omp_get_thread_num();
        if (id == 0) *threads = omp_get_num_threads();
        #pragma omp barrier
        double start = omp_get_wtime();
        unsigned long long pairs = 0;

        #pragma omp for schedule(runtime) nowait
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                double dx = x[i] - x[j], dy = y[i] - y[j], dz = z[i] - z[j], c;
                energy += pair(lj, q[i], q[j], dx * dx + dy * dy + dz * dz, &c);
                fx[i] += c * dx; fy[i] += c * dy; fz[i] += c * dz;
                fx[j] -= c * dx; fy[j] -= c * dy; fz[j] -= c * dz;
            }
            pairs += (unsigned long long)(n - 1 - i);
        }
        times[id] = omp_get_wtime() - start;
        counts[id] = pairs;

        /* Все частные суммы готовы. Оба отсчёта хвоста делает master.
         * Интервал включает последний барьер, слияние и завершение команды.
         */
        #pragma omp barrier
        if (id == 0) start_tail = omp_get_wtime();
        #pragma omp barrier
    }
    *tail = omp_get_wtime() - start_tail;
    return energy;
}

int main(int argc, char **argv)
{
    int V = argc > 1 ? atoi(argv[1]) : 8;
    if (V < 1 || V > 17) return 1;
    variant_t p = variant(V);
    if (argc > 2) p.n_side = atoi(argv[2]);
    int repeat = argc > 3 ? atoi(argv[3]) : 5;
    if (p.n_side < 2 || p.n_side > 100 || repeat < 1) return 1;
    int n = p.n_side * p.n_side * p.n_side;
    int capacity = omp_get_max_threads();
    double *data = malloc((size_t)10 * n * sizeof *data);
    double *times = calloc((size_t)capacity, sizeof *times);
    unsigned long long *counts = calloc((size_t)capacity, sizeof *counts);
    if (!data || !times || !counts) {
        fprintf(stderr, "мало памяти\n");
        free(data); free(times); free(counts);
        return 1;
    }
    double *x = data, *y = x + n, *z = y + n, *q = z + n;
    double *fx = q + n, *fy = fx + n, *fz = fy + n;
    double *rx = fz + n, *ry = rx + n, *rz = ry + n;
    make_system(V, p, n, x, y, z, q);
    double reference = forces_seq(n, p.lj, x, y, z, q, rx, ry, rz);
    double scale = 0.0;
    for (int i = 0; i < n; i++)
        scale = fmax(scale, fmax(fabs(rx[i]), fmax(fabs(ry[i]), fabs(rz[i]))));
    unsigned long long total_pairs = (unsigned long long)n * (n - 1) / 2;

    puts("schedule,rep,thread,threads,n,pairs,pair_share,work_seconds,merge_join_seconds,control_join_seconds,relative_error");
    for (int s = 0; s < 2; s++) {
        omp_set_schedule(s == 0 ? omp_sched_static : omp_sched_dynamic, s == 0 ? 0 : 16);
        const char *schedule = s == 0 ? "static" : "dynamic,16";
        for (int rep = 0; rep <= repeat; rep++) {
            int threads = 0;
            double tail;
            double energy = diagnose(n, p.lj, x, y, z, q, fx, fy, fz,
                                     times, counts, &threads, &tail);
            double error = fabs(energy - reference) / fmax(1.0, fabs(reference));
            unsigned long long measured_pairs = 0;
            for (int id = 0; id < threads; id++) measured_pairs += counts[id];
            for (int i = 0; i < n; i++) {
                if (!isfinite(fx[i]) || !isfinite(fy[i]) || !isfinite(fz[i])) return 1;
                double difference = fmax(fabs(fx[i] - rx[i]),
                                        fmax(fabs(fy[i] - ry[i]), fabs(fz[i] - rz[i])));
                error = fmax(error, difference / fmax(1.0, scale));
            }
            if (!isfinite(energy) || error > 1e-10 || measured_pairs != total_pairs) return 1;
            double control = control_tail();
            if (rep == 0) continue;
            for (int id = 0; id < threads; id++) {
                printf("\"%s\",%d,%d,%d,%d,%llu,%.12f,%.9f,%.9f,%.9f,%.3e\n",
                       schedule, rep, id, threads, n, counts[id],
                       (double)counts[id] / total_pairs, times[id], tail, control, error);
            }
        }
    }
    free(data); free(times); free(counts);
    return 0;
}
