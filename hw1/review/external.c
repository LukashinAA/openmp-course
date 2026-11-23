#include <math.h>
#include <omp.h>

static inline double external_pair(int lj, double qi, double qj,
                                   double r2, double *c)
{
    if (lj) {
        double ir2 = 1.0 / r2, ir6 = ir2 * ir2 * ir2;
        *c = 24.0 * ir2 * ir6 * (2.0 * ir6 - 1.0);
        return 4.0 * ir6 * (ir6 - 1.0);
    } else {
        double ir = 1.0 / sqrt(r2), e = qi * qj * ir;
        *c = e * ir * ir;
        return e;
    }
}

double forces_external(int n, int lj, const double *x, const double *y,
                       const double *z, const double *q,
                       double *fx, double *fy, double *fz)
{
    double e = 0.0;
    for (int i = 0; i < n; i++)
        fx[i] = fy[i] = fz[i] = 0.0;

    #pragma omp parallel for default(none) \
        shared(n, lj, x, y, z, q, fx, fy, fz) \
        reduction(+:e) schedule(runtime)
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            double dx = x[i] - x[j];
            double dy = y[i] - y[j];
            double dz = z[i] - z[j];
            double c;
            e += external_pair(lj, q[i], q[j],
                               dx * dx + dy * dy + dz * dz, &c);
            double dfx = c * dx, dfy = c * dy, dfz = c * dz;

            #pragma omp atomic update
            fx[i] += dfx;
            #pragma omp atomic update
            fy[i] += dfy;
            #pragma omp atomic update
            fz[i] += dfz;
            #pragma omp atomic update
            fx[j] -= dfx;
            #pragma omp atomic update
            fy[j] -= dfy;
            #pragma omp atomic update
            fz[j] -= dfz;
        }
    }
    return e;
}
