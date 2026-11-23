#define main newton_main
#include "newton.c"
#undef main

int main(void)
{
    variant_t p = variant(8);
    long bands[20] = {0}, total = 0;
    for (int j = 0; j < 2000; j++) {
        for (int i = 0; i < 2000; i++) {
            double x = p.cx - p.w + 2.0 * p.w * (i + 0.5) / 2000;
            double y = p.cy - p.w + 2.0 * p.w * (j + 0.5) / 2000;
            int root;
            int it = newton_point(x, y, p.m, &root);
            bands[j / 100] += it;
            total += it;
        }
    }
    puts("band,first_row,last_row,iterations,percent");
    for (int i = 0; i < 20; i++)
        printf("%d,%d,%d,%ld,%.6f\n", i, i * 100, i * 100 + 99,
               bands[i], 100.0 * bands[i] / total);
    return 0;
}
