/* GCC 13.3 loop-vectoriser miscompile: interleaved store-group permutation.
 *
 * Context: SpeckleBem WP2, src/geometry/rough_surface.cpp (detail::rough_box_arrays). Writing
 * the four triangles of each grid cell (top face and bottom plate of the rough-surface box)
 * with the corner indices computed inside the store expressions produced wrong bottom-plate
 * triangles at -O3. The generator now computes the corner indices into locals first; the
 * unit tests check the emitted topology and orientation in the release build.
 *
 * Verified: no UB (all values small, all stores in bounds). GCC 13.3 x86-64 -O3 prints BAD;
 * -O2, -O3 -fno-tree-loop-vectorize and clang 18 -O3 print ok.
 *
 *   gcc -O3 gcc13_vectorizer_store_permutation.c && ./a.out
 *
 * Minimal reproducer saved during review (its original comment follows). Not part of the
 * build. */
/* GCC 13.3 x86-64: gcc -O3 min5.c && ./a.out prints BAD (wrong stores from the loop
   vectorizer); -O2, -O3 -fno-tree-loop-vectorize and clang -O3 print ok. No UB: all values
   are small, all stores in bounds. */
#include <stdio.h>
#define N 5
__attribute__((noinline)) void fill(long *f, long n) {
    long g = n * n;
    for (long i = 0; i + 1 < n; ++i)
        for (long j = 0; j + 1 < n; ++j) {
            long *o = f + 12 * (i * (n - 1) + j);
            o[0] = i * n + j;  o[1] = (i + 1) * n + (j + 1);     o[2] = (i + 1) * n + j;
            o[3] = i * n + j;  o[4] = i * n + (j + 1);           o[5] = (i + 1) * n + (j + 1);
            o[6] = g + i * n + j;  o[7] = g + (i + 1) * n + j;   o[8] = g + (i + 1) * n + (j + 1);
            o[9] = g + i * n + j;  o[10] = g + (i + 1) * n + (j + 1); o[11] = g + i * n + (j + 1);
        }
}
int main(void) {
    long f[12 * (N - 1) * (N - 1)];
    fill(f, N);
    int bad = 0;
    for (long i = 0; i + 1 < N; ++i)
        for (long j = 0; j + 1 < N; ++j) {
            long a = i * N + j, b = a + N, c = b + 1, d = a + 1, g = N * N;
            long e[12] = {a, c, b, a, d, c, g + a, g + b, g + c, g + a, g + c, g + d};
            for (int k = 0; k < 12; ++k) bad |= f[12 * (i * (N - 1) + j) + k] != e[k];
        }
    printf("%s: bottom of cell 0 = %ld %ld %ld / %ld %ld %ld (expected 25 30 31 / 25 31 26)\n",
           bad ? "BAD" : "ok", f[6], f[7], f[8], f[9], f[10], f[11]);
    return bad;
}
