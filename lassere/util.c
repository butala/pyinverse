#include <math.h>

#include "util.h"


bool fequal(double a, double b) {
    return fabs(a - b) < TOLERANCE;
}


bool allclose_scaled(size_t N, double *a, double scale_a, double *b, double scale_b) {
    size_t i;

    for (i = 0; i < N; i++) {
        if (!fequal(a[i] * scale_a, b[i] * scale_b)) {
            return false;
        }
    }
    return true;
}
