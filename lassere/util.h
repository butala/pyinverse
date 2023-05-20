#ifndef UTIL_H
#define UTIL_H

#include <stddef.h>
#include <stdbool.h>

#define TOLERANCE 1e-6


bool fequal(double a, double b);

bool allclose_scaled(size_t N, double *a, double scale_a, double *b, double scale_b);

#endif
