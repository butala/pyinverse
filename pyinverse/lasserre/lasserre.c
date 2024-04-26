#include <stdlib.h>
#include <stddef.h>
#include <float.h>
#include <math.h>
#include <assert.h>
#include <stdbool.h>
#include <string.h>

#include "lasserre.h"
#include "util.h"


static double lasserre_vol_base_case(size_t M, double *A, double *b) {
    double min_val =  DBL_MAX;
    double max_val = -DBL_MAX;
    size_t i;

    for (i = 0; i < M; i++) {
        if (A[i] > 0) {
            if (b[i] / A[i] < min_val) {
                min_val = b[i] / A[i];
            }
        }
        else if (A[i] < 0) {
            if (b[i] / A[i] > max_val) {
                max_val = b[i] / A[i];
            }
        }
        else {
            assert(0);
        }
    }

    if ((min_val == DBL_MAX) || (max_val == -DBL_MAX)) {
        return INFINITY;
    }
    else {
        return fmax(0, min_val - max_val);
    }
}


static void normalize_constraints(size_t M, size_t N, double *A, double *b) {
    size_t i, j, k;
    bool j_found;
    double scale_factor;

    for (i = 0; i < M; i++) {
        j_found = false;
        for (j = 0; j < N; j++) {
            if (!fequal(A[i*N + j], 0)) {
                j_found = true;
                break;
            }
        }
        if (!j_found) {
            continue;
        }

        scale_factor = fabs(A[i*N + j]);
        for (k = 0; k < N; k++) {
            A[i*N + k] /= scale_factor;
        }
        b[i] /= scale_factor;
    }
}


static bool filter_parallel_constraints(size_t *M, size_t N, double *A, double *b) {
    bool *parallel_halfspaces;
    size_t i, j, k;
    double *smallest_b;
    bool j_found;
    double scale_factor_i, scale_factor_k;
    bool *rows_to_keep;
    size_t M_out;

    parallel_halfspaces = malloc(*M * sizeof(bool));
    rows_to_keep = malloc(*M * sizeof(bool));
    assert(parallel_halfspaces);
    assert(rows_to_keep);
    for (i = 0; i < *M; i++) {
        parallel_halfspaces[i] = false;
        rows_to_keep[i] = false;
    }

    smallest_b = malloc(*M * sizeof(double));
    assert(smallest_b);

    for (i = 0; i < *M; i++) {
        if (parallel_halfspaces[i]) {
            continue;
        }
        smallest_b[i] = NAN;

        j_found = false;
        for (j = 0; j < N; j++) {
            if (!fequal(A[i*N + j], 0)) {
                j_found = true;
                break;
            }
        }
        if (!j_found) {
            continue;
        }

        scale_factor_i = fabs(A[i*N + j]);
        for (k = i+1; k < *M; k++) {
            if (fequal(A[k*N + j], 0)) {
                continue;
            }
            scale_factor_k = fabs(A[k*N + j]);

            if (allclose_scaled(N, &A[i*N], 1/scale_factor_i, &A[k*N], 1/scale_factor_k)) {
                parallel_halfspaces[k] = true;
                if (b[i] <= b[k]) {
                    smallest_b[i] = b[i];
                }
                else {
                    smallest_b[i] = b[k];
                }
            }
            else if (allclose_scaled(N, &A[i*N], 1/scale_factor_i, &A[k*N], -1/scale_factor_k)) {
                if (-b[k] > b[i]) {
                    free(parallel_halfspaces);
                    free(rows_to_keep);
                    free(smallest_b);
                    return false;
                }
            }
        }
        rows_to_keep[i] = true;
    }

    M_out = 0;
    for (i = 0; i < *M; i++) {
        if (rows_to_keep[i]) {
            for (k = 0; k < N; k++) {
                A[M_out*N + k] = A[i*N + k];
            }
            if (isnan(smallest_b[i])) {
                b[M_out] = b[i];
            }
            else {
                b[M_out] = smallest_b[i];
            }
            M_out++;
        }
    }

    *M = M_out;

    free(parallel_halfspaces);
    free(rows_to_keep);
    free(smallest_b);

    return true;
}


static double lasserre_vol_helper(size_t *M, size_t N, double *A, double *b) {
    double vol, vol_i;
    size_t i, j, k, k_prime, l, l_prime, M_tilde, N_tilde;
    double *A_tilde = NULL;
    double *b_tilde = NULL;
    bool j_found;

    if (*M == 1) {
        return INFINITY;
    }

    if (N == 1) {
        return lasserre_vol_base_case(*M, A, b);
    }

    normalize_constraints(*M, N, A, b);

    if (!filter_parallel_constraints(M, N, A, b)) {
        return NAN;
    }

    vol = 0;

    M_tilde = *M - 1;
    N_tilde = N - 1;

    A_tilde = malloc(M_tilde * N_tilde * sizeof(double));
    assert(A_tilde);

    b_tilde = malloc(M_tilde * sizeof(double));
    assert(b_tilde);

    for (i=0; i < *M; i++) {
        if (fequal(b[i], 0)) {
            continue;
        }

        j_found = false;
        for (j = 0; j < N; j++) {
            if (!fequal(A[i*N + j], 0)) {
                j_found = true;
                break;
            }
        }
        if (!j_found) {
            continue;
        }

        k_prime = 0;
        for (k = 0; k < *M; k++) {
            if (k == i) {
                continue;
            }

            l_prime = 0;
            for (l = 0; l < N; l++) {
                if (l == j) {
                    continue;
                }
                A_tilde[k_prime*N_tilde + l_prime] = A[k*N + l] - A[k*N + j] * A[i*N + l] / A[i*N + j];
                l_prime++;
            }
            b_tilde[k_prime] = b[k] - A[k*N + j] / A[i*N + j] * b[i];
            k_prime++;
        }
        vol_i = lasserre_vol(M_tilde, N_tilde, A_tilde, b_tilde);
        if (isinf(vol_i)) {
            vol = INFINITY;
            break;
        }
        else if (isnan(vol_i)) {
            continue;
        }
        vol += b[i] / fabs(A[i*N + j]) * vol_i;
    }

    free(A_tilde);
    free(b_tilde);

    return vol / N;
}


double lasserre_vol(size_t M, size_t N, double *A, double *b) {
    size_t M_copy = M;
    double *A_copy;
    double *b_copy;
    double vol;

    A_copy = malloc(M * N * sizeof(double));
    assert(A_copy);

    b_copy = malloc(M * sizeof(double));
    assert(b_copy);

    memcpy(A_copy, A, sizeof(double)*M*N);
    memcpy(b_copy, b, sizeof(double)*M);

    vol = lasserre_vol_helper(&M_copy, N, A, b);
    if (isnan(vol)) {
        return 0;
    }

    free(A_copy);
    free(b_copy);

    return vol;
}
