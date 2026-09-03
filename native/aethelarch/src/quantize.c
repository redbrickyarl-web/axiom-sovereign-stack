#include "aethelarch/aethelarch.h"
#include "aethelarch/internal.h"
#include <stdlib.h>
#include <string.h>

aethelarch_matrix_t* aethelarch_matrix_create(size_t rows, size_t cols) {
    if (rows == 0 || cols == 0) return NULL;

    aethelarch_matrix_t* mat = (aethelarch_matrix_t*)malloc(sizeof(aethelarch_matrix_t));
    if (!mat) return NULL;

    size_t raw_stride = (cols + 7) / 8;
    mat->rows = rows;
    mat->cols = cols;
    mat->row_stride_bytes = (raw_stride + 63) & ~(size_t)63;
    mat->scale = 1.0f;
    mat->w_pos = NULL;
    mat->w_neg = NULL;

    size_t total_bytes = rows * mat->row_stride_bytes;
    mat->w_pos = (uint8_t*)aeth_aligned_alloc(64, total_bytes);
    mat->w_neg = (uint8_t*)aeth_aligned_alloc(64, total_bytes);

    if (!mat->w_pos || !mat->w_neg) {
        aethelarch_matrix_free(mat);
        return NULL;
    }

    memset(mat->w_pos, 0, total_bytes);
    memset(mat->w_neg, 0, total_bytes);
    return mat;
}

void aethelarch_matrix_free(aethelarch_matrix_t* mat) {
    if (!mat) return;
    if (mat->w_pos) aeth_aligned_free(mat->w_pos);
    if (mat->w_neg) aeth_aligned_free(mat->w_neg);
    free(mat);
}

bool aethelarch_encode_dense(
    aethelarch_matrix_t* dst,
    const int8_t* src_ternary,
    size_t rows,
    size_t cols
) {
    if (!dst || !src_ternary) return false;
    if (!dst->w_pos || !dst->w_neg) return false;
    if (dst->rows != rows || dst->cols != cols) return false;

    size_t total = rows * dst->row_stride_bytes;
    memset(dst->w_pos, 0, total);
    memset(dst->w_neg, 0, total);

    for (size_t r = 0; r < rows; ++r) {
        uint8_t* pos_row = dst->w_pos + r * dst->row_stride_bytes;
        uint8_t* neg_row = dst->w_neg + r * dst->row_stride_bytes;

        for (size_t c = 0; c < cols; ++c) {
            int8_t val = src_ternary[r * cols + c];
            if (val == 1) {
                pos_row[c / 8] |= (uint8_t)(1u << (c % 8));
            } else if (val == -1) {
                neg_row[c / 8] |= (uint8_t)(1u << (c % 8));
            } else if (val != 0) {
                return false;
            }
        }

        for (size_t b = 0; b < dst->row_stride_bytes; ++b) {
            if (pos_row[b] & neg_row[b]) return false;
        }
    }
    return true;
}

void aethelarch_quantize_activation(
    uint8_t* dst_bits,
    const float* src_activations,
    size_t dim
) {
    if (!dst_bits || !src_activations || dim == 0) return;

    size_t nbytes = aethelarch_act_bytes(dim);
    memset(dst_bits, 0, nbytes);

    for (size_t i = 0; i < dim; ++i) {
        if (src_activations[i] > 0.0f) {
            dst_bits[i / 8] |= (uint8_t)(1u << (i % 8));
        }
    }
}

bool aethelarch_gemv(
    float* dst_out,
    const aethelarch_matrix_t* mat,
    const uint8_t* act_bits
) {
    if (!dst_out || !mat || !act_bits) return false;
    if (!mat->w_pos || !mat->w_neg) return false;

    for (size_t r = 0; r < mat->rows; ++r) {
        const uint8_t* pos_row = mat->w_pos + r * mat->row_stride_bytes;
        const uint8_t* neg_row = mat->w_neg + r * mat->row_stride_bytes;
        int32_t dot = aethelarch_dot_product(pos_row, neg_row, act_bits, mat->cols);
        dst_out[r] = (float)dot * mat->scale;
    }
    return true;
}
