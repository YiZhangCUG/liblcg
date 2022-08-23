/******************************************************
 * C++ Library of the Linear Conjugate Gradient Methods (LibLCG)
 * 
 * Copyright (C) 2022  Yi Zhang (yizhang-geo@zju.edu.cn)
 * 
 * LibLCG is distributed under a dual licensing scheme. You can
 * redistribute it and/or modify it under the terms of the GNU Lesser
 * General Public License (LGPL) as published by the Free Software Foundation,
 * either version 2 of the License, or (at your option) any later version. 
 * You should have received a copy of the GNU Lesser General Public 
 * License along with this program. If not, see <http://www.gnu.org/licenses/>. 
 * 
 * If the terms and conditions of the LGPL v.2. would prevent you from
 * using the LibLCG, please consider the option to obtain a commercial
 * license for a fee. These licenses are offered by the LibLCG developing 
 * team. As a rule, licenses are provided "as-is", unlimited in time for 
 * a one time fee. Please send corresponding requests to: yizhang-geo@zju.edu.cn. 
 * Please do not forget to include some description of your company and the 
 * realm of its activities. Also add information on how to contact you by 
 * electronic and paper mail.
 ******************************************************/

#ifndef _PRECONDITIONER_H
#define _PRECONDITIONER_H

#include "algebra.h"

/**
 * @brief Return the number of non-zero elements in the lower triangular part of the input matrix
 * 
 * @param row[in]        Row index of the input sparse matrix.
 * @param col[in]        Column index of the input sparse matrix.
 * @param nz_size[in]    Length of the non-zero elements.
 * @param lnz_size[out]  Legnth of the non-zero elements in the lower triangle
 */
void lcg_incomplete_Cholesky_half_buffsize_coo(const int *row, const int *col, int nz_size, int *lnz_size);

/**
 * @brief Preform the incomplete Cholesky factorization for a sparse matrix that is saved in the COO format.
 * 
 * @note  Only the factorized lower triangular matrix is stored in the lower part of the output matrix accordingly.
 * 
 * @param row        Row index of the input sparse matrix.
 * @param col        Column index of the input sparse matrix.
 * @param val        Non-zero values of the input sparse matrix.
 * @param N          Row/Column size of the sparse matrix.
 * @param nz_size    Length of the non-zero elements.
 * @param lnz_size   Legnth of the non-zero elements in the lower triangle
 * @param IC_row     Row index of the factorized triangular sparse matrix.
 * @param IC_col     Column index of the factorized triangular sparse matrix.
 * @param IC_val     Non-zero values of the factorized triangular sparse matrix.
 */
void lcg_incomplete_Cholesky_half_coo(const int *row, const int *col, const lcg_float *val, int N, int nz_size, int lnz_size, int *IC_row, int *IC_col, lcg_float *IC_val);

/**
 * @brief Preform the incomplete Cholesky factorization for a sparse matrix that is saved in the COO format.
 * 
 * @note  The factorized lower and upper triangular matrixes are stored in the lower and upper triangular parts of the output matrix accordingly.
 * 
 * @param row        Row index of the input sparse matrix.
 * @param col        Column index of the input sparse matrix.
 * @param val        Non-zero values of the input sparse matrix.
 * @param N          Row/Column size of the sparse matrix.
 * @param nz_size    Length of the non-zeor elements.
 * @param IC_row     Row index of the factorized triangular sparse matrix.
 * @param IC_col     Column index of the factorized triangular sparse matrix.
 * @param IC_val     Non-zero values of the factorized triangular sparse matrix.
 */
void lcg_incomplete_Cholesky_full_coo(const int *row, const int *col, const lcg_float *val, int N, int nz_size, int *IC_row, int *IC_col, lcg_float *IC_val);

/**
 * @brief Solve the linear system Ux = B, in which U is a upper triangle matrix.
 * 
 * @param row        Row index of the input sparse matrix.
 * @param col        Column index of the input sparse matrix.
 * @param U          Non-zero values of the input sparse matrix.
 * @param B          Object array.
 * @param x          The returned solution.
 * @param N          Row/Column size of the sparse matrix.
 * @param nz_size    Length of the non-zeor elements.
 */
void lcg_solve_upper_triangle_coo(const int *row, const int *col, const lcg_float *U, const lcg_float *B, lcg_float *x, int N, int nz_size);

/**
 * @brief Solve the linear system Lx = B, in which L is a lower triangle matrix.
 * 
 * @param row        Row index of the input sparse matrix.
 * @param col        Column index of the input sparse matrix.
 * @param L          Non-zero values of the input sparse matrix.
 * @param B          Object array.
 * @param x          The returned solution.
 * @param N          Row/Column size of the sparse matrix.
 * @param nz_size    Length of the non-zeor elements.
 */
void lcg_solve_lower_triangle_coo(const int *row, const int *col, const lcg_float *L, const lcg_float *B, lcg_float *x, int N, int nz_size);

/**
 * @brief Check to see if a square matrix is full ranked or not. The sparse matrix is stored in the COO format.
 * 
 * @param row        Row index of the input sparse matrix.
 * @param col        Column index of the input sparse matrix.
 * @param M          Non-zero values of the input sparse matrix.
 * @param N          Row/Column size of the sparse matrix.
 * @param nz_size    Length of the non-zeor elements.
 * @return true      The matrix is full ranked.
 * @return false     The matrix is not full ranked.
 */
bool lcg_full_rank_coo(const int *row, const int *col, const lcg_float *M, int N, int nz_size);

#endif // _PRECONDITIONER_H