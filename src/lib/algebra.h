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

#ifndef _ALGEBRA_H
#define _ALGEBRA_H

#include "config.h"

/**
 * @brief      Matrix layouts.
 */
enum lcg_matrix_e
{
	MatNormal,
	MatTranspose,
};

/**
 * @brief      Conjugate types for a complex number.
 */
enum clcg_complex_e
{
	NonConjugate,
	Conjugate,
};

/**
 * @brief      A simple definition of the float type we use here. 
 * Easy to change in the future. Right now it is just an alias of double
 */
typedef double lcg_float;

/**
 * @brief      Return absolute value
 *
 * @param[in]  a     input value
 *
 * @return     The absolute value
 */
lcg_float lcg_abs(lcg_float a);

/**
 * @brief      Return the bigger value
 *
 * @param[in]  a     input value
 * @param[in]  b     input value
 *
 * @return     The bigger value
 */
lcg_float lcg_max(lcg_float a, lcg_float b);

/**
 * @brief      Return the smaller value
 *
 * @param[in]  a     input value
 * @param[in]  b     input value
 *
 * @return     The smaller value
 */
lcg_float lcg_min(lcg_float a, lcg_float b);

/**
 * @brief      Set the input value within a box constraint
 *
 * @param      a     low boundary
 * @param      b     high boundary
 * @param      in    input value
 * @param      low_bound    Whether to include the low boundary value
 * @param      hig_bound    Whether to include the high boundary value
 *
 * @return     box constrained value
 */
lcg_float lcg_set2box(lcg_float low, lcg_float hig, lcg_float a, 
	bool low_bound = true, bool hig_bound = true);

/**
 * @brief      Locate memory for a lcg_float pointer type.
 *
 * @param[in]  n     Size of the lcg_float array.
 *
 * @return     Pointer of the array's location.
 */
lcg_float* lcg_malloc(int n);

/**
 * @brief      Locate memory for a lcg_float second pointer type.
 *
 * @param[in]  n     Size of the lcg_float array.
 *
 * @return     Pointer of the array's location.
 */
lcg_float** lcg_malloc(int m, int n);

/**
 * @brief      Destroy memory used by the lcg_float type array.
 *
 * @param      x     Pointer of the array.
 */
void lcg_free(lcg_float* x);

/**
 * @brief      Destroy memory used by the 2D lcg_float type array.
 *
 * @param      x     Pointer of the array.
 */
void lcg_free(lcg_float **x, int m);

/**
 * @brief      set a vector's value
 *
 * @param      a     pointer of the vector
 * @param[in]  b     initial value
 * @param[in]  size  vector size
 */
void lcg_vecset(lcg_float *a, lcg_float b, int size);

/**
 * @brief      set a 2d vector's value
 *
 * @param      a     pointer of the matrix
 * @param[in]  b     initial value
 * @param[in]  m     row size of the matrix
 * @param[in]  n     column size of the matrix
 */
void lcg_vecset(lcg_float **a, lcg_float b, int m, int n);

/**
 * @brief      set a vector using random values
 *
 * @param      a     pointer of the vector
 * @param[in]  l     the lower bound of random values
 * @param[in]  h     the higher bound of random values
 * @param[in]  size  size of the vector
 */
void lcg_vecrnd(lcg_float *a, lcg_float l, lcg_float h, int size);

/**
 * @brief      set a 2D vector using random values
 *
 * @param      a     pointer of the vector
 * @param[in]  l     the lower bound of random values
 * @param[in]  h     the higher bound of random values
 * @param[in]  m     row size of the vector
 * @param[in]  n     column size of the vector
 */
void lcg_vecrnd(lcg_float **a, lcg_float l, lcg_float h, int m, int n);

/**
 * @brief    calculate the squared L2 norm of the input vector
 * 
 * @param a         pointer of the vector
 * @param n         size of the vector
 * @return double   L2 norm
 */
double lcg_squaredl2norm(lcg_float *a, int n);

/**
 * @brief      calculate dot product of two real vectors
 *
 * @param[in]  a       pointer of the vector a
 * @param[in]  b       pointer of the vector b
 * @param[in]  size    size of the vector
 *
 * @return     dot product
 */
void lcg_dot(lcg_float &ret, const lcg_float *a, const lcg_float *b, int size);

/**
 * @brief      calculate product of a real matrix and a vector
 * 
 * Different configurations:
 * layout=Normal -> A
 * layout=Transpose -> A^T
 *
 * @param      A          matrix A
 * @param[in]  x          vector x
 * @param      Ax         product of Ax
 * @param[in]  m_size     row size of A
 * @param[in]  n_size     column size of A
 * @param[in]  layout     layout of A used for multiplication. Must be Normal or Transpose
 */
void lcg_matvec(lcg_float **A, const lcg_float *x, lcg_float *Ax, int m_size, int n_size, 
	lcg_matrix_e layout = MatNormal);

/**
 * @brief     Calculate the product of a sparse matrix multipled by a vector. The matrix is stored in the COO format.
 * 
 * @param row             Row index of the input sparse matrix.
 * @param col             Column index of the input sparse matrix.
 * @param Mat             Non-zero values of the input sparse matrix.
 * @param V               Multipler vector
 * @param p               Output prodcut
 * @param M               Row number of the sparse matrix
 * @param N               Column number of the sparse matrix
 * @param nz_size         Non-zero size of the matrix
 * @param pre_position    If ture, the multipler is seen as a row vector. Otherwise, it is treated as a column vector.
 */
void lcg_matvec_coo(const int *row, const int *col, const lcg_float *Mat, const lcg_float *V, lcg_float *p, int M, int N, int nz_size, bool pre_position = false);

#endif //_ALGEBRA_H