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

#ifndef _LCG_COMPLEX_CUDA_H
#define _LCG_COMPLEX_CUDA_H

#include "lcg_complex.h"

#ifdef LibLCG_CUDA

#include <cuda_runtime.h>
#include <cuComplex.h>

/**
 * @brief  Convert cuda complex number to lcg complex number
 * 
 * @param a CUDA complex number
 * @return lcg_complex  lcg complex number
 */
lcg_complex cuda2lcg_complex(cuDoubleComplex a);

/**
 * @brief Convert lcg complex number to CUDA complex number
 * 
 * @param a lcg complex number
 * @return cuDoubleComplex CUDA complex number
 */
cuDoubleComplex lcg2cuda_complex(lcg_complex a);

/**
 * @brief      Locate memory for a cuDoubleComplex pointer type.
 *
 * @param[in]  n     Size of the lcg_float array.
 *
 * @return     Pointer of the array's location.
 */
cuDoubleComplex* clcg_malloc_cuda(size_t n);

/**
 * @brief      Destroy memory used by the cuDoubleComplex type array.
 *
 * @param      x     Pointer of the array.
 */
void clcg_free_cuda(cuDoubleComplex *x);

/**
 * @brief      set a complex vector's value
 *
 * @param      a     pointer of the vector
 * @param[in]  b     initial value
 * @param[in]  size  vector size
 */
void clcg_vecset_cuda(cuDoubleComplex *a, cuDoubleComplex b, size_t size);

/**
 * @brief    Host side function for scale a cuDoubleComplex object
 * 
 * @param s  scale factor
 * @param a  Complex number
 * @return cuComplex  scaled complex number
 */
cuComplex clcg_Cscale(lcg_float s, cuComplex a);

/**
 * @brief   Calculate the sum of two cuda complex number. This is a host side function.
 * 
 * @param a Complex number
 * @param b Complex number
 * @return cuComplex Sum of the input complex number 
 */
cuComplex clcg_Csum(cuComplex a, cuComplex b);

/**
 * @brief   Calculate the difference of two cuda complex number. This is a host side function.
 * 
 * @param a Complex number
 * @param b Complex number
 * @return cuComplex Difference of the input complex number 
 */
cuComplex clcg_Cdiff(cuComplex a, cuComplex b);

/**
 * @brief   Calculate the sqrt() of a cuda complex number
 * 
 * @param a Complex number
 * @return cuComplex root value
 */
cuComplex clcg_Csqrt(cuComplex a);

/**
 * @brief    Host side function for scale a cuDoubleComplex object
 * 
 * @param s  scale factor
 * @param a  Complex number
 * @return cuDoubleComplex  scaled complex number
 */
cuDoubleComplex clcg_Zscale(lcg_float s, cuDoubleComplex a);

/**
 * @brief   Calculate the sum of two cuda complex number. This is a host side function.
 * 
 * @param a Complex number
 * @param b Complex number
 * @return cuDoubleComplex Sum of the input complex number 
 */
cuDoubleComplex clcg_Zsum(cuDoubleComplex a, cuDoubleComplex b);

/**
 * @brief   Calculate the difference of two cuda complex number. This is a host side function.
 * 
 * @param a Complex number
 * @param b Complex number
 * @return cuDoubleComplex Difference of the input complex number 
 */
cuDoubleComplex clcg_Zdiff(cuDoubleComplex a, cuDoubleComplex b);

/**
 * @brief   Calculate the sqrt() of a cuda complex number
 * 
 * @param a Complex number
 * @return cuDoubleComplex root value
 */
cuDoubleComplex clcg_Zsqrt(cuDoubleComplex a);

/**
 * @brief   Convert the indexing sequence of a sparse matrix from the row-major to col-major format.
 * 
 * @note    The sparse matrix is stored in the COO foramt. This is a host side function.
 * 
 * @param A_row      Row index
 * @param A_col      Column index
 * @param A          Non-zero values of the matrix
 * @param N          Row/column length of A
 * @param nz         Number of the non-zero values in A
 * @param Ac_row     Output row index
 * @param Ac_col     Output column index
 * @param Ac_val     Non-zero values of the output matrix
 */
void clcg_smCcoo_row2col(const int *A_row, const int *A_col, const cuComplex *A, int N, int nz, int *Ac_row, int *Ac_col, cuComplex *Ac_val);

/**
 * @brief   Convert the indexing sequence of a sparse matrix from the row-major to col-major format.
 * 
 * @note    The sparse matrix is stored in the COO foramt. This is a host side function.
 * 
 * @param A_row      Row index
 * @param A_col      Column index
 * @param A          Non-zero values of the matrix
 * @param N          Row/column length of A
 * @param nz         Number of the non-zero values in A
 * @param Ac_row     Output row index
 * @param Ac_col     Output column index
 * @param Ac_val     Non-zero values of the output matrix
 */
void clcg_smZcoo_row2col(const int *A_row, const int *A_col, const cuDoubleComplex *A, int N, int nz, int *Ac_row, int *Ac_col, cuDoubleComplex *Ac_val);

/**
 * @brief      Extract diagonal elements from a square CUDA sparse matrix that is formatted in the CSR format
 * 
 * @note       This is a device side function. All memories must be allocated on the GPU device.
 *
 * @param[in]  A_ptr   Row index pointer
 * @param[in]  A_col   Column index
 * @param[in]  A_val   Non-zero values of the matrix
 * @param[in]  A_len   Dimension of the matrix
 * @param      A_diag  Output digonal elements
 * @param[in]  bk_size Default CUDA block size.
 */
void clcg_smCcsr_get_diagonal(const int *A_ptr, const int *A_col, const cuComplex *A_val, const int A_len, cuComplex *A_diag, int bk_size = 1024);

/**
 * @brief      Extract diagonal elements from a square CUDA sparse matrix that is formatted in the CSR format
 * 
 * @note       This is a device side function. All memories must be allocated on the GPU device.
 *
 * @param[in]  A_ptr   Row index pointer
 * @param[in]  A_col   Column index
 * @param[in]  A_val   Non-zero values of the matrix
 * @param[in]  A_len   Dimension of the matrix
 * @param      A_diag  Output digonal elements
 * @param[in]  bk_size Default CUDA block size.
 */
void clcg_smZcsr_get_diagonal(const int *A_ptr, const int *A_col, const cuDoubleComplex *A_val, const int A_len, cuDoubleComplex *A_diag, int bk_size = 1024);

/**
 * @brief      Element-wise muplication between two CUDA arries.
 * 
 * @note       This is a device side function. All memories must be allocated on the GPU device.
 *
 * @param[in]  a     Pointer of the input array
 * @param[in]  b     Pointer of the input array
 * @param      c     Pointer of the output array
 * @param[in]  n     Length of the arraies
 * @param[in]  bk_size Default CUDA block size.
 */
void clcg_vecMvecC_element_wise(const cuComplex *a, const cuComplex *b, cuComplex *c, int n, int bk_size = 1024);

/**
 * @brief      Element-wise muplication between two CUDA arries.
 * 
 * @note       This is a device side function. All memories must be allocated on the GPU device.
 *
 * @param[in]  a     Pointer of the input array
 * @param[in]  b     Pointer of the input array
 * @param      c     Pointer of the output array
 * @param[in]  n     Length of the arraies
 * @param[in]  bk_size Default CUDA block size.
 */
void clcg_vecMvecZ_element_wise(const cuDoubleComplex *a, const cuDoubleComplex *b, cuDoubleComplex *c, int n, int bk_size = 1024);

/**
 * @brief      Element-wise division between two CUDA arries.
 * 
 * @note       This is a device side function. All memories must be allocated on the GPU device.
 *
 * @param[in]  a     Pointer of the input array
 * @param[in]  b     Pointer of the input array
 * @param      c     Pointer of the output array
 * @param[in]  n     Length of the arraies
 * @param[in]  bk_size Default CUDA block size.
 */
void clcg_vecDvecC_element_wise(const cuComplex *a, const cuComplex *b, cuComplex *c, int n, int bk_size = 1024);

/**
 * @brief      Element-wise division between two CUDA arries.
 * 
 * @note       This is a device side function. All memories must be allocated on the GPU device.
 *
 * @param[in]  a     Pointer of the input array
 * @param[in]  b     Pointer of the input array
 * @param      c     Pointer of the output array
 * @param[in]  n     Length of the arraies
 * @param[in]  bk_size Default CUDA block size.
 */
void clcg_vecDvecZ_element_wise(const cuDoubleComplex *a, const cuDoubleComplex *b, cuDoubleComplex *c, int n, int bk_size = 1024);

/**
 * @brief      Return complex conjugates of an input CUDA complex array
 * 
 * @param a    Pointer of the input arra
 * @param ca   Pointer of the output array
 * @param n    Length of the arraies
 * @param[in]  bk_size Default CUDA block size.
 */
void clcg_vecC_conjugate(const cuComplex *a, cuComplex *ca, int n, int bk_size = 1024);

/**
 * @brief      Return complex conjugates of an input CUDA complex array
 * 
 * @param a    Pointer of the input arra
 * @param ca   Pointer of the output array
 * @param n    Length of the arraies
 * @param[in]  bk_size Default CUDA block size.
 */
void clcg_vecZ_conjugate(const cuDoubleComplex *a, cuDoubleComplex *ca, int n, int bk_size = 1024);

#endif // LibLCG_CUDA

#endif // _LCG_COMPLEX_CUDA_H