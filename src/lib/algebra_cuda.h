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

#ifndef _ALGEBRA_CUDA_H
#define _ALGEBRA_CUDA_H

#include "algebra.h"

#ifdef LibLCG_CUDA

#include <cuda_runtime.h>

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
void lcg_set2box_cuda(const lcg_float *low, const lcg_float *hig, lcg_float *a, 
    int n, bool low_bound = true, bool hig_bound = true);

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
void lcg_smDcsr_get_diagonal(const int *A_ptr, const int *A_col, const lcg_float *A_val, const int A_len, lcg_float *A_diag, int bk_size = 1024);

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
void lcg_vecMvecD_element_wise(const lcg_float *a, const lcg_float *b, lcg_float *c, int n, int bk_size = 1024);

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
void lcg_vecDvecD_element_wise(const lcg_float *a, const lcg_float *b, lcg_float *c, int n, int bk_size = 1024);

#endif // LibLCG_CUDA

#endif //_ALGEBRA_CUDA_H