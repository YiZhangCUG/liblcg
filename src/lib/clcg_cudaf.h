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

#ifndef _CLCG_CUDA_FLOAT_H
#define _CLCG_CUDA_FLOAT_H

#include "util.h"
#include "lcg_complex_cuda.h"

#ifdef LibLCG_CUDA

#include <cublas_v2.h>
#include <cusparse_v2.h>

/**
 * @brief  Callback interface for calculating the product of a N*N matrix 'A' multiplied 
 * by a vertical vector 'x'. Note that both A and x are hosted on the GPU device.
 * 
 * @param  instance    The user data sent for the lcg_solver_cuda() functions by the client.
 * @param  cub_handle  Handler of the cublas object.
 * @param  cus_handle  Handlee of the cusparse object.
 * @param  x           Multiplier of the Ax product.
 * @param  Ax          Product of A multiplied by x.
 * @param  n_size      Size of x and column/row numbers of A.
 */
typedef void (*clcg_axfunc_cudaf_ptr)(void* instance, cublasHandle_t cub_handle, cusparseHandle_t cus_handle, 
    cusparseDnVecDescr_t x, cusparseDnVecDescr_t prod_Ax, const int n_size, const int nz_size, cusparseOperation_t oper_t);

/**
 * @brief     Callback interface for monitoring the progress and terminate the iteration 
 * if necessary. Note that m is hosted on the GPU device.
 * 
 * @param    instance    The user data sent for the lcg_solver() functions by the client.
 * @param    m           The current solutions.
 * @param    converge    The current value evaluating the iteration progress.
 * @param    n_size      The size of the variables
 * @param    k           The iteration count.
 * 
 * @retval   int         Zero to continue the optimization process. Returning a
 *                       non-zero value will terminate the optimization process.
 */
typedef int (*clcg_progress_cudaf_ptr)(void* instance, const cuComplex* m, const float converge, 
	const clcg_para* param, const int n_size, const int nz_size, const int k);

/**
 * @brief      A combined conjugate gradient solver function. Note that both m and B are hosted on the GPU device.
 *
 * @param[in]  Afp         Callback function for calculating the product of 'Ax'.
 * @param[in]  Pfp         Callback function for monitoring the iteration progress.
 * @param      m           Initial solution vector.
 * @param      B           Objective vector of the linear system.
 * @param[in]  n_size      Size of the solution vector and objective vector.
 * @param      param       Parameter setup for the conjugate gradient methods.
 * @param      instance    The user data sent for the lcg_solver() function by the client. 
 * @param      cub_handle  Handler of the cublas object.
 * @param      cus_handle  Handlee of the cusparse object.
 * This variable is either 'this' for class member functions or 'NULL' for global functions.
 * @param      solver_id   Solver type used to solve the linear system. The default value is LCG_BICG.
 *
 * @return     Status of the function.
 */
int clcg_solver_cuda(clcg_axfunc_cudaf_ptr Afp, clcg_progress_cudaf_ptr Pfp, cuComplex* m, const cuComplex* B, 
    const int n_size, const int nz_size, const clcg_para* param, void* instance, cublasHandle_t cub_handle, 
    cusparseHandle_t cus_handle, clcg_solver_enum solver_id = CLCG_BICG);

/**
 * @brief      A combined conjugate gradient solver function. Note that both m and B are hosted on the GPU device.
 *
 * @param[in]  Afp         Callback function for calculating the product of 'Ax'.
 * @param[in]  Mfp         Callback function for calculating the product of 'Mx' for preconditioning.
 * @param[in]  Pfp         Callback function for monitoring the iteration progress.
 * @param      m           Initial solution vector.
 * @param      B           Objective vector of the linear system.
 * @param[in]  n_size      Size of the solution vector and objective vector.
 * @param      param       Parameter setup for the conjugate gradient methods.
 * @param      instance    The user data sent for the lcg_solver() function by the client. 
 * @param      cub_handle  Handler of the cublas object.
 * @param      cus_handle  Handlee of the cusparse object.
 * This variable is either 'this' for class member functions or 'NULL' for global functions.
 * @param      solver_id   Solver type used to solve the linear system. The default value is LCG_CGS.
 *
 * @return     Status of the function.
 */
int clcg_solver_preconditioned_cuda(clcg_axfunc_cudaf_ptr Afp, clcg_axfunc_cudaf_ptr Mfp, clcg_progress_cudaf_ptr Pfp, 
    cuComplex* m, const cuComplex* B, const int n_size, const int nz_size, const clcg_para* param, void* instance, 
    cublasHandle_t cub_handle, cusparseHandle_t cus_handle, clcg_solver_enum solver_id = CLCG_PCG);

#endif // LibLCG_CUDA

#endif // _CLCG_CUDA_FLOAT_H