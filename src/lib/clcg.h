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

#ifndef _CLCG_H
#define _CLCG_H

#include "lcg_complex.h"
#include "util.h"

/**
 * @brief  Callback interface for calculating the complex product of a N*N matrix 'A' multiplied 
 * by a complex vertical vector 'x'.
 * 
 * @param  instance    The user data sent for the clcg_solver() functions by the client.
 * @param  x           Multiplier of the Ax product.
 * @param  Ax          Product of A multiplied by x.
 * @param  x_size      Size of x and column/row numbers of A.
 * @param  layout      Whether to use the transpose of A for calculation.
 * @param  conjugate   Whether to use the conjugate of A for calculation.
 */
typedef void (*clcg_axfunc_ptr)(void *instance, const lcg_complex *x, lcg_complex *prod_Ax, 
	const int x_size, lcg_matrix_e layout, clcg_complex_e conjugate);

/**
 * @brief     Callback interface for monitoring the progress and terminate the iteration 
 * if necessary.
 * 
 * @param    instance    The user data sent for the clcg_solver() functions by the client.
 * @param    m           The current solutions.
 * @param    converge    The current value evaluating the iteration progress.
 * @param    n_size      The size of the variables
 * @param    k           The iteration count.
 * 
 * @retval   int         Zero to continue the optimization process. Returning a
 *                       non-zero value will terminate the optimization process.
 */
typedef int (*clcg_progress_ptr)(void* instance, const lcg_complex* m, 
	const lcg_float converge, const clcg_para* param, const int n_size, const int k);

/**
 * @brief      A combined complex conjugate gradient solver function.
 *
 * @param[in]  Afp         Callback function for calculating the product of 'Ax'.
 * @param[in]  Pfp         Callback function for monitoring the iteration progress.
 * @param      m           Initial solution vector.
 * @param      B           Objective vector of the linear system.
 * @param[in]  n_size      Size of the solution vector and objective vector.
 * @param      param       Parameter setup for the conjugate gradient methods.
 * @param      instance    The user data sent for the lcg_solver() function by the client. 
 * This variable is either 'this' for class member functions or 'NULL' for global functions.
 * @param      solver_id   Solver type used to solve the linear system. The default value is LCG_CGS.
 *
 * @return     Status of the function.
 */
int clcg_solver(clcg_axfunc_ptr Afp, clcg_progress_ptr Pfp, lcg_complex* m, 
	const lcg_complex* B, const int n_size, const clcg_para* param, void* instance, 
	clcg_solver_enum solver_id = CLCG_BICG);

#endif // _CLCG_H