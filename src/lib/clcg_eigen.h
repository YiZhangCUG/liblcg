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

#ifndef _CLCG_EIGEN_H
#define _CLCG_EIGEN_H

#include "util.h"
#include "complex"
#include "Eigen/Dense"

/**
 * @brief  Callback interface for calculating the product of a N*N matrix 'A' multiplied 
 * by a vertical vector 'x'.
 * 
 * @param  instance    The user data sent for the solver functions by the client.
 * @param  x           Multiplier of the Ax product.
 * @param  Ax          Product of A multiplied by x.
 * @param  layout      layout information of the matrix A passed by the solver functions.
 * @param  conjugate   Layout information of the matrix A passed by the solver functions.
 */
typedef void (*clcg_axfunc_eigen_ptr)(void* instance, const Eigen::VectorXcd &x, Eigen::VectorXcd &prod_Ax, 
	lcg_matrix_e layout, clcg_complex_e conjugate);

/**
 * @brief     Callback interface for monitoring the progress and terminate the iteration 
 * if necessary.
 * 
 * @param    instance    The user data sent for the solver functions by the client.
 * @param    m           The current solutions.
 * @param    converge    The current value evaluating the iteration progress.
 * @param    param       The parameter object passed by the solver functions.
 * @param    k           The iteration count.
 * 
 * @retval   int         Zero to continue the optimization process. Returning a
 *                       non-zero value will terminate the optimization process.
 */
typedef int (*clcg_progress_eigen_ptr)(void* instance, const Eigen::VectorXcd *m, const lcg_float converge, 
	const clcg_para *param, const int k);

/**
 * @brief      A combined conjugate gradient solver function.
 *
 * @param[in]  Afp         Callback function for calculating the product of 'Ax'.
 * @param[in]  Pfp         Callback function for monitoring the iteration progress.
 * @param      m           Initial solution vector.
 * @param      B           Objective vector of the linear system.
 * @param      param       Parameter setup for the conjugate gradient methods.
 * @param      instance    The user data sent for the solver function by the client. 
 * This variable is either 'this' for class member functions or 'nullptr' for global functions.
 * @param      solver_id   Solver type used to solve the linear system. The default value is CLCG_CGS.
 *
 * @return     Status of the function.
 */
int clcg_solver_eigen(clcg_axfunc_eigen_ptr Afp, clcg_progress_eigen_ptr Pfp, Eigen::VectorXcd &m, 
	const Eigen::VectorXcd &B, const clcg_para* param, void* instance, clcg_solver_enum solver_id = CLCG_CGS);

/**
 * @brief      A combined conjugate gradient solver function.
 *
 * @param[in]  Afp         Callback function for calculating the product of 'Ax'.
 * @param[in]  Mfp         Callback function for calculating the product of 'M^{-1}x', in which M is the preconditioning matrix
 * @param[in]  Pfp         Callback function for monitoring the iteration progress.
 * @param      m           Initial solution vector.
 * @param      B           Objective vector of the linear system.
 * @param      param       Parameter setup for the conjugate gradient methods.
 * @param      instance    The user data sent for the solver function by the client. 
 * This variable is either 'this' for class member functions or 'nullptr' for global functions.
 * @param      solver_id   Solver type used to solve the linear system. the value must CLCG_PBICG (default) or CLCG_PCG.
 *
 * @return     Status of the function.
 */
int clcg_solver_preconditioned_eigen(clcg_axfunc_eigen_ptr Afp, clcg_axfunc_eigen_ptr Mfp, clcg_progress_eigen_ptr Pfp, 
    Eigen::VectorXcd &m, const Eigen::VectorXcd &B, const clcg_para* param, void* instance, clcg_solver_enum solver_id = CLCG_PBICG);

#endif // _CLCG_EIGEN_H