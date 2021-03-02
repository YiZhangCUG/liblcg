/******************************************************//**
 *    C++ library of linear conjugate gradient.
 *
 * Copyright (c) 2019-2029 Yi Zhang (zhangyiss@icloud.com)
 * All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *********************************************************/

#ifndef _LCG_H
#define _LCG_H

#include "algebra.h"

/**
 * @brief      Types of method that could be recognized by the lcg_solver() function.
 */
enum lcg_solver_enum
{
	/**
	 * Conjugate gradient method.
	 */
	LCG_CG,
	/**
	 * Preconditioned conjugate gradient method.
	 */
	LCG_PCG,
	/**
	 * Conjugate gradient squared method.
	 */
	LCG_CGS,
	/**
	 * Biconjugate gradient method.
	 */
	LCG_BICGSTAB,
	/**
	 * Biconjugate gradient method with restart.
	 */
	LCG_BICGSTAB2,
	/**
	 * Conjugate gradient method with projected gradient for inequality constraints.
	 * This algorithm comes without non-monotonic linear search for the step length.
	 */
	LCG_PG,
	/**
	 * Conjugate gradient method with spectral projected gradient for inequality constraints.
	 * This algorithm comes with non-monotonic linear search for the step length.
	 */
	LCG_SPG,
};

/**
 * @brief      return value of the lcg_solver() function
 */
enum lcg_return_enum
{
	LCG_SUCCESS = 0, ///< The solver function terminated successfully.
	LCG_CONVERGENCE = 0, ///< The iteration reached convergence.
	LCG_STOP, ///< The iteration is stopped by the monitoring function.
	LCG_ALREADY_OPTIMIZIED, ///< The initial solution is already optimized.
	// A negative number means a error
	LCG_UNKNOWN_ERROR = -1024, ///< Unknown error.
	LCG_INVILAD_VARIABLE_SIZE, ///< The variable size is negative
	LCG_INVILAD_MAX_ITERATIONS, ///< The maximal iteration times is negative.
	LCG_INVILAD_EPSILON, ///< The epsilon is negative.
	LCG_INVILAD_RESTART_EPSILON, ///< The restart epsilon is negative.
	LCG_REACHED_MAX_ITERATIONS, ///< Iteration reached maximal limit.
	LCG_NULL_PRECONDITION_MATRIX, ///< Null precondition matrix.
	LCG_NAN_VALUE, ///< Nan value.
	LCG_INVALID_POINTER, ///< Invalid pointer.
	LCG_INVALID_LAMBDA, ///< Invalid range for lambda.
	LCG_INVALID_SIGMA, ///< Invalid range for sigma.
	LCG_INVALID_BETA, ///< Invalid range for beta.
	LCG_INVALID_MAXIM, ///< Invalid range for maxi_m.
};

/**
 * @brief      Parameters of the conjugate gradient methods.
 */
struct lcg_para
{
	/**
	 * Maximal iteration times. The default value is 100. one adjust this parameter 
	 * by passing a lcg_para type to the lcg_solver() function.
	*/
	int max_iterations;

	/**
	 * Epsilon for convergence test.
	 * This parameter determines the accuracy with which the solution is to be found. 
	 * A minimization terminates when ||g||/||b|| <= epsilon or |Ax - B| <= epsilon for 
	 * the lcg_solver() function, where ||.|| denotes the Euclidean (L2) norm and | | 
	 * denotes the L1 norm. The default value of epsilon is 1e-6. For box-constrained methods,
	 * the convergence test is implemented using ||P(m-g) - m|| <= epsilon, in which P is the
	 * projector that transfers m into the constrained domain.
	*/
	lcg_float epsilon;

	/**
	 * Whether to use absolute mean differences (AMD) between |Ax - B| to evaluate the process. 
	 * The default value is false which means the gradient based evaluating method is used. 
	 * The AMD based method will be used if this variable is set to true. This parameter is only 
	 * applied to the non-constrained methods.
	 */
	int abs_diff;

	/**
	 * Restart epsilon for the LCG_BICGSTAB2 algorithm. The default value is 1e-6
	 */
	lcg_float restart_epsilon;

	/**
	 * Initial step length for the project gradient method. The default is 1.0
	 */
	lcg_float lambda;

	/**
	 * multiplier for updating solutions with the spectral projected gradient method. The range of
	 * this variable is (0, 1). The default is given as 0.95
	 */
	lcg_float sigma;

	/**
	 * descending ratio for conducting the non-monotonic linear search. The range of
	 * this variable is (0, 1). The default is given as 0.9
	 */
	lcg_float beta;

	/**
	 * The maximal record times of the objective values for the SPG method. The method use the 
	 * objective values from the most recent maxi_m times to preform the non-monotonic linear search.
	 * The default value is 10.
	 */
	int maxi_m;
};

/**
 * Default parameter for conjugate gradient methods
 */
static const lcg_para defparam = {100, 1e-6, 0, 1e-6, 1.0, 0.95, 0.9, 10};

/**
 * @brief      Return a lcg_para type instance with default values.
 * 
 * Users can use this function to get default parameters' value for the conjugate gradient methods.
 * 
 * @return     A lcg_para type instance.
 */
lcg_para lcg_default_parameters();

/**
 * @brief      Display or throw out a string explanation for the lcg_solver() function's return values.
 *
 * @param[in]  er_index  The error index returned by the lcg_solver() function.
 * @param[in]  er_throw  throw out a char string of the explanation.
 *
 * @return     A string explanation of the error.
 */
void lcg_error_str(int er_index, bool er_throw = false);

#endif //_LCG_H