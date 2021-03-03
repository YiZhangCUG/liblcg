/******************************************************//**
 *    C++ library of complex linear conjugate gradient.
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

#ifndef _CLCG_CMN_H
#define _CLCG_CMN_H

#include "algebra.h"

/**
 * @brief      Types of method that could be recognized by the clcg_solver() function.
 */
enum clcg_solver_enum
{
	/**
	 * Jacob's Bi-Conjugate Gradient Method
	 */
	CLCG_BICG,
	/**
	 * Bi-Conjugate Gradient Method accelerated for complex symmetric A
	 */
	CLCG_BICG_SYM,
	/**
	 * Conjugate Gradient Squared Method with real coefficients.
	 */
	CLCG_CGS,
	/**
	 * Transpose Free Quasi-Minimal Residual Method
	 */
	CLCG_TFQMR,
};

/**
 * @brief      return value of the clcg_solver() function
 */
enum clcg_return_enum
{
	CLCG_SUCCESS = 0, ///< The solver function terminated successfully.
	CLCG_CONVERGENCE = 0, ///< The iteration reached convergence.
	CLCG_STOP, ///< The iteration is stopped by the monitoring function.
	CLCG_ALREADY_OPTIMIZIED, ///< The initial solution is already optimized.
	// A negative number means a error
	CLCG_UNKNOWN_ERROR = -1024, ///< Unknown error.
	CLCG_INVILAD_VARIABLE_SIZE, ///< The variable size is negative
	CLCG_INVILAD_MAX_ITERATIONS, ///< The maximal iteration times is negative.
	CLCG_INVILAD_EPSILON, ///< The epsilon is negative.
	CLCG_REACHED_MAX_ITERATIONS, ///< Iteration reached maximal limit.
	CLCG_NAN_VALUE, ///< Nan value.
	CLCG_INVALID_POINTER, ///< Invalid pointer.
};

/**
 * @brief      Parameters of the conjugate gradient methods.
 */
struct clcg_para
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
};

/**
 * Default parameter for conjugate gradient methods
 */
static const clcg_para defparam = {100, 1e-6, 0};

/**
 * @brief      Return a clcg_para type instance with default values.
 * 
 * Users can use this function to get default parameters' value for the complex conjugate gradient methods.
 * 
 * @return     A clcg_para type instance.
 */
clcg_para clcg_default_parameters();

/**
 * @brief      Display or throw out a string explanation for the clcg_solver() function's return values.
 *
 * @param[in]  er_index  The error index returned by the lcg_solver() function.
 * @param[in]  er_throw  throw out a char string of the explanation.
 *
 * @return     A string explanation of the error.
 */
void clcg_error_str(int er_index, bool er_throw = false);


#endif //_CLCG_CMN_H