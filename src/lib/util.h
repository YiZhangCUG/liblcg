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

#ifndef _LCG_UTIL_H
#define _LCG_UTIL_H

#include "string"
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
	LCG_SIZE_NOT_MATCH, ///< Sizes of m and B do not match
};

/**
 * @brief      Parameters of the conjugate gradient methods.
 */
struct lcg_para
{
	/**
	 * Maximal iteration times. The process will continue till the convergance is met
	 * if this option is set to zero (default).
	*/
	int max_iterations;

	/**
	 * Epsilon for convergence test.
	 * This parameter determines the accuracy with which the solution is to be 
	 * found. A minimization terminates when ||g||/max(||x||, 1.0) <= epsilon or 
	 * sqrt(||g||)/N <= epsilon for the lcg_solver() function, where ||.|| denotes 
	 * the Euclidean (L2) norm. The default value of epsilon is 1e-6.
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
	lcg_float step;

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
static const lcg_para defparam = {0, 1e-6, 0, 1e-6, 1.0, 0.95, 0.9, 10};

/**
 * @brief      Return a lcg_para type instance with default values.
 * 
 * Users can use this function to get default parameters' value for the conjugate gradient methods.
 * 
 * @return     A lcg_para type instance.
 */
lcg_para lcg_default_parameters();

/**
 * @brief      Select a type of solver according to the name
 *
 * @param[in]  slr_char  Name of the solver
 *
 * @return     The lcg solver enum.
 */
lcg_solver_enum lcg_select_solver(std::string slr_char);

/**
 * @brief      Display or throw out a string explanation for the lcg_solver() function's return values.
 *
 * @param[in]  er_index  The error index returned by the lcg_solver() function.
 * @param[in]  er_throw  throw out a char string of the explanation.
 *
 * @return     A string explanation of the error.
 */
void lcg_error_str(int er_index, bool er_throw = false);


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
	 * Biconjugate gradient method.
	 */
	CLCG_BICGSTAB,
	/**
	 * Quasi-Minimal Residual Method
	 */
	//CLCG_QMR,
	/**
	 * Transpose Free Quasi-Minimal Residual Method
	 */
	CLCG_TFQMR,
	/**
	 * Preconditioned conjugate gradient
	 */
	CLCG_PCG,
	/**
	 * Preconditioned Bi-Conjugate Gradient Method
	 */
	CLCG_PBICG,
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
	CLCG_SIZE_NOT_MATCH, ///< Sizes of m and B do not match
	CLCG_UNKNOWN_SOLVER, ///< Unknown solver
};

/**
 * @brief      Parameters of the conjugate gradient methods.
 */
struct clcg_para
{
	/**
	 * Maximal iteration times. The process will continue till the convergance is met
	 * if this option is set to zero (default).
	*/
	int max_iterations;

	/**
	 * Epsilon for convergence test.
	 * This parameter determines the accuracy with which the solution is to be found. 
	 * A minimization terminates when ||g||/max(||x||, 1.0) <= epsilon or |Ax - B| <= epsilon for 
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
static const clcg_para defparam2 = {0, 1e-6, 0};

/**
 * @brief      Return a clcg_para type instance with default values.
 * 
 * Users can use this function to get default parameters' value for the complex conjugate gradient methods.
 * 
 * @return     A clcg_para type instance.
 */
clcg_para clcg_default_parameters();

/**
 * @brief      Select a type of solver according to the name
 *
 * @param[in]  slr_char  Name of the solver
 *
 * @return     The clcg solver enum.
 */
clcg_solver_enum clcg_select_solver(std::string slr_char);

/**
 * @brief      Display or throw out a string explanation for the clcg_solver() function's return values.
 *
 * @param[in]  er_index  The error index returned by the lcg_solver() function.
 * @param[in]  er_throw  throw out a char string of the explanation.
 *
 * @return     A string explanation of the error.
 */
void clcg_error_str(int er_index, bool er_throw = false);

#endif // _LCG_UTIL_H