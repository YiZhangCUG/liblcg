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

#ifndef _CLCG_H
#define _CLCG_H

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
	const int x_size, matrix_layout_e layout, complex_conjugate_e conjugate);

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
	clcg_solver_enum solver_id = CLCG_CGS);

/**
 * @brief      Complex linear conjugate gradient solver class
 */
class CLCG_Solver
{
protected:
	clcg_para param_;

public:
	CLCG_Solver()
	{
		param_ = clcg_default_parameters();
	}

	virtual ~CLCG_Solver(){}

	/**
	 * 因为类的成员函数指针不能直接被调用，所以我们在这里定义一个静态的中转函数来辅助Ax函数的调用
	 * 这里我们利用reinterpret_cast将_Ax的指针转换到Ax上，需要注意的是成员函数的指针只能通过
	 * 实例对象进行调用，因此需要void* instance变量。
	*/
	static void _AxProduct(void *instance, const lcg_complex *x, lcg_complex *prod_Ax, 
		const int x_size, matrix_layout_e layout, complex_conjugate_e conjugate)
	{
		return reinterpret_cast<CLCG_Solver*>(instance)->AxProduct(x, prod_Ax, x_size, layout, conjugate);
	}
	virtual void AxProduct(const lcg_complex *x, lcg_complex *prod_Ax, 
		const int x_size, matrix_layout_e layout, complex_conjugate_e conjugate) = 0;

	static int _Progress(void* instance, const lcg_complex* m, const lcg_float converge, 
		const clcg_para* param, const int n_size, const int k)
	{
		return reinterpret_cast<CLCG_Solver*>(instance)->Progress(m, converge, param, n_size, k);
	}
	virtual int Progress(const lcg_complex* m, const lcg_float converge, 
		const clcg_para* param, const int n_size, const int k)
	{
		std::clog << "\rIteration-times: " << k << "\tconvergence: " << converge;
		return 0;
	}

	void set_clcg_parameter(const clcg_para &in_param)
	{
		param_ = in_param;
		return;
	}

	void Minimize(lcg_complex *m, const lcg_complex *b, int x_size, 
		clcg_solver_enum solver_id = CLCG_CGS, bool verbose = true, 
		bool er_throw = false)
	{
		if (!er_throw)
		{
			switch (solver_id)
			{
				case CLCG_BICG:
					std::clog << "Solver: Bi-Conjugate Gradient" << std::endl;
					break;
				case CLCG_BICG_SYM:
					std::clog << "Solver: Bi-Conjugate Gradient (symmetrically accelerated)" << std::endl;
					break;
				case CLCG_CGS:
					std::clog << "Solver: Conjugate Gradient Squared" << std::endl;
					break;
				case CLCG_TFQMR:
					std::clog << "Solver: Transpose Free Quasi-Minimal Residual" << std::endl;
					break;
				default:
					std::clog << "Solver: Unknown" << std::endl;
					break;
			}
		}
		

		// 使用lcg求解 注意当我们使用函数指针来调用求解函数时默认参数不可以省略
		int ret = clcg_solver(_AxProduct, _Progress, m, b, x_size, &param_, this, solver_id);
		if (verbose)
		{
			if (!er_throw) std::clog << std::endl;
			clcg_error_str(ret, er_throw);
		}
		else if (ret < 0)
		{
			if (!er_throw) std::clog << std::endl;
			clcg_error_str(ret, er_throw);
		}
		return;
	}
};

#endif //_CLCG_H