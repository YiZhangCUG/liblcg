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

#include "lcg_cmn.h"
#include "ctime"

/**
 * @brief  Callback interface for calculating the product of a N*N matrix 'A' multiplied 
 * by a vertical vector 'x'.
 * 
 * @param  instance    The user data sent for the lcg_solver() functions by the client.
 * @param  x           Multiplier of the Ax product.
 * @param  Ax          Product of A multiplied by x.
 * @param  n_size      Size of x and column/row numbers of A.
 */
typedef void (*lcg_axfunc_ptr)(void* instance, const lcg_float* x, lcg_float* prod_Ax, 
	const int n_size);

/**
 * @brief     Callback interface for monitoring the progress and terminate the iteration 
 * if necessary.
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
typedef int (*lcg_progress_ptr)(void* instance, const lcg_float* m, const lcg_float converge, 
	const lcg_para* param, const int n_size, const int k);

/**
 * @brief      A combined conjugate gradient solver function.
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
 * @param      P           Precondition vector (optional expect for the LCG_PCG method). The default value is NULL.
 *
 * @return     Status of the function.
 */
int lcg_solver(lcg_axfunc_ptr Afp, lcg_progress_ptr Pfp, lcg_float* m, const lcg_float* B, const int n_size, 
	const lcg_para* param, void* instance, lcg_solver_enum solver_id = LCG_CGS, const lcg_float* P = nullptr);

/**
 * @brief      A combined conjugate gradient solver function with inequality constraints.
 *
 * @param[in]  Afp         Callback function for calculating the product of 'Ax'.
 * @param[in]  Pfp         Callback function for monitoring the iteration progress.
 * @param      m           Initial solution vector.
 * @param      B           Objective vector of the linear system.
 * @param[in]  low         The lower boundary of the acceptable solution.
 * @param[in]  hig         The higher boundary of the acceptable solution.
 * @param[in]  n_size      Size of the solution vector and objective vector.
 * @param      param       Parameter setup for the conjugate gradient methods.
 * @param      instance    The user data sent for the lcg_solver() function by the client. 
 * This variable is either 'this' for class member functions or 'NULL' for global functions.
 * @param      solver_id   Solver type used to solve the linear system. The default value is LCG_CGS.
 * @param      P           Precondition vector (optional expect for the LCG_PCG method). The default value is NULL.
 *
 * @return     Status of the function.
 */
int lcg_solver_constrained(lcg_axfunc_ptr Afp, lcg_progress_ptr Pfp, lcg_float* m, const lcg_float* B, 
	const lcg_float* low, const lcg_float *hig, const int n_size, const lcg_para* param, 
	void* instance, lcg_solver_enum solver_id = LCG_PG);

/**
 * @brief      Linear conjugate gradient solver class
 */
class LCG_Solver
{
protected:
	lcg_para param_;

public:
	LCG_Solver()
	{
		param_ = lcg_default_parameters();
	}

	virtual ~LCG_Solver(){}

	/**
	 * ?????????????????????????????????????????????????????????????????????????????????????????????????????????????????????Ax???????????????
	 * ??????????????????reinterpret_cast???_Ax??????????????????Ax?????????????????????????????????????????????????????????
	 * ???????????????????????????????????????void* instance?????????
	*/
	static void _AxProduct(void* instance, const lcg_float* a, lcg_float* b, const int num)
	{
		return reinterpret_cast<LCG_Solver*>(instance)->AxProduct(a, b, num);
	}
	virtual void AxProduct(const lcg_float* a, lcg_float* b, const int num) = 0;

	static int _Progress(void* instance, const lcg_float* m, const lcg_float converge, 
		const lcg_para *param, const int n_size, const int k)
	{
		return reinterpret_cast<LCG_Solver*>(instance)->Progress(m, converge, param, n_size, k);
	}
	virtual int Progress(const lcg_float* m, const lcg_float converge, 
		const lcg_para *param, const int n_size, const int k)
	{
		std::clog << "\rIteration-times: " << k << "\tconvergence: " << converge;
		return 0;
	}

	void set_lcg_parameter(const lcg_para &in_param)
	{
		param_ = in_param;
		return;
	}

	void Minimize(lcg_float *m, const lcg_float *b, int x_size, 
		lcg_solver_enum solver_id = LCG_CG, const lcg_float *p = NULL, 
		bool verbose = true, bool er_throw = false)
	{
		// ??????lcg?????? ????????????????????????????????????????????????????????????????????????????????????
		clock_t start = clock();
		int ret = lcg_solver(_AxProduct, _Progress, m, b, x_size, &param_, this, solver_id, p);
		clock_t end = clock();

		lcg_float costime = 1000*(end-start)/(double)CLOCKS_PER_SEC;
		if (!er_throw)
		{
			std::clog << std::endl;
			switch (solver_id)
			{
				case LCG_CG:
					std::clog << "Solver: CG. Time cost: " << costime << " ms" << std::endl;
					break;
				case LCG_PCG:
					std::clog << "Solver: PCG. Time cost: " << costime << " ms" << std::endl;
					break;
				case LCG_CGS:
					std::clog << "Solver: CGS. Time cost: " << costime << " ms" << std::endl;
					break;
				case LCG_BICGSTAB:
					std::clog << "Solver: BICGSTAB. Times cost: " << costime << " ms" << std::endl;
					break;
				case LCG_BICGSTAB2:
					std::clog << "Solver: BICGSTAB2. Time cost: " << costime << " ms" << std::endl;
					break;
				default:
					std::clog << "Solver: Unknown. Time cost: " << costime << std::endl;
					break;
			}	
		}

		if (verbose) lcg_error_str(ret, er_throw);
		else if (ret < 0) lcg_error_str(ret, er_throw);
		return;
	}

	void MinimizeConstrained(lcg_float *m, const lcg_float *b, const lcg_float* low, 
		const lcg_float *hig, int x_size, lcg_solver_enum solver_id = LCG_PG, 
		bool verbose = true, bool er_throw = false)
	{
		// ??????lcg?????? ????????????????????????????????????????????????????????????????????????????????????
		clock_t start = clock();
		int ret = lcg_solver_constrained(_AxProduct, _Progress, m, b, low, hig, x_size, &param_, this, solver_id);
		clock_t end = clock();

		lcg_float costime = 1000*(end-start)/(double)CLOCKS_PER_SEC;
		if (!er_throw)
		{
			std::clog << std::endl;
			switch (solver_id)
			{
				case LCG_PG:
					std::clog << "Solver: PG-CG. Time cost: " << costime << " ms" << std::endl;
					break;
				case LCG_SPG:
					std::clog << "Solver: SPG-CG. Time cost: " << costime << " ms" << std::endl;
					break;
				default:
					std::clog << "Solver: Unknown. Time cost: " << costime << std::endl;
					break;
			}	
		}

		if (verbose) lcg_error_str(ret, er_throw);
		else if (ret < 0) lcg_error_str(ret, er_throw);
		return;
	}
};

#endif //_LCG_H