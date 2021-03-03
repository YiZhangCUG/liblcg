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

#ifndef _LCG_EIGEN_H
#define _LCG_EIGEN_H

#include "lcg_cmn.h"
#include "ctime"
#include "Eigen/Dense"

/**
 * @brief  Callback interface for calculating the product of a N*N matrix 'A' multiplied 
 * by a vertical vector 'x'.
 * 
 * @param  instance    The user data sent for the lcg_solver() functions by the client.
 * @param  x           Multiplier of the Ax product.
 * @param  Ax          Product of A multiplied by x.
 */
typedef void (*lcg_axfunc_eigen_ptr)(void* instance, const Eigen::VectorXd &x, Eigen::VectorXd &prod_Ax);

/**
 * @brief     Callback interface for monitoring the progress and terminate the iteration 
 * if necessary.
 * 
 * @param    instance    The user data sent for the lcg_solver() functions by the client.
 * @param    m           The current solutions.
 * @param    converge    The current value evaluating the iteration progress.
 * @param    k           The iteration count.
 * 
 * @retval   int         Zero to continue the optimization process. Returning a
 *                       non-zero value will terminate the optimization process.
 */
typedef int (*lcg_progress_eigen_ptr)(void* instance, const Eigen::VectorXd *m, const lcg_float converge, 
	const lcg_para *param, const int k);

/**
 * @brief      A combined conjugate gradient solver function.
 *
 * @param[in]  Afp         Callback function for calculating the product of 'Ax'.
 * @param[in]  Pfp         Callback function for monitoring the iteration progress.
 * @param      m           Initial solution vector.
 * @param      B           Objective vector of the linear system.
 * @param      param       Parameter setup for the conjugate gradient methods.
 * @param      instance    The user data sent for the lcg_solver() function by the client. 
 * This variable is either 'this' for class member functions or 'NULL' for global functions.
 * @param      solver_id   Solver type used to solve the linear system. The default value is LCG_CGS.
 * @param      P           Precondition vector (optional expect for the LCG_PCG method). The default value is NULL.
 *
 * @return     Status of the function.
 */
int lcg_solver_eigen(lcg_axfunc_eigen_ptr Afp, lcg_progress_eigen_ptr Pfp, Eigen::VectorXd &m, 
	const Eigen::VectorXd &B, const lcg_para* param, void* instance, lcg_solver_enum solver_id = LCG_CG, 
	const Eigen::VectorXd *P = nullptr);

/**
 * @brief      A combined conjugate gradient solver function with inequality constraints.
 *
 * @param[in]  Afp         Callback function for calculating the product of 'Ax'.
 * @param[in]  Pfp         Callback function for monitoring the iteration progress.
 * @param      m           Initial solution vector.
 * @param      B           Objective vector of the linear system.
 * @param[in]  low         The lower boundary of the acceptable solution.
 * @param[in]  hig         The higher boundary of the acceptable solution.
 * @param      param       Parameter setup for the conjugate gradient methods.
 * @param      instance    The user data sent for the lcg_solver() function by the client. 
 * This variable is either 'this' for class member functions or 'NULL' for global functions.
 * @param      solver_id   Solver type used to solve the linear system. The default value is LCG_CGS.
 * @param      P           Precondition vector (optional expect for the LCG_PCG method). The default value is NULL.
 *
 * @return     Status of the function.
 */
int lcg_solver_constrained_eigen(lcg_axfunc_eigen_ptr Afp, lcg_progress_eigen_ptr Pfp, Eigen::VectorXd &m, 
	const Eigen::VectorXd &B, const Eigen::VectorXd &low, const Eigen::VectorXd &hig, 
	const lcg_para* param, void* instance, lcg_solver_enum solver_id = LCG_PG);

/**
 * @brief      Linear conjugate gradient solver class
 */
class LCG_EIGEN_Solver
{
protected:
	lcg_para param_;

public:
	LCG_EIGEN_Solver()
	{
		param_ = lcg_default_parameters();
	}

	virtual ~LCG_EIGEN_Solver(){}

	/**
	 * 因为类的成员函数指针不能直接被调用，所以我们在这里定义一个静态的中转函数来辅助Ax函数的调用
	 * 这里我们利用reinterpret_cast将_Ax的指针转换到Ax上，需要注意的是成员函数的指针只能通过
	 * 实例对象进行调用，因此需要void* instance变量。
	*/
	static void _AxProduct(void* instance, const Eigen::VectorXd &x, Eigen::VectorXd &prod_Ax)
	{
		return reinterpret_cast<LCG_EIGEN_Solver*>(instance)->AxProduct(x, prod_Ax);
	}
	virtual void AxProduct(const Eigen::VectorXd &x, Eigen::VectorXd &prod_Ax) = 0;

	static int _Progress(void* instance, const Eigen::VectorXd *m, const lcg_float converge, 
		const lcg_para *param, const int k)
	{
		return reinterpret_cast<LCG_EIGEN_Solver*>(instance)->Progress(m, converge, param, k);
	}
	virtual int Progress(const Eigen::VectorXd *m, const lcg_float converge, const lcg_para *param, 
		const int k)
	{
		std::clog << "\rIteration-times: " << k << "\tconvergence: " << converge;
		return 0;
	}

	void set_lcg_parameter(const lcg_para &in_param)
	{
		param_ = in_param;
		return;
	}

	void Minimize(Eigen::VectorXd &m, const Eigen::VectorXd &b, lcg_solver_enum solver_id = LCG_CG, 
		const Eigen::VectorXd *p = nullptr, bool verbose = true, bool er_throw = false)
	{
		// 使用lcg求解 注意当我们使用函数指针来调用求解函数时默认参数不可以省略
		clock_t start = clock();
		int ret = lcg_solver_eigen(_AxProduct, _Progress, m, b, &param_, this, solver_id, p);
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
};

#endif //_LCG_EIGEN_H