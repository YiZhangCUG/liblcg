/******************************************************//**
 *    C/C++ library of complex linear conjugate gradient.
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

#ifndef _CLCG_CXX_H
#define _CLCG_CXX_H

#include "clcg.h"
#include "iostream"

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
	static void _AxProduct(void *instance, const clcg_complex *x, clcg_complex *prod_Ax, 
		const int x_size, matrix_layout_e layout, complex_conjugate_e conjugate)
	{
		return reinterpret_cast<CLCG_Solver*>(instance)->AxProduct(x, prod_Ax, x_size, layout, conjugate);
	}
	virtual void AxProduct(const clcg_complex *x, clcg_complex *prod_Ax, 
		const int x_size, matrix_layout_e layout, complex_conjugate_e conjugate) = 0;

	static int _Progress(void* instance, const clcg_complex* m, const lcg_float converge, 
		const clcg_para* param, const int n_size, const int k)
	{
		return reinterpret_cast<CLCG_Solver*>(instance)->Progress(m, converge, param, n_size, k);
	}
	virtual int Progress(const clcg_complex* m, const lcg_float converge, 
		const clcg_para* param, const int n_size, const int k)
	{
		// 如果迭代达不到收敛值则会出现一个bug 后面再解决
		if (converge > param->epsilon)
			std::clog << "\rIteration-times: " << k << "\tconvergence: " << converge;
		else std::clog << "\rIteration-times: " << k << "\tconvergence: " << converge << std::endl;
		return 0;
	}

	void set_clcg_parameter(const clcg_para &in_param)
	{
		param_ = in_param;
		return;
	}

	void Minimize(clcg_complex *m, const clcg_complex *b, int x_size, 
		clcg_solver_enum solver_id = CLCG_CGS, bool verbose = true)
	{
		// 使用lcg求解 注意当我们使用函数指针来调用求解函数时默认参数不可以省略
		int ret = clcg_solver(_AxProduct, _Progress, m, b, x_size, &param_, this, solver_id);
		if (verbose)
		{
			switch (solver_id)
			{
				case CLCG_BICG:
					std::cerr << "Solver: Bi-Conjugate Gradient" << std::endl;
					break;
				case CLCG_CGS:
					std::cerr << "Solver: Conjugate Gradient Squared" << std::endl;
					break;
				case CLCG_TFQMR:
					std::cerr << "Solver: Transpose Free Quasi-Minimal Residual" << std::endl;
					break;
				default:
					std::cerr << "Solver: Unknown" << std::endl;
					break;
			}
			std::cerr << clcg_error_str(ret) << std::endl;
		}
		else if (ret < 0) std::cerr << clcg_error_str(ret) << std::endl;
		return;
	}
};

#endif //_CLCG_CXX_H