/******************************************************//**
 *    C/C++ library of linear conjugate gradient.
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

#ifndef _LCG_CXX_H
#define _LCG_CXX_H

#include "lcg.h"
#include "iostream"

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
		std::clog << "Iteration-times: " << k << "\tconvergence: " << converge << std::endl;
#ifdef __linux__
		if (converge > param->epsilon) std::clog << "\033[1A\033[K";
#elif defined __APPLE__
		if (converge > param->epsilon) std::clog << "\033[1A\033[K";
#endif
		return 0;
	}

	void set_lcg_parameter(const lcg_para &in_param)
	{
		param_ = in_param;
		return;
	}

	void Minimize(lcg_float *m, const lcg_float *b, int x_size, 
		lcg_solver_enum solver_id = LCG_CG, const lcg_float *p = NULL)
	{
		// 使用lcg求解 注意当我们使用函数指针来调用求解函数时默认参数不可以省略
		int ret = lcg_solver(_AxProduct, _Progress, m, b, x_size, &param_, this, solver_id, p);
		if (ret < 0) std::cerr << lcg_error_str(ret) << std::endl;
		return;
	}

	void MinimizeConstrained(lcg_float *m, const lcg_float *b, const lcg_float* low, 
		const lcg_float *hig, int x_size, lcg_solver_enum solver_id = LCG_PG)
	{
		// 使用lcg求解 注意当我们使用函数指针来调用求解函数时默认参数不可以省略
		int ret = lcg_solver_constrained(_AxProduct, _Progress, m, b, low, hig, x_size, &param_, this, solver_id);
		if (ret < 0) std::cerr << lcg_error_str(ret) << std::endl;
		return;
	}
};

#endif //_LCG_CXX_H