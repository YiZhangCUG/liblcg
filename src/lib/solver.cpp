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

#include "solver.h"

#include "ctime"
#include "iostream"

#include "config.h"
#ifdef LibLCG_OPENMP
#include "omp.h"
#endif

LCG_Solver::LCG_Solver()
{
	param_ = lcg_default_parameters();
	inter_ = 1;
	silent_ = false;
}

int LCG_Solver::Progress(const lcg_float* m, const lcg_float converge, 
	const lcg_para *param, const int n_size, const int k)
{
	if (inter_ > 0 && k%inter_ == 0)
	{
		std::clog << "\rIteration-times: " << k << "\tconvergence: " << converge;
		return 0;
	}

	if (converge <= param->epsilon)
	{
		std::clog << "\rIteration-times: " << k << "\tconvergence: " << converge;
	}
	return 0;
}

void LCG_Solver::silent()
{
	silent_ = true;
	return;
}

void LCG_Solver::set_report_interval(unsigned int inter)
{
	inter_ = inter;
	return;
}

void LCG_Solver::set_lcg_parameter(const lcg_para &in_param)
{
	param_ = in_param;
	return;
}

void LCG_Solver::Minimize(lcg_float *m, const lcg_float *b, int x_size, 
	lcg_solver_enum solver_id, bool verbose, bool er_throw)
{
	if (silent_)
	{
		int ret = lcg_solver(_AxProduct, nullptr, m, b, x_size, &param_, this, solver_id);
		if (ret < 0) lcg_error_str(ret, true);
		return;
	}
	
	// 使用lcg求解 注意当我们使用函数指针来调用求解函数时默认参数不可以省略
#ifdef LibLCG_OPENMP
	double start = omp_get_wtime();
	int ret = lcg_solver(_AxProduct, _Progress, m, b, x_size, &param_, this, solver_id);
	double end = omp_get_wtime();

	lcg_float costime = 1000*(end-start);
#else
	clock_t start = clock();
	int ret = lcg_solver(_AxProduct, _Progress, m, b, x_size, &param_, this, solver_id);
	clock_t end = clock();

	lcg_float costime = 1000*(end-start)/(double)CLOCKS_PER_SEC;
#endif
	
	if (!er_throw)
	{
		std::clog << std::endl;
		switch (solver_id)
		{
			case LCG_CG:
				std::clog << "Solver: CG. Time cost: " << costime << " ms" << std::endl;
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
				std::clog << "Solver: Unknown. Time cost: " << costime << " ms" << std::endl;
				break;
		}	
	}

	if (verbose) lcg_error_str(ret, er_throw);
	else if (ret < 0) lcg_error_str(ret, er_throw);
	return;
}

void LCG_Solver::MinimizePreconditioned(lcg_float *m, const lcg_float *b, int x_size, 
	lcg_solver_enum solver_id, bool verbose, bool er_throw)
{
	if (silent_)
	{
		int ret = lcg_solver_preconditioned(_AxProduct, _MxProduct, nullptr, m, b, x_size, &param_, this, solver_id);
		if (ret < 0) lcg_error_str(ret, true);
		return;
	}
	
	// 使用lcg求解 注意当我们使用函数指针来调用求解函数时默认参数不可以省略
#ifdef LibLCG_OPENMP
	double start = omp_get_wtime();
	int ret = lcg_solver_preconditioned(_AxProduct, _MxProduct, _Progress, m, b, x_size, &param_, this, solver_id);
	double end = omp_get_wtime();

	lcg_float costime = 1000*(end-start);
#else
	clock_t start = clock();
	int ret = lcg_solver_preconditioned(_AxProduct, _MxProduct, _Progress, m, b, x_size, &param_, this, solver_id);
	clock_t end = clock();

	lcg_float costime = 1000*(end-start)/(double)CLOCKS_PER_SEC;
#endif
	
	if (!er_throw)
	{
		std::clog << std::endl;
		switch (solver_id)
		{
			case LCG_PCG:
				std::clog << "Solver: PCG. Time cost: " << costime << " ms" << std::endl;
				break;
			default:
				std::clog << "Solver: Unknown. Time cost: " << costime << " ms" << std::endl;
				break;
		}	
	}

	if (verbose) lcg_error_str(ret, er_throw);
	else if (ret < 0) lcg_error_str(ret, er_throw);
	return;
}

void LCG_Solver::MinimizeConstrained(lcg_float *m, const lcg_float *b, const lcg_float* low, 
	const lcg_float *hig, int x_size, lcg_solver_enum solver_id, bool verbose, bool er_throw)
{
	if (silent_)
	{
		int ret = lcg_solver_constrained(_AxProduct, nullptr, m, b, low, hig, x_size, &param_, this, solver_id);
		if (ret < 0) lcg_error_str(ret, true);
		return;
	}

	// 使用lcg求解 注意当我们使用函数指针来调用求解函数时默认参数不可以省略
#ifdef LibLCG_OPENMP
	double start = omp_get_wtime();
	int ret = lcg_solver_constrained(_AxProduct, _Progress, m, b, low, hig, x_size, &param_, this, solver_id);
	double end = omp_get_wtime();

	lcg_float costime = 1000*(end-start);
#else
	clock_t start = clock();
	int ret = lcg_solver_constrained(_AxProduct, _Progress, m, b, low, hig, x_size, &param_, this, solver_id);
	clock_t end = clock();

	lcg_float costime = 1000*(end-start)/(double)CLOCKS_PER_SEC;
#endif

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
				std::clog << "Solver: Unknown. Time cost: " << costime << " ms" << std::endl;
				break;
		}
	}

	if (verbose) lcg_error_str(ret, er_throw);
	else if (ret < 0) lcg_error_str(ret, er_throw);
	return;
}


CLCG_Solver::CLCG_Solver()
{
	param_ = clcg_default_parameters();
	inter_ = 1;
	silent_ = false;
}

int CLCG_Solver::Progress(const lcg_complex* m, const lcg_float converge, 
	const clcg_para* param, const int n_size, const int k)
{
	if (inter_ > 0 && k%inter_ == 0)
	{
		std::clog << "\rIteration-times: " << k << "\tconvergence: " << converge;
		return 0;
	}

	if (converge <= param->epsilon)
	{
		std::clog << "\rIteration-times: " << k << "\tconvergence: " << converge;
	}
	return 0;
}

void CLCG_Solver::silent()
{
	silent_ = true;
	return;
}

void CLCG_Solver::set_report_interval(unsigned int inter)
{
	inter_ = inter;
	return;
}

void CLCG_Solver::set_clcg_parameter(const clcg_para &in_param)
{
	param_ = in_param;
	return;
}

void CLCG_Solver::Minimize(lcg_complex *m, const lcg_complex *b, int x_size, 
	clcg_solver_enum solver_id, bool verbose, bool er_throw)
{
	if (silent_)
	{
		int ret = clcg_solver(_AxProduct, nullptr, m, b, x_size, &param_, this, solver_id);
		if (ret < 0) clcg_error_str(ret, true);
		return;
	}

	// 使用lcg求解 注意当我们使用函数指针来调用求解函数时默认参数不可以省略
#ifdef LibLCG_OPENMP
	double start = omp_get_wtime();
	int ret = clcg_solver(_AxProduct, _Progress, m, b, x_size, &param_, this, solver_id);
	double end = omp_get_wtime();

	lcg_float costime = 1000*(end-start);
#else
	clock_t start = clock();
	int ret = clcg_solver(_AxProduct, _Progress, m, b, x_size, &param_, this, solver_id);
	clock_t end = clock();

	lcg_float costime = 1000*(end-start)/(double)CLOCKS_PER_SEC;
#endif

	if (!er_throw)
	{
		std::clog << std::endl;
		switch (solver_id)
		{
			case CLCG_BICG:
				std::clog << "Solver: Bi-CG. Times cost: " << costime << " ms" << std::endl;
				break;
			case CLCG_BICG_SYM:
				std::clog << "Solver: Bi-CG (symmetrically accelerated). Times cost: " << costime << " ms" << std::endl;
				break;
			case CLCG_CGS:
				std::clog << "Solver: CGS. Times cost: " << costime << " ms" << std::endl;
				break;
			case CLCG_TFQMR:
				std::clog << "Solver: TFQMR. Times cost: " << costime << " ms" << std::endl;
				break;
			default:
				std::clog << "Solver: Unknown. Times cost: " << costime << " ms" << std::endl;
				break;
		}
	}

	if (verbose) clcg_error_str(ret, er_throw);
	else if (ret < 0) clcg_error_str(ret, er_throw);
	return;
}