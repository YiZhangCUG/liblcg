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

#include "solver_cuda.h"

#include "cmath"
#include "ctime"
#include "iostream"

LCG_CUDA_Solver::LCG_CUDA_Solver()
{
	param_ = lcg_default_parameters();
	inter_ = 1;
	silent_ = false;
}

int LCG_CUDA_Solver::Progress(const lcg_float* m, const lcg_float converge, 
	const lcg_para* param, const int n_size, const int nz_size, const int k)
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

void LCG_CUDA_Solver::silent()
{
	silent_ = true;
	return;
}

void LCG_CUDA_Solver::set_report_interval(unsigned int inter)
{
	inter_ = inter;
	return;
}

void LCG_CUDA_Solver::set_lcg_parameter(const lcg_para &in_param)
{
	param_ = in_param;
	return;
}

void LCG_CUDA_Solver::Minimize(cublasHandle_t cub_handle, cusparseHandle_t cus_handle, lcg_float *x, lcg_float *b, 
	const int n_size, const int nz_size, lcg_solver_enum solver_id, bool verbose, bool er_throw)
{
	if (silent_)
	{
		int ret = lcg_solver_cuda(_AxProduct, nullptr, x, b, n_size, nz_size, &param_, this, cub_handle, cus_handle, solver_id);
		if (ret < 0) lcg_error_str(ret, true);
		return;
	}
	
	// 使用lcg求解 注意当我们使用函数指针来调用求解函数时默认参数不可以省略
	clock_t start = clock();
	int ret = lcg_solver_cuda(_AxProduct, _Progress, x, b, n_size, nz_size, &param_, this, cub_handle, cus_handle, solver_id);
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
			case LCG_CGS:
				std::clog << "Solver: CGS. Time cost: " << costime << " ms" << std::endl;
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

void LCG_CUDA_Solver::MinimizePreconditioned(cublasHandle_t cub_handle, cusparseHandle_t cus_handle, lcg_float *x, lcg_float *b, 
    const int n_size, const int nz_size, lcg_solver_enum solver_id, bool verbose, bool er_throw)
{
	if (silent_)
	{
		int ret = lcg_solver_preconditioned_cuda(_AxProduct, _MxProduct, nullptr, x, b, n_size, nz_size, &param_, this, cub_handle, cus_handle, solver_id);
		if (ret < 0) lcg_error_str(ret, true);
		return;
	}
	
	// 使用lcg求解 注意当我们使用函数指针来调用求解函数时默认参数不可以省略
	clock_t start = clock();
	int ret = lcg_solver_preconditioned_cuda(_AxProduct, _MxProduct, _Progress, x, b, n_size, nz_size, &param_, this, cub_handle, cus_handle, solver_id);
	clock_t end = clock();

	lcg_float costime = 1000*(end-start)/(double)CLOCKS_PER_SEC;
	
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

void LCG_CUDA_Solver::MinimizeConstrained(cublasHandle_t cub_handle, cusparseHandle_t cus_handle, lcg_float *x, const lcg_float *b, 
    const lcg_float* low, const lcg_float *hig, const int n_size, const int nz_size, lcg_solver_enum solver_id, 
    bool verbose, bool er_throw)
{
	if (silent_)
	{
		int ret = lcg_solver_constrained_cuda(_AxProduct, nullptr, x, b, low, hig, n_size, nz_size, &param_, this, cub_handle, cus_handle, solver_id);
		if (ret < 0) lcg_error_str(ret, true);
		return;
	}
	
	// 使用lcg求解 注意当我们使用函数指针来调用求解函数时默认参数不可以省略
	clock_t start = clock();
	int ret = lcg_solver_constrained_cuda(_AxProduct, _Progress, x, b, low, hig, n_size, nz_size, &param_, this, cub_handle, cus_handle, solver_id);
	clock_t end = clock();

	lcg_float costime = 1000*(end-start)/(double)CLOCKS_PER_SEC;
	
	if (!er_throw)
	{
		std::clog << std::endl;
		switch (solver_id)
		{
			case LCG_PG:
				std::clog << "Solver: PG. Time cost: " << costime << " ms" << std::endl;
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


CLCG_CUDAF_Solver::CLCG_CUDAF_Solver()
{
	param_ = clcg_default_parameters();
	inter_ = 1;
	silent_ = false;
}

int CLCG_CUDAF_Solver::Progress(const cuComplex* m, const float converge, 
	const clcg_para* param, const int n_size, const int nz_size, const int k)
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

void CLCG_CUDAF_Solver::silent()
{
	silent_ = true;
	return;
}

void CLCG_CUDAF_Solver::set_report_interval(unsigned int inter)
{
	inter_ = inter;
	return;
}

void CLCG_CUDAF_Solver::set_clcg_parameter(const clcg_para &in_param)
{
	param_ = in_param;
	return;
}

void CLCG_CUDAF_Solver::Minimize(cublasHandle_t cub_handle, cusparseHandle_t cus_handle, cuComplex *x, cuComplex *b, 
	const int n_size, const int nz_size, clcg_solver_enum solver_id, bool verbose, bool er_throw)
{
	if (silent_)
	{
		int ret = clcg_solver_cuda(_AxProduct, nullptr, x, b, n_size, nz_size, &param_, this, cub_handle, cus_handle, solver_id);
		if (ret < 0) lcg_error_str(ret, true);
		return;
	}
	
	// 使用lcg求解 注意当我们使用函数指针来调用求解函数时默认参数不可以省略
	clock_t start = clock();
	int ret = clcg_solver_cuda(_AxProduct, _Progress, x, b, n_size, nz_size, &param_, this, cub_handle, cus_handle, solver_id);
	clock_t end = clock();

	float costime = 1000*(end-start)/(double)CLOCKS_PER_SEC;
	
	if (!er_throw)
	{
		std::clog << std::endl;
		switch (solver_id)
		{
			case CLCG_BICG:
				std::clog << "Solver: BI-CG. Time cost: " << costime << " ms" << std::endl;
				break;
			case CLCG_BICG_SYM:
				std::clog << "Solver: BI-CG (symmetrically accelerated). Time cost: " << costime << " ms" << std::endl;
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

void CLCG_CUDAF_Solver::MinimizePreconditioned(cublasHandle_t cub_handle, cusparseHandle_t cus_handle, cuComplex *x, cuComplex *b, 
	const int n_size, const int nz_size, clcg_solver_enum solver_id, bool verbose, bool er_throw)
{
	if (silent_)
	{
		int ret = clcg_solver_preconditioned_cuda(_AxProduct, _MxProduct, nullptr, x, b, n_size, nz_size, &param_, this, cub_handle, cus_handle, solver_id);
		if (ret < 0) lcg_error_str(ret, true);
		return;
	}
	
	// 使用lcg求解 注意当我们使用函数指针来调用求解函数时默认参数不可以省略
	clock_t start = clock();
	int ret = clcg_solver_preconditioned_cuda(_AxProduct, _MxProduct, _Progress, x, b, n_size, nz_size, &param_, this, cub_handle, cus_handle, solver_id);
	clock_t end = clock();

	float costime = 1000*(end-start)/(double)CLOCKS_PER_SEC;
	
	if (!er_throw)
	{
		std::clog << std::endl;
		switch (solver_id)
		{
			case CLCG_PCG:
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


CLCG_CUDA_Solver::CLCG_CUDA_Solver()
{
	param_ = clcg_default_parameters();
	inter_ = 1;
	silent_ = false;
}

int CLCG_CUDA_Solver::Progress(const cuDoubleComplex* m, const lcg_float converge, 
	const clcg_para* param, const int n_size, const int nz_size, const int k)
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

void CLCG_CUDA_Solver::silent()
{
	silent_ = true;
	return;
}

void CLCG_CUDA_Solver::set_report_interval(unsigned int inter)
{
	inter_ = inter;
	return;
}

void CLCG_CUDA_Solver::set_clcg_parameter(const clcg_para &in_param)
{
	param_ = in_param;
	return;
}

void CLCG_CUDA_Solver::Minimize(cublasHandle_t cub_handle, cusparseHandle_t cus_handle, cuDoubleComplex *x, cuDoubleComplex *b, 
	const int n_size, const int nz_size, clcg_solver_enum solver_id, bool verbose, bool er_throw)
{
	if (silent_)
	{
		int ret = clcg_solver_cuda(_AxProduct, nullptr, x, b, n_size, nz_size, &param_, this, cub_handle, cus_handle, solver_id);
		if (ret < 0) lcg_error_str(ret, true);
		return;
	}
	
	// 使用lcg求解 注意当我们使用函数指针来调用求解函数时默认参数不可以省略
	clock_t start = clock();
	int ret = clcg_solver_cuda(_AxProduct, _Progress, x, b, n_size, nz_size, &param_, this, cub_handle, cus_handle, solver_id);
	clock_t end = clock();

	lcg_float costime = 1000*(end-start)/(double)CLOCKS_PER_SEC;
	
	if (!er_throw)
	{
		std::clog << std::endl;
		switch (solver_id)
		{
			case CLCG_BICG:
				std::clog << "Solver: BI-CG. Time cost: " << costime << " ms" << std::endl;
				break;
			case CLCG_BICG_SYM:
				std::clog << "Solver: BI-CG (symmetrically accelerated). Time cost: " << costime << " ms" << std::endl;
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

void CLCG_CUDA_Solver::MinimizePreconditioned(cublasHandle_t cub_handle, cusparseHandle_t cus_handle, cuDoubleComplex *x, cuDoubleComplex *b, 
	const int n_size, const int nz_size, clcg_solver_enum solver_id, bool verbose, bool er_throw)
{
	if (silent_)
	{
		int ret = clcg_solver_preconditioned_cuda(_AxProduct, _MxProduct, nullptr, x, b, n_size, nz_size, &param_, this, cub_handle, cus_handle, solver_id);
		if (ret < 0) lcg_error_str(ret, true);
		return;
	}
	
	// 使用lcg求解 注意当我们使用函数指针来调用求解函数时默认参数不可以省略
	clock_t start = clock();
	int ret = clcg_solver_preconditioned_cuda(_AxProduct, _MxProduct, _Progress, x, b, n_size, nz_size, &param_, this, cub_handle, cus_handle, solver_id);
	clock_t end = clock();

	lcg_float costime = 1000*(end-start)/(double)CLOCKS_PER_SEC;
	
	if (!er_throw)
	{
		std::clog << std::endl;
		switch (solver_id)
		{
			case CLCG_PCG:
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