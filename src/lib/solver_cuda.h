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

#ifndef _SOLVER_CUDA_H
#define _SOLVER_CUDA_H

#include "lcg_cuda.h"
#include "clcg_cuda.h"
#include "clcg_cudaf.h"

#ifdef LibLCG_CUDA

/**
 * @brief      Linear conjugate gradient solver class
 */
class LCG_CUDA_Solver
{
protected:
	lcg_para param_;
	unsigned int inter_;
	bool silent_;

public:
	LCG_CUDA_Solver();
	virtual ~LCG_CUDA_Solver(){}

	/**
	 * @brief       Interface of the virtual function of the product of A*x
	 * 
	 * @param instance   User data sent to identify the function address
	 * @param cub_handle  Handler of the CuBLAS library
	 * @param cus_handle  Handler of the CuSparse library
	 * @param x[in]      Pointer of the multiplier
	 * @param prod_Ax[out]     Pointer of the product
	 * @param n_size      Size of the solution
	 * @param nz_size     Non-zero size of the sparse kernel matrix. This parameter is not need by the algorithm. It is passed for CUDA usages
	 */
	static void _AxProduct(void* instance, cublasHandle_t cub_handle, cusparseHandle_t cus_handle, 
        cusparseDnVecDescr_t x, cusparseDnVecDescr_t prod_Ax, const int n_size, const int nz_size)
	{
		return reinterpret_cast<LCG_CUDA_Solver*>(instance)->AxProduct(cub_handle, cus_handle, x, prod_Ax, n_size, nz_size);
	}

	/**
	 * @brief       Virtual function of the product of A*x
	 * 
	 * @param cub_handle  Handler of the CuBLAS library
	 * @param cus_handle  Handler of the CuSparse library
	 * @param x[in]     Pointer of the multiplier
	 * @param prod_Ax[out]    Pointer of the product
	 * @param n_size      Size of the solution
	 * @param nz_size     Non-zero size of the sparse kernel matrix. This parameter is not need by the algorithm. It is passed for CUDA usages
	 */
	virtual void AxProduct(cublasHandle_t cub_handle, cusparseHandle_t cus_handle, 
        cusparseDnVecDescr_t x, cusparseDnVecDescr_t prod_Ax, const int n_size, const int nz_size) = 0;

	/**
	 * @brief       Interface of the virtual function of the product of M^-1*x
	 * 
	 * @param instance   User data sent to identify the function address
	 * @param cub_handle  Handler of the CuBLAS library
	 * @param cus_handle  Handler of the CuSparse library
	 * @param x[in]      Pointer of the multiplier
	 * @param prod_Mx[out]     Pointer of the product
	 * @param n_size      Size of the solution
	 * @param nz_size     Non-zero size of the sparse kernel matrix. This parameter is not need by the algorithm. It is passed for CUDA usages
	 */
	static void _MxProduct(void* instance, cublasHandle_t cub_handle, cusparseHandle_t cus_handle, 
        cusparseDnVecDescr_t x, cusparseDnVecDescr_t prod_Mx, const int n_size, const int nz_size)
	{
		return reinterpret_cast<LCG_CUDA_Solver*>(instance)->AxProduct(cub_handle, cus_handle, x, prod_Mx, n_size, nz_size);
	}

	/**
	 * @brief       Virtual function of the product of M^-1*x
	 * 
	 * @param cub_handle  Handler of the CuBLAS library
	 * @param cus_handle  Handler of the CuSparse library
	 * @param x[in]     Pointer of the multiplier
	 * @param prod_Mx[out]    Pointer of the product
	 * @param n_size      Size of the solution
	 * @param nz_size     Non-zero size of the sparse kernel matrix. This parameter is not need by the algorithm. It is passed for CUDA usages
	 */
	virtual void MxProduct(cublasHandle_t cub_handle, cusparseHandle_t cus_handle, 
        cusparseDnVecDescr_t x, cusparseDnVecDescr_t prod_Mx, const int n_size, const int nz_size) = 0;

	/**
	 * @brief       Interface of the virtual function of the process monitoring
	 * 
	 * @param instance    User data sent to identify the function address
	 * @param m           Pointer of the current solution
	 * @param converge    Current value of the convergence
	 * @param param       Pointer of the parameters used in the algorithms
	 * @param n_size      Size of the solution
	 * @param nz_size     Non-zero size of the sparse kernel matrix. This parameter is not need by the algorithm. It is passed for CUDA usages
	 * @param k           Current iteration times
	 * @return int        Status of the process
	 */
	static int _Progress(void* instance, const lcg_float* m, const lcg_float converge, 
	    const lcg_para* param, const int n_size, const int nz_size, const int k)
	{
		return reinterpret_cast<LCG_CUDA_Solver*>(instance)->Progress(m, converge, param, n_size, nz_size, k);
	}
	
	/**
	 * @brief       Virtual function of the process monitoring
	 * 
	 * @param m           Pointer of the current solution
	 * @param converge    Current value of the convergence
	 * @param param       Pointer of the parameters used in the algorithms
	 * @param n_size      Size of the solution
	 * @param nz_size     Non-zero size of the sparse kernel matrix. This parameter is not need by the algorithm. It is passed for CUDA usages
	 * @param k           Current iteration times
	 * @return int        Status of the process
	 */
	virtual int Progress(const lcg_float* m, const lcg_float converge, 
	    const lcg_para* param, const int n_size, const int nz_size, const int k);

	/**
	 * @brief      Do not report any processes
	 */
	void silent();

	/**
	 * @brief      Set the interval to run the process monitoring function
	 * 
	 * @param inter      the interval
	 */
	void set_report_interval(unsigned int inter);

	/**
	 * @brief      Set the parameters of the algorithms
	 * 
	 * @param in_param   the input parameters
	 */
	void set_lcg_parameter(const lcg_para &in_param);

	/**
	 * @brief      Run the constrained minimizing process
	 * 
	 * @param cub_handle  Handler of the CuBLAS library
	 * @param cus_handle  Handler of the CuSparse library
	 * @param x          Pointer of the solution vector
	 * @param b          Pointer of the targeting vector
	 * @param n_size     Size of the solution vector
	 * @param nz_size    Non-zero size of the sparse kernel matrix. This parameter is not need by the algorithm. It is passed for CUDA usages
	 * @param solver_id  Solver type
	 * @param verbose    Report more information of the full process
	 * @param er_throw   Instead of showing error messages on screen, throw them out using std::exception
	 */
	void Minimize(cublasHandle_t cub_handle, cusparseHandle_t cus_handle, lcg_float *x, lcg_float *b, 
        const int n_size, const int nz_size, lcg_solver_enum solver_id = LCG_CG, bool verbose = true, bool er_throw = false);
	
	/**
	 * @brief      Run the preconditioned minimizing process
	 * 
	 * @param cub_handle  Handler of the CuBLAS library
	 * @param cus_handle  Handler of the CuSparse library
	 * @param x          Pointer of the solution vector
	 * @param b          Pointer of the targeting vector
	 * @param n_size     Size of the solution vector
	 * @param nz_size    Non-zero size of the sparse kernel matrix. This parameter is not need by the algorithm. It is passed for CUDA usages
	 * @param solver_id  Solver type
	 * @param verbose    Report more information of the full process
	 * @param er_throw   Instead of showing error messages on screen, throw them out using std::exception
	 */
	void MinimizePreconditioned(cublasHandle_t cub_handle, cusparseHandle_t cus_handle, lcg_float *x, lcg_float *b, 
        const int n_size, const int nz_size, lcg_solver_enum solver_id = LCG_CG, bool verbose = true, bool er_throw = false);
	
	/**
	 * @brief      Run the constrained minimizing process
	 * 
	 * @param cub_handle  Handler of the CuBLAS library
	 * @param cus_handle  Handler of the CuSparse library
	 * @param x          Pointer of the solution vector
	 * @param b          Pointer of the targeting vector
	 * @param low        Lower bound of the solution vector
	 * @param hig        Higher bound of the solution vector
	 * @param n_size     Size of the solution vector
	 * @param nz_size    Non-zero size of the sparse kernel matrix. This parameter is not need by the algorithm. It is passed for CUDA usages
	 * @param solver_id  Solver type
	 * @param verbose    Report more information of the full process
	 * @param er_throw   Instead of showing error messages on screen, throw them out using std::exception
	 */
    void MinimizeConstrained(cublasHandle_t cub_handle, cusparseHandle_t cus_handle, lcg_float *x, const lcg_float *b, 
        const lcg_float* low, const lcg_float *hig, const int n_size, const int nz_size, lcg_solver_enum solver_id = LCG_PG, 
        bool verbose = true, bool er_throw = false);
};


/**
 * @brief      Complex linear conjugate gradient solver class
 */
class CLCG_CUDAF_Solver
{
protected:
	clcg_para param_;
	unsigned int inter_;
	bool silent_;

public:
	CLCG_CUDAF_Solver();
	virtual ~CLCG_CUDAF_Solver(){}

	/**
	 * @brief       Interface of the virtual function of the product of A*x
	 * 
	 * @param instance   User data sent to identify the function address
	 * @param cub_handle  Handler of the CuBLAS library
	 * @param cus_handle  Handler of the CuSparse library
	 * @param x[in]      Pointer of the multiplier
	 * @param prod_Ax[out]     Pointer of the product
	 * @param n_size      Size of the solution
	 * @param nz_size     Non-zero size of the sparse kernel matrix. This parameter is not need by the algorithm. It is passed for CUDA usages
	 * @param oper_t      Cusparse operator. This parameter is not need by the algorithm. It is passed for CUDA usages
	 */
	static void _AxProduct(void* instance, cublasHandle_t cub_handle, cusparseHandle_t cus_handle, 
		cusparseDnVecDescr_t x, cusparseDnVecDescr_t prod_Ax, 
		const int n_size, const int nz_size, cusparseOperation_t oper_t)
	{
		return reinterpret_cast<CLCG_CUDAF_Solver*>(instance)->AxProduct(cub_handle, cus_handle, x, prod_Ax, n_size, nz_size, oper_t);
	}

	/**
	 * @brief       Virtual function of the product of A*x
	 * 
	 * @param cub_handle  Handler of the CuBLAS library
	 * @param cus_handle  Handler of the CuSparse library
	 * @param x[in]      Pointer of the multiplier
	 * @param prod_Ax[out]     Pointer of the product
	 * @param n_size      Size of the solution
	 * @param nz_size     Non-zero size of the sparse kernel matrix. This parameter is not need by the algorithm. It is passed for CUDA usages
	 * @param oper_t      Cusparse operator. This parameter is not need by the algorithm. It is passed for CUDA usages
	 */
	virtual void AxProduct(cublasHandle_t cub_handle, cusparseHandle_t cus_handle, 
		cusparseDnVecDescr_t x, cusparseDnVecDescr_t prod_Ax, 
		const int n_size, const int nz_size, cusparseOperation_t oper_t) = 0;

	/**
	 * @brief       Interface of the virtual function of the product of M^-1*x
	 * 
	 * @param instance   User data sent to identify the function address
	 * @param cub_handle  Handler of the CuBLAS library
	 * @param cus_handle  Handler of the CuSparse library
	 * @param x[in]      Pointer of the multiplier
	 * @param prod_Mx[out]     Pointer of the product
	 * @param n_size      Size of the solution
	 * @param nz_size     Non-zero size of the sparse kernel matrix. This parameter is not need by the algorithm. It is passed for CUDA usages
	 * @param oper_t      Cusparse operator. This parameter is not need by the algorithm. It is passed for CUDA usages
	 */
	static void _MxProduct(void* instance, cublasHandle_t cub_handle, cusparseHandle_t cus_handle, 
		cusparseDnVecDescr_t x, cusparseDnVecDescr_t prod_Mx, 
		const int n_size, const int nz_size, cusparseOperation_t oper_t)
	{
		return reinterpret_cast<CLCG_CUDAF_Solver*>(instance)->MxProduct(cub_handle, cus_handle, x, prod_Mx, n_size, nz_size, oper_t);
	}

	/**
	 * @brief       Virtual function of the product of M^-1*x
	 * 
	 * @param cub_handle  Handler of the CuBLAS library
	 * @param cus_handle  Handler of the CuSparse library
	 * @param x[in]      Pointer of the multiplier
	 * @param prod_Mx[out]     Pointer of the product
	 * @param n_size      Size of the solution
	 * @param nz_size     Non-zero size of the sparse kernel matrix. This parameter is not need by the algorithm. It is passed for CUDA usages
	 * @param oper_t      Cusparse operator. This parameter is not need by the algorithm. It is passed for CUDA usages
	 */
	virtual void MxProduct(cublasHandle_t cub_handle, cusparseHandle_t cus_handle, 
		cusparseDnVecDescr_t x, cusparseDnVecDescr_t prod_Mx, 
		const int n_size, const int nz_size, cusparseOperation_t oper_t) = 0;

	/**
	 * @brief       Interface of the virtual function of the process monitoring
	 * 
	 * @param instance    User data sent to identify the function address
	 * @param m           Pointer of the current solution
	 * @param converge    Current value of the convergence
	 * @param param       Pointer of the parameters used in the algorithms
	 * @param n_size      Size of the solution
	 * @param nz_size     Non-zero size of the sparse kernel matrix. This parameter is not need by the algorithm. It is passed for CUDA usages
	 * @param k           Current iteration times
	 * @return int        Status of the process
	 */
	static int _Progress(void* instance, const cuComplex* m, const float converge, 
	    const clcg_para* param, const int n_size, const int nz_size, const int k)
	{
		return reinterpret_cast<CLCG_CUDAF_Solver*>(instance)->Progress(m, converge, param, n_size, nz_size, k);
	}

	/**
	 * @brief       Virtual function of the process monitoring
	 * 
	 * @param m           Pointer of the current solution
	 * @param converge    Current value of the convergence
	 * @param param       Pointer of the parameters used in the algorithms
	 * @param n_size      Size of the solution
	 * @param nz_size     Non-zero size of the sparse kernel matrix. This parameter is not need by the algorithm. It is passed for CUDA usages
	 * @param k           Current iteration times
	 * @return int        Status of the process
	 */
	virtual int Progress(const cuComplex* m, const float converge, 
	    const clcg_para* param, const int n_size, const int nz_size, const int k);

	/**
	 * @brief      Do not report any processes
	 */
	void silent();

	/**
	 * @brief      Set the interval to run the process monitoring function
	 * 
	 * @param inter      the interval
	 */
	void set_report_interval(unsigned int inter);

	/**
	 * @brief      Set the parameters of the algorithms
	 * 
	 * @param in_param   the input parameters
	 */
	void set_clcg_parameter(const clcg_para &in_param);

	/**
	 * @brief      Run the constrained minimizing process
	 * 
	 * @param cub_handle  Handler of the CuBLAS library
	 * @param cus_handle  Handler of the CuSparse library
	 * @param x          Pointer of the solution vector
	 * @param b          Pointer of the targeting vector
	 * @param n_size     Size of the solution vector
	 * @param nz_size    Non-zero size of the sparse kernel matrix. This parameter is not need by the algorithm. It is passed for CUDA usages
	 * @param solver_id  Solver type
	 * @param verbose    Report more information of the full process
	 * @param er_throw   Instead of showing error messages on screen, throw them out using std::exception
	 */
	void Minimize(cublasHandle_t cub_handle, cusparseHandle_t cus_handle, cuComplex *x, cuComplex *b, 
		const int n_size, const int nz_size, clcg_solver_enum solver_id = CLCG_BICG, bool verbose = true, bool er_throw = false);
	
	/**
	 * @brief      Run the preconditioned minimizing process
	 * 
	 * @param cub_handle  Handler of the CuBLAS library
	 * @param cus_handle  Handler of the CuSparse library
	 * @param x          Pointer of the solution vector
	 * @param b          Pointer of the targeting vector
	 * @param n_size     Size of the solution vector
	 * @param nz_size    Non-zero size of the sparse kernel matrix. This parameter is not need by the algorithm. It is passed for CUDA usages
	 * @param solver_id  Solver type
	 * @param verbose    Report more information of the full process
	 * @param er_throw   Instead of showing error messages on screen, throw them out using std::exception
	 */
	void MinimizePreconditioned(cublasHandle_t cub_handle, cusparseHandle_t cus_handle, cuComplex *x, cuComplex *b, 
        const int n_size, const int nz_size, clcg_solver_enum solver_id = CLCG_PCG, bool verbose = true, bool er_throw = false);
};


/**
 * @brief      Complex linear conjugate gradient solver class
 */
class CLCG_CUDA_Solver
{
protected:
	clcg_para param_;
	unsigned int inter_;
	bool silent_;

public:
	CLCG_CUDA_Solver();
	virtual ~CLCG_CUDA_Solver(){}

	/**
	 * @brief       Interface of the virtual function of the product of A*x
	 * 
	 * @param instance   User data sent to identify the function address
	 * @param cub_handle  Handler of the CuBLAS library
	 * @param cus_handle  Handler of the CuSparse library
	 * @param x[in]      Pointer of the multiplier
	 * @param prod_Ax[out]     Pointer of the product
	 * @param n_size      Size of the solution
	 * @param nz_size     Non-zero size of the sparse kernel matrix. This parameter is not need by the algorithm. It is passed for CUDA usages
	 * @param oper_t      Cusparse operator. This parameter is not need by the algorithm. It is passed for CUDA usages
	 */
	static void _AxProduct(void* instance, cublasHandle_t cub_handle, cusparseHandle_t cus_handle, 
		cusparseDnVecDescr_t x, cusparseDnVecDescr_t prod_Ax, 
		const int n_size, const int nz_size, cusparseOperation_t oper_t)
	{
		return reinterpret_cast<CLCG_CUDA_Solver*>(instance)->AxProduct(cub_handle, cus_handle, x, prod_Ax, n_size, nz_size, oper_t);
	}

	/**
	 * @brief       Virtual function of the product of A*x
	 * 
	 * @param cub_handle  Handler of the CuBLAS library
	 * @param cus_handle  Handler of the CuSparse library
	 * @param x[in]      Pointer of the multiplier
	 * @param prod_Ax[out]     Pointer of the product
	 * @param n_size      Size of the solution
	 * @param nz_size     Non-zero size of the sparse kernel matrix. This parameter is not need by the algorithm. It is passed for CUDA usages
	 * @param oper_t      Cusparse operator. This parameter is not need by the algorithm. It is passed for CUDA usages
	 */
	virtual void AxProduct(cublasHandle_t cub_handle, cusparseHandle_t cus_handle, 
		cusparseDnVecDescr_t x, cusparseDnVecDescr_t prod_Ax, 
		const int n_size, const int nz_size, cusparseOperation_t oper_t) = 0;

	/**
	 * @brief       Interface of the virtual function of the product of M^-1*x
	 * 
	 * @param instance   User data sent to identify the function address
	 * @param cub_handle  Handler of the CuBLAS library
	 * @param cus_handle  Handler of the CuSparse library
	 * @param x[in]      Pointer of the multiplier
	 * @param prod_Mx[out]     Pointer of the product
	 * @param n_size      Size of the solution
	 * @param nz_size     Non-zero size of the sparse kernel matrix. This parameter is not need by the algorithm. It is passed for CUDA usages
	 * @param oper_t      Cusparse operator. This parameter is not need by the algorithm. It is passed for CUDA usages
	 */
	static void _MxProduct(void* instance, cublasHandle_t cub_handle, cusparseHandle_t cus_handle, 
		cusparseDnVecDescr_t x, cusparseDnVecDescr_t prod_Mx, 
		const int n_size, const int nz_size, cusparseOperation_t oper_t)
	{
		return reinterpret_cast<CLCG_CUDA_Solver*>(instance)->MxProduct(cub_handle, cus_handle, x, prod_Mx, n_size, nz_size, oper_t);
	}

	/**
	 * @brief       Virtual function of the product of M^-1*x
	 * 
	 * @param cub_handle  Handler of the CuBLAS library
	 * @param cus_handle  Handler of the CuSparse library
	 * @param x[in]      Pointer of the multiplier
	 * @param prod_Mx[out]     Pointer of the product
	 * @param n_size      Size of the solution
	 * @param nz_size     Non-zero size of the sparse kernel matrix. This parameter is not need by the algorithm. It is passed for CUDA usages
	 * @param oper_t      Cusparse operator. This parameter is not need by the algorithm. It is passed for CUDA usages
	 */
	virtual void MxProduct(cublasHandle_t cub_handle, cusparseHandle_t cus_handle, 
		cusparseDnVecDescr_t x, cusparseDnVecDescr_t prod_Mx, 
		const int n_size, const int nz_size, cusparseOperation_t oper_t) = 0;

	/**
	 * @brief       Interface of the virtual function of the process monitoring
	 * 
	 * @param instance    User data sent to identify the function address
	 * @param m           Pointer of the current solution
	 * @param converge    Current value of the convergence
	 * @param param       Pointer of the parameters used in the algorithms
	 * @param n_size      Size of the solution
	 * @param nz_size     Non-zero size of the sparse kernel matrix. This parameter is not need by the algorithm. It is passed for CUDA usages
	 * @param k           Current iteration times
	 * @return int        Status of the process
	 */
	static int _Progress(void* instance, const cuDoubleComplex* m, const lcg_float converge, 
	    const clcg_para* param, const int n_size, const int nz_size, const int k)
	{
		return reinterpret_cast<CLCG_CUDA_Solver*>(instance)->Progress(m, converge, param, n_size, nz_size, k);
	}

	/**
	 * @brief       Virtual function of the process monitoring
	 * 
	 * @param m           Pointer of the current solution
	 * @param converge    Current value of the convergence
	 * @param param       Pointer of the parameters used in the algorithms
	 * @param n_size      Size of the solution
	 * @param nz_size     Non-zero size of the sparse kernel matrix. This parameter is not need by the algorithm. It is passed for CUDA usages
	 * @param k           Current iteration times
	 * @return int        Status of the process
	 */
	virtual int Progress(const cuDoubleComplex* m, const lcg_float converge, 
	    const clcg_para* param, const int n_size, const int nz_size, const int k);

	/**
	 * @brief      Do not report any processes
	 */
	void silent();

	/**
	 * @brief      Set the interval to run the process monitoring function
	 * 
	 * @param inter      the interval
	 */
	void set_report_interval(unsigned int inter);

	/**
	 * @brief      Set the parameters of the algorithms
	 * 
	 * @param in_param   the input parameters
	 */
	void set_clcg_parameter(const clcg_para &in_param);

	/**
	 * @brief      Run the constrained minimizing process
	 * 
	 * @param cub_handle  Handler of the CuBLAS library
	 * @param cus_handle  Handler of the CuSparse library
	 * @param x          Pointer of the solution vector
	 * @param b          Pointer of the targeting vector
	 * @param n_size     Size of the solution vector
	 * @param nz_size    Non-zero size of the sparse kernel matrix. This parameter is not need by the algorithm. It is passed for CUDA usages
	 * @param solver_id  Solver type
	 * @param verbose    Report more information of the full process
	 * @param er_throw   Instead of showing error messages on screen, throw them out using std::exception
	 */
	void Minimize(cublasHandle_t cub_handle, cusparseHandle_t cus_handle, cuDoubleComplex *x, cuDoubleComplex *b, 
		const int n_size, const int nz_size, clcg_solver_enum solver_id = CLCG_BICG, bool verbose = true, bool er_throw = false);
	
	/**
	 * @brief      Run the preconditioned minimizing process
	 * 
	 * @param cub_handle  Handler of the CuBLAS library
	 * @param cus_handle  Handler of the CuSparse library
	 * @param x          Pointer of the solution vector
	 * @param b          Pointer of the targeting vector
	 * @param n_size     Size of the solution vector
	 * @param nz_size    Non-zero size of the sparse kernel matrix. This parameter is not need by the algorithm. It is passed for CUDA usages
	 * @param solver_id  Solver type
	 * @param verbose    Report more information of the full process
	 * @param er_throw   Instead of showing error messages on screen, throw them out using std::exception
	 */
	void MinimizePreconditioned(cublasHandle_t cub_handle, cusparseHandle_t cus_handle, cuDoubleComplex *x, cuDoubleComplex *b, 
        const int n_size, const int nz_size, clcg_solver_enum solver_id = CLCG_PCG, bool verbose = true, bool er_throw = false);
};

#endif // LibLCG_CUDA

#endif // _SOLVER_CUDA_H