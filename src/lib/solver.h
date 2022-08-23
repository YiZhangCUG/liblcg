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

#ifndef _SOLVER_H
#define _SOLVER_H

#include "lcg.h"
#include "clcg.h"

/**
 * @brief      Linear conjugate gradient solver class
 */
class LCG_Solver
{
protected:
	lcg_para param_;
	unsigned int inter_;
	bool silent_;

public:
	LCG_Solver();
	virtual ~LCG_Solver(){}

	/**
	 * @brief       Interface of the virtual function of the product of A*x
	 * 
	 * @param instance   User data sent to identify the function address
	 * @param a[in]      Pointer of the multiplier
	 * @param b[out]     Pointer of the product
	 * @param num        Size of the array
	 */
	static void _AxProduct(void* instance, const lcg_float* a, lcg_float* b, const int num)
	{
		return reinterpret_cast<LCG_Solver*>(instance)->AxProduct(a, b, num);
	}

	/**
	 * @brief       Virtual function of the product of A*x
	 * 
	 * @param a[in]     Pointer of the multiplier
	 * @param b[out]    Pointer of the product
	 * @param num   Size of the array
	 */
	virtual void AxProduct(const lcg_float* a, lcg_float* b, const int num) = 0;

	/**
	 * @brief       Interface of the virtual function of the product of M^-1*x
	 * 
	 * @param instance   User data sent to identify the function address
	 * @param a[in]      Pointer of the multiplier
	 * @param b[out]     Pointer of the product
	 * @param num        Size of the array
	 */
	static void _MxProduct(void* instance, const lcg_float* a, lcg_float* b, const int num)
	{
		return reinterpret_cast<LCG_Solver*>(instance)->MxProduct(a, b, num);
	}

	/**
	 * @brief       Virtual function of the product of M^-1*x
	 * 
	 * @param a[in]     Pointer of the multiplier
	 * @param b[out]    Pointer of the product
	 * @param num   Size of the array
	 */
	virtual void MxProduct(const lcg_float* a, lcg_float* b, const int num) = 0;

	/**
	 * @brief       Interface of the virtual function of the process monitoring
	 * 
	 * @param instance    User data sent to identify the function address
	 * @param m           Pointer of the current solution
	 * @param converge    Current value of the convergence
	 * @param param       Pointer of the parameters used in the algorithms
	 * @param n_size      Size of the solution
	 * @param k           Current iteration times
	 * @return int        Status of the process
	 */
	static int _Progress(void* instance, const lcg_float* m, const lcg_float converge, 
		const lcg_para *param, const int n_size, const int k)
	{
		return reinterpret_cast<LCG_Solver*>(instance)->Progress(m, converge, param, n_size, k);
	}

	/**
	 * @brief       Virtual function of the process monitoring
	 * 
	 * @param m           Pointer of the current solution
	 * @param converge    Current value of the convergence
	 * @param param       Pointer of the parameters used in the algorithms
	 * @param n_size      Size of the solution
	 * @param k           Current iteration times
	 * @return int        Status of the process
	 */
	virtual int Progress(const lcg_float* m, const lcg_float converge, 
		const lcg_para *param, const int n_size, const int k);

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
	 * @brief      Run the minimizing process
	 * 
	 * @param m          Pointer of the solution vector
	 * @param b          Pointer of the targeting vector
	 * @param x_size     Size of the solution vector
	 * @param solver_id  Solver type
	 * @param verbose    Report more information of the full process
	 * @param er_throw   Instead of showing error messages on screen, throw them out using std::exception
	 */
	void Minimize(lcg_float *m, const lcg_float *b, int x_size, 
		lcg_solver_enum solver_id = LCG_CG, bool verbose = true, bool er_throw = false);

	/**
	 * @brief      Run the preconitioned minimizing process
	 * 
	 * @param m          Pointer of the solution vector
	 * @param b          Pointer of the targeting vector
	 * @param x_size     Size of the solution vector
	 * @param solver_id  Solver type
	 * @param verbose    Report more information of the full process
	 * @param er_throw   Instead of showing error messages on screen, throw them out using std::exception
	 */
	void MinimizePreconditioned(lcg_float *m, const lcg_float *b, int x_size, 
		lcg_solver_enum solver_id = LCG_PCG, bool verbose = true, bool er_throw = false);

	/**
	 * @brief      Run the constrained minimizing process
	 * 
	 * @param m          Pointer of the solution vector
	 * @param b          Pointer of the targeting vector
	 * @param low        Lower bound of the solution vector
	 * @param hig        Higher bound of the solution vector
	 * @param x_size     Size of the solution vector
	 * @param solver_id  Solver type
	 * @param verbose    Report more information of the full process
	 * @param er_throw   Instead of showing error messages on screen, throw them out using std::exception
	 */
	void MinimizeConstrained(lcg_float *m, const lcg_float *b, const lcg_float* low, 
		const lcg_float *hig, int x_size, lcg_solver_enum solver_id = LCG_PG, 
		bool verbose = true, bool er_throw = false);
};

/**
 * @brief      Complex linear conjugate gradient solver class
 */
class CLCG_Solver
{
protected:
	clcg_para param_;
	unsigned int inter_;
	bool silent_;

public:
	CLCG_Solver();
	virtual ~CLCG_Solver(){}

	/**
	 * @brief       Interface of the virtual function of the product of A*x
	 * 
	 * @param instance   User data sent to identify the function address
	 * @param x[in]      Pointer of the multiplier
	 * @param prod_Ax[out]     Pointer of the product
	 * @param x_size     Size of the array
	 * @param layout     Layout of the kernel matrix. This is passed for the clcg_matvec() function
	 * @param conjugate  Welther to use conjugate of the kernel matrix. This is passed for the clcg_matvec() function
	 */
	static void _AxProduct(void *instance, const lcg_complex *x, lcg_complex *prod_Ax, 
		const int x_size, lcg_matrix_e layout, clcg_complex_e conjugate)
	{
		return reinterpret_cast<CLCG_Solver*>(instance)->AxProduct(x, prod_Ax, x_size, layout, conjugate);
	}

	/**
	 * @brief       Interface of the virtual function of the product of A*x
	 * 
	 * @param x[in]      Pointer of the multiplier
	 * @param prod_Ax[out]     Pointer of the product
	 * @param x_size     Size of the array
	 * @param layout     Layout of the kernel matrix. This is passed for the clcg_matvec() function
	 * @param conjugate  Welther to use conjugate of the kernel matrix. This is passed for the clcg_matvec() function
	 */
	virtual void AxProduct(const lcg_complex *x, lcg_complex *prod_Ax, 
		const int x_size, lcg_matrix_e layout, clcg_complex_e conjugate) = 0;

	/**
	 * @brief       Interface of the virtual function of the process monitoring
	 * 
	 * @param instance    User data sent to identify the function address
	 * @param m           Pointer of the current solution
	 * @param converge    Current value of the convergence
	 * @param param       Pointer of the parameters used in the algorithms
	 * @param n_size      Size of the solution
	 * @param k           Current iteration times
	 * @return int        Status of the process
	 */
	static int _Progress(void* instance, const lcg_complex* m, const lcg_float converge, 
		const clcg_para* param, const int n_size, const int k)
	{
		return reinterpret_cast<CLCG_Solver*>(instance)->Progress(m, converge, param, n_size, k);
	}

	/**
	 * @brief       Interface of the virtual function of the process monitoring
	 * 
	 * @param m           Pointer of the current solution
	 * @param converge    Current value of the convergence
	 * @param param       Pointer of the parameters used in the algorithms
	 * @param n_size      Size of the solution
	 * @param k           Current iteration times
	 * @return int        Status of the process
	 */
	virtual int Progress(const lcg_complex* m, const lcg_float converge, 
		const clcg_para* param, const int n_size, const int k);

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
	 * @brief      Run the minimizing process
	 * 
	 * @param m          Pointer of the solution vector
	 * @param b          Pointer of the targeting vector
	 * @param x_size     Size of the solution vector
	 * @param solver_id  Solver type
	 * @param verbose    Report more information of the full process
	 * @param er_throw   Instead of showing error messages on screen, throw them out using std::exception
	 */
	void Minimize(lcg_complex *m, const lcg_complex *b, int x_size, 
		clcg_solver_enum solver_id = CLCG_CGS, bool verbose = true, 
		bool er_throw = false);
};

#endif // _SOLVER_H