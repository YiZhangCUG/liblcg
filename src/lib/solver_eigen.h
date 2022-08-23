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

#ifndef _SOLVER_EIGEN_H
#define _SOLVER_EIGEN_H

#include "lcg_eigen.h"
#include "clcg_eigen.h"

/**
 * @brief      Linear conjugate gradient solver class
 */
class LCG_EIGEN_Solver
{
protected:
	lcg_para param_;
	unsigned int inter_;
	bool silent_;

public:
	LCG_EIGEN_Solver();
	virtual ~LCG_EIGEN_Solver(){}

	/**
	 * @brief       Interface of the virtual function of the product of A*x
	 * 
	 * @param instance   User data sent to identify the function address
	 * @param x[in]      Pointer of the multiplier
	 * @param prod_Ax[out]     Pointer of the product
	 */
	static void _AxProduct(void* instance, const Eigen::VectorXd &x, Eigen::VectorXd &prod_Ax)
	{
		return reinterpret_cast<LCG_EIGEN_Solver*>(instance)->AxProduct(x, prod_Ax);
	}

	/**
	 * @brief       Virtual function of the product of A*x
	 * 
	 * @param x[in]     Pointer of the multiplier
	 * @param prod_Ax[out]    Pointer of the product
	 */
	virtual void AxProduct(const Eigen::VectorXd &x, Eigen::VectorXd &prod_Ax) = 0;

	/**
	 * @brief       Interface of the virtual function of the product of M^-1*x
	 * 
	 * @param instance   User data sent to identify the function address
	 * @param x[in]      Pointer of the multiplier
	 * @param prod_Mx[out]     Pointer of the product
	 */
	static void _MxProduct(void* instance, const Eigen::VectorXd &x, Eigen::VectorXd &prod_Mx)
	{
		return reinterpret_cast<LCG_EIGEN_Solver*>(instance)->MxProduct(x, prod_Mx);
	}

	/**
	 * @brief       Virtual function of the product of M^-1*x
	 * 
	 * @param x[in]     Pointer of the multiplier
	 * @param prod_Mx[out]    Pointer of the product
	 */
	virtual void MxProduct(const Eigen::VectorXd &x, Eigen::VectorXd &prod_Mx) = 0;

	/**
	 * @brief       Interface of the virtual function of the process monitoring
	 * 
	 * @param instance    User data sent to identify the function address
	 * @param m           Pointer of the current solution
	 * @param converge    Current value of the convergence
	 * @param param       Pointer of the parameters used in the algorithms
	 * @param k           Current iteration times
	 * @return int        Status of the process
	 */
	static int _Progress(void* instance, const Eigen::VectorXd *m, const lcg_float converge, 
		const lcg_para *param, const int k)
	{
		return reinterpret_cast<LCG_EIGEN_Solver*>(instance)->Progress(m, converge, param, k);
	}

	/**
	 * @brief       Virtual function of the process monitoring
	 * 
	 * @param m           Pointer of the current solution
	 * @param converge    Current value of the convergence
	 * @param param       Pointer of the parameters used in the algorithms
	 * @param k           Current iteration times
	 * @return int        Status of the process
	 */
	virtual int Progress(const Eigen::VectorXd *m, const lcg_float converge, const lcg_para *param, 
		const int k);

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
	 * @param solver_id  Solver type
	 * @param verbose    Report more information of the full process
	 * @param er_throw   Instead of showing error messages on screen, throw them out using std::exception
	 */
	void Minimize(Eigen::VectorXd &m, const Eigen::VectorXd &b, lcg_solver_enum solver_id = LCG_CG, 
		bool verbose = true, bool er_throw = false);
	
	/**
	 * @brief      Run the preconitioned minimizing process
	 * 
	 * @param m          Pointer of the solution vector
	 * @param b          Pointer of the targeting vector
	 * @param solver_id  Solver type
	 * @param verbose    Report more information of the full process
	 * @param er_throw   Instead of showing error messages on screen, throw them out using std::exception
	 */
	void MinimizePreconditioned(Eigen::VectorXd &m, const Eigen::VectorXd &b, lcg_solver_enum solver_id = LCG_PCG, 
		bool verbose = true, bool er_throw = false);

	/**
	 * @brief      Run the constrained minimizing process
	 * 
	 * @param m          Pointer of the solution vector
	 * @param b          Pointer of the targeting vector
	 * @param low        Lower bound of the solution vector
	 * @param hig        Higher bound of the solution vector
	 * @param solver_id  Solver type
	 * @param verbose    Report more information of the full process
	 * @param er_throw   Instead of showing error messages on screen, throw them out using std::exception
	 */
	void MinimizeConstrained(Eigen::VectorXd &m, const Eigen::VectorXd &B, const Eigen::VectorXd &low, 
		const Eigen::VectorXd &hig, lcg_solver_enum solver_id = LCG_PG, bool verbose = true, 
		bool er_throw = false);
};

/**
 * @brief      Complex linear conjugate gradient solver class
 */
class CLCG_EIGEN_Solver
{
protected:
	clcg_para param_;
	unsigned int inter_;
	bool silent_;

public:
	CLCG_EIGEN_Solver();
	virtual ~CLCG_EIGEN_Solver(){}

	/**
	 * @brief       Interface of the virtual function of the product of A*x
	 * 
	 * @param instance   User data sent to identify the function address
	 * @param x[in]      Pointer of the multiplier
	 * @param prod_Ax[out]     Pointer of the product
	 * @param layout     Layout of the kernel matrix. This is passed for the clcg_matvec() function
	 * @param conjugate  Welther to use conjugate of the kernel matrix. This is passed for the clcg_matvec() function
	 */
	static void _AxProduct(void* instance, const Eigen::VectorXcd &x, Eigen::VectorXcd &prod_Ax, 
		lcg_matrix_e layout, clcg_complex_e conjugate)
	{
		return reinterpret_cast<CLCG_EIGEN_Solver*>(instance)->AxProduct(x, prod_Ax, layout, conjugate);
	}

	/**
	 * @brief       Interface of the virtual function of the product of A*x
	 * 
	 * @param x[in]      Pointer of the multiplier
	 * @param prod_Ax[out]     Pointer of the product
	 * @param layout     Layout of the kernel matrix. This is passed for the clcg_matvec() function
	 * @param conjugate  Welther to use conjugate of the kernel matrix. This is passed for the clcg_matvec() function
	 */
	virtual void AxProduct(const Eigen::VectorXcd &x, Eigen::VectorXcd &prod_Ax, 
		lcg_matrix_e layout, clcg_complex_e conjugate) = 0;

	/**
	 * @brief       Interface of the virtual function of the product of M^-1*x
	 * 
	 * @param instance   User data sent to identify the function address
	 * @param x[in]      Pointer of the multiplier
	 * @param prod_Mx[out]     Pointer of the product
	 * @param layout     Layout of the kernel matrix. This is passed for the clcg_matvec() function
	 * @param conjugate  Welther to use conjugate of the kernel matrix. This is passed for the clcg_matvec() function
	 */
    static void _MxProduct(void* instance, const Eigen::VectorXcd &x, Eigen::VectorXcd &prod_Mx, 
        lcg_matrix_e layout, clcg_complex_e conjugate)
    {
        return reinterpret_cast<CLCG_EIGEN_Solver*>(instance)->MxProduct(x, prod_Mx, layout, conjugate);
    }

	/**
	 * @brief       Interface of the virtual function of the product of M^-1*x
	 * 
	 * @param x[in]      Pointer of the multiplier
	 * @param prod_Mx[out]     Pointer of the product
	 * @param layout     Layout of the kernel matrix. This is passed for the clcg_matvec() function
	 * @param conjugate  Welther to use conjugate of the kernel matrix. This is passed for the clcg_matvec() function
	 */
    virtual void MxProduct(const Eigen::VectorXcd &x, Eigen::VectorXcd &prod_Mx, 
        lcg_matrix_e layout, clcg_complex_e conjugate) = 0;

	/**
	 * @brief       Interface of the virtual function of the process monitoring
	 * 
	 * @param instance    User data sent to identify the function address
	 * @param m           Pointer of the current solution
	 * @param converge    Current value of the convergence
	 * @param param       Pointer of the parameters used in the algorithms
	 * @param k           Current iteration times
	 * @return int        Status of the process
	 */
	static int _Progress(void* instance, const Eigen::VectorXcd *m, const lcg_float converge, 
		const clcg_para *param, const int k)
	{
		return reinterpret_cast<CLCG_EIGEN_Solver*>(instance)->Progress(m, converge, param, k);
	}

	/**
	 * @brief       Virtual function of the process monitoring
	 * 
	 * @param m           Pointer of the current solution
	 * @param converge    Current value of the convergence
	 * @param param       Pointer of the parameters used in the algorithms
	 * @param k           Current iteration times
	 * @return int        Status of the process
	 */
	virtual int Progress(const Eigen::VectorXcd *m, const lcg_float converge, const clcg_para *param, 
		const int k);

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
	 * @brief      Set the interval to run the process monitoring function
	 * 
	 * @param inter      the interval
	 */
	void set_clcg_parameter(const clcg_para &in_param);
	
	/**
	 * @brief      Run the minimizing process
	 * 
	 * @param m          Pointer of the solution vector
	 * @param b          Pointer of the targeting vector
	 * @param solver_id  Solver type
	 * @param verbose    Report more information of the full process
	 * @param er_throw   Instead of showing error messages on screen, throw them out using std::exception
	 */
	void Minimize(Eigen::VectorXcd &m, const Eigen::VectorXcd &b, clcg_solver_enum solver_id = CLCG_CGS, 
		bool verbose = true, bool er_throw = false);

	/**
	 * @brief      Run the preconitioned minimizing process
	 * 
	 * @param m          Pointer of the solution vector
	 * @param b          Pointer of the targeting vector
	 * @param solver_id  Solver type
	 * @param verbose    Report more information of the full process
	 * @param er_throw   Instead of showing error messages on screen, throw them out using std::exception
	 */
    void MinimizePreconditioned(Eigen::VectorXcd &m, const Eigen::VectorXcd &b, clcg_solver_enum solver_id = CLCG_PBICG, 
        bool verbose = true, bool er_throw = false);
};

#endif // _SOLVER_EIGEN_H