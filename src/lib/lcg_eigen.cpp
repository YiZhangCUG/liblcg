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

#include "lcg_eigen.h"

#include "cmath"

#include "config.h"
#ifdef LibLCG_OPENMP
#include "omp.h"
#endif

/**
 * @brief      Callback interface of the conjugate gradient solver
 *
 * @param[in]  Afp         Callback function for calculating the product of 'Ax'.
 * @param[in]  Pfp         Callback function for monitoring the iteration progress.
 * @param      m           Initial solution vector.
 * @param      B           Objective vector of the linear system.
 * @param      param       Parameter setup for the conjugate gradient methods.
 * @param      instance    The user data sent for the lcg_solver() function by the client. 
 * This variable is either 'this' for class member functions or 'nullptr' for global functions.
 *
 * @return     Status of the function.
 */
typedef int (*eigen_solver_ptr)(lcg_axfunc_eigen_ptr Afp, lcg_progress_eigen_ptr Pfp, Eigen::VectorXd &m, 
	const Eigen::VectorXd &B, const lcg_para* param, void* instance);

int lcg(lcg_axfunc_eigen_ptr Afp, lcg_progress_eigen_ptr Pfp, Eigen::VectorXd &m, const Eigen::VectorXd &B, 
	const lcg_para* param, void* instance);
int lcgs(lcg_axfunc_eigen_ptr Afp, lcg_progress_eigen_ptr Pfp, Eigen::VectorXd &m, const Eigen::VectorXd &B, 
	const lcg_para* param, void* instance);
int lbicgstab(lcg_axfunc_eigen_ptr Afp, lcg_progress_eigen_ptr Pfp, Eigen::VectorXd &m, const Eigen::VectorXd &B, 
	const lcg_para* param, void* instance);
int lbicgstab2(lcg_axfunc_eigen_ptr Afp, lcg_progress_eigen_ptr Pfp, Eigen::VectorXd &m, const Eigen::VectorXd &B, 
	const lcg_para* param, void* instance);

int lcg_solver_eigen(lcg_axfunc_eigen_ptr Afp, lcg_progress_eigen_ptr Pfp, Eigen::VectorXd &m, const Eigen::VectorXd &B, 
	const lcg_para* param, void* instance, lcg_solver_enum solver_id)
{
	eigen_solver_ptr cg_solver;
	switch (solver_id)
	{
		case LCG_CG:
			cg_solver = lcg;
			break;
		case LCG_CGS:
			cg_solver = lcgs;
			break;
		case LCG_BICGSTAB:
			cg_solver = lbicgstab;
			break;
		case LCG_BICGSTAB2:
			cg_solver = lbicgstab2;
			break;
		default:
			cg_solver = lcg;
			break;
	}

	return cg_solver(Afp, Pfp, m, B, param, instance);
}

int lpcg(lcg_axfunc_eigen_ptr Afp, lcg_axfunc_eigen_ptr Mfp, lcg_progress_eigen_ptr Pfp, Eigen::VectorXd &m, 
	const Eigen::VectorXd &B, const lcg_para* param, void* instance);

int lcg_solver_preconditioned_eigen(lcg_axfunc_eigen_ptr Afp, lcg_axfunc_eigen_ptr Mfp, lcg_progress_eigen_ptr Pfp, 
	Eigen::VectorXd &m, const Eigen::VectorXd &B, const lcg_para* param, void* instance, lcg_solver_enum solver_id)
{
	return lpcg(Afp, Mfp, Pfp, m, B, param, instance);
}

/**
 * @brief      A combined conjugate gradient solver function.
 *
 * @param[in]  Afp         Callback function for calculating the product of 'Ax'.
 * @param[in]  Pfp         Callback function for monitoring the iteration progress.
 * @param      m           Initial solution vector.
 * @param      B           Objective vector of the linear system.
 * @param[in]  low         The lower boundary of the acceptable solution.
 * @param[in]  hig         The higher boundary of the acceptable solution.
 * @param      param       Parameter setup for the conjugate gradient methods.
 * @param      instance    The user data sent for the lcg_solver() function by the client. 
 * This variable is either 'this' for class member functions or 'nullptr' for global functions.
 * @param      solver_id   Solver type used to solve the linear system. The default value is LCG_CGS.
 * @param      P           Precondition vector (optional expect for the LCG_PCG method). The default value is nullptr.
 *
 * @return     Status of the function.
 */
typedef int (*eigen_solver_ptr2)(lcg_axfunc_eigen_ptr Afp, lcg_progress_eigen_ptr Pfp, 
	Eigen::VectorXd &m, const Eigen::VectorXd &B, const Eigen::VectorXd &low, const Eigen::VectorXd &hig, 
	const lcg_para* param, void* instance);

int lpg(lcg_axfunc_eigen_ptr Afp, lcg_progress_eigen_ptr Pfp, 
	Eigen::VectorXd &m, const Eigen::VectorXd &B, const Eigen::VectorXd &low, const Eigen::VectorXd &hig, 
	const lcg_para* param, void* instance);
int lspg(lcg_axfunc_eigen_ptr Afp, lcg_progress_eigen_ptr Pfp, 
	Eigen::VectorXd &m, const Eigen::VectorXd &B, const Eigen::VectorXd &low, const Eigen::VectorXd &hig, 
	const lcg_para* param, void* instance);

int lcg_solver_constrained_eigen(lcg_axfunc_eigen_ptr Afp, lcg_progress_eigen_ptr Pfp, Eigen::VectorXd &m, 
	const Eigen::VectorXd &B, const Eigen::VectorXd &low, const Eigen::VectorXd &hig, 
	const lcg_para* param, void* instance, lcg_solver_enum solver_id)
{
	eigen_solver_ptr2 cg_solver;
	switch (solver_id)
	{
		case LCG_PG:
			cg_solver = lpg;
			break;
		case LCG_SPG:
			cg_solver = lspg;
			break;
		default:
			cg_solver = lpg;
			break;
	}

	return cg_solver(Afp, Pfp, m, B, low, hig, param, instance);
}


/**
 * @brief      Conjugate gradient method
 *
 * @param[in]  Afp         Callback function for calculating the product of 'Ax'.
 * @param[in]  Pfp         Callback function for monitoring the iteration progress.
 * @param      m           Initial solution vector.
 * @param      B           Objective vector of the linear system.
 * @param      param       Parameter setup for the conjugate gradient methods.
 * @param      instance    The user data sent for the lcg_solver() function by the client. 
 * This variable is either 'this' for class member functions or 'nullptr' for global functions.
 * @param      P           Precondition vector (optional expect for the LCG_PCG method). The default value is nullptr.
 *
 * @return     Status of the function.
 */
int lcg(lcg_axfunc_eigen_ptr Afp, lcg_progress_eigen_ptr Pfp, Eigen::VectorXd &m, 
	const Eigen::VectorXd &B, const lcg_para* param, void* instance)
{
	// set CG parameters
	lcg_para para = (param != nullptr) ? (*param) : defparam;

	int n_size = B.size();
	//check parameters
	if (n_size <= 0) return LCG_INVILAD_VARIABLE_SIZE;
	if (n_size != m.size()) return LCG_SIZE_NOT_MATCH;
	if (para.max_iterations < 0) return LCG_INVILAD_MAX_ITERATIONS;
	if (para.epsilon <= 0.0 || para.epsilon >= 1.0) return LCG_INVILAD_EPSILON;

	// locate memory
	Eigen::VectorXd gk(n_size), dk(n_size), Adk(n_size);

	Afp(instance, m, Adk);

	gk = Adk - B;
	dk = -1.0*gk;

	lcg_float m_mod = m.dot(m);
	if (m_mod < 1.0) m_mod = 1.0;

	lcg_float gk_mod = gk.dot(gk);

	int ret, t = 0;
	if (para.abs_diff && sqrt(gk_mod)/n_size <= para.epsilon)
	{
		ret = LCG_ALREADY_OPTIMIZIED;
		if (Pfp != nullptr)
		{
			Pfp(instance, &m, sqrt(gk_mod)/n_size, &para, 0);
		}
		goto func_ends;
	}	
	else if (gk_mod/m_mod <= para.epsilon)
	{
		ret = LCG_ALREADY_OPTIMIZIED;
		if (Pfp != nullptr)
		{
			Pfp(instance, &m, gk_mod/m_mod, &para, 0);
		}
		goto func_ends;
	}

	lcg_float dTAd, ak, betak, gk1_mod, residual;
	while(1)
	{
		if (para.abs_diff) residual = sqrt(gk_mod)/n_size;
		else residual = gk_mod/m_mod;

		if (Pfp != nullptr)
		{
			if (Pfp(instance, &m, residual, &para, t))
			{
				ret = LCG_STOP; goto func_ends;
			}
		}

		if (residual <= para.epsilon)
		{
			ret = LCG_CONVERGENCE; goto func_ends;
		}

		if (para.max_iterations > 0 && t+1 > para.max_iterations)
		{
			ret = LCG_REACHED_MAX_ITERATIONS;
			break;
		}
		
		t++;

		Afp(instance , dk, Adk);

		dTAd = dk.dot(Adk);
		ak = gk_mod/dTAd;

		m += ak*dk;
		gk += ak*Adk;

		m_mod = m.dot(m);
		if (m_mod < 1.0) m_mod = 1.0;

		gk1_mod = gk.dot(gk);
		betak = gk1_mod/gk_mod;
		gk_mod = gk1_mod;

		dk = (betak*dk - gk);
	}

	func_ends:
	{
		dk.resize(0);
		gk.resize(0);
		Adk.resize(0);
	}

	return ret;
}

/**
 * @brief      Preconditioned conjugate gradient method
 * 
 * @note       Algorithm 1 in "Preconditioned conjugate gradients for singular systems" by Kaasschieter (1988).
 *
 * @param[in]  Afp         Callback function for calculating the product of 'Ax'.
 * @param[in]  Pfp         Callback function for monitoring the iteration progress.
 * @param      m           Initial solution vector.
 * @param      B           Objective vector of the linear system.
 * @param[in]  n_size      Size of the solution vector and objective vector.
 * @param      param       Parameter setup for the conjugate gradient methods.
 * @param      instance    The user data sent for the lcg_solver() function by the client. 
 * This variable is either 'this' for class member functions or 'nullptr' for global functions.
 * @param      P           Precondition vector (optional expect for the LCG_PCG method). The default value is nullptr.
 *
 * @return     Status of the function.
 */
int lpcg(lcg_axfunc_eigen_ptr Afp, lcg_axfunc_eigen_ptr Mfp, lcg_progress_eigen_ptr Pfp, Eigen::VectorXd &m, const Eigen::VectorXd &B, 
	const lcg_para* param, void* instance)
{
	// set CG parameters
	lcg_para para = (param != nullptr) ? (*param) : defparam;

	int n_size = B.size();
	//check parameters
	if (n_size <= 0) return LCG_INVILAD_VARIABLE_SIZE;
	if (n_size != m.size()) return LCG_SIZE_NOT_MATCH;
	if (para.max_iterations < 0) return LCG_INVILAD_MAX_ITERATIONS;
	if (para.epsilon <= 0.0 || para.epsilon >= 1.0) return LCG_INVILAD_EPSILON;

	// locate memory
	Eigen::VectorXd rk(n_size), zk(n_size), dk(n_size), Adk(n_size);

	Afp(instance, m, Adk);

	rk = B - Adk;
	Mfp(instance, rk, zk);
	dk = zk;

	lcg_float rk_mod = rk.dot(rk);
	lcg_float zTr = zk.dot(rk);
	lcg_float m_mod = m.dot(m);
	if (m_mod < 1.0) m_mod = 1.0;

	int ret, t = 0;
	if (para.abs_diff && sqrt(rk_mod)/n_size <= para.epsilon)
	{
		ret = LCG_ALREADY_OPTIMIZIED;
		if (Pfp != nullptr)
		{
			Pfp(instance, &m, sqrt(rk_mod)/n_size, &para, 0);
		}
		goto func_ends;
	}	
	else if (rk_mod/m_mod <= para.epsilon)
	{
		ret = LCG_ALREADY_OPTIMIZIED;
		if (Pfp != nullptr)
		{
			Pfp(instance, &m, rk_mod/m_mod, &para, 0);
		}
		goto func_ends;
	}

	lcg_float dTAd, ak, betak, zTr1, residual;
	while(1)
	{
		if (para.abs_diff) residual = sqrt(rk_mod)/n_size;
		else residual = rk_mod/m_mod;

		if (Pfp != nullptr)
		{
			if (Pfp(instance, &m, residual, &para, t))
			{
				ret = LCG_STOP; goto func_ends;
			}
		}

		if (residual <= para.epsilon)
		{
			ret = LCG_CONVERGENCE; goto func_ends;
		}

		if (para.max_iterations > 0 && t+1 > para.max_iterations)
		{
			ret = LCG_REACHED_MAX_ITERATIONS;
			break;
		}
		
		t++;

		Afp(instance, dk, Adk);

		dTAd = dk.dot(Adk);
		ak = zTr/dTAd;

		m += ak*dk;
		rk -= ak*Adk;
		Mfp(instance, rk, zk);

		m_mod = m.dot(m);
		if (m_mod < 1.0) m_mod = 1.0;

		rk_mod = rk.dot(rk);

		zTr1 = zk.dot(rk);
		betak = zTr1/zTr;
		zTr = zTr1;

		dk = (zk + betak*dk);
	}

	func_ends:
	{
		rk.resize(0);
		zk.resize(0);
		dk.resize(0);
		Adk.resize(0);
	}

	return ret;
}

/**
 * @brief      Conjugate gradient squared method.
 * 
 * @note       Algorithm 2 in "Generalized conjugate gradient method" by Fokkema et al. (1996).
 *
 * @param[in]  Afp         Callback function for calculating the product of 'Ax'.
 * @param[in]  Pfp         Callback function for monitoring the iteration progress.
 * @param      m           Initial solution vector.
 * @param      B           Objective vector of the linear system.
 * @param[in]  n_size      Size of the solution vector and objective vector.
 * @param      param       Parameter setup for the conjugate gradient methods.
 * @param      instance    The user data sent for the lcg_solver() function by the client. 
 * This variable is either 'this' for class member functions or 'nullptr' for global functions.
 * @param      P           Precondition vector (optional expect for the LCG_PCG method). The default value is nullptr.
 *
 * @return     Status of the function.
 */
int lcgs(lcg_axfunc_eigen_ptr Afp, lcg_progress_eigen_ptr Pfp, Eigen::VectorXd &m, const Eigen::VectorXd &B, 
	const lcg_para* param, void* instance)
{
	// set CGS parameters
	lcg_para para = (param != nullptr) ? (*param) : defparam;

	int n_size = B.size();
	//check parameters
	if (n_size <= 0) return LCG_INVILAD_VARIABLE_SIZE;
	if (n_size != m.size()) return LCG_SIZE_NOT_MATCH;
	if (para.max_iterations < 0) return LCG_INVILAD_MAX_ITERATIONS;
	if (para.epsilon <= 0.0 || para.epsilon >= 1.0) return LCG_INVILAD_EPSILON;

	Eigen::VectorXd rk(n_size), r0_T(n_size), pk(n_size), Ax(n_size);
	Eigen::VectorXd uk(n_size), qk(n_size), wk(n_size);

	Afp(instance, m, Ax);

	// 假设p0和q0为零向量 则在第一次迭代是pk和uk都等于rk
	// 所以我们能直接从计算Apk开始迭代
	pk = uk = r0_T = rk = (B - Ax);

	lcg_float rkr0_T = rk.dot(r0_T);

	lcg_float m_mod = m.dot(m);
	if (m_mod < 1.0) m_mod = 1.0;

	lcg_float rk_mod = rk.dot(rk);

	int ret, t = 0;
	if (para.abs_diff && sqrt(rk_mod)/n_size <= para.epsilon)
	{
		ret = LCG_ALREADY_OPTIMIZIED;
		if (Pfp != nullptr)
		{
			Pfp(instance, &m, sqrt(rk_mod)/n_size, &para, 0);
		}
		goto func_ends;
	}	
	else if (rk_mod/m_mod <= para.epsilon)
	{
		ret = LCG_ALREADY_OPTIMIZIED;
		if (Pfp != nullptr)
		{
			Pfp(instance, &m, sqrt(rk_mod)/m_mod, &para, 0);
		}
		goto func_ends;
	}

	lcg_float ak, rkr0_T1, Apr_T, betak, residual;
	while(1)
	{
		if (para.abs_diff) residual = sqrt(rk_mod)/n_size;
		else residual = rk_mod/m_mod;

		if (Pfp != nullptr)
		{
			if (Pfp(instance, &m, residual, &para, t))
			{
				ret = LCG_STOP; goto func_ends;
			}
		}

		if (residual <= para.epsilon)
		{
			ret = LCG_CONVERGENCE; goto func_ends;
		}

		if (para.max_iterations > 0 && t+1 > para.max_iterations)
		{
			ret = LCG_REACHED_MAX_ITERATIONS;
			break;
		}
		
		t++;

		Afp(instance, pk, Ax);

		Apr_T = Ax.dot(r0_T);
		ak = rkr0_T/Apr_T;
		qk = uk - ak*Ax;
		wk = uk + qk;

		Afp(instance, wk, Ax);

		m += ak*wk;
		rk -= ak*Ax;

		m_mod = m.dot(m);
		if (m_mod < 1.0) m_mod = 1.0;

		rk_mod = rk.dot(rk);

		rkr0_T1 = rk.dot(r0_T);
		betak = rkr0_T1/rkr0_T;
		rkr0_T = rkr0_T1;

		uk = rk + betak*qk;
		pk = uk + betak*(qk + betak*pk);
	}

	func_ends:
	{
		rk.resize(0);
		r0_T.resize(0);
		pk.resize(0);
		Ax.resize(0);
		uk.resize(0);
		qk.resize(0);
		wk.resize(0);
	}

	return ret;
}

/**
 * @brief      Biconjugate gradient method.
 *
 * @param[in]  Afp         Callback function for calculating the product of 'Ax'.
 * @param[in]  Pfp         Callback function for monitoring the iteration progress.
 * @param      m           Initial solution vector.
 * @param      B           Objective vector of the linear system.
 * @param[in]  n_size      Size of the solution vector and objective vector.
 * @param      param       Parameter setup for the conjugate gradient methods.
 * @param      instance    The user data sent for the lcg_solver() function by the client. 
 * This variable is either 'this' for class member functions or 'nullptr' for global functions.
 * @param      P           Precondition vector (optional expect for the LCG_PCG method). The default value is nullptr.
 *
 * @return     Status of the function.
 */
int lbicgstab(lcg_axfunc_eigen_ptr Afp, lcg_progress_eigen_ptr Pfp, Eigen::VectorXd &m, const Eigen::VectorXd &B, 
	const lcg_para* param, void* instance)
{
	// set CGS parameters
	lcg_para para = (param != nullptr) ? (*param) : defparam;

	int n_size = B.size();
	//check parameters
	if (n_size <= 0) return LCG_INVILAD_VARIABLE_SIZE;
	if (n_size != m.size()) return LCG_SIZE_NOT_MATCH;
	if (para.max_iterations < 0) return LCG_INVILAD_MAX_ITERATIONS;
	if (para.epsilon <= 0.0 || para.epsilon >= 1.0) return LCG_INVILAD_EPSILON;

	Eigen::VectorXd rk(n_size), r0_T(n_size), pk(n_size);
	Eigen::VectorXd Ax(n_size), sk(n_size), Apk(n_size);

	Afp(instance, m, Ax);

	pk = r0_T = rk = (B - Ax);

	lcg_float rkr0_T = rk.dot(r0_T);

	lcg_float m_mod = m.dot(m);
	if (m_mod < 1.0) m_mod = 1.0;

	lcg_float rk_mod = rk.dot(rk);

	int ret, t = 0;
	if (para.abs_diff && sqrt(rk_mod)/n_size <= para.epsilon)
	{
		ret = LCG_ALREADY_OPTIMIZIED;
		if (Pfp != nullptr)
		{
			Pfp(instance, &m, sqrt(rk_mod)/n_size, &para, 0);
		}
		goto func_ends;
	}	
	else if (rk_mod/m_mod <= para.epsilon)
	{
		ret = LCG_ALREADY_OPTIMIZIED;
		if (Pfp != nullptr)
		{
			Pfp(instance, &m, rk_mod/m_mod, &para, 0);
		}
		goto func_ends;
	}

	lcg_float ak, wk, rkr0_T1, Apr_T, betak, Ass, AsAs, residual;
	while(1)
	{
		if (para.abs_diff) residual = sqrt(rk_mod)/n_size;
		else residual = rk_mod/m_mod;

		if (Pfp != nullptr)
		{
			if (Pfp(instance, &m, residual, &para, t))
			{
				ret = LCG_STOP; goto func_ends;
			}
		}

		if (residual <= para.epsilon)
		{
			ret = LCG_CONVERGENCE; goto func_ends;
		}

		if (para.max_iterations > 0 && t+1 > para.max_iterations)
		{
			ret = LCG_REACHED_MAX_ITERATIONS;
			break;
		}
		
		t++;

		Afp(instance, pk, Apk);

		Apr_T = Apk.dot(r0_T);
		ak = rkr0_T/Apr_T;

		sk = rk - ak*Apk;

		Afp(instance, sk, Ax);

		Ass = Ax.dot(sk);
		AsAs = Ax.dot(Ax);
		wk = Ass/AsAs;

		m += (ak*pk + wk*sk);
		rk = sk - wk*Ax;

		m_mod = m.dot(m);
		if (m_mod < 1.0) m_mod = 1.0;

		rk_mod = rk.dot(rk);

		rkr0_T1 = rk.dot(r0_T);
		betak = (ak/wk)*rkr0_T1/rkr0_T;
		rkr0_T = rkr0_T1;

		pk = rk + betak*(pk - wk*Apk);
	}

	func_ends:
	{
		rk.resize(0);
		r0_T.resize(0);
		pk.resize(0);
		Ax.resize(0);
		sk.resize(0);
		Apk.resize(0);
	}

	return ret;
}


/**
 * @brief      Biconjugate gradient method 2.
 *
 * @param[in]  Afp         Callback function for calculating the product of 'Ax'.
 * @param[in]  Pfp         Callback function for monitoring the iteration progress.
 * @param      m           Initial solution vector.
 * @param      B           Objective vector of the linear system.
 * @param[in]  n_size      Size of the solution vector and objective vector.
 * @param      param       Parameter setup for the conjugate gradient methods.
 * @param      instance    The user data sent for the lcg_solver() function by the client. 
 * This variable is either 'this' for class member functions or 'nullptr' for global functions.
 * @param      P           Precondition vector (optional expect for the LCG_PCG method). The default value is nullptr.
 *
 * @return     Status of the function.
 */
int lbicgstab2(lcg_axfunc_eigen_ptr Afp, lcg_progress_eigen_ptr Pfp, Eigen::VectorXd &m, const Eigen::VectorXd &B, 
	const lcg_para* param, void* instance)
{
	// set CGS parameters
	lcg_para para = (param != nullptr) ? (*param) : defparam;

	int n_size = B.size();
	//check parameters
	if (n_size <= 0) return LCG_INVILAD_VARIABLE_SIZE;
	if (n_size != m.size()) return LCG_SIZE_NOT_MATCH;
	if (para.max_iterations < 0) return LCG_INVILAD_MAX_ITERATIONS;
	if (para.epsilon <= 0.0) return LCG_INVILAD_EPSILON;
	if (para.restart_epsilon <= 0.0 || para.epsilon >= 1.0) return LCG_INVILAD_RESTART_EPSILON;

	Eigen::VectorXd rk(n_size), r0_T(n_size), pk(n_size);
	Eigen::VectorXd Ax(n_size), sk(n_size), Apk(n_size);

	Afp(instance, m, Ax);

	pk = r0_T = rk = B - Ax;

	lcg_float rkr0_T = rk.dot(r0_T);

	lcg_float m_mod = m.dot(m);
	if (m_mod < 1.0) m_mod = 1.0;

	lcg_float rk_mod = rk.dot(rk);

	int ret, t = 0;
	if (para.abs_diff && sqrt(rk_mod)/n_size <= para.epsilon)
	{
		ret = LCG_ALREADY_OPTIMIZIED;
		if (Pfp != nullptr)
		{
			Pfp(instance, &m, sqrt(rk_mod)/n_size, &para, 0);
		}
		goto func_ends;
	}	
	else if (rk_mod/m_mod <= para.epsilon)
	{
		ret = LCG_ALREADY_OPTIMIZIED;
		if (Pfp != nullptr)
		{
			Pfp(instance, &m, sqrt(rk_mod)/m_mod, &para, 0);
		}
		goto func_ends;
	}

	lcg_float ak, wk, rkr0_T1, Apr_T, betak;
	lcg_float Ass, AsAs, rr1_abs, residual;
	while(1)
	{
		if (para.abs_diff) residual = sqrt(rk_mod)/n_size;
		else residual = rk_mod/m_mod;

		if (Pfp != nullptr)
		{
			if (Pfp(instance, &m, residual, &para, t))
			{
				ret = LCG_STOP; goto func_ends;
			}
		}

		if (residual <= para.epsilon)
		{
			ret = LCG_CONVERGENCE; goto func_ends;
		}

		if (para.max_iterations > 0 && t+1 > para.max_iterations)
		{
			ret = LCG_REACHED_MAX_ITERATIONS;
			break;
		}
		
		t++;

		Afp(instance, pk, Apk);

		Apr_T = Apk.dot(r0_T);
		ak = rkr0_T/Apr_T;

		sk = rk - ak*Apk;

		if (para.abs_diff)
		{
			residual = sqrt(sk.dot(sk))/n_size;
			if (Pfp != nullptr)
			{
				if (Pfp(instance, &m, residual, &para, t))
				{
					ret = LCG_STOP; goto func_ends;
				}
			}
			if (residual <= para.epsilon)
			{
				m += ak*pk;
				ret = LCG_CONVERGENCE; goto func_ends;
			}

			if (para.max_iterations > 0 && t+1 > para.max_iterations)
			{
				ret = LCG_REACHED_MAX_ITERATIONS;
				break;
			}
			
			t++;
		}

		Afp(instance, sk, Ax);

		Ass = Ax.dot(sk);
		AsAs = Ax.dot(Ax);
		wk = Ass/AsAs;

		m += ak*pk + wk*sk;
		rk = sk - wk*Ax;

		m_mod = m.dot(m);
		if (m_mod < 1.0) m_mod = 1.0;

		rk_mod = rk.dot(rk);

		rkr0_T1 = rk.dot(r0_T);
		rr1_abs = fabs(rkr0_T1);
		if (rr1_abs < para.restart_epsilon)
		{
			r0_T = rk;
			pk = rk;

			rkr0_T1 = rk.dot(r0_T);
			betak = (ak/wk)*rkr0_T1/rkr0_T;
			rkr0_T = rkr0_T1;
		}
		else
		{
			betak = (ak/wk)*rkr0_T1/rkr0_T;
			rkr0_T = rkr0_T1;

			pk = rk + betak*(pk - wk*Apk);
		}
	}

	func_ends:
	{
		rk.resize(0);
		r0_T.resize(0);
		pk.resize(0);
		Ax.resize(0);
		sk.resize(0);
		Apk.resize(0);
	}

	return ret;
}

/**
 * @brief      Conjugate gradient method with projected gradient for inequality constraints.
 *
 * @param[in]  Afp         Callback function for calculating the product of 'Ax'.
 * @param[in]  Pfp         Callback function for monitoring the iteration progress.
 * @param      m           Initial solution vector.
 * @param      B           Objective vector of the linear system.
 * @param[in]  low         The lower boundary of the acceptable solution.
 * @param[in]  hig         The higher boundary of the acceptable solution.
 * @param      param       Parameter setup for the conjugate gradient methods.
 * @param      instance    The user data sent for the lcg_solver() function by the client. 
 * This variable is either 'this' for class member functions or 'nullptr' for global functions.
 * @param      solver_id   Solver type used to solve the linear system. The default value is LCG_CGS.
 * @param      P           Precondition vector (optional expect for the LCG_PCG method). The default value is nullptr.
 *
 * @return     Status of the function.
 */
int lpg(lcg_axfunc_eigen_ptr Afp, lcg_progress_eigen_ptr Pfp, Eigen::VectorXd &m, 
	const Eigen::VectorXd &B, const Eigen::VectorXd &low, const Eigen::VectorXd &hig, 
	const lcg_para* param, void* instance)
{
	// set CG parameters
	lcg_para para = (param != nullptr) ? (*param) : defparam;

	int n_size = B.size();
	//check parameters
	if (n_size <= 0) return LCG_INVILAD_VARIABLE_SIZE;
	if (n_size != m.size()) return LCG_SIZE_NOT_MATCH;
	if (n_size != low.size()) return LCG_SIZE_NOT_MATCH;
	if (n_size != hig.size()) return LCG_SIZE_NOT_MATCH;
	if (para.max_iterations < 0) return LCG_INVILAD_MAX_ITERATIONS;
	if (para.epsilon <= 0.0 || para.epsilon >= 1.0) return LCG_INVILAD_EPSILON;
	if (para.step <= 0.0) return LCG_INVALID_LAMBDA;

	// locate memory
	Eigen::VectorXd gk(n_size), Adk(n_size), m_new(n_size), dm(n_size);
	Eigen::VectorXd gk_new(n_size), sk(n_size), yk(n_size);

	lcg_float alpha_k = para.step;

	lcg_set2box_eigen(low, hig, m);
	Afp(instance, m, Adk);

	gk = Adk - B;

	lcg_float m_mod = m.dot(m);
	if (m_mod < 1.0) m_mod = 1.0;

	lcg_float gk_mod = gk.dot(gk);

	int ret, t = 0;
	if (para.abs_diff && sqrt(gk_mod)/n_size <= para.epsilon)
	{
		ret = LCG_ALREADY_OPTIMIZIED;
		if (Pfp != nullptr)
		{
			Pfp(instance, &m, sqrt(gk_mod)/n_size, &para, 0);
		}
		goto func_ends;
	}	
	else if (gk_mod/m_mod <= para.epsilon)
	{
		ret = LCG_ALREADY_OPTIMIZIED;
		if (Pfp != nullptr)
		{
			Pfp(instance, &m, gk_mod/m_mod, &para, 0);
		}
		goto func_ends;
	}

	lcg_float p_mod, sk_mod, syk_mod, residual;
	while(1)
	{
		if (para.abs_diff) residual = sqrt(gk_mod)/n_size;
		else residual = gk_mod/m_mod;

		if (Pfp != nullptr)
		{
			if (Pfp(instance, &m, residual, &para, t))
			{
				ret = LCG_STOP; goto func_ends;
			}
		}

		if (residual <= para.epsilon)
		{
			ret = LCG_CONVERGENCE; goto func_ends;
		}

		if (para.max_iterations > 0 && t+1 > para.max_iterations)
		{
			ret = LCG_REACHED_MAX_ITERATIONS;
			break;
		}
		
		t++;

		m_new = m - gk;
		lcg_set2box_eigen(low, hig, m_new);

		sk = m_new - m;
		p_mod = sqrt(sk.dot(sk))/n_size;

		/*
		if (Pfp != nullptr)
		{
			if (Pfp(instance, &m, p_mod, &para, time))
			{
				ret = LCG_STOP; goto func_ends;
			}
		}
		if (p_mod <= para.epsilon)
		{
			ret = LCG_CONVERGENCE; goto func_ends;
		}
		*/

		// project the model
		m_new = m - alpha_k*gk;
		lcg_set2box_eigen(low, hig, m_new);

		Afp(instance, m_new, Adk);

		gk_new = Adk - B;
		sk = m_new - m;
		yk = gk_new - gk;

		sk_mod = sk.dot(sk);
		syk_mod = sk.dot(yk);
		alpha_k = sk_mod/syk_mod;

		m = m_new;
		gk = gk_new;

		m_mod = m.dot(m);
		if (m_mod < 1.0) m_mod = 1.0;

		gk_mod = gk.dot(gk);
	}

	func_ends:
	{
		gk.resize(0);
		gk_new.resize(0);
		m_new.resize(0);
		sk.resize(0);
		yk.resize(0);
		Adk.resize(0);
	}

	return ret;
}

/**
 * @brief      Conjugate gradient method with projected gradient for inequality constraints.
 *
 * @param[in]  Afp         Callback function for calculating the product of 'Ax'.
 * @param[in]  Pfp         Callback function for monitoring the iteration progress.
 * @param      m           Initial solution vector.
 * @param      B           Objective vector of the linear system.
 * @param[in]  low         The lower boundary of the acceptable solution.
 * @param[in]  hig         The higher boundary of the acceptable solution.
 * @param      param       Parameter setup for the conjugate gradient methods.
 * @param      instance    The user data sent for the lcg_solver() function by the client. 
 * This variable is either 'this' for class member functions or 'nullptr' for global functions.
 * @param      solver_id   Solver type used to solve the linear system. The default value is LCG_CGS.
 * @param      P           Precondition vector (optional expect for the LCG_PCG method). The default value is nullptr.
 *
 * @return     Status of the function.
 */
int lspg(lcg_axfunc_eigen_ptr Afp, lcg_progress_eigen_ptr Pfp, Eigen::VectorXd &m, 
	const Eigen::VectorXd &B, const Eigen::VectorXd &low, const Eigen::VectorXd &hig, 
	const lcg_para* param, void* instance)
{
	// set CG parameters
	lcg_para para = (param != nullptr) ? (*param) : defparam;

	int i;
	int n_size = B.size();
	//check parameters
	if (n_size <= 0) return LCG_INVILAD_VARIABLE_SIZE;
	if (n_size != m.size()) return LCG_SIZE_NOT_MATCH;
	if (n_size != low.size()) return LCG_SIZE_NOT_MATCH;
	if (n_size != hig.size()) return LCG_SIZE_NOT_MATCH;
	if (para.max_iterations < 0) return LCG_INVILAD_MAX_ITERATIONS;
	if (para.epsilon <= 0.0 || para.epsilon >= 1.0) return LCG_INVILAD_EPSILON;
	if (para.step <= 0.0) return LCG_INVALID_LAMBDA;
	if (para.sigma <= 0.0 || para.sigma >= 1.0) return LCG_INVALID_SIGMA;
	if (para.beta <= 0.0 || para.beta >= 1.0) return LCG_INVALID_BETA;
	if (para.maxi_m <= 0) return LCG_INVALID_MAXIM;

	// locate memory
	Eigen::VectorXd gk(n_size), Adk(n_size), m_new(n_size), gk_new(n_size);
	Eigen::VectorXd sk(n_size), yk(n_size), dk(n_size), qk_m(para.maxi_m);

	lcg_float lambda_k = para.step;

	// project the initial model
	lcg_set2box_eigen(low, hig, m);

	Afp(instance, m, Adk);

	gk = Adk - B;

	lcg_float qk = 0.5*m.dot(Adk) - B.dot(m);
	qk_m = Eigen::VectorXd::Constant(para.maxi_m, -1e+30);
	qk_m[0] = qk;

	lcg_float m_mod = m.dot(m);
	if (m_mod < 1.0) m_mod = 1.0;

	lcg_float gk_mod = gk.dot(gk);

	int ret, t = 0;
	if (para.abs_diff && sqrt(gk_mod)/n_size <= para.epsilon)
	{
		ret = LCG_ALREADY_OPTIMIZIED;
		if (Pfp != nullptr)
		{
			Pfp(instance, &m, sqrt(gk_mod)/n_size, &para, 0);
		}
		goto func_ends;
	}	
	else if (gk_mod/m_mod <= para.epsilon)
	{
		ret = LCG_ALREADY_OPTIMIZIED;
		if (Pfp != nullptr)
		{
			Pfp(instance, &m, gk_mod/m_mod, &para, 0);
		}
		goto func_ends;
	}

	lcg_float alpha_k, maxi_qk;
	lcg_float p_mod, alpha_mod, sk_mod, syk_mod, residual;
	while(1)
	{
		if (para.abs_diff) residual = sqrt(gk_mod)/n_size;
		else residual = gk_mod/m_mod;

		if (Pfp != nullptr)
		{
			if (Pfp(instance, &m, residual, &para, t))
			{
				ret = LCG_STOP; goto func_ends;
			}
		}

		if (residual <= para.epsilon)
		{
			ret = LCG_CONVERGENCE; goto func_ends;
		}

		if (para.max_iterations > 0 && t+1 > para.max_iterations)
		{
			ret = LCG_REACHED_MAX_ITERATIONS;
			break;
		}
		
		t++;

		m_new = m - gk;
		lcg_set2box_eigen(low, hig, m_new);

		sk = m_new - m;
		p_mod = sqrt(sk.dot(sk))/n_size;

		/*
		if (Pfp != nullptr)
		{
			if (Pfp(instance, &m, p_mod, &para, time))
			{
				ret = LCG_STOP; goto func_ends;
			}
		}
		if (p_mod <= para.epsilon)
		{
			ret = LCG_CONVERGENCE; goto func_ends;
		}
		*/

		dk = m - lambda_k*gk;
		lcg_set2box_eigen(low, hig, dk);
		dk -= m;

		alpha_k = 1.0;
		m_new = m + alpha_k*dk;

		Afp(instance, m_new, Adk);

		qk = 0.5*m_new.dot(Adk) - B.dot(m_new);

		alpha_mod = para.sigma*alpha_k*gk.dot(dk);

		maxi_qk = qk_m[0];
		for (i = 1; i < para.maxi_m; i++)
		{
			maxi_qk = lcg_max(maxi_qk, qk_m[i]);
		}

		while(qk > maxi_qk + alpha_mod)
		{
			alpha_k = alpha_k*para.beta;
			m_new = m + alpha_k*dk;

			Afp(instance, m_new, Adk);

			qk = 0.5*m_new.dot(Adk) - B.dot(m_new);
			alpha_mod = para.sigma*alpha_k*gk.dot(dk);
		}

		// put new values in qk_m
		qk_m[(t+1)%para.maxi_m] = qk;

		gk_new = Adk - B;
		sk = m_new - m;
		yk = gk_new - gk;

		sk_mod = sk.dot(sk);
		syk_mod = sk.dot(yk);
		lambda_k = sk_mod/syk_mod;

		m = m_new;
		gk = gk_new;

		m_mod = m.dot(m);
		if (m_mod < 1.0) m_mod = 1.0;

		gk_mod = gk.dot(gk);
	}

	func_ends:
	{
		gk.resize(0);
		gk_new.resize(0);
		m_new.resize(0);
		sk.resize(0);
		yk.resize(0);
		Adk.resize(0);
		dk.resize(0);
		qk_m.resize(0);
	}

	return ret;
}