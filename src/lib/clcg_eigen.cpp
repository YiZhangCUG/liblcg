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

#include "cmath"
#include "ctime"
#include "iostream"

#include "clcg_eigen.h"

#include "config.h"
#ifdef LibLCG_OPENMP
#include "omp.h"
#endif


typedef int (*eigen_solver_ptr)(clcg_axfunc_eigen_ptr Afp, clcg_progress_eigen_ptr Pfp, Eigen::VectorXcd &m, 
	const Eigen::VectorXcd &B, const clcg_para* param, void* instance);

int clbicg(clcg_axfunc_eigen_ptr Afp, clcg_progress_eigen_ptr Pfp, Eigen::VectorXcd &m, 
	const Eigen::VectorXcd &B, const clcg_para* param, void* instance);
int clbicg_symmetric(clcg_axfunc_eigen_ptr Afp, clcg_progress_eigen_ptr Pfp, Eigen::VectorXcd &m, 
	const Eigen::VectorXcd &B, const clcg_para* param, void* instance);
int clcgs(clcg_axfunc_eigen_ptr Afp, clcg_progress_eigen_ptr Pfp, Eigen::VectorXcd &m, 
	const Eigen::VectorXcd &B, const clcg_para* param, void* instance);
int cltfqmr(clcg_axfunc_eigen_ptr Afp, clcg_progress_eigen_ptr Pfp, Eigen::VectorXcd &m, 
	const Eigen::VectorXcd &B, const clcg_para* param, void* instance);

int clcg_solver_eigen(clcg_axfunc_eigen_ptr Afp, clcg_progress_eigen_ptr Pfp, Eigen::VectorXcd &m, 
	const Eigen::VectorXcd &B, const clcg_para* param, void* instance, clcg_solver_enum solver_id)
{
	eigen_solver_ptr cg_solver;
	switch (solver_id)
	{
		case CLCG_BICG:
			cg_solver = clbicg;
			break;
		case CLCG_BICG_SYM:
			cg_solver = clbicg_symmetric;
			break;
		case CLCG_CGS:
			cg_solver = clcgs;
			break;
		case CLCG_TFQMR:
			cg_solver = cltfqmr;
			break;
		default:
			return CLCG_UNKNOWN_SOLVER;
	}

	return cg_solver(Afp, Pfp, m, B, param, instance);
}


typedef int (*eigen_preconditioned_solver_ptr)(clcg_axfunc_eigen_ptr Afp, clcg_axfunc_eigen_ptr Mfp, clcg_progress_eigen_ptr Pfp, 
	Eigen::VectorXcd &m, const Eigen::VectorXcd &B, const clcg_para* param, void* instance);

int clpcg(clcg_axfunc_eigen_ptr Afp, clcg_axfunc_eigen_ptr Mfp, clcg_progress_eigen_ptr Pfp, 
	Eigen::VectorXcd &m, const Eigen::VectorXcd &B, const clcg_para* param, void* instance);
int clpbicg(clcg_axfunc_eigen_ptr Afp, clcg_axfunc_eigen_ptr Mfp, clcg_progress_eigen_ptr Pfp, 
	Eigen::VectorXcd &m, const Eigen::VectorXcd &B, const clcg_para* param, void* instance);

int clcg_solver_preconditioned_eigen(clcg_axfunc_eigen_ptr Afp, clcg_axfunc_eigen_ptr Mfp, clcg_progress_eigen_ptr Pfp, 
	Eigen::VectorXcd &m, const Eigen::VectorXcd &B, const clcg_para* param, void* instance, clcg_solver_enum solver_id)
{
	eigen_preconditioned_solver_ptr cgp_solver;
	switch (solver_id)
	{
		case CLCG_PCG:
			cgp_solver = clpcg; break;
		case CLCG_PBICG:
			cgp_solver = clpbicg; break;
		default:
			return CLCG_UNKNOWN_SOLVER;
	}

	return cgp_solver(Afp, Mfp, Pfp, m, B, param, instance);
}


int clbicg(clcg_axfunc_eigen_ptr Afp, clcg_progress_eigen_ptr Pfp, Eigen::VectorXcd &m, 
	const Eigen::VectorXcd &B, const clcg_para* param, void* instance)
{
	// set CGS parameters
	clcg_para para = (param != nullptr) ? (*param) : defparam2;

	int n_size = B.size();
	//check parameters
	if (n_size <= 0) return CLCG_INVILAD_VARIABLE_SIZE;
	if (n_size != m.size()) return CLCG_SIZE_NOT_MATCH;
	if (para.max_iterations < 0) return CLCG_INVILAD_MAX_ITERATIONS;
	if (para.epsilon <= 0.0 || para.epsilon >= 1.0) return CLCG_INVILAD_EPSILON;

	std::complex<lcg_float> ak, Ad1d2, r1r2_next, betak;
	Eigen::VectorXcd r1k(n_size), r2k(n_size), d1k(n_size), d2k(n_size);
	Eigen::VectorXcd Ax(n_size);

	Afp(instance, m, Ax, MatNormal, NonConjugate);

	d1k = r1k = B - Ax;
	d2k = r2k = r1k.conjugate();

	// Eigen's dot is inner product
	std::complex<lcg_float> r1r2 = r2k.dot(r1k);

	lcg_float m_mod = std::norm(m.dot(m));
	if (m_mod < 1.0) m_mod = 1.0;

	lcg_float rk_mod = std::norm(r1k.dot(r1k));

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

	lcg_float residual;
	while(1)
	{
		if (para.abs_diff) residual = std::sqrt(rk_mod)/n_size;
		else residual = rk_mod/m_mod;

		if (Pfp != nullptr)
		{
			if (Pfp(instance, &m, residual, &para, t))
			{
				ret = CLCG_STOP; goto func_ends;
			}
		}

		if (residual <= para.epsilon)
		{
			ret = CLCG_CONVERGENCE; goto func_ends;
		}

		if (para.max_iterations > 0 && t+1 > para.max_iterations)
		{
			ret = LCG_REACHED_MAX_ITERATIONS;
			break;
		}
		
		t++;

		Afp(instance, d1k, Ax, MatNormal, NonConjugate);
		Ad1d2 = d2k.dot(Ax);
		ak = r1r2/Ad1d2;

		m = m + ak*d1k;
		r1k = r1k - ak*Ax;

		m_mod = std::norm(m.dot(m));
		if (m_mod < 1.0) m_mod = 1.0;

		rk_mod = std::norm(r1k.dot(r1k));

		Afp(instance, d2k, Ax, MatTranspose, Conjugate);

		r2k = r2k - std::conj(ak)*Ax;

		r1r2_next = r2k.dot(r1k);
		betak = r1r2_next/r1r2;
		r1r2 = r1r2_next;

		d1k = r1k + betak*d1k;
		d2k = r2k + std::conj(betak)*d2k;
	}

	func_ends:
	{
		r1k.resize(0);
		r2k.resize(0);
		d1k.resize(0);
		d2k.resize(0);
		Ax.resize(0);
	}

	return ret;
}

int clbicg_symmetric(clcg_axfunc_eigen_ptr Afp, clcg_progress_eigen_ptr Pfp, Eigen::VectorXcd &m, 
	const Eigen::VectorXcd &B, const clcg_para* param, void* instance)
{
	// set CGS parameters
	clcg_para para = (param != nullptr) ? (*param) : defparam2;

	int n_size = B.size();
	//check parameters
	if (n_size <= 0) return CLCG_INVILAD_VARIABLE_SIZE;
	if (n_size != m.size()) return CLCG_SIZE_NOT_MATCH;
	if (para.max_iterations < 0) return CLCG_INVILAD_MAX_ITERATIONS;
	if (para.epsilon <= 0.0 || para.epsilon >= 1.0) return CLCG_INVILAD_EPSILON;

	std::complex<lcg_float> ak, rkrk2, betak, dkAx;
	Eigen::VectorXcd rk(n_size), dk(n_size), Ax(n_size);

	Afp(instance, m, Ax, MatNormal, NonConjugate);

	dk = rk = (B - Ax);

	std::complex<lcg_float> rkrk = rk.conjugate().dot(rk);

	lcg_float m_mod = std::norm(m.dot(m));
	if (m_mod < 1.0) m_mod = 1.0;

	lcg_float rk_mod = std::norm(rk.dot(rk));

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

	lcg_float residual;
	while(1)
	{
		if (para.abs_diff) residual = std::sqrt(rk_mod)/n_size;
		else residual = rk_mod/m_mod;

		if (Pfp != nullptr)
		{
			if (Pfp(instance, &m, residual, &para, t))
			{
				ret = CLCG_STOP; goto func_ends;
			}
		}

		if (residual <= para.epsilon)
		{
			ret = CLCG_CONVERGENCE; goto func_ends;
		}

		if (para.max_iterations > 0 && t+1 > para.max_iterations)
		{
			ret = LCG_REACHED_MAX_ITERATIONS;
			break;
		}
		
		t++;

		Afp(instance, dk, Ax, MatNormal, NonConjugate);
		dkAx = dk.conjugate().dot(Ax);
		ak = rkrk/dkAx;

		m += ak*dk;
		rk -= ak*Ax;

		m_mod = std::norm(m.dot(m));
		if (m_mod < 1.0) m_mod = 1.0;

		rk_mod = std::norm(rk.dot(rk));

		rkrk2 = rk.conjugate().dot(rk);
		betak = rkrk2/rkrk;
		rkrk = rkrk2;

		dk = rk + betak*dk;
	}

	func_ends:
	{
		rk.resize(0);
		dk.resize(0);
		Ax.resize(0);
	}

	return ret;
}

int clcgs(clcg_axfunc_eigen_ptr Afp, clcg_progress_eigen_ptr Pfp, Eigen::VectorXcd &m, 
	const Eigen::VectorXcd &B, const clcg_para* param, void* instance)
{
	// set CGS parameters
	clcg_para para = (param != nullptr) ? (*param) : defparam2;

	int n_size = B.size();
	//check parameters
	if (n_size <= 0) return CLCG_INVILAD_VARIABLE_SIZE;
	if (n_size != m.size()) return CLCG_SIZE_NOT_MATCH;
	if (para.max_iterations < 0) return CLCG_INVILAD_MAX_ITERATIONS;
	if (para.epsilon <= 0.0 || para.epsilon >= 1.0) return CLCG_INVILAD_EPSILON;

	std::complex<lcg_float> ak, rhok2, sigma, betak, rkmod;
	Eigen::VectorXcd rk(n_size), s0, pk(n_size);
	Eigen::VectorXcd Ax(n_size), uk(n_size), qk(n_size), wk(n_size);

	Afp(instance, m, Ax, MatNormal, NonConjugate);

	pk = uk = rk = (B - Ax);

	std::complex<lcg_float> rhok;
	do
	{
		s0 = Eigen::VectorXcd::Random(n_size);
		rhok = s0.conjugate().dot(rk);
	} while (std::sqrt(std::norm(rhok)) < 1e-8);

	lcg_float m_mod = std::norm(m.dot(m));
	if (m_mod < 1.0) m_mod = 1.0;

	lcg_float rk_mod = std::norm(rk.dot(rk));

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

	lcg_float residual;
	while(1)
	{
		if (para.abs_diff) residual = std::sqrt(rk_mod)/n_size;
		else residual = rk_mod/m_mod;

		if (Pfp != nullptr)
		{
			if (Pfp(instance, &m, residual, &para, t))
			{
				ret = CLCG_STOP; goto func_ends;
			}
		}

		if (residual <= para.epsilon)
		{
			ret = CLCG_CONVERGENCE; goto func_ends;
		}

		if (para.max_iterations > 0 && t+1 > para.max_iterations)
		{
			ret = LCG_REACHED_MAX_ITERATIONS;
			break;
		}
		
		t++;

		Afp(instance, pk, Ax, MatNormal, NonConjugate);
		sigma = s0.conjugate().dot(Ax);
		ak = rhok/sigma;

		qk = uk - ak*Ax;
		wk = uk + qk;

		Afp(instance, wk, Ax, MatNormal, NonConjugate);

		m += ak*wk;
		rk -= ak*Ax;

		m_mod = std::norm(m.dot(m));
		if (m_mod < 1.0) m_mod = 1.0;

		rk_mod = std::norm(rk.dot(rk));

		rhok2 = s0.conjugate().dot(rk);
		betak = rhok2/rhok;
		rhok = rhok2;

		uk = rk + betak*qk;
		pk = uk + betak*(qk + betak*pk);
	}

	func_ends:
	{
		rk.resize(0);
		s0.resize(0);
		pk.resize(0);
		Ax.resize(0);
		uk.resize(0);
		qk.resize(0);
		wk.resize(0);
	}

	return ret;
}

int cltfqmr(clcg_axfunc_eigen_ptr Afp, clcg_progress_eigen_ptr Pfp, Eigen::VectorXcd &m, 
	const Eigen::VectorXcd &B, const clcg_para* param, void* instance)
{
	// set CGS parameters
	clcg_para para = (param != nullptr) ? (*param) : defparam2;

	int n_size = B.size();
	//check parameters
	if (n_size <= 0) return CLCG_INVILAD_VARIABLE_SIZE;
	if (n_size != m.size()) return CLCG_SIZE_NOT_MATCH;
	if (para.max_iterations < 0) return CLCG_INVILAD_MAX_ITERATIONS;
	if (para.epsilon <= 0.0 || para.epsilon >= 1.0) return CLCG_INVILAD_EPSILON;

	int j;
	Eigen::VectorXcd pk(n_size), uk(n_size), vk(n_size), dk(n_size);
	Eigen::VectorXcd r0(n_size), rk(n_size), Ax(n_size), qk(n_size);
	Eigen::VectorXcd uqk(n_size);

	Afp(instance, m, Ax, MatNormal, NonConjugate);

	pk = uk = r0 = rk = (B - Ax);
	dk.setZero();

	std::complex<lcg_float> rk_mod = rk.dot(rk);

	lcg_float theta = 0.0, omega = sqrt(rk_mod.real());
	lcg_float residual, tao = omega;
	std::complex<lcg_float> rk_mod2, sigma, alpha, betak, rho, rho2, sign, eta(0.0, 0.0);

	rho = r0.dot(r0);

	lcg_float m_mod = std::norm(m.dot(m));
	if (m_mod < 1.0) m_mod = 1.0;

	int ret, t = 0;
	if (para.abs_diff && sqrt(std::norm(rk_mod))/n_size <= para.epsilon)
	{
		ret = LCG_ALREADY_OPTIMIZIED;
		if (Pfp != nullptr)
		{
			Pfp(instance, &m, sqrt(std::norm(rk_mod))/n_size, &para, 0);
		}
		goto func_ends;
	}	
	else if (std::norm(rk_mod)/m_mod <= para.epsilon)
	{
		ret = LCG_ALREADY_OPTIMIZIED;
		if (Pfp != nullptr)
		{
			Pfp(instance, &m, std::norm(rk_mod)/m_mod, &para, 0);
		}
		goto func_ends;
	}

	while(1)
	{
		Afp(instance, pk, vk, MatNormal, NonConjugate);

		sigma = r0.dot(vk);
		alpha = rho/sigma;

		qk = uk - alpha*vk;
		uqk = uk + qk;

		Afp(instance, uqk, Ax, MatNormal, NonConjugate);

		rk -= alpha*Ax;
		rk_mod2 = rk.dot(rk);

		for (j = 1; j <= 2; j++)
		{
			if (para.abs_diff) residual = std::sqrt(std::norm(rk_mod))/n_size;
			else residual = std::norm(rk_mod)/m_mod;

			if (Pfp != nullptr)
			{
				if (Pfp(instance, &m, residual, &para, t))
				{
					ret = CLCG_STOP; goto func_ends;
				}
			}

			if (residual <= para.epsilon)
			{
				ret = CLCG_CONVERGENCE; goto func_ends;
			}

			if (para.max_iterations > 0 && t+1 > para.max_iterations)
			{
				ret = LCG_REACHED_MAX_ITERATIONS;
				break;
			}
			
			t++;

			sign = theta*theta*(eta/alpha);

			if (j == 1)
			{
				omega = sqrt(sqrt(rk_mod.real())*sqrt(rk_mod2.real()));
				dk = uk + sign*dk;
			}
			else
			{
				omega = sqrt(rk_mod2.real());
				dk = qk + sign*dk;
			}

			theta = omega/tao;
			tao = omega/sqrt(1.0+theta*theta);
			eta = (1.0/(1.0+theta*theta))*alpha;

			m += eta*dk;

			m_mod = std::norm(m.dot(m));
			if (m_mod < 1.0) m_mod = 1.0;
		}
		rk_mod = rk_mod2;

		rho2 = r0.dot(rk);
		betak = rho2/rho;
		rho = rho2;

		uk = rk + betak*qk;
		pk = uk + betak*(qk + betak*pk);
	}

	func_ends:
	{
		pk.resize(0);
		uk.resize(0);
		vk.resize(0);
		dk.resize(0);
		r0.resize(0);
		rk.resize(0);
		Ax.resize(0);
		qk.resize(0);
		uqk.resize(0);
	}

	return ret;
}

int clpcg(clcg_axfunc_eigen_ptr Afp, clcg_axfunc_eigen_ptr Mfp, clcg_progress_eigen_ptr Pfp, 
	Eigen::VectorXcd &m, const Eigen::VectorXcd &B, const clcg_para* param, void* instance)
{
	// set CGS parameters
	clcg_para para = (param != nullptr) ? (*param) : defparam2;

	int n_size = B.size();
	//check parameters
	if (n_size <= 0) return CLCG_INVILAD_VARIABLE_SIZE;
	if (n_size != m.size()) return CLCG_SIZE_NOT_MATCH;
	if (para.max_iterations < 0) return CLCG_INVILAD_MAX_ITERATIONS;
	if (para.epsilon <= 0.0 || para.epsilon >= 1.0) return CLCG_INVILAD_EPSILON;

	Eigen::VectorXcd rk(n_size), dk(n_size), sk(n_size), Ax(n_size);

	Afp(instance, m, Ax, MatNormal, NonConjugate);

	rk = (B - Ax);
	Mfp(instance, rk, dk, MatNormal, NonConjugate);

	std::complex<lcg_float> ak, d_old, betak, dkAx;
	std::complex<lcg_float> d_new = rk.conjugate().dot(dk);

	lcg_float m_mod = std::norm(m.dot(m));
	if (m_mod < 1.0) m_mod = 1.0;

	lcg_float rk_mod = std::norm(rk.dot(rk));

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

	lcg_float residual;
	while(1)
	{
		if (para.abs_diff) residual = std::sqrt(rk_mod)/n_size;
		else residual = rk_mod/m_mod;

		if (Pfp != nullptr)
		{
			if (Pfp(instance, &m, residual, &para, t))
			{
				ret = CLCG_STOP; goto func_ends;
			}
		}

		if (residual <= para.epsilon)
		{
			ret = CLCG_CONVERGENCE; goto func_ends;
		}

		if (para.max_iterations > 0 && t+1 > para.max_iterations)
		{
			ret = LCG_REACHED_MAX_ITERATIONS;
			break;
		}
		
		t++;

		Afp(instance, dk, Ax, MatNormal, NonConjugate);
		dkAx = dk.conjugate().dot(Ax);
		ak = d_new/dkAx;

		m += ak*dk;
		rk -= ak*Ax;

		m_mod = std::norm(m.dot(m));
		if (m_mod < 1.0) m_mod = 1.0;

		rk_mod = std::norm(rk.dot(rk));

		Mfp(instance, rk, sk, MatNormal, NonConjugate);

		d_old = d_new;
		d_new = rk.conjugate().dot(sk);

		betak = d_new/d_old;

		dk = sk + betak*dk;
	}

	func_ends:
	{
		rk.resize(0);
		dk.resize(0);
		sk.resize(0);
		Ax.resize(0);
	}

	return ret;
}

int clpbicg(clcg_axfunc_eigen_ptr Afp, clcg_axfunc_eigen_ptr Mfp, clcg_progress_eigen_ptr Pfp, 
	Eigen::VectorXcd &m, const Eigen::VectorXcd &B, const clcg_para* param, void* instance)
{
	// set CGS parameters
	clcg_para para = (param != nullptr) ? (*param) : defparam2;

	int n_size = B.size();
	//check parameters
	if (n_size <= 0) return CLCG_INVILAD_VARIABLE_SIZE;
	if (n_size != m.size()) return CLCG_SIZE_NOT_MATCH;
	if (para.max_iterations < 0) return CLCG_INVILAD_MAX_ITERATIONS;
	if (para.epsilon <= 0.0 || para.epsilon >= 1.0) return CLCG_INVILAD_EPSILON;

	std::complex<lcg_float> ak, betak, pkAx, rhok2;
	Eigen::VectorXcd rk(n_size), rsk(n_size), zk(n_size), pk(n_size), psk(n_size), Ax(n_size), Asx(n_size);

	Afp(instance, m, Ax, MatNormal, NonConjugate);

	rk = (B - Ax);
	Mfp(instance, rk, zk, MatNormal, NonConjugate);

	pk = zk;
	rsk = rk.conjugate();
	psk = pk.conjugate();

	std::complex<lcg_float> rhok = rsk.dot(zk);

	lcg_float m_mod = std::norm(m.dot(m));
	if (m_mod < 1.0) m_mod = 1.0;

	lcg_float rk_mod = std::norm(rk.dot(rk));

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

	lcg_float residual;
	while(1)
	{
		if (para.abs_diff) residual = std::sqrt(rk_mod)/n_size;
		else residual = rk_mod/m_mod;

		if (Pfp != nullptr)
		{
			if (Pfp(instance, &m, residual, &para, t))
			{
				ret = CLCG_STOP; goto func_ends;
			}
		}

		if (residual <= para.epsilon)
		{
			ret = CLCG_CONVERGENCE; goto func_ends;
		}

		if (para.max_iterations > 0 && t+1 > para.max_iterations)
		{
			ret = LCG_REACHED_MAX_ITERATIONS;
			break;
		}
		
		t++;

		Afp(instance, pk, Ax, MatNormal, NonConjugate);
		Afp(instance, psk, Asx, MatNormal, Conjugate);

		pkAx = psk.dot(Ax);
		ak = rhok/pkAx;

		m += ak*pk;
		rsk = rk.conjugate() - std::conj(ak)*Asx;
		rk -= ak*Ax;

		m_mod = std::norm(m.dot(m));
		if (m_mod < 1.0) m_mod = 1.0;

		rk_mod = std::norm(rk.dot(rk));

		Mfp(instance, rk, zk, MatNormal, NonConjugate);

		rhok2 = rsk.dot(zk);
		betak = rhok2/rhok;
		rhok = rhok2;

		pk = zk + betak*pk;
		psk = zk.conjugate() + std::conj(betak)*psk;
	}

	func_ends:
	{
		rk.resize(0);
		rsk.resize(0);
		zk.resize(0);
		pk.resize(0);
		psk.resize(0);
		Ax.resize(0);
		Asx.resize(0);
	}

	return ret;
}