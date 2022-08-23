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

#include "clcg.h"

#include "cmath"

#include "config.h"
#ifdef LibLCG_OPENMP
#include "omp.h"
#endif

typedef int (*clcg_solver_ptr)(clcg_axfunc_ptr Afp, clcg_progress_ptr Pfp, lcg_complex* m, 
	const lcg_complex* B, const int n_size, const clcg_para* param, void* instance);

int clbicg(clcg_axfunc_ptr Afp, clcg_progress_ptr Pfp, lcg_complex* m, const lcg_complex* B, 
	const int n_size, const clcg_para* param, void* instance);
int clbicg_symmetric(clcg_axfunc_ptr Afp, clcg_progress_ptr Pfp, lcg_complex* m, const lcg_complex* B, 
	const int n_size, const clcg_para* param, void* instance);
int clcgs(clcg_axfunc_ptr Afp, clcg_progress_ptr Pfp, lcg_complex* m, const lcg_complex* B, 
	const int n_size, const clcg_para* param, void* instance);
int clbicgstab(clcg_axfunc_ptr Afp, clcg_progress_ptr Pfp, lcg_complex* m, const lcg_complex* B, 
	const int n_size, const clcg_para* param, void* instance);
int cltfqmr(clcg_axfunc_ptr Afp, clcg_progress_ptr Pfp, lcg_complex* m, const lcg_complex* B, 
	const int n_size, const clcg_para* param, void* instance);

int clcg_solver(clcg_axfunc_ptr Afp, clcg_progress_ptr Pfp, lcg_complex* m, 
	const lcg_complex* B, const int n_size, const clcg_para* param, void* instance, 
	clcg_solver_enum solver_id)
{
	clcg_solver_ptr cg_solver;
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
		case CLCG_BICGSTAB:
			cg_solver = clbicgstab;
			break;
		case CLCG_TFQMR:
			cg_solver = cltfqmr;
			break;
		default:
			cg_solver = clcgs;
			break;
	}

	return cg_solver(Afp, Pfp, m, B, n_size, param, instance);
}


int clbicg(clcg_axfunc_ptr Afp, clcg_progress_ptr Pfp, lcg_complex* m, const lcg_complex* B, 
	const int n_size, const clcg_para* param, void* instance)
{
	// set CGS parameters
	clcg_para para = (param != nullptr) ? (*param) : defparam2;

	//check parameters
	if (n_size <= 0) return CLCG_INVILAD_VARIABLE_SIZE;
	if (para.max_iterations < 0) return CLCG_INVILAD_MAX_ITERATIONS;
	if (para.epsilon <= 0.0 || para.epsilon >= 1.0) return CLCG_INVILAD_EPSILON;

	if (m == nullptr) return CLCG_INVALID_POINTER;
	if (B == nullptr) return CLCG_INVALID_POINTER;

	int i;
	lcg_complex *r1k = nullptr, *r2k = nullptr, *d1k = nullptr, *d2k = nullptr;
	lcg_complex *Ax = nullptr;
	r1k = clcg_malloc(n_size); r2k = clcg_malloc(n_size);
	d1k = clcg_malloc(n_size); d2k = clcg_malloc(n_size);
	Ax  = clcg_malloc(n_size);

	lcg_complex ak, Ad1d2, r1r2_next, betak;

	Afp(instance, m, Ax, n_size, MatNormal, NonConjugate);

#pragma omp parallel for private (i) schedule(guided)
	for (i = 0; i < n_size; i++)
	{
		d1k[i] = r1k[i] = B[i] - Ax[i];
		d2k[i] = r2k[i] = clcg_conjugate(&r1k[i]);
	}

	lcg_complex r1r2;
	clcg_inner(r1r2, r2k, r1k, n_size);

	lcg_float m_square;
	lcg_complex m_mod;
	clcg_inner(m_mod, m, m, n_size);
	m_square = clcg_square(&m_mod);
	if (m_square < 1.0) m_square = 1.0;

	lcg_float rk_square;
	lcg_complex rk_mod;
	clcg_inner(rk_mod, r1k, r1k, n_size);
	rk_square = clcg_square(&rk_mod);

	int ret, t = 0;
	if (para.abs_diff && sqrt(rk_square)/n_size <= para.epsilon)
	{
		ret = LCG_ALREADY_OPTIMIZIED;
		if (Pfp != nullptr)
		{
			Pfp(instance, m, sqrt(rk_square)/n_size, &para, n_size, 0);
		}
		goto func_ends;
	}	
	else if (rk_square/m_square <= para.epsilon)
	{
		ret = LCG_ALREADY_OPTIMIZIED;
		if (Pfp != nullptr)
		{
			Pfp(instance, m, rk_square/m_square, &para, n_size, 0);
		}
		goto func_ends;
	}

	lcg_float residual;
	while(1)
	{
		if (para.abs_diff) residual = sqrt(rk_square)/n_size;
		else residual = rk_square/m_square;

		if (Pfp != nullptr)
		{
			if (Pfp(instance, m, residual, &para, n_size, t))
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

		Afp(instance, d1k, Ax, n_size, MatNormal, NonConjugate);
		clcg_inner(Ad1d2, d2k, Ax, n_size);
		ak = r1r2/Ad1d2;

#pragma omp parallel for private (i) schedule(guided)
		for (i = 0; i < n_size; i++)
		{
			m[i] = m[i] + ak*d1k[i];
			r1k[i] = r1k[i] - ak*Ax[i];
		}

		clcg_inner(m_mod, m, m, n_size);
		m_square = clcg_square(&m_mod);
		if (m_square < 1.0) m_square = 1.0;

		clcg_inner(rk_mod, r1k, r1k, n_size);
		rk_square = clcg_square(&rk_mod);

		Afp(instance, d2k, Ax, n_size, MatTranspose, Conjugate);

#pragma omp parallel for private (i) schedule(guided)
		for (i = 0; i < n_size; i++)
		{
			r2k[i] = r2k[i] - clcg_conjugate(&ak)*Ax[i];
		}

		for (i = 0; i < n_size; i++)
		{
			if (m[i] != m[i])
			{
				ret = CLCG_NAN_VALUE; goto func_ends;
			}
		}

		clcg_inner(r1r2_next, r2k, r1k, n_size);
		betak = r1r2_next/r1r2;
		r1r2 = r1r2_next;

#pragma omp parallel for private (i) schedule(guided)
		for (i = 0; i < n_size; i++)
		{
			d1k[i] = r1k[i] + betak*d1k[i];
			d2k[i] = r2k[i] + clcg_conjugate(&betak)*d2k[i];
		}
	}

	func_ends:
	{
		clcg_free(r1k);
		clcg_free(r2k);
		clcg_free(d1k);
		clcg_free(d2k);
		clcg_free(Ax);
	}

	return ret;
}

int clbicg_symmetric(clcg_axfunc_ptr Afp, clcg_progress_ptr Pfp, lcg_complex* m, const lcg_complex* B, 
	const int n_size, const clcg_para* param, void* instance)
{
	// set CGS parameters
	clcg_para para = (param != nullptr) ? (*param) : defparam2;

	//check parameters
	if (n_size <= 0) return CLCG_INVILAD_VARIABLE_SIZE;
	if (para.max_iterations < 0) return CLCG_INVILAD_MAX_ITERATIONS;
	if (para.epsilon <= 0.0 || para.epsilon >= 1.0) return CLCG_INVILAD_EPSILON;

	if (m == nullptr) return CLCG_INVALID_POINTER;
	if (B == nullptr) return CLCG_INVALID_POINTER;

	int i;
	lcg_complex *rk = nullptr, *dk = nullptr;
	lcg_complex *Ax = nullptr;
	rk = clcg_malloc(n_size); dk = clcg_malloc(n_size);
	Ax = clcg_malloc(n_size);

	lcg_complex ak, rkrk2, betak, dkAx;

	Afp(instance, m, Ax, n_size, MatNormal, NonConjugate);

#pragma omp parallel for private (i) schedule(guided)
	for (i = 0; i < n_size; i++)
	{
		dk[i] = rk[i] = B[i] - Ax[i];
	}

	lcg_complex rkrk;
	clcg_dot(rkrk, rk, rk, n_size);

	lcg_float m_square;
	lcg_complex m_mod;
	clcg_inner(m_mod, m, m, n_size);
	m_square = clcg_square(&m_mod);
	if (m_square < 1.0) m_square = 1.0;

	lcg_float rk_square;
	lcg_complex rk_mod;
	clcg_inner(rk_mod, rk, rk, n_size);
	rk_square = clcg_square(&rk_mod);

	int ret, t = 0;
	if (para.abs_diff && sqrt(rk_square)/n_size <= para.epsilon)
	{
		ret = LCG_ALREADY_OPTIMIZIED;
		if (Pfp != nullptr)
		{
			Pfp(instance, m, sqrt(rk_square)/n_size, &para, n_size, 0);
		}
		goto func_ends;
	}	
	else if (rk_square/m_square <= para.epsilon)
	{
		ret = LCG_ALREADY_OPTIMIZIED;
		if (Pfp != nullptr)
		{
			Pfp(instance, m, rk_square/m_square, &para, n_size, 0);
		}
		goto func_ends;
	}

	lcg_float residual;
	while(1)
	{
		if (para.abs_diff) residual = sqrt(rk_square)/n_size;
		else residual = rk_square/m_square;

		if (Pfp != nullptr)
		{
			if (Pfp(instance, m, residual, &para, n_size, t))
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

		Afp(instance, dk, Ax, n_size, MatNormal, NonConjugate);
		clcg_dot(dkAx, dk, Ax, n_size);
		ak = rkrk/dkAx;

#pragma omp parallel for private (i) schedule(guided)
		for (i = 0; i < n_size; i++)
		{
			m[i] = m[i] + ak*dk[i];
			rk[i] = rk[i] - ak*Ax[i];
		}

		clcg_inner(m_mod, m, m, n_size);
		m_square = clcg_square(&m_mod);
		if (m_square < 1.0) m_square = 1.0;

		clcg_inner(rk_mod, rk, rk, n_size);
		rk_square = clcg_square(&rk_mod);

		for (i = 0; i < n_size; i++)
		{
			if (m[i] != m[i])
			{
				ret = CLCG_NAN_VALUE; goto func_ends;
			}
		}

		clcg_dot(rkrk2, rk, rk, n_size);
		betak = rkrk2/rkrk;
		rkrk = rkrk2;

#pragma omp parallel for private (i) schedule(guided)
		for (i = 0; i < n_size; i++)
		{
			dk[i] = rk[i] + betak*dk[i];
		}
	}

	func_ends:
	{
		clcg_free(rk);
		clcg_free(dk);
		clcg_free(Ax);
	}

	return ret;
}

int clcgs(clcg_axfunc_ptr Afp, clcg_progress_ptr Pfp, lcg_complex* m, const lcg_complex* B, 
	const int n_size, const clcg_para* param, void* instance)
{
	// set CGS parameters
	clcg_para para = (param != nullptr) ? (*param) : defparam2;

	//check parameters
	if (n_size <= 0) return CLCG_INVILAD_VARIABLE_SIZE;
	if (para.max_iterations < 0) return CLCG_INVILAD_MAX_ITERATIONS;
	if (para.epsilon <= 0.0 || para.epsilon >= 1.0) return CLCG_INVILAD_EPSILON;

	if (m == nullptr) return CLCG_INVALID_POINTER;
	if (B == nullptr) return CLCG_INVALID_POINTER;

	int i;
	lcg_complex *rk = nullptr, *rbar0 = nullptr, *pk = nullptr;
	lcg_complex *Ax = nullptr, *uk = nullptr, *qk = nullptr, *wk = nullptr; // w_k = u_{k-1} + q_k
	rk = clcg_malloc(n_size); rbar0 = clcg_malloc(n_size);
	pk = clcg_malloc(n_size); Ax  = clcg_malloc(n_size);
	uk = clcg_malloc(n_size); qk  = clcg_malloc(n_size);
	wk = clcg_malloc(n_size);

	lcg_complex ak, rhok2, sigma, betak;

	Afp(instance, m, Ax, n_size, MatNormal, NonConjugate);

#pragma omp parallel for private (i) schedule(guided)
	for (i = 0; i < n_size; i++)
	{
		pk[i] = uk[i] = rk[i] = B[i] - Ax[i];
	}

	lcg_complex rhok;
	do
	{
		clcg_vecrnd(rbar0, lcg_complex(1.0, 0.0), lcg_complex(2.0, 0.0), n_size);
		clcg_inner(rhok, rbar0, rk, n_size);
	} while (clcg_module(&rhok) < 1e-8);

	lcg_float m_square;
	lcg_complex m_mod;
	clcg_inner(m_mod, m, m, n_size);
	m_square = clcg_square(&m_mod);
	if (m_square < 1.0) m_square = 1.0;

	lcg_float rk_square;
	lcg_complex rk_mod;
	clcg_inner(rk_mod, rk, rk, n_size);
	rk_square = clcg_square(&rk_mod);

	int ret, t = 0;
	if (para.abs_diff && sqrt(rk_square)/n_size <= para.epsilon)
	{
		ret = LCG_ALREADY_OPTIMIZIED;
		if (Pfp != nullptr)
		{
			Pfp(instance, m, sqrt(rk_square)/n_size, &para, n_size, 0);
		}
		goto func_ends;
	}	
	else if (rk_square/m_square <= para.epsilon)
	{
		ret = LCG_ALREADY_OPTIMIZIED;
		if (Pfp != nullptr)
		{
			Pfp(instance, m, rk_square/m_square, &para, n_size, 0);
		}
		goto func_ends;
	}

	lcg_float residual;
	while(1)
	{
		if (para.abs_diff) residual = sqrt(rk_square)/n_size;
		else residual = rk_square/m_square;

		if (Pfp != nullptr)
		{
			if (Pfp(instance, m, residual, &para, n_size, t))
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

		Afp(instance, pk, Ax, n_size, MatNormal, NonConjugate); // vk = Apk
		clcg_inner(sigma, rbar0, Ax, n_size);
		ak = rhok/sigma;

#pragma omp parallel for private (i) schedule(guided)
		for (i = 0; i < n_size; i++)
		{
			qk[i] = uk[i] - ak*Ax[i];
			wk[i] = uk[i] + qk[i];
		}

		Afp(instance, wk, Ax, n_size, MatNormal, NonConjugate);

#pragma omp parallel for private (i) schedule(guided)
		for (i = 0; i < n_size; i++)
		{
			m[i] = m[i] + ak*wk[i];
			rk[i] = rk[i] - ak*Ax[i];
		}

		clcg_inner(m_mod, m, m, n_size);
		m_square = clcg_square(&m_mod);
		if (m_square < 1.0) m_square = 1.0;

		clcg_inner(rk_mod, rk, rk, n_size);
		rk_square = clcg_square(&rk_mod);

		for (i = 0; i < n_size; i++)
		{
			if (m[i] != m[i])
			{
				ret = CLCG_NAN_VALUE; goto func_ends;
			}
		}

		clcg_inner(rhok2, rbar0, rk, n_size);
		betak = rhok2/rhok;
		rhok = rhok2;

#pragma omp parallel for private (i) schedule(guided)
		for (i = 0; i < n_size; i++)
		{
			uk[i] = rk[i] + betak*qk[i];
			pk[i] = uk[i] + betak*(qk[i] + betak*pk[i]);
		}
	}

	func_ends:
	{
		clcg_free(rk);
		clcg_free(rbar0);
		clcg_free(pk);
		clcg_free(Ax);
		clcg_free(uk);
		clcg_free(qk);
		clcg_free(wk);
	}

	return ret;
}

int clbicgstab(clcg_axfunc_ptr Afp, clcg_progress_ptr Pfp, lcg_complex* m, const lcg_complex* B, 
	const int n_size, const clcg_para* param, void* instance)
{
	// set BICGSTAB parameters
	clcg_para para = (param != nullptr) ? (*param) : defparam2;

	//check parameters
	if (n_size <= 0) return CLCG_INVILAD_VARIABLE_SIZE;
	if (para.max_iterations < 0) return CLCG_INVILAD_MAX_ITERATIONS;
	if (para.epsilon <= 0.0 || para.epsilon >= 1.0) return CLCG_INVILAD_EPSILON;

	if (m == nullptr) return CLCG_INVALID_POINTER;
	if (B == nullptr) return CLCG_INVALID_POINTER;

	int i;
	lcg_complex *rk = nullptr, *rbar0 = nullptr, *pk = nullptr, *sk = nullptr;
	lcg_complex *Ap = nullptr, *As = nullptr;
	rk = clcg_malloc(n_size); rbar0 = clcg_malloc(n_size);
	pk = clcg_malloc(n_size); sk = clcg_malloc(n_size);
	Ap = clcg_malloc(n_size); As = clcg_malloc(n_size);

	lcg_complex ak, rhok2, sigma, omega, betak, Ass, AsAs;

	Afp(instance, m, Ap, n_size, MatNormal, NonConjugate);

#pragma omp parallel for private (i) schedule(guided)
	for (i = 0; i < n_size; i++)
	{
		pk[i] = rk[i] = B[i] - Ap[i];
	}

	lcg_complex rhok;
	do
	{
		clcg_vecrnd(rbar0, lcg_complex(1.0, 0.0), lcg_complex(2.0, 0.0), n_size);
		clcg_inner(rhok, rbar0, rk, n_size);
	} while (clcg_module(&rhok) < 1e-8);

	lcg_float m_square;
	lcg_complex m_mod;
	clcg_inner(m_mod, m, m, n_size);
	m_square = clcg_square(&m_mod);
	if (m_square < 1.0) m_square = 1.0;

	lcg_float rk_square;
	lcg_complex rk_mod;
	clcg_inner(rk_mod, rk, rk, n_size);
	rk_square = clcg_square(&rk_mod);

	int ret, t = 0;
	if (para.abs_diff && sqrt(rk_square)/n_size <= para.epsilon)
	{
		ret = LCG_ALREADY_OPTIMIZIED;
		if (Pfp != nullptr)
		{
			Pfp(instance, m, sqrt(rk_square)/n_size, &para, n_size, 0);
		}
		goto func_ends;
	}	
	else if (rk_square/m_square <= para.epsilon)
	{
		ret = LCG_ALREADY_OPTIMIZIED;
		if (Pfp != nullptr)
		{
			Pfp(instance, m, rk_square/m_square, &para, n_size, 0);
		}
		goto func_ends;
	}

	lcg_float residual;
	while(1)
	{
		if (para.abs_diff) residual = sqrt(rk_square)/n_size;
		else residual = rk_square/m_square;

		if (Pfp != nullptr)
		{
			if (Pfp(instance, m, residual, &para, n_size, t))
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

		Afp(instance, pk, Ap, n_size, MatNormal, NonConjugate);
		clcg_inner(sigma, rbar0, Ap, n_size);
		ak = rhok/sigma;

#pragma omp parallel for private (i) schedule(guided)
		for (i = 0; i < n_size; i++)
		{
			sk[i] = rk[i] - ak*Ap[i];
		}

		Afp(instance, sk, As, n_size, MatNormal, NonConjugate);
		clcg_inner(Ass, As, sk, n_size);
		clcg_inner(AsAs, As, As, n_size);
		omega = Ass/AsAs;

#pragma omp parallel for private (i) schedule(guided)
		for (i = 0; i < n_size; i++)
		{
			m[i] = m[i] + ak*pk[i] + omega*sk[i];
			rk[i] = sk[i] - omega*As[i];
		}

		clcg_inner(m_mod, m, m, n_size);
		m_square = clcg_square(&m_mod);
		if (m_square < 1.0) m_square = 1.0;

		clcg_inner(rk_mod, rk, rk, n_size);
		rk_square = clcg_square(&rk_mod);

		for (i = 0; i < n_size; i++)
		{
			if (m[i] != m[i])
			{
				ret = CLCG_NAN_VALUE; goto func_ends;
			}
		}

		clcg_inner(rhok2, rbar0, rk, n_size);
		betak = rhok2*ak/(rhok*omega);
		rhok = rhok2;

#pragma omp parallel for private (i) schedule(guided)
		for (i = 0; i < n_size; i++)
		{
			pk[i] = rk[i] + betak*(pk[i] - omega*Ap[i]);
		}
	}

	func_ends:
	{
		clcg_free(rk);
		clcg_free(rbar0);
		clcg_free(pk);
		clcg_free(sk);
		clcg_free(Ap);
		clcg_free(As);
	}

	return ret;
}

int cltfqmr(clcg_axfunc_ptr Afp, clcg_progress_ptr Pfp, lcg_complex* m, const lcg_complex* B, 
	const int n_size, const clcg_para* param, void* instance)
{
	// set CGS parameters
	clcg_para para = (param != nullptr) ? (*param) : defparam2;

	//check parameters
	if (n_size <= 0) return CLCG_INVILAD_VARIABLE_SIZE;
	if (para.max_iterations < 0) return CLCG_INVILAD_MAX_ITERATIONS;
	if (para.epsilon <= 0.0 || para.epsilon >= 1.0) return CLCG_INVILAD_EPSILON;

	if (m == nullptr) return CLCG_INVALID_POINTER;
	if (B == nullptr) return CLCG_INVALID_POINTER;

	int i, j;
	lcg_complex *pk = nullptr, *uk = nullptr;
	lcg_complex *vk = nullptr, *dk = nullptr;
	lcg_complex *rbar0 = nullptr, *rk = nullptr;
	lcg_complex *Ax = nullptr, *qk = nullptr;
	lcg_complex *uqk = nullptr;
	pk = clcg_malloc(n_size); uk = clcg_malloc(n_size);
	vk = clcg_malloc(n_size); dk = clcg_malloc(n_size);
	rbar0 = clcg_malloc(n_size); rk = clcg_malloc(n_size);
	Ax = clcg_malloc(n_size); qk = clcg_malloc(n_size);
	uqk = clcg_malloc(n_size);

	Afp(instance, m, Ax, n_size, MatNormal, NonConjugate);

#pragma omp parallel for private (i) schedule(guided)
	for (i = 0; i < n_size; i++)
	{
		pk[i] = uk[i] = rk[i] = B[i] - Ax[i];
		clcg_set(&dk[i], 0.0, 0.0);
	}

	lcg_complex rho, rk_mod, rk_mod2;
	lcg_float rk_square;
	clcg_inner(rk_mod, rk, rk, n_size);
	rk_square = clcg_square(&rk_mod);

	do
	{
		clcg_vecrnd(rbar0, lcg_complex(1.0, 0.0), lcg_complex(2.0, 0.0), n_size);
		clcg_inner(rho, rbar0, rk, n_size);
	} while (clcg_module(&rho) < 1e-8);

	lcg_float theta = 0.0, omega = clcg_module(&rk_mod);
	lcg_float residual, tao = omega;
	lcg_complex sigma, alpha, betak, rho2, sign, eta(0.0, 0.0);

	lcg_float m_square;
	lcg_complex m_mod;
	clcg_inner(m_mod, m, m, n_size);
	m_square = clcg_square(&m_mod);
	if (m_square < 1.0) m_square = 1.0;

	int ret, t = 0;
	if (para.abs_diff && sqrt(rk_square)/n_size <= para.epsilon)
	{
		ret = LCG_ALREADY_OPTIMIZIED;
		if (Pfp != nullptr)
		{
			Pfp(instance, m, sqrt(rk_square)/n_size, &para, n_size, 0);
		}
		goto func_ends;
	}	
	else if (rk_square/m_square <= para.epsilon)
	{
		ret = LCG_ALREADY_OPTIMIZIED;
		if (Pfp != nullptr)
		{
			Pfp(instance, m, rk_square/m_square, &para, n_size, 0);
		}
		goto func_ends;
	}

	while(1)
	{
		Afp(instance, pk, vk, n_size, MatNormal, NonConjugate);

		clcg_inner(sigma, rbar0, vk, n_size);
		alpha = rho/sigma;

#pragma omp parallel for private (i) schedule(guided)
		for (i = 0; i < n_size; i++)
		{
			qk[i] = uk[i] - alpha*vk[i];
			uqk[i] = uk[i] + qk[i];
		}

		Afp(instance, uqk, Ax, n_size, MatNormal, NonConjugate);

#pragma omp parallel for private (i) schedule(guided)
		for (i = 0; i < n_size; i++)
		{
			rk[i] = rk[i] - alpha*Ax[i];
		}

		clcg_inner(rk_mod2, rk, rk, n_size);

		for (j = 1; j <= 2; j++)
		{

			if (para.abs_diff) residual = sqrt(rk_square)/n_size;
			else residual = rk_square/m_square;

			if (Pfp != nullptr)
			{
				if (Pfp(instance, m, residual, &para, n_size, t))
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
				omega = sqrt(clcg_module(&rk_mod)*clcg_module(&rk_mod2));

#pragma omp parallel for private (i) schedule(guided)
				for (i = 0; i < n_size; i++)
				{
					dk[i] = uk[i] + sign*dk[i];
				}
			}
			else
			{
				omega = clcg_module(&rk_mod2);

#pragma omp parallel for private (i) schedule(guided)
				for (i = 0; i < n_size; i++)
				{
					dk[i] = qk[i] + sign*dk[i];
				}
			}

			theta = omega/tao;
			tao = omega/sqrt(1.0+theta*theta);
			eta = (1.0/(1.0+theta*theta))*alpha;

#pragma omp parallel for private (i) schedule(guided)
			for (i = 0; i < n_size; i++)
			{
				m[i] = m[i] + eta*dk[i];
			}

			clcg_inner(m_mod, m, m, n_size);
			m_square = clcg_square(&m_mod);
			if (m_square < 1.0) m_square = 1.0;

			for (i = 0; i < n_size; i++)
			{
				if (m[i] != m[i])
				{
					ret = CLCG_NAN_VALUE; goto func_ends;
				}
			}
		}
		rk_mod = rk_mod2;
		rk_square = clcg_square(&rk_mod);

		clcg_inner(rho2, rbar0, rk, n_size);
		betak = rho2/rho;
		rho = rho2;

#pragma omp parallel for private (i) schedule(guided)
		for (i = 0; i < n_size; i++)
		{
			uk[i] = rk[i] + betak*qk[i];
			pk[i] = uk[i] + betak*(qk[i] + betak*pk[i]);
		}
	}

	func_ends:
	{
		clcg_free(pk);
		clcg_free(uk);
		clcg_free(vk);
		clcg_free(dk);
		clcg_free(rbar0);
		clcg_free(rk);
		clcg_free(Ax);
		clcg_free(qk);
		clcg_free(uqk);
	}

	return ret;
}