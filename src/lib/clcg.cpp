#include "config.h"
#include "clcg.h"
#include "cmath"

#ifdef LCG_OPENMP
#include "omp.h"
#endif

/**
 * @brief      return value of the clcg_solver() function
 */
enum clcg_return_enum
{
	CLCG_SUCCESS = 0, ///< The solver function terminated successfully.
	CLCG_CONVERGENCE = 0, ///< The iteration reached convergence.
	CLCG_STOP, ///< The iteration is stopped by the monitoring function.
	CLCG_ALREADY_OPTIMIZIED, ///< The initial solution is already optimized.
	// A negative number means a error
	CLCG_UNKNOWN_ERROR = -1024, ///< Unknown error.
	CLCG_INVILAD_VARIABLE_SIZE, ///< The variable size is negative
	CLCG_INVILAD_MAX_ITERATIONS, ///< The maximal iteration times is negative.
	CLCG_INVILAD_EPSILON, ///< The epsilon is negative.
	CLCG_REACHED_MAX_ITERATIONS, ///< Iteration reached maximal limit.
	CLCG_NAN_VALUE, ///< Nan value.
	CLCG_INVALID_POINTER, ///< Invalid pointer.
};

/**
 * Default parameter for conjugate gradient methods
 */
static const clcg_para defparam = {100, 1e-6, 0};

clcg_para clcg_default_parameters()
{
	clcg_para param = defparam;
	return param;
}

const char* clcg_error_str(int er_index)
{
#if defined(__linux__) || defined(__APPLE__)
	switch (er_index)
	{
		case CLCG_SUCCESS:
			return "\033[1m\033[32mSuccess\033[0m Iteration reached convergence.";
		case CLCG_STOP:
			return "\033[1m\033[32mSuccess\033[0m Iteration is stopped by the progress evaluation function.";
		case CLCG_ALREADY_OPTIMIZIED:
			return "\033[1m\033[32mSuccess\033[0m Variables are already optimized.";
		case CLCG_UNKNOWN_ERROR:
			return "\033[1m\033[31mFail\033[0m Unknown error.";
		case CLCG_INVILAD_VARIABLE_SIZE:
			return "\033[1m\033[31mFail\033[0m Size of the variables is negative.";
		case CLCG_INVILAD_MAX_ITERATIONS:
			return "\033[1m\033[31mFail\033[0m The maximal iteration times is negative.";
		case CLCG_INVILAD_EPSILON:
			return "\033[1m\033[31mFail\033[0m The epsilon is negative.";
		case CLCG_REACHED_MAX_ITERATIONS:
			return "\033[1m\033[31mFail\033[0m The maximal iteration has been reached.";
		case CLCG_NAN_VALUE:
			return "\033[1m\033[31mFail\033[0m The model values are NaN.";
		case CLCG_INVALID_POINTER:
			return "\033[1m\033[31mFail\033[0m Invalid pointer.";
		default:
			return "\033[1m\033[31mFail\033[0m Unknown error.";
	}
#else
	switch (er_index)
	{
		case CLCG_SUCCESS:
			return "Iteration reached convergence.";
		case CLCG_STOP:
			return "Iteration is stopped by the progress evaluation function.";
		case CLCG_ALREADY_OPTIMIZIED:
			return "Variables are already optimized.";
		case CLCG_UNKNOWN_ERROR:
			return "Unknown error.";
		case CLCG_INVILAD_VARIABLE_SIZE:
			return "Size of the variables is negative.";
		case CLCG_INVILAD_MAX_ITERATIONS:
			return "The maximal iteration times is negative.";
		case CLCG_INVILAD_EPSILON:
			return "The epsilon is negative.";
		case CLCG_REACHED_MAX_ITERATIONS:
			return "The maximal iteration has been reached.";
		case CLCG_NAN_VALUE:
			return "The model values are NaN.";
		case CLCG_INVALID_POINTER:
			return "Invalid pointer.";
		default:
			return "Unknown error.";
	}
#endif
}

typedef int (*clcg_solver_ptr)(clcg_axfunc_ptr Afp, clcg_progress_ptr Pfp, lcg_complex* m, 
	const lcg_complex* B, const int n_size, const clcg_para* param, void* instance);

int clbicg(clcg_axfunc_ptr Afp, clcg_progress_ptr Pfp, lcg_complex* m, const lcg_complex* B, 
	const int n_size, const clcg_para* param, void* instance);

int clbicg_symmetric(clcg_axfunc_ptr Afp, clcg_progress_ptr Pfp, lcg_complex* m, const lcg_complex* B, 
	const int n_size, const clcg_para* param, void* instance);

int clcgs(clcg_axfunc_ptr Afp, clcg_progress_ptr Pfp, lcg_complex* m, const lcg_complex* B, 
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
	clcg_para para = (param != nullptr) ? (*param) : defparam;

	//check parameters
	if (n_size <= 0) return CLCG_INVILAD_VARIABLE_SIZE;
	if (para.max_iterations <= 0) return CLCG_INVILAD_MAX_ITERATIONS;
	if (para.epsilon <= 0.0) return CLCG_INVILAD_EPSILON;

	if (m == nullptr) return CLCG_INVALID_POINTER;
	if (B == nullptr) return CLCG_INVALID_POINTER;

	int i;
	lcg_complex *r1k = nullptr, *r2k = nullptr, *d1k = nullptr, *d2k = nullptr;
	lcg_complex *Ax = nullptr;
	r1k = lcg_malloc_complex(n_size); r2k = lcg_malloc_complex(n_size);
	d1k = lcg_malloc_complex(n_size); d2k = lcg_malloc_complex(n_size);
	Ax  = lcg_malloc_complex(n_size);

	Afp(instance, m, Ax, n_size, Normal, NonConjugate);

#pragma omp parallel for private (i) schedule(guided)
	for (i = 0; i < n_size; i++)
	{
		d1k[i] = r1k[i] = B[i] - Ax[i];
		d2k[i] = r2k[i] = r1k[i].conjugate();
	}

	lcg_float B_mod = lcg_inner_complex(B, B, n_size).rel;
	lcg_complex r1r2 = lcg_inner_complex(r2k, r1k, n_size);

	int time, ret;
	lcg_complex ak, Ad1d2, r1r2_next, betak;
	lcg_float rk_mod;
	for (time = 0; time < para.max_iterations; time++)
	{
		rk_mod = lcg_inner_complex(r1k, r1k, n_size).rel;

		if (para.abs_diff)
		{
			if (Pfp != nullptr)
			{
				if (Pfp(instance, m, rk_mod, &para, n_size, time))
				{
					ret = CLCG_STOP; goto func_ends;
				}
			}
			if (rk_mod <= para.epsilon)
			{
				ret = CLCG_CONVERGENCE; goto func_ends;
			}
		}
		else
		{
			if (Pfp != nullptr)
			{
				if (Pfp(instance, m, rk_mod/B_mod, &para, n_size, time))
				{
					ret = CLCG_STOP; goto func_ends;
				}
			}
			if (rk_mod/B_mod <= para.epsilon)
			{
				ret = CLCG_CONVERGENCE; goto func_ends;
			}
		}

		Afp(instance, d1k, Ax, n_size, Normal, NonConjugate);
		Ad1d2 = lcg_inner_complex(d2k, Ax, n_size);
		ak = r1r2/Ad1d2;

#pragma omp parallel for private (i) schedule(guided)
		for (i = 0; i < n_size; i++)
		{
			m[i] = m[i] + ak*d1k[i];
			r1k[i] = r1k[i] - ak*Ax[i];
		}

		Afp(instance, d2k, Ax, n_size, Transpose, Conjugate);

#pragma omp parallel for private (i) schedule(guided)
		for (i = 0; i < n_size; i++)
		{
			r2k[i] = r2k[i] - ak.conjugate()*Ax[i];
		}

		for (i = 0; i < n_size; i++)
		{
			if (m[i] != m[i])
			{
				ret = CLCG_NAN_VALUE; goto func_ends;
			}
		}

		r1r2_next = lcg_inner_complex(r2k, r1k, n_size);
		betak = r1r2_next/r1r2;
		r1r2 = r1r2_next;

#pragma omp parallel for private (i) schedule(guided)
		for (i = 0; i < n_size; i++)
		{
			d1k[i] = r1k[i] + betak*d1k[i];
			d2k[i] = r2k[i] + betak.conjugate()*d2k[i];
		}
	}

	func_ends:
	{
		lcg_free(r1k);
		lcg_free(r2k);
		lcg_free(d1k);
		lcg_free(d2k);
		lcg_free(Ax);
	}

	if (time == para.max_iterations)
		return CLCG_REACHED_MAX_ITERATIONS;
	else if (ret == CLCG_CONVERGENCE)
		return CLCG_SUCCESS;
	else return ret;
}

int clbicg_symmetric(clcg_axfunc_ptr Afp, clcg_progress_ptr Pfp, lcg_complex* m, const lcg_complex* B, 
	const int n_size, const clcg_para* param, void* instance)
{
	// set CGS parameters
	clcg_para para = (param != nullptr) ? (*param) : defparam;

	//check parameters
	if (n_size <= 0) return CLCG_INVILAD_VARIABLE_SIZE;
	if (para.max_iterations <= 0) return CLCG_INVILAD_MAX_ITERATIONS;
	if (para.epsilon <= 0.0) return CLCG_INVILAD_EPSILON;

	if (m == nullptr) return CLCG_INVALID_POINTER;
	if (B == nullptr) return CLCG_INVALID_POINTER;

	int i;
	lcg_complex *rk = nullptr, *dk = nullptr;
	lcg_complex *Ax = nullptr;
	rk = lcg_malloc_complex(n_size); dk = lcg_malloc_complex(n_size);
	Ax = lcg_malloc_complex(n_size);

	Afp(instance, m, Ax, n_size, Normal, NonConjugate);

#pragma omp parallel for private (i) schedule(guided)
	for (i = 0; i < n_size; i++)
	{
		dk[i] = rk[i] = B[i] - Ax[i];
	}

	lcg_float B_mod = lcg_inner_complex(B, B, n_size).rel;
	lcg_complex rkrk = lcg_dot_complex(rk, rk, n_size);

	int time, ret;
	lcg_float rk_mod;
	lcg_complex ak, rkrk2, betak;
	for (time = 0; time < para.max_iterations; time++)
	{
		rk_mod = lcg_inner_complex(rk, rk, n_size).rel;
		if (para.abs_diff)
		{
			if (Pfp != nullptr)
			{
				if (Pfp(instance, m, rk_mod, &para, n_size, time))
				{
					ret = CLCG_STOP; goto func_ends;
				}
			}
			if (rk_mod <= para.epsilon)
			{
				ret = CLCG_CONVERGENCE; goto func_ends;
			}
		}
		else
		{
			if (Pfp != nullptr)
			{
				if (Pfp(instance, m, rk_mod/B_mod, &para, n_size, time))
				{
					ret = CLCG_STOP; goto func_ends;
				}
			}
			if (rk_mod/B_mod <= para.epsilon)
			{
				ret = CLCG_CONVERGENCE; goto func_ends;
			}
		}

		Afp(instance, dk, Ax, n_size, Normal, NonConjugate);
		ak = rkrk/lcg_dot_complex(dk, Ax, n_size);

#pragma omp parallel for private (i) schedule(guided)
		for (i = 0; i < n_size; i++)
		{
			m[i] = m[i] + ak*dk[i];
			rk[i] = rk[i] - ak*Ax[i];
		}

		for (i = 0; i < n_size; i++)
		{
			if (m[i] != m[i])
			{
				ret = CLCG_NAN_VALUE; goto func_ends;
			}
		}

		rkrk2 = lcg_dot_complex(rk, rk, n_size);
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
		lcg_free(rk);
		lcg_free(dk);
		lcg_free(Ax);
	}

	if (time == para.max_iterations)
		return CLCG_REACHED_MAX_ITERATIONS;
	else if (ret == CLCG_CONVERGENCE)
		return CLCG_SUCCESS;
	else return ret;
}

int clcgs(clcg_axfunc_ptr Afp, clcg_progress_ptr Pfp, lcg_complex* m, const lcg_complex* B, 
	const int n_size, const clcg_para* param, void* instance)
{
	// set CGS parameters
	clcg_para para = (param != nullptr) ? (*param) : defparam;

	//check parameters
	if (n_size <= 0) return CLCG_INVILAD_VARIABLE_SIZE;
	if (para.max_iterations <= 0) return CLCG_INVILAD_MAX_ITERATIONS;
	if (para.epsilon <= 0.0) return CLCG_INVILAD_EPSILON;

	if (m == nullptr) return CLCG_INVALID_POINTER;
	if (B == nullptr) return CLCG_INVALID_POINTER;

	int i;
	lcg_complex *rk = nullptr, *r0 = nullptr, *pk = nullptr;
	lcg_complex *Ax = nullptr, *uk = nullptr, *qk = nullptr, *wk = nullptr;
	rk = lcg_malloc_complex(n_size); r0 = lcg_malloc_complex(n_size);
	pk = lcg_malloc_complex(n_size); Ax  = lcg_malloc_complex(n_size);
	uk = lcg_malloc_complex(n_size); qk  = lcg_malloc_complex(n_size);
	wk = lcg_malloc_complex(n_size);

	Afp(instance, m, Ax, n_size, Normal, NonConjugate);

#pragma omp parallel for private (i) schedule(guided)
	for (i = 0; i < n_size; i++)
	{
		pk[i] = qk[i] = r0[i] = rk[i] = B[i] - Ax[i];
	}

	lcg_float B_mod = lcg_inner_complex(B, B, n_size).rel;
	lcg_complex r0Hrk = lcg_inner_complex(r0, rk, n_size);

	int time, ret;
	lcg_float rk_mod;
	lcg_complex ak, r0Hrk1, r0HAp, betak;
	for (time = 0; time < para.max_iterations; time++)
	{
		rk_mod = lcg_inner_complex(rk, rk, n_size).rel;

		if (para.abs_diff)
		{
			if (Pfp != nullptr)
			{
				if (Pfp(instance, m, rk_mod, &para, n_size, time))
				{
					ret = CLCG_STOP; goto func_ends;
				}
			}
			if (rk_mod <= para.epsilon)
			{
				ret = CLCG_CONVERGENCE; goto func_ends;
			}
		}
		else
		{
			if (Pfp != nullptr)
			{
				if (Pfp(instance, m, rk_mod/B_mod, &para, n_size, time))
				{
					ret = CLCG_STOP; goto func_ends;
				}
			}
			if (rk_mod/B_mod <= para.epsilon)
			{
				ret = CLCG_CONVERGENCE; goto func_ends;
			}
		}

		Afp(instance, qk, Ax, n_size, Normal, NonConjugate);
		r0HAp = lcg_inner_complex(r0, Ax, n_size);
		ak = r0Hrk/r0HAp;

#pragma omp parallel for private (i) schedule(guided)
		for (i = 0; i < n_size; i++)
		{
			uk[i] = pk[i] - ak*Ax[i];
			wk[i] = uk[i] + pk[i];
		}

		Afp(instance, wk, Ax, n_size, Normal, NonConjugate);

#pragma omp parallel for private (i) schedule(guided)
		for (i = 0; i < n_size; i++)
		{
			m[i] = m[i] + ak*wk[i];
			rk[i] = rk[i] - ak*Ax[i];
		}

		for (i = 0; i < n_size; i++)
		{
			if (m[i] != m[i])
			{
				ret = CLCG_NAN_VALUE; goto func_ends;
			}
		}

		r0Hrk1 = lcg_inner_complex(r0, rk, n_size);
		betak = r0Hrk1/r0Hrk;
		r0Hrk = r0Hrk1;

#pragma omp parallel for private (i) schedule(guided)
		for (i = 0; i < n_size; i++)
		{
			pk[i] = rk[i] + betak*uk[i];
			qk[i] = pk[i] + betak*(uk[i] + betak*qk[i]);
		}
	}

	func_ends:
	{
		lcg_free(rk);
		lcg_free(r0);
		lcg_free(pk);
		lcg_free(Ax);
		lcg_free(uk);
		lcg_free(qk);
		lcg_free(wk);
	}

	if (time == para.max_iterations)
		return CLCG_REACHED_MAX_ITERATIONS;
	else if (ret == CLCG_CONVERGENCE)
		return CLCG_SUCCESS;
	else return ret;
}

int cltfqmr(clcg_axfunc_ptr Afp, clcg_progress_ptr Pfp, lcg_complex* m, const lcg_complex* B, 
	const int n_size, const clcg_para* param, void* instance)
{
	// set CGS parameters
	clcg_para para = (param != nullptr) ? (*param) : defparam;

	//check parameters
	if (n_size <= 0) return CLCG_INVILAD_VARIABLE_SIZE;
	if (para.max_iterations <= 0) return CLCG_INVILAD_MAX_ITERATIONS;
	if (para.epsilon <= 0.0) return CLCG_INVILAD_EPSILON;

	if (m == nullptr) return CLCG_INVALID_POINTER;
	if (B == nullptr) return CLCG_INVALID_POINTER;

	int i;
	lcg_complex *yk = nullptr, *vk = nullptr, *v2k = nullptr;
	lcg_complex *r0 = nullptr, *wk = nullptr, *dk = nullptr;
	yk = lcg_malloc_complex(n_size); vk = lcg_malloc_complex(n_size); v2k = lcg_malloc_complex(n_size);
	r0 = lcg_malloc_complex(n_size); wk = lcg_malloc_complex(n_size); dk = lcg_malloc_complex(n_size);

#pragma omp parallel for private (i) schedule(guided)
	for (i = 0; i < n_size; i++)
	{
		dk[i].set(0.0, 0.0);
	}

	Afp(instance, m, v2k, n_size, Normal, NonConjugate);
#pragma omp parallel for private (i) schedule(guided)
	for (i = 0; i < n_size; i++)
	{
		wk[i] = yk[i] = r0[i] = B[i] - v2k[i];
		v2k[i].set(0.0, 0.0);
	}

	lcg_complex rho = lcg_inner_complex(r0, r0, n_size);
	lcg_float tao = sqrt(rho.rel);
	lcg_float B_mod = lcg_inner_complex(B, B, n_size).rel;

	int time, ret;
	lcg_float ck, w_mod = 0.0;
	lcg_complex betak, rho2, alphak, eta(0.0, 0.0);
	for (time = 0; time < para.max_iterations; time++)
	{
		if (para.abs_diff)
		{
			if (Pfp != nullptr)
			{
				if (Pfp(instance, m, tao*tao, &para, n_size, time))
				{
					ret = CLCG_STOP; goto func_ends;
				}
			}
			if (tao*tao <= para.epsilon)
			{
				ret = CLCG_CONVERGENCE; goto func_ends;
			}
		}
		else
		{
			if (Pfp != nullptr)
			{
				if (Pfp(instance, m, tao*tao/B_mod, &para, n_size, time))
				{
					ret = CLCG_STOP; goto func_ends;
				}
			}
			if (tao*tao/B_mod <= para.epsilon)
			{
				ret = CLCG_CONVERGENCE; goto func_ends;
			}
		}

		Afp(instance, yk, vk, n_size, Normal, NonConjugate);
		for (i = 0; i < n_size; i++)
		{
			vk[i] = vk[i] + v2k[i];
		}

		alphak = rho/lcg_inner_complex(r0, vk, n_size);

#pragma omp parallel for private (i) schedule(guided)
		for (i = 0; i < n_size; i++)
		{
			wk[i] = wk[i] - alphak*vk[i];
			dk[i] = yk[i] + (w_mod*w_mod)*(eta/alphak)*dk[i];
		}

		w_mod = sqrt(lcg_inner_complex(wk, wk, n_size).rel)/tao;
		ck = 1.0/sqrt(1.0+w_mod*w_mod);
		tao = tao*w_mod*ck;
		eta = ck*ck*alphak;

#pragma omp parallel for private (i) schedule(guided)
		for (i = 0; i < n_size; i++)
		{
			m[i] = m[i] + eta*dk[i];
		}

		for (i = 0; i < n_size; i++)
		{
			if (m[i] != m[i])
			{
				ret = CLCG_NAN_VALUE; goto func_ends;
			}
		}

#pragma omp parallel for private (i) schedule(guided)
		for (i = 0; i < n_size; i++)
		{
			yk[i] = yk[i] - alphak*vk[i];
		}

		Afp(instance, yk, v2k, n_size, Normal, NonConjugate);

		#pragma omp parallel for private (i) schedule(guided)
		for (i = 0; i < n_size; i++)
		{
			wk[i] = wk[i] - alphak*v2k[i];
			dk[i] = yk[i] + (w_mod*w_mod)*(eta/alphak)*dk[i];
		}

		w_mod = sqrt(lcg_inner_complex(wk, wk, n_size).rel)/tao;
		ck = 1.0/sqrt(1.0+w_mod*w_mod);
		tao = tao*w_mod*ck;
		eta = ck*ck*alphak;

#pragma omp parallel for private (i) schedule(guided)
		for (i = 0; i < n_size; i++)
		{
			m[i] = m[i] + eta*dk[i];
		}

		for (i = 0; i < n_size; i++)
		{
			if (m[i] != m[i])
			{
				ret = CLCG_NAN_VALUE; goto func_ends;
			}
		}

		rho2 = lcg_inner_complex(r0, wk, n_size);
		betak = rho2/rho;
		rho = rho2;

#pragma omp parallel for private (i) schedule(guided)
		for (i = 0; i < n_size; i++)
		{
			yk[i] = wk[i] + betak*yk[i];
			v2k[i] = betak*(v2k[i] + betak*vk[i]);
		}
	}

	func_ends:
	{
		lcg_free(yk);
		lcg_free(vk);
		lcg_free(v2k);
		lcg_free(dk);
		lcg_free(r0);
		lcg_free(wk);
	}

	if (time == para.max_iterations)
		return CLCG_REACHED_MAX_ITERATIONS;
	else if (ret == CLCG_CONVERGENCE)
		return CLCG_SUCCESS;
	else return ret;
}