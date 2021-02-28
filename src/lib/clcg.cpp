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

	lcg_complex B_mod, r1r2;
	lcg_inner(B_mod, B, B, n_size);
	lcg_inner(r1r2, r2k, r1k, n_size);

	int time, ret;
	lcg_float residual;
	lcg_complex ak, Ad1d2, r1r2_next, betak, rk_mod;
	for (time = 0; time < para.max_iterations; time++)
	{
		lcg_inner(rk_mod, r1k, r1k, n_size);
		if (para.abs_diff) residual = rk_mod.rel;
		else residual = rk_mod.rel/B_mod.rel;

		if (Pfp != nullptr)
		{
			if (Pfp(instance, m, residual, &para, n_size, time))
			{
				ret = CLCG_STOP; goto func_ends;
			}
		}

		if (residual <= para.epsilon)
		{
			ret = CLCG_CONVERGENCE; goto func_ends;
		}

		Afp(instance, d1k, Ax, n_size, Normal, NonConjugate);
		lcg_inner(Ad1d2, d2k, Ax, n_size);
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

		lcg_inner(r1r2_next, r2k, r1k, n_size);
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

	lcg_complex B_mod, rkrk;
	lcg_inner(B_mod, B, B, n_size);
	lcg_dot(rkrk, rk, rk, n_size);

	int time, ret;
	lcg_float residual;
	lcg_complex ak, rkrk2, betak, rk_mod, dkAx;
	for (time = 0; time < para.max_iterations; time++)
	{
		lcg_inner(rk_mod, rk, rk, n_size);
		if (para.abs_diff) residual = rk_mod.rel;
		else residual = rk_mod.rel/B_mod.rel;

		if (Pfp != nullptr)
		{
			if (Pfp(instance, m, residual, &para, n_size, time))
			{
				ret = CLCG_STOP; goto func_ends;
			}
		}

		if (residual <= para.epsilon)
		{
			ret = CLCG_CONVERGENCE; goto func_ends;
		}

		Afp(instance, dk, Ax, n_size, Normal, NonConjugate);
		lcg_dot(dkAx, dk, Ax, n_size);
		ak = rkrk/dkAx;

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

		lcg_dot(rkrk2, rk, rk, n_size);
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

	lcg_complex B_mod, r0Hrk;
	lcg_inner(B_mod, B, B, n_size);
	lcg_inner(r0Hrk, r0, rk, n_size);

	int time, ret;
	lcg_float residual;
	lcg_complex ak, r0Hrk1, r0HAp, betak, rk_mod;
	for (time = 0; time < para.max_iterations; time++)
	{
		lcg_inner(rk_mod, rk, rk, n_size);
		if (para.abs_diff) residual = rk_mod.rel;
		else residual = rk_mod.rel/B_mod.rel;

		if (Pfp != nullptr)
		{
			if (Pfp(instance, m, residual, &para, n_size, time))
			{
				ret = CLCG_STOP; goto func_ends;
			}
		}

		if (residual <= para.epsilon)
		{
			ret = CLCG_CONVERGENCE; goto func_ends;
		}

		Afp(instance, qk, Ax, n_size, Normal, NonConjugate);
		lcg_inner(r0HAp, r0, Ax, n_size);
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

		lcg_inner(r0Hrk1, r0, rk, n_size);
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

	int i, j;
	lcg_complex *pk = nullptr, *uk = nullptr;
	lcg_complex *vk = nullptr, *dk = nullptr;
	lcg_complex *r0 = nullptr, *rk = nullptr;
	lcg_complex *Ax = nullptr, *qk = nullptr;
	lcg_complex *uqk = nullptr;
	pk = lcg_malloc_complex(n_size); uk = lcg_malloc_complex(n_size);
	vk = lcg_malloc_complex(n_size); dk = lcg_malloc_complex(n_size);
	r0 = lcg_malloc_complex(n_size); rk = lcg_malloc_complex(n_size);
	Ax = lcg_malloc_complex(n_size); qk = lcg_malloc_complex(n_size);
	uqk = lcg_malloc_complex(n_size);

	Afp(instance, m, Ax, n_size, Normal, NonConjugate);

#pragma omp parallel for private (i) schedule(guided)
	for (i = 0; i < n_size; i++)
	{
		pk[i] = uk[i] = r0[i] = rk[i] = B[i] - Ax[i];
		dk[i].set(0.0, 0.0);
	}

	lcg_complex B_mod;
	lcg_inner(B_mod, B, B, n_size);

	lcg_complex rk_mod, rk_mod2;
	lcg_inner(rk_mod, rk, rk, n_size);

	lcg_float theta = 0.0, omega = sqrt(rk_mod.rel);
	lcg_float residual, tao = omega;
	lcg_complex sigma, alpha, betak, rho, rho2, sign, eta(0.0, 0.0);
	lcg_inner(rho, r0, r0, n_size);

	int time, ret;
	for (time = 0; time < para.max_iterations; time++)
	{
		if (para.abs_diff) residual = (time+1)*tao*tao;
		else residual = (time+1)*tao*tao/B_mod.rel;

		if (Pfp != nullptr)
		{
			if (Pfp(instance, m, residual, &para, n_size, time))
			{
				ret = CLCG_STOP; goto func_ends;
			}
		}

		if (residual <= para.epsilon)
		{
			ret = CLCG_CONVERGENCE; goto func_ends;
		}

		Afp(instance, pk, vk, n_size, Normal, NonConjugate);

		lcg_inner(sigma, r0, vk, n_size);
		alpha = rho/sigma;

#pragma omp parallel for private (i) schedule(guided)
		for (i = 0; i < n_size; i++)
		{
			qk[i] = uk[i] - alpha*vk[i];
			uqk[i] = uk[i] + qk[i];
		}

		Afp(instance, uqk, Ax, n_size, Normal, NonConjugate);

#pragma omp parallel for private (i) schedule(guided)
		for (i = 0; i < n_size; i++)
		{
			rk[i] = rk[i] - alpha*Ax[i];
		}

		lcg_inner(rk_mod2, rk, rk, n_size);

		for (j = 1; j <= 2; j++)
		{
			sign = theta*theta*(eta/alpha);

			if (j == 1)
			{
				omega = sqrt(sqrt(rk_mod.rel)*sqrt(rk_mod2.rel));

#pragma omp parallel for private (i) schedule(guided)
				for (i = 0; i < n_size; i++)
				{
					dk[i] = uk[i] + sign*dk[i];
				}
			}
			else
			{
				omega = sqrt(rk_mod2.rel);

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

			for (i = 0; i < n_size; i++)
			{
				if (m[i] != m[i])
				{
					ret = CLCG_NAN_VALUE; goto func_ends;
				}
			}
		}
		rk_mod = rk_mod2;

		lcg_inner(rho2, r0, rk, n_size);
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
		lcg_free(pk);
		lcg_free(uk);
		lcg_free(vk);
		lcg_free(dk);
		lcg_free(r0);
		lcg_free(rk);
		lcg_free(Ax);
		lcg_free(qk);
		lcg_free(uqk);
	}

	if (time == para.max_iterations)
		return CLCG_REACHED_MAX_ITERATIONS;
	else if (ret == CLCG_CONVERGENCE)
		return CLCG_SUCCESS;
	else return ret;
}