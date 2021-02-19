#include "config.h"
#include "clcg.h"
#include "cmath"
#include "iostream"

#ifdef LCG_OPENMP
#include "omp.h"
#endif

bool operator==(const clcg_complex &a, const clcg_complex &b)
{
	if (a.rel == b.rel && a.img == b.img)
	{
		return true;
	}

	return false;
}

bool operator!=(const clcg_complex &a, const clcg_complex &b)
{
	if (a.rel != b.rel || a.img != b.img)
	{
		return true;
	}

	return false;
}

clcg_complex operator+(const clcg_complex &a, const clcg_complex &b)
{
	clcg_complex ret;
	ret.rel = a.rel + b.rel;
	ret.img = a.img + b.img;
	return ret;
}

clcg_complex operator-(const clcg_complex &a, const clcg_complex &b)
{
	clcg_complex ret;
	ret.rel = a.rel - b.rel;
	ret.img = a.img - b.img;
	return ret;
}

clcg_complex operator*(const clcg_complex &a, const clcg_complex &b)
{
	clcg_complex ret;
	ret.rel = a.rel*b.rel - a.img*b.img;
	ret.img = a.rel*b.img + a.img*b.rel;
	return ret;
}

clcg_complex operator/(const clcg_complex &a, const clcg_complex &b)
{
	clcg_complex ret;
	if (b.rel == 0 && b.img == 0)
	{
		ret.rel = ret.img = NAN;
		return ret;
	}

	ret.rel = (a.rel*b.rel + a.img*b.img)/(b.rel*b.rel + b.img*b.img);
	ret.img = (a.img*b.rel - a.rel*b.img)/(b.rel*b.rel + b.img*b.img);
	return ret;
}

clcg_complex conjugate(const clcg_complex &a)
{
	clcg_complex ret;
	ret.rel = a.rel;
	ret.img = -1.0*a.img;
	return ret;
}

clcg_complex real_product(const lcg_float &a, const clcg_complex &b)
{
	clcg_complex ret;
	ret.rel = a*b.rel;
	ret.img = a*b.img;
	return ret;
}

clcg_complex inner_product(const clcg_complex *a, const clcg_complex *b, int x_size)
{
	clcg_complex ret;
	ret.rel = 0.0; ret.img = 0.0;
	for (int i = 0; i < x_size; i++)
	{
		// <a,b> = a* \cdot b
		ret.rel += (a[i].rel*b[i].rel + a[i].img*b[i].img);
		ret.img += (a[i].rel*b[i].img - a[i].img*b[i].rel);
	}
	return ret;
}

void matrix_product(clcg_complex **A, const clcg_complex *x, clcg_complex *Ax, 
	int m_size, int n_size, matrix_layout_e layout, complex_conjugate_e conjugate)
{
	int i, j;
	if (conjugate == Conjugate)
	{
		if (layout == Normal)
		{
#pragma omp parallel for private (i, j) schedule(guided)
			for (i = 0; i < m_size; i++)
			{
				Ax[i].rel = 0.0;
				Ax[i].img = 0.0;
				for (j = 0; j < n_size; j++)
				{
					// <a,b> = a* \cdot b
					Ax[i].rel += (A[i][j].rel*x[j].rel - A[i][j].img*x[j].img);
					Ax[i].img += (A[i][j].img*x[j].rel + A[i][j].rel*x[j].img);
				}
			}
			return;
		}

#pragma omp parallel for private (i, j) schedule(guided)
		for (j = 0; j < n_size; j++)
		{
			Ax[j].rel = 0.0;
			Ax[j].img = 0.0;
			for (i = 0; i < m_size; i++)
			{
				Ax[j].rel += (A[i][j].rel*x[i].rel - A[i][j].img*x[i].img);
				Ax[j].img += (A[i][j].img*x[i].rel + A[i][j].rel*x[i].img);
			}
		}
		return;
	}

	if (layout == Normal)
	{
#pragma omp parallel for private (i, j) schedule(guided)
		for (i = 0; i < m_size; i++)
		{
			Ax[i].rel = 0.0;
			Ax[i].img = 0.0;
			for (j = 0; j < n_size; j++)
			{
				Ax[i].rel += (A[i][j].rel*x[j].rel + A[i][j].img*x[j].img);
				Ax[i].img += (A[i][j].img*x[j].rel - A[i][j].rel*x[j].img);
			}
		}
		return;
	}

#pragma omp parallel for private (i, j) schedule(guided)
	for (j = 0; j < n_size; j++)
	{
		Ax[j].rel = 0.0;
		Ax[j].img = 0.0;
		for (i = 0; i < m_size; i++)
		{
			Ax[j].rel += (A[i][j].rel*x[i].rel + A[i][j].img*x[i].img);
			Ax[j].img += (A[i][j].img*x[i].rel - A[i][j].rel*x[i].img);
		}
	}
	return;
}


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

clcg_complex* clcg_malloc(const int n)
{
	clcg_complex* x = new clcg_complex [n];
	return x;
}

void clcg_free(clcg_complex* x)
{
	if (x != nullptr) delete[] x;
	x = nullptr;
	return;
}

clcg_para clcg_default_parameters()
{
	clcg_para param = defparam;
	return param;
}

const char* clcg_error_str(int er_index)
{
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
}

typedef int (*clcg_solver_ptr)(clcg_axfunc_ptr Afp, clcg_progress_ptr Pfp, clcg_complex* m, 
	const clcg_complex* B, const int n_size, const clcg_para* param, void* instance);

int clbicg(clcg_axfunc_ptr Afp, clcg_progress_ptr Pfp, clcg_complex* m, const clcg_complex* B, 
	const int n_size, const clcg_para* param, void* instance);

int clcgs(clcg_axfunc_ptr Afp, clcg_progress_ptr Pfp, clcg_complex* m, const clcg_complex* B, 
	const int n_size, const clcg_para* param, void* instance);

int cltfqmr(clcg_axfunc_ptr Afp, clcg_progress_ptr Pfp, clcg_complex* m, const clcg_complex* B, 
	const int n_size, const clcg_para* param, void* instance);

int clcg_solver(clcg_axfunc_ptr Afp, clcg_progress_ptr Pfp, clcg_complex* m, 
	const clcg_complex* B, const int n_size, const clcg_para* param, void* instance, 
	clcg_solver_enum solver_id)
{
	clcg_solver_ptr cg_solver;
	switch (solver_id)
	{
		case CLCG_BICG:
			cg_solver = clbicg;
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


int clbicg(clcg_axfunc_ptr Afp, clcg_progress_ptr Pfp, clcg_complex* m, const clcg_complex* B, 
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
	clcg_complex *r1k = nullptr, *r2k = nullptr, *d1k = nullptr, *d2k = nullptr;
	clcg_complex *Ax = nullptr;
	r1k = clcg_malloc(n_size); r2k = clcg_malloc(n_size);
	d1k = clcg_malloc(n_size); d2k = clcg_malloc(n_size);
	Ax  = clcg_malloc(n_size);

	Afp(instance, m, Ax, n_size, Normal, NonConjugate);

#pragma omp parallel for private (i) schedule(guided)
	for (i = 0; i < n_size; i++)
	{
		d1k[i] = r1k[i] = B[i] - Ax[i];
		d2k[i] = r2k[i] = conjugate(r1k[i]);
	}

	lcg_float B_mod = 0.0;
	for (i = 0; i < n_size; i++)
	{
		B_mod += (B[i].rel*B[i].rel + B[i].img*B[i].img);
	}

	int time, ret;
	clcg_complex ak, Ad1d2, r1r2, r1r2_next, betak;
	lcg_float rk_mod;
	for (time = 0; time < para.max_iterations; time++)
	{
		// 我们在迭代开始的时候先检查m是否符合终止条件以避免不必要的迭代
		rk_mod = 0.0;
		for (i = 0; i < n_size; i++)
		{
			rk_mod += (r1k[i].rel*r1k[i].rel + r1k[i].img*r1k[i].img);
		}

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

		r1r2 = inner_product(r1k, r2k, n_size);
		Afp(instance, d1k, Ax, n_size, Normal, NonConjugate);
		Ad1d2 = inner_product(Ax, d2k, n_size);
		ak = r1r2/Ad1d2;

#pragma omp parallel for private (i) schedule(guided)
		for (i = 0; i < n_size; i++)
		{
			m[i] = m[i] + ak*d1k[i];
			r1k[i] = r1k[i] - ak*Ax[i];
		}

		Afp(instance, d2k, Ax, n_size, Normal, Conjugate);

#pragma omp parallel for private (i) schedule(guided)
		for (i = 0; i < n_size; i++)
		{
			r2k[i] = r2k[i] - conjugate(ak)*Ax[i];
		}

		for (i = 0; i < n_size; i++)
		{
			if (m[i] != m[i])
			{
				ret = CLCG_NAN_VALUE; goto func_ends;
			}
		}

		betak = real_product(-1.0, inner_product(Ax, r1k, n_size)/Ad1d2);

#pragma omp parallel for private (i) schedule(guided)
		for (i = 0; i < n_size; i++)
		{
			d1k[i] = r1k[i] + betak*d1k[i];
			d2k[i] = r2k[i] + conjugate(betak)*d2k[i];
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

	if (time == para.max_iterations)
		return CLCG_REACHED_MAX_ITERATIONS;
	else if (ret == CLCG_CONVERGENCE)
		return CLCG_SUCCESS;
	else return ret;
}

int clcgs(clcg_axfunc_ptr Afp, clcg_progress_ptr Pfp, clcg_complex* m, const clcg_complex* B, 
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
	clcg_complex *rk = nullptr, *r0_T = nullptr, *pk = nullptr;
	clcg_complex *Ax = nullptr, *uk = nullptr,   *qk = nullptr, *wk = nullptr;
	rk   = clcg_malloc(n_size); r0_T = clcg_malloc(n_size);
	pk   = clcg_malloc(n_size); Ax  = clcg_malloc(n_size);
	uk   = clcg_malloc(n_size); qk   = clcg_malloc(n_size);
	wk  = clcg_malloc(n_size);

	Afp(instance, m, Ax, n_size, Normal, NonConjugate);

#pragma omp parallel for private (i) schedule(guided)
	for (i = 0; i < n_size; i++)
	{
		pk[i] = qk[i] = r0_T[i] = rk[i] = B[i] - Ax[i];
	}

	lcg_float B_mod = 0.0;
	for (i = 0; i < n_size; i++)
	{
		B_mod += (B[i].rel*B[i].rel + B[i].img*B[i].img);
	}

	lcg_float rkr0_T = inner_product(rk, r0_T, n_size).rel;

	int time, ret;
	lcg_float ak, rkr0_T1, Apr_T, betak, rk_mod;
	for (time = 0; time < para.max_iterations; time++)
	{
		// 我们在迭代开始的时候先检查m是否符合终止条件以避免不必要的迭代
		rk_mod = 0.0;
		for (i = 0; i < n_size; i++)
		{
			rk_mod += (rk[i].rel*rk[i].rel + rk[i].img*rk[i].img);
		}

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

		Apr_T = inner_product(Ax, r0_T, n_size).rel;
		ak = rkr0_T/Apr_T;

#pragma omp parallel for private (i) schedule(guided)
		for (i = 0; i < n_size; i++)
		{
			uk[i] = pk[i] - real_product(ak, Ax[i]);
			wk[i] = uk[i] + pk[i];
		}

		Afp(instance, wk, Ax, n_size, Normal, NonConjugate);

#pragma omp parallel for private (i) schedule(guided)
		for (i = 0; i < n_size; i++)
		{
			m[i] = m[i] + real_product(ak, wk[i]);
			rk[i] = rk[i] - real_product(ak, Ax[i]);
		}

		for (i = 0; i < n_size; i++)
		{
			if (m[i] != m[i])
			{
				ret = CLCG_NAN_VALUE; goto func_ends;
			}
		}

		rkr0_T1 = inner_product(rk, r0_T, n_size).rel;
		betak = rkr0_T1/rkr0_T;
		rkr0_T = rkr0_T1;

#pragma omp parallel for private (i) schedule(guided)
		for (i = 0; i < n_size; i++)
		{
			pk[i] = rk[i] + real_product(betak, uk[i]);
			qk[i] = pk[i] + real_product(betak, uk[i] + real_product(betak, qk[i]));
		}
	}

	func_ends:
	{
		clcg_free(rk);
		clcg_free(r0_T);
		clcg_free(pk);
		clcg_free(Ax);
		clcg_free(uk);
		clcg_free(qk);
		clcg_free(wk);
	}

	if (time == para.max_iterations)
		return CLCG_REACHED_MAX_ITERATIONS;
	else if (ret == CLCG_CONVERGENCE)
		return CLCG_SUCCESS;
	else return ret;
}

int cltfqmr(clcg_axfunc_ptr Afp, clcg_progress_ptr Pfp, clcg_complex* m, const clcg_complex* B, 
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
	clcg_complex *yk = nullptr, *y2k = nullptr, *vk = nullptr, *v2k = nullptr;
	clcg_complex *r0 = nullptr, *dk = nullptr, *wk = nullptr;
	yk = clcg_malloc(n_size); y2k = clcg_malloc(n_size);
	vk = clcg_malloc(n_size); v2k = clcg_malloc(n_size);
	r0 = clcg_malloc(n_size); dk  = clcg_malloc(n_size);
	wk = clcg_malloc(n_size);

#pragma omp parallel for private (i) schedule(guided)
	for (i = 0; i < n_size; i++)
	{
		dk[i].rel = 0.0; dk[i].img = 0.0;
	}

	Afp(instance, m, v2k, n_size, Normal, NonConjugate);
#pragma omp parallel for private (i) schedule(guided)
	for (i = 0; i < n_size; i++)
	{
		wk[i] = yk[i] = r0[i] = B[i] - v2k[i];
	}

	Afp(instance, yk, vk, n_size, Normal, NonConjugate);

	//clcg_complex rho = inner_product(r0, r0, n_size);
	lcg_float rho = inner_product(r0, r0, n_size).rel;
	//lcg_float tao = sqrt(rho.rel);
	lcg_float tao = sqrt(rho);

	lcg_float B_mod = 0.0;
	for (i = 0; i < n_size; i++)
	{
		B_mod += (B[i].rel*B[i].rel + B[i].img*B[i].img);
	}

	int time, ret;
	lcg_float ck, w_mod = 0.0;
	//clcg_complex betak, rho2, ak, eta, alpha;
	//eta.rel = 0.0; eta.img = 0.0;
	lcg_float betak, rho2, ak, alpha, eta = 0.0;
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

		//alpha = rho/inner_product(r0, vk, n_size);
		alpha = rho/inner_product(r0, vk, n_size).rel;

#pragma omp parallel for private (i) schedule(guided)
		for (i = 0; i < n_size; i++)
		{
			//y2k[i] = yk[i] - alpha*vk[i];
			y2k[i] = yk[i] - real_product(alpha, vk[i]);
		}

		// iteration with A*yk
		//ak = real_product(w_mod*w_mod, eta/alpha);
		ak = w_mod*w_mod*eta/alpha;
#pragma omp parallel for private (i) schedule(guided)
		for (i = 0; i < n_size; i++)
		{
			//dk[i] = yk[i] + ak*dk[i];
			dk[i] = yk[i] + real_product(ak, dk[i]);
		}

#pragma omp parallel for private (i) schedule(guided)
		for (i = 0; i < n_size; i++)
		{
			//wk[i] = wk[i] - alpha*vk[i];
			wk[i] = wk[i] - real_product(alpha, vk[i]);
		}

		w_mod = 0.0;
		for (i = 0; i < n_size; i++)
		{
			w_mod += (wk[i].rel*wk[i].rel + wk[i].img*wk[i].img);
		}
		w_mod = sqrt(w_mod)/tao;
		ck = 1.0/sqrt(1.0+w_mod*w_mod);
		//eta = real_product(ck*ck, alpha);
		eta = ck*ck*alpha;
		tao = tao*w_mod*ck;

#pragma omp parallel for private (i) schedule(guided)
		for (i = 0; i < n_size; i++)
		{
			//m[i] = m[i] + eta*dk[i];
			m[i] = m[i] + real_product(eta, dk[i]);
		}

		for (i = 0; i < n_size; i++)
		{
			if (m[i] != m[i])
			{
				ret = CLCG_NAN_VALUE; goto func_ends;
			}
		}
		// end iteration with A*yk

		Afp(instance, y2k, v2k, n_size, Normal, NonConjugate);

		// iteration with A*y2k
		ak = w_mod*w_mod*eta/alpha;
#pragma omp parallel for private (i) schedule(guided)
		for (i = 0; i < n_size; i++)
		{
			//dk[i] = yk[i] + ak*dk[i];
			dk[i] = y2k[i] + real_product(ak, dk[i]);
		}

#pragma omp parallel for private (i) schedule(guided)
		for (i = 0; i < n_size; i++)
		{
			//wk[i] = wk[i] - alpha*vk[i];
			wk[i] = wk[i] - real_product(alpha, v2k[i]);
		}

		w_mod = 0.0;
		for (i = 0; i < n_size; i++)
		{
			w_mod += (wk[i].rel*wk[i].rel + wk[i].img*wk[i].img);
		}
		w_mod = sqrt(w_mod)/tao;
		ck = 1.0/sqrt(1.0+w_mod*w_mod);
		//eta = real_product(ck*ck, alpha);
		eta = ck*ck*alpha;
		tao = tao*w_mod*ck;

#pragma omp parallel for private (i) schedule(guided)
		for (i = 0; i < n_size; i++)
		{
			//m[i] = m[i] + eta*dk[i];
			m[i] = m[i] + real_product(eta, dk[i]);
		}

		for (i = 0; i < n_size; i++)
		{
			if (m[i] != m[i])
			{
				ret = CLCG_NAN_VALUE; goto func_ends;
			}
		}
		// end iteration with A*y2k

		//rho2 = inner_product(r0, wk, n_size);
		rho2 = inner_product(r0, wk, n_size).rel;
		betak = rho2/rho;
		rho = rho2;

#pragma omp parallel for private (i) schedule(guided)
		for (i = 0; i < n_size; i++)
		{
			//yk[i] = wk[i] + betak*y2k[i];
			//v2k[i] = betak*(v2k[i] + betak*vk[i]);
			yk[i] = wk[i] + real_product(betak, y2k[i]);
			v2k[i] = real_product(betak, v2k[i] + real_product(betak, vk[i]));
		}

		Afp(instance, yk, vk, n_size, Normal, NonConjugate);
#pragma omp parallel for private (i) schedule(guided)
		for (i = 0; i < n_size; i++)
		{
			vk[i] = vk[i] + v2k[i];
		}
	}

	func_ends:
	{
		clcg_free(yk);
		clcg_free(y2k);
		clcg_free(vk);
		clcg_free(v2k);
		clcg_free(r0);
		clcg_free(dk);
		clcg_free(wk);
	}

	if (time == para.max_iterations)
		return CLCG_REACHED_MAX_ITERATIONS;
	else if (ret == CLCG_CONVERGENCE)
		return CLCG_SUCCESS;
	else return ret;
}