#include "config.h"
#include "lcg_eigen.h"
#include "cmath"

#ifdef LCG_OPENMP
#include "omp.h"
#endif

/**
 * @brief      Callback interface of the conjugate gradient solver
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
typedef int (*eigen_solver_ptr)(eigen_axfunc_ptr Afp, eigen_progress_ptr Pfp, Eigen::VectorXd &m, const Eigen::VectorXd &B, 
	const lcg_para* param, void* instance);

int lcg(eigen_axfunc_ptr Afp, eigen_progress_ptr Pfp, Eigen::VectorXd &m, const Eigen::VectorXd &B, 
	const lcg_para* param, void* instance);

int eigen_solver(eigen_axfunc_ptr Afp, eigen_progress_ptr Pfp, Eigen::VectorXd &m, const Eigen::VectorXd &B, 
	const lcg_para* param, void* instance, lcg_solver_enum solver_id)
{
	eigen_solver_ptr cg_solver;
	switch (solver_id)
	{
		case LCG_CG:
			cg_solver = lcg;
			break;
		default:
			cg_solver = lcg;
			break;
	}

	return cg_solver(Afp, Pfp, m, B, param, instance);
}


/**
 * @brief      Conjugate gradient method
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
int lcg(eigen_axfunc_ptr Afp, eigen_progress_ptr Pfp, Eigen::VectorXd &m, const Eigen::VectorXd &B, 
	const lcg_para* param, void* instance)
{
	// set CG parameters
	lcg_para para = (param != nullptr) ? (*param) : defparam;

	int n_size = B.size();
	//check parameters
	if (n_size <= 0) return LCG_INVILAD_VARIABLE_SIZE;
	if (para.max_iterations <= 0) return LCG_INVILAD_MAX_ITERATIONS;
	if (para.epsilon <= 0.0) return LCG_INVILAD_EPSILON;

	// locate memory
	Eigen::VectorXd gk(n_size), dk(n_size), Adk(n_size);

	Afp(instance, m, Adk);

	gk = Adk - B;
	dk = -1.0*gk;

	lcg_float B_mod = B.dot(B);
	lcg_float gk_mod = gk.dot(gk);

	int time, ret;
	lcg_float dTAd, ak, betak, gk1_mod, residual;
	for (time = 0; time < para.max_iterations; time++)
	{
		if (para.abs_diff) residual = sqrt(gk_mod)/n_size;
		else residual = gk_mod/B_mod;

		if (Pfp != nullptr)
		{
			if (Pfp(instance, &m, residual, &para, time))
			{
				ret = LCG_STOP; goto func_ends;
			}
		}

		if (residual <= para.epsilon)
		{
			ret = LCG_CONVERGENCE; goto func_ends;
		}

		Afp(instance , dk, Adk);

		dTAd = dk.dot(Adk);
		ak = gk_mod/dTAd;

		m += ak*dk;
		gk += ak*Adk;

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

	if (time == para.max_iterations)
		return LCG_REACHED_MAX_ITERATIONS;
	else if (ret == LCG_CONVERGENCE)
		return LCG_SUCCESS;
	else return ret;
}
