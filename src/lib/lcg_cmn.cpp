#include "config.h"
#include "lcg_cmn.h"
#include "cmath"
#include "ctime"

#ifdef LCG_OPENMP
#include "omp.h"
#endif

#ifdef __WIN32__
#include "windows.h"
#endif

lcg_para lcg_default_parameters()
{
	lcg_para param = defparam;
	return param;
}

void lcg_error_str(int er_index, bool er_throw)
{
#if defined(__linux__) || defined(__APPLE__)
	if (!er_throw)
	{
		if (er_index >= 0)
			std::cerr << "\033[1m\033[32mSuccess! ";
		else
			std::cerr << "\033[1m\033[31mFail! ";
	}
#elif defined(__WIN32__)
	if (!er_throw)
	{
		if (er_index >= 0)
		{
			SetConsoleTextAttribute(GetStdHandle(STD_ERROR_HANDLE), FOREGROUND_INTENSITY | FOREGROUND_GREEN);
			std::cerr << "Success! ";
		}
		else
		{
			SetConsoleTextAttribute(GetStdHandle(STD_ERROR_HANDLE), FOREGROUND_INTENSITY | FOREGROUND_RED);
			std::cerr << "Fail! ";
		}	
	}
#endif

	std::string err_str;
	switch (er_index)
	{
		case LCG_SUCCESS:
			err_str = "Iteration reached convergence."; break;
		case LCG_STOP:
			err_str = "Iteration is stopped by the progress evaluation function."; break;
		case LCG_ALREADY_OPTIMIZIED:
			err_str = "Variables are already optimized."; break;
		case LCG_UNKNOWN_ERROR:
			err_str = "Unknown error."; break;
		case LCG_INVILAD_VARIABLE_SIZE:
			err_str = "Size of the variables is negative."; break;
		case LCG_INVILAD_MAX_ITERATIONS:
			err_str = "The maximal iteration times can't be negative."; break;
		case LCG_INVILAD_EPSILON:
			err_str = "The convergence threshold can't be negative."; break;
		case LCG_INVILAD_RESTART_EPSILON:
			err_str = "The restart threshold can't be negative."; break;
		case LCG_REACHED_MAX_ITERATIONS:
			err_str = "The maximal iteration has been reached."; break;
		case LCG_NULL_PRECONDITION_MATRIX:
			err_str = "The precondition matrix can't be null."; break;
		case LCG_NAN_VALUE:
			err_str = "The model values are NaN."; break;
		case LCG_INVALID_POINTER:
			err_str = "Invalid pointer."; break;
		case LCG_INVALID_LAMBDA:
			err_str = "Invalid value for lambda."; break;
		case LCG_INVALID_SIGMA:
			err_str = "Invalid value for sigma."; break;
		case LCG_INVALID_BETA:
			err_str = "Invalid value for beta."; break;
		case LCG_INVALID_MAXIM:
			err_str = "Invalid value for maxi_m."; break;
		case LCG_SIZE_NOT_MATCH:
			err_str = "The sizes of solution and target do not match."; break;
		default:
			err_str = "Unknown error."; break;
	}

	if (er_throw && er_index < 0) throw err_str;
	else std::cerr << err_str;

#if defined(__linux__) || defined(__APPLE__)
	if (!er_throw)
	{
		if (er_index >= 0)
			std::cerr << "\033[0m" << std::endl;
		else
			std::cerr << "\033[0m" << std::endl;	
	}
#elif defined(__WIN32__)
	if (!er_throw)
	{
		if (er_index >= 0)
		{
			SetConsoleTextAttribute(GetStdHandle(STD_ERROR_HANDLE), 7);
			std::cerr << std::endl;
		}
		else
		{
			SetConsoleTextAttribute(GetStdHandle(STD_ERROR_HANDLE), 7);
			std::cerr << std::endl;
		}	
	}
#endif

	return;
}
