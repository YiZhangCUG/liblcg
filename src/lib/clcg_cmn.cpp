#include "config.h"
#include "clcg_cmn.h"

#ifdef __WIN32__
#include "windows.h"
#endif

clcg_para clcg_default_parameters()
{
	clcg_para param = defparam;
	return param;
}

void clcg_error_str(int er_index, bool er_throw)
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
		case CLCG_SUCCESS:
			err_str = "Iteration reached convergence."; break;
		case CLCG_STOP:
			err_str = "Iteration is stopped by the progress evaluation function."; break;
		case CLCG_ALREADY_OPTIMIZIED:
			err_str = "Variables are already optimized."; break;
		case CLCG_UNKNOWN_ERROR:
			err_str = "Unknown error."; break;
		case CLCG_INVILAD_VARIABLE_SIZE:
			err_str = "Size of the variables is negative."; break;
		case CLCG_INVILAD_MAX_ITERATIONS:
			err_str = "The maximal iteration times is negative."; break;
		case CLCG_INVILAD_EPSILON:
			err_str = "The epsilon is negative."; break;
		case CLCG_REACHED_MAX_ITERATIONS:
			err_str = "The maximal iteration has been reached."; break;
		case CLCG_NAN_VALUE:
			err_str = "The model values are NaN."; break;
		case CLCG_INVALID_POINTER:
			err_str = "Invalid pointer."; break;
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