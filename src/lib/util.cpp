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

#include "iostream"
#include "exception"
#include "stdexcept"

#include "util.h"

#if defined _WINDOWS || __WIN32__
#include "windows.h"
#endif

lcg_para lcg_default_parameters()
{
	lcg_para param = defparam;
	return param;
}

lcg_solver_enum lcg_select_solver(std::string slr_char)
{
	lcg_solver_enum slr_id;
	if (slr_char == "LCG_CG") slr_id = LCG_CG;
	else if (slr_char == "LCG_PCG") slr_id = LCG_PCG;
	else if (slr_char == "LCG_CGS") slr_id = LCG_CGS;
	else if (slr_char == "LCG_BICGSTAB") slr_id = LCG_BICGSTAB;
	else if (slr_char == "LCG_BICGSTAB2") slr_id = LCG_BICGSTAB2;
	else if (slr_char == "LCG_PG") slr_id = LCG_PG;
	else if (slr_char == "LCG_SPG") slr_id = LCG_SPG;
	else throw std::invalid_argument("Invalid solver type.");
	return slr_id;
}

void lcg_error_str(int er_index, bool er_throw)
{
#if defined _WINDOWS || __WIN32__
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
#else
	if (!er_throw)
	{
		if (er_index >= 0)
			std::cerr << "\033[1m\033[32mSuccess! ";
		else
			std::cerr << "\033[1m\033[31mFail! ";
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
			err_str = "The variables are already optimized."; break;
		case LCG_UNKNOWN_ERROR:
			err_str = "Unknown error."; break;
		case LCG_INVILAD_VARIABLE_SIZE:
			err_str = "The size of the variables is negative."; break;
		case LCG_INVILAD_MAX_ITERATIONS:
			err_str = "The maximal iteration times can't be negative."; break;
		case LCG_INVILAD_EPSILON:
			err_str = "The epsilon is not in the range (0, 1)."; break;
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

	if (er_throw && er_index < 0) throw  std::runtime_error(err_str.c_str());
	else std::cerr << err_str;

#if defined _WINDOWS || __WIN32__
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
#else
	if (!er_throw)
	{
		if (er_index >= 0)
			std::cerr << "\033[0m" << std::endl;
		else
			std::cerr << "\033[0m" << std::endl;	
	}
#endif

	return;
}


clcg_para clcg_default_parameters()
{
	clcg_para param = defparam2;
	return param;
}

clcg_solver_enum clcg_select_solver(std::string slr_char)
{
	clcg_solver_enum slr_id;
	if (slr_char == "CLCG_BICG") slr_id = CLCG_BICG;
	else if (slr_char == "CLCG_BICG_SYM") slr_id = CLCG_BICG_SYM;
	else if (slr_char == "CLCG_CGS") slr_id = CLCG_CGS;
	else if (slr_char == "CLCG_TFQMR") slr_id = CLCG_TFQMR;
	else throw std::invalid_argument("Invalid solver type.");
	return slr_id;
}

void clcg_error_str(int er_index, bool er_throw)
{
#if defined _WINDOWS || __WIN32__
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
#else
	if (!er_throw)
	{
		if (er_index >= 0)
			std::cerr << "\033[1m\033[32mSuccess! ";
		else
			std::cerr << "\033[1m\033[31mFail! ";
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
			err_str = "The variables are already optimized."; break;
		case CLCG_UNKNOWN_ERROR:
			err_str = "Unknown error."; break;
		case CLCG_INVILAD_VARIABLE_SIZE:
			err_str = "The size of the variables is negative."; break;
		case CLCG_INVILAD_MAX_ITERATIONS:
			err_str = "The maximal iteration times is negative."; break;
		case CLCG_INVILAD_EPSILON:
			err_str = "The epsilon is not in the range (0, 1)."; break;
		case CLCG_REACHED_MAX_ITERATIONS:
			err_str = "The maximal iteration has been reached."; break;
		case CLCG_NAN_VALUE:
			err_str = "The model values are NaN."; break;
		case CLCG_INVALID_POINTER:
			err_str = "Invalid pointer."; break;
		case CLCG_SIZE_NOT_MATCH:
			err_str = "The sizes of the solution and target do not match."; break;
		case CLCG_UNKNOWN_SOLVER:
			err_str = "Unknown solver."; break;
		default:
			err_str = "Unknown error."; break;
	}

	if (er_throw && er_index < 0) throw std::runtime_error(err_str.c_str());
	else std::cerr << err_str;

#if defined _WINDOWS || __WIN32__
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
#else
	if (!er_throw)
	{
		if (er_index >= 0)
			std::cerr << "\033[0m" << std::endl;
		else
			std::cerr << "\033[0m" << std::endl;	
	}
#endif

	return;
}