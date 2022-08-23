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

#include "../lib/lcg_eigen.h"
#include "iostream"
#include "Eigen/Dense"

#define M 1000
#define N 800

lcg_float max_diff(const Eigen::VectorXd &a, const Eigen::VectorXd &b)
{
	lcg_float max = -1;
	for (int i = 0; i < a.size(); i++)
	{
		max = lcg_max(sqrt((a[i] - b[i])*(a[i] - b[i])), max);
	}
	return max;
}

// 普通二维数组做核矩阵
Eigen::MatrixXd kernel = Eigen::MatrixXd::Random(M, N);
// 中间结果数组
Eigen::VectorXd tmp_arr(M);
Eigen::VectorXd p = Eigen::VectorXd::Constant(N, 1.0);

// 计算核矩阵乘向量的乘积
void CalAx(void* instance, const Eigen::VectorXd &x, Eigen::VectorXd &prod_Ax)
{
	tmp_arr = kernel * x;
	prod_Ax = kernel.transpose() * tmp_arr;
	return;
}

void CalMx(void* instance, const Eigen::VectorXd &x, Eigen::VectorXd &prod_Mx)
{
	prod_Mx = p.cwiseProduct(x);
	return;
}

//定义共轭梯度监控函数
int Prog(void* instance, const Eigen::VectorXd *m, const lcg_float converge, 
	const lcg_para *param, const int k)
{
	std::clog << "\rIteration-times: " << k << "\tconvergence: " << converge;
	return 0;
}

int main(int argc, char const *argv[])
{
	// 生成一组正演解
	lcg_float LO = 1.0, HI = 2.0, Range = HI - LO;
	Eigen::VectorXd fm = Eigen::VectorXd::Random(N);
	fm = (fm + Eigen::VectorXd::Constant(N, 1.0))*0.5*Range;
	fm = (fm + Eigen::VectorXd::Constant(N, LO));

	// 计算共轭梯度B项
	Eigen::VectorXd B(N);
	tmp_arr = kernel * fm;
	B = kernel.transpose() * tmp_arr;

	/********************准备工作完成************************/
	lcg_para self_para = lcg_default_parameters();
	self_para.epsilon = 1e-5;
	self_para.abs_diff = 0;

	// 声明一组解
	Eigen::VectorXd m = Eigen::VectorXd::Zero(N);
	//Eigen::VectorXd p = Eigen::VectorXd::Constant(N, 1.0);
	Eigen::VectorXd low = Eigen::VectorXd::Constant(N, LO);
	Eigen::VectorXd hig = Eigen::VectorXd::Constant(N, HI);

	std::clog << "solver: cg" << std::endl;
	clock_t start = clock();
	int ret = lcg_solver_eigen(CalAx, Prog, m, B, &self_para, NULL, LCG_CG);
	clock_t end = clock();
	std::clog << std::endl; lcg_error_str(ret);
	std::clog << "maximal difference: " << max_diff(fm, m) << std::endl;
	std::clog << "time use: "<<1000*(end-start)/(double)CLOCKS_PER_SEC<<" ms" << std::endl;

	m.setZero();
	std::clog << "solver: pcg" << std::endl;
	start = clock();
	ret = lcg_solver_preconditioned_eigen(CalAx, CalMx, Prog, m, B, &self_para, NULL, LCG_PCG);
	end = clock();
	std::clog << std::endl; lcg_error_str(ret);
	std::clog << "maximal difference: " << max_diff(fm, m) << std::endl;
	std::clog << "time use: "<<1000*(end-start)/(double)CLOCKS_PER_SEC<<" ms" << std::endl;

	m.setZero();
	std::clog << "solver: cgs" << std::endl;
	start = clock();
	ret = lcg_solver_eigen(CalAx, Prog, m, B, &self_para, NULL, LCG_CGS);
	end = clock();
	std::clog << std::endl; lcg_error_str(ret);
	std::clog << "maximal difference: " << max_diff(fm, m) << std::endl;
	std::clog << "time use: "<<1000*(end-start)/(double)CLOCKS_PER_SEC<<" ms" << std::endl;

	m.setZero();
	std::clog << "solver: bicgstab" << std::endl;
	start = clock();
	ret = lcg_solver_eigen(CalAx, Prog, m, B, &self_para, NULL, LCG_BICGSTAB);
	end = clock();
	std::clog << std::endl; lcg_error_str(ret);
	std::clog << "maximal difference: " << max_diff(fm, m) << std::endl;
	std::clog << "time use: "<<1000*(end-start)/(double)CLOCKS_PER_SEC<<" ms" << std::endl;

	m.setZero();
	std::clog << "solver: bicgstab2" << std::endl;
	start = clock();
	ret = lcg_solver_eigen(CalAx, Prog, m, B, &self_para, NULL, LCG_BICGSTAB2);
	end = clock();
	std::clog << std::endl; lcg_error_str(ret);
	std::clog << "maximal difference: " << max_diff(fm, m) << std::endl;
	std::clog << "time use: "<<1000*(end-start)/(double)CLOCKS_PER_SEC<<" ms" << std::endl;

	m.setZero();
	std::clog << "solver: pg" << std::endl;
	start = clock();
	ret = lcg_solver_constrained_eigen(CalAx, Prog, m, B, low, hig, &self_para, NULL, LCG_PG);
	end = clock();
	std::clog << std::endl; lcg_error_str(ret);
	std::clog << "maximal difference: " << max_diff(fm, m) << std::endl;
	std::clog << "time use: "<<1000*(end-start)/(double)CLOCKS_PER_SEC<<" ms" << std::endl;

	m.setZero();
	std::clog << "solver: spg" << std::endl;
	start = clock();
	ret = lcg_solver_constrained_eigen(CalAx, Prog, m, B, low, hig, &self_para, NULL, LCG_SPG);
	end = clock();
	std::clog << std::endl; lcg_error_str(ret);
	std::clog << "maximal difference: " << max_diff(fm, m) << std::endl;
	std::clog << "time use: "<<1000*(end-start)/(double)CLOCKS_PER_SEC<<" ms" << std::endl;

	return 0;
}