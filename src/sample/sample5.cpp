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

// 计算核矩阵乘向量的乘积
void CalAx(void* instance, const Eigen::VectorXd &x, Eigen::VectorXd &prod_Ax)
{
	tmp_arr = kernel * x;
	prod_Ax = kernel.transpose() * tmp_arr;
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
	Eigen::VectorXd fm = Eigen::VectorXd::Random(N);

	// 计算共轭梯度B项
	Eigen::VectorXd B(N);
	tmp_arr = kernel * fm;
	B = kernel.transpose() * tmp_arr;

	/********************准备工作完成************************/
	lcg_para self_para = lcg_default_parameters();
	self_para.max_iterations = 1000;
	self_para.epsilon = 1e-3;
	self_para.abs_diff = 1;

	// 声明一组解
	Eigen::VectorXd m = Eigen::VectorXd::Zero(N);
	Eigen::VectorXd p = Eigen::VectorXd::Constant(N, 1.0);

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
	ret = lcg_solver_eigen(CalAx, Prog, m, B, &self_para, NULL, LCG_PCG, &p);
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

	return 0;
}