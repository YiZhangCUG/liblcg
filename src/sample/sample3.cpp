#include "../lib/clcg.h"
#include "ctime"
#include "random"
#include "iostream"
#include "iomanip"

#define N 100

//返回范围内的随机浮点值 注意调取函数之前要调用srand(time(0));
lcg_float random_lcg_float(lcg_float L,lcg_float T)
{
	return (T-L)*rand()*1.0/RAND_MAX + L;
}

//返回范围内的随机整数 注意调取函数之前要调用srand(time(0));
int random_int(int small, int big)
{
	return (rand() % (big - small))+ small;
}

// 普通二维数组做核矩阵
clcg_complex **kernel;

// 计算核矩阵乘向量的乘积
void CalAx(void *instance, const clcg_complex *x, clcg_complex *prod_Ax, 
	const int x_size, matrix_layout_e layout, complex_conjugate_e conjugate)
{
	matrix_product(kernel, x, prod_Ax, N, x_size, layout, conjugate);
	return;
}


//定义共轭梯度监控函数
int Prog(void* instance, const clcg_complex* m, const lcg_float converge, 
	const clcg_para* param, const int n_size, const int k)
{
	std::clog << "\rIteration-times: " << k << "\tconvergence: " << converge;
	return 0;
}

int main(int argc, char const *argv[])
{
	srand(time(0));

	kernel = new clcg_complex *[N];
	for (int i = 0; i < N; i++)
	{
		kernel[i] = new clcg_complex [N];
	}

	for (int i = 0; i < N; i++)
	{
		for (int j = i; j < N; j++)
		{
			kernel[i][j].rel = random_lcg_float(-1.0, 1.0);
			kernel[i][j].img = random_lcg_float(-1.0, 1.0);
			kernel[j][i] = kernel[i][j];
		}
	}

	// 添加一些大数
	int tmp_id, tmp_size;
	for (int i = 0; i < N; i++)
	{
		tmp_size = random_int(20, 35);
		for (int j = 0; j < tmp_size; j++)
		{
			tmp_id = random_int(0, N);
			kernel[i][tmp_id].rel = random_lcg_float(-10, 10);
			kernel[i][tmp_id].img = random_lcg_float(-10, 10);
			kernel[tmp_id][i] = kernel[i][tmp_id];
		}
	}

	// 生成一组正演解
	clcg_complex *fm = new clcg_complex [N];
	for (int i = 0; i < N; i++)
	{
		fm[i].rel = random_lcg_float(1, 2);
		fm[i].img = random_lcg_float(1, 2);
	}

	// 计算共轭梯度B项
	clcg_complex *B = new clcg_complex [N];
	matrix_product(kernel, fm, B, N, N, Normal, NonConjugate);

	/********************准备工作完成************************/
	clcg_para self_para = clcg_default_parameters();
	self_para.max_iterations = 1000;
	self_para.epsilon = 1e-3;
	self_para.abs_diff = 1;

	// 声明一组解
	clcg_complex *m = new clcg_complex [N];
	for (int i = 0; i < N; i++)
	{
		m[i].rel = 0.0;
		m[i].img = 0.0;
	}

	int ret = clcg_solver(CalAx, Prog, m, B, N, &self_para, NULL, CLCG_CGS);
	std::cerr << std::endl << clcg_error_str(ret) << std::endl;

	for (int i = 0; i < N; i++)
	{
		if (fm[i].img >= 0)
		{
			std::cout << std::setw(8) << fm[i].rel << "+" << fm[i].img << "i\t";
		}
		else
		{
			std::cout << std::setw(8) << fm[i].rel << fm[i].img << "i\t";
		}

		if (m[i].img >= 0)
		{
			std::cout << std::setw(8) << m[i].rel << "+" << m[i].img << "i" << std::endl;
		}
		else
		{
			std::cout << std::setw(8) << m[i].rel << m[i].img << "i" << std::endl;
		}
	}

	delete[] kernel;
	delete[] fm;
	delete[] B;
	delete[] m;
	return 0;
}