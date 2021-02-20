#include "../lib/clcg.h"
#include "ctime"
#include "random"
#include "iostream"
#include "iomanip"

#define M 100
#define N 60

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
// 中间结果数组
clcg_complex *tmp_arr;

// 计算核矩阵乘向量的乘积
void CalAx(void *instance, const clcg_complex *x, clcg_complex *prod_Ax, 
	const int x_size, matrix_layout_e layout, complex_conjugate_e conjugate)
{
	matrix_product(kernel, x, tmp_arr, M, x_size, Normal, conjugate);
	matrix_product(kernel, tmp_arr, prod_Ax, M, x_size, Transpose, conjugate);
	return;
}


//定义共轭梯度监控函数
int Prog(void* instance, const clcg_complex* m, const lcg_float converge, 
	const clcg_para* param, const int n_size, const int k)
{
#if defined(__linux__) || defined(__APPLE__)
	std::clog << "Iteration-times: " << k << "\tconvergence: " << converge << std::endl;
	if (converge > param->epsilon) std::clog << "\033[1A\033[K";
#elif defined (__WIN32__)
	if (converge > param->epsilon)
		std::clog << "\rIteration-times: " << k << "\tconvergence: " << converge;
	else std::clog << "\rIteration-times: " << k << "\tconvergence: " << converge << std::endl;
#endif
	return 0;
}

int main(int argc, char const *argv[])
{
	srand(time(0));

	kernel = new clcg_complex *[M];
	for (int i = 0; i < M; i++)
	{
		kernel[i] = new clcg_complex [N];
	}
	tmp_arr = new clcg_complex [M];

	for (int i = 0; i < M; i++)
	{
		for (int j = 0; j < N; j++)
		{
			kernel[i][j].rel = random_lcg_float(-1.0, 1.0);
			kernel[i][j].img = random_lcg_float(-1.0, 1.0);
		}
	}

	// 添加一些大数
	int tmp_id, tmp_size;
	for (int i = 0; i < M; i++)
	{
		tmp_size = random_int(25, 35);
		for (int j = 0; j < tmp_size; j++)
		{
			tmp_id = random_int(0, N);
			kernel[i][tmp_id].rel = random_lcg_float(-10, 10);
			kernel[i][tmp_id].img = random_lcg_float(-10, 10);
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
	matrix_product(kernel, fm, tmp_arr, M, N, Normal);
	matrix_product(kernel, tmp_arr, B, M, N, Transpose);

	/********************准备工作完成************************/
	clcg_para self_para = clcg_default_parameters();
	self_para.max_iterations = 1000;
	self_para.epsilon = 1e-8;
	self_para.abs_diff = 0;

	// 声明一组解
	clcg_complex *m = new clcg_complex [N];
	for (int i = 0; i < N; i++)
	{
		m[i].rel = 0.0;
		m[i].img = 0.0;
	}

	int ret = clcg_solver(CalAx, Prog, m, B, N, &self_para, NULL, CLCG_BICG);
	std::cerr << clcg_error_str(ret) << std::endl;

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
	delete[] tmp_arr;
	delete[] fm;
	delete[] B;
	delete[] m;
	return 0;
}