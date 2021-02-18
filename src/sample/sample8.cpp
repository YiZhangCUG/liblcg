#include "../lib/clcg_cxx.h"
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

class TESTFUNC : public CLCG_Solver
{
public:
	TESTFUNC();
	~TESTFUNC();

	// 计算共轭梯度的B项
	void cal_partb(clcg_complex *B, const clcg_complex *x);

	//定义共轭梯度中Ax的算法
	void AxProduct(const clcg_complex *x, clcg_complex *prod_Ax, const int x_size, 
		matrix_layout_e layout, complex_conjugate_e conjugate)
	{
		matrix_product(kernel, x, tmp_arr, M, x_size, Normal, conjugate);
		matrix_product(kernel, tmp_arr, prod_Ax, M, x_size, Transpose, conjugate);
		return;
	}

private:
	// 普通二维数组做核矩阵
	clcg_complex **kernel;
	// 中间结果数组
	clcg_complex *tmp_arr;
};

TESTFUNC::TESTFUNC()
{
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
}

TESTFUNC::~TESTFUNC()
{
	for (int i = 0; i < M; i++)
	{
		delete[] kernel[i];
	}
	delete[] kernel;
	delete[] tmp_arr;
}

void TESTFUNC::cal_partb(clcg_complex *B, const clcg_complex *x)
{
	matrix_product(kernel, x, tmp_arr, M, N, Normal);
	matrix_product(kernel, tmp_arr, B, M, N, Transpose);
	return;
}

int main(int argc, char const *argv[])
{
	srand(time(0));

	// 声明一组解
	clcg_complex *fm = new clcg_complex [N];
	for (int i = 0; i < N; i++)
	{
		fm[i].rel = random_lcg_float(1, 2);
		fm[i].img = random_lcg_float(1, 2);
	}

	TESTFUNC test;

	// 计算共轭梯度B项
	clcg_complex *B = new clcg_complex [N];
	test.cal_partb(B, fm);

	/********************准备工作完成************************/
	clcg_para self_para = clcg_default_parameters();
	self_para.max_iterations = 1000;
	self_para.epsilon = 1e-6;
	self_para.abs_diff = 0;
	test.set_clcg_parameter(self_para);

	// 声明一组解
	clcg_complex *m = new clcg_complex [N];
	for (int i = 0; i < N; i++)
	{
		m[i].rel = 0.0;
		m[i].img = 0.0;
	}

	test.Minimize(m, B, N, CLCG_TFQMR);

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

	delete[] fm;
	delete[] B;
	delete[] m;
	return 0;
}