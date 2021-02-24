#include "../lib/clcg.h"
#include "random"
#include "iostream"
#include "iomanip"

#define N 5

lcg_float random_lcg_float(lcg_float L,lcg_float T)
{
	return (T-L)*rand()*1.0/RAND_MAX + L;
}

clcg_complex **kernel;

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
		for (int j = 0; j < N; j++)
		{
			kernel[i][j].rel = random_lcg_float(-1.0, 1.0);
			kernel[i][j].img = random_lcg_float(-1.0, 1.0);
			if (kernel[i][j].img >= 0)
			{
				std::cout << std::setprecision(3) << kernel[i][j].rel << "+" << kernel[i][j].img << "i\t";
			}
			else
			{
				std::cout << std::setprecision(3) << kernel[i][j].rel << kernel[i][j].img << "i\t";
			}
		}
		std::cout << std::endl;
	}

	std::cout << std::endl;

	clcg_complex a;
	for (int i = 0; i < N; i++)
	{
		a.rel = a.img = 0.0;
		for (int j = 0; j < N; j++)
		{
			a.rel += (kernel[j][i].rel*kernel[i][j].rel + kernel[j][i].img*kernel[i][j].img);
			a.img += -1.0*(kernel[j][i].rel*kernel[i][j].img - kernel[j][i].img*kernel[i][j].rel);
			if (a.img >= 0)
			{
				std::cout << std::setprecision(3) << a.rel << "+" << a.img << "i\t";
			}
			else
			{
				std::cout << std::setprecision(3) << a.rel << a.img << "i\t";
			}
		}
		std::cout << std::endl;
	}

	delete[] kernel;
	return 0;
}