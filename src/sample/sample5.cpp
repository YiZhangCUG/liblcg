#include "../lib/algebra.h"
#include "iostream"

int main(int argc, char const *argv[])
{
	lcg_complex a(2, 4);
	lcg_complex b(-3, 5);

	lcg_complex c;
	c = a + b;
	std::cout << c << std::endl;
	c = a - b;
	std::cout << c << std::endl;
	c = a * b;
	std::cout << c << std::endl;
	c = a / b;
	std::cout << c << std::endl;
	c = a.conjugate();
	std::cout << c << std::endl;
	lcg_float a_rel = a.module();
	std::cout << a_rel << std::endl;
	std::cout << "===================" << std::endl;

	lcg_complex d[3];
	lcg_complex e[3];
	for (int i = 0; i < 3; i++)
	{
		d[i].set(1.1*i-5.1, -0.6*i+3.32);
		std::cout << d[i] << std::endl;
	}
	std::cout << "===================" << std::endl;

	for (int i = 0; i < 3; i++)
	{
		e[i].set(-0.9*i+4.65, 0.7*i-4.35);
		std::cout << e[i] << std::endl;
	}
	std::cout << "===================" << std::endl;

	lcg_complex f = complex_inner(d,e,3);
	std::cout << f << std::endl;
	lcg_complex g = complex_inner(e,d,3);
	std::cout << g << std::endl;

	return 0;
}