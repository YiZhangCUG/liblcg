#include "../lib/clcg.h"
#include "iostream"

void display(const lcg_complex &a)
{
	if (a.img >= 0)
		std::cout << a.rel << "+" << a.img << "i" << std::endl;
	else
		std::cout << a.rel << a.img << "i" << std::endl;
	return;
}

int main(int argc, char const *argv[])
{
	lcg_complex a = complex(2, 4);
	lcg_complex b = complex(-3, 5);

	lcg_complex c;
	c = a + b; display(c);
	c = a - b; display(c);
	c = a * b; display(c);
	c = a / b; display(c);
	c = complex_conjugate(a); display(c);
	lcg_float a_rel = complex_module(a);
	std::cout << a_rel << std::endl;
	std::cout << "===================" << std::endl;

	lcg_complex *d = clcg_malloc(3);
	lcg_complex *e = clcg_malloc(3);
	for (int i = 0; i < 3; i++)
	{
		d[i] = complex(1.1*i-5.1, -0.6*i+3.32);
		display(d[i]);
	}
	std::cout << "===================" << std::endl;

	for (int i = 0; i < 3; i++)
	{
		e[i] = complex(-0.9*i+4.65, 0.7*i-4.35);
		display(e[i]);
	}
	std::cout << "===================" << std::endl;

	lcg_complex f = inner_product(d,e,3); display(f);
	lcg_complex g = inner_product(e,d,3); display(g);

	clcg_free(d);
	clcg_free(e);
	return 0;
}