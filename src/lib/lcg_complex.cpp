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

#include "cmath"
#include "ctime"
#include "random"

#include "lcg_complex.h"

#ifdef LibLCG_OPENMP
#include "omp.h"
#endif

lcg_complex* clcg_malloc(int n)
{
	lcg_complex *x = new lcg_complex [n];
	return x;
}

lcg_complex** clcg_malloc(int m, int n)
{
	lcg_complex **x = new lcg_complex* [m];
	for (int i = 0; i < m; i++)
	{
		x[i] = new lcg_complex [n];
	}
	return x;
}

void clcg_free(lcg_complex* x)
{
	if (x != nullptr)
	{
		delete[] x; x = nullptr;
	}
	return;
}

void clcg_free(lcg_complex **x, int m)
{
	if (x != nullptr)
	{
		for (int i = 0; i < m; i++)
		{
			delete[] x[i];
		}
		delete[] x;
		x = nullptr;
	}
	return;
}

void clcg_vecset(lcg_complex *a, lcg_complex b, int size)
{
	for (int i = 0; i < size; i++)
	{
		a[i] = b;
	}
	return;
}

void clcg_vecset(lcg_complex **a, lcg_complex b, int m, int n)
{
	for (int i = 0; i < m; ++i)
	{
		for (int j = 0; j < n; ++j)
		{
			a[i][j] = b;
		}
	}
	return;
}

#ifdef LibLCG_STD_COMPLEX

void clcg_set(lcg_complex *a, lcg_float r, lcg_float i)
{
	a->real(r);
	a->imag(i);
	return;
}

lcg_float clcg_square(const lcg_complex *a)
{
	return std::norm(*a);
}

lcg_float clcg_module(const lcg_complex *a)
{
	return sqrt(std::norm(*a));
}

lcg_complex clcg_conjugate(const lcg_complex *a)
{
	lcg_complex b = std::conj(*a);
	return b;
}

void clcg_vecrnd(lcg_complex *a, lcg_complex l, lcg_complex h, int size)
{
	srand(time(0));
	for (int i = 0; i < size; i++)
	{
		a[i].real((h.real()-l.real())*rand()*1.0/RAND_MAX + l.real());
		a[i].imag((h.imag()-l.imag())*rand()*1.0/RAND_MAX + l.imag());
	}
	return;
}

void clcg_vecrnd(lcg_complex **a, lcg_complex l, lcg_complex h, int m, int n)
{
	srand(time(0));
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			a[i][j].real((h.real()-l.real())*rand()*1.0/RAND_MAX + l.real());
			a[i][j].imag((h.imag()-l.imag())*rand()*1.0/RAND_MAX + l.imag());	
		}
	}
	return;
}

void clcg_dot(lcg_complex &ret, const lcg_complex *a, const lcg_complex *b, int size)
{
	lcg_float re = 0.0, im = 0.0;
	// <a,b> = \sum{a_i \cdot b_i}
	for (int i = 0; i < size; i++)
	{
		re += (a[i].real()*b[i].real() - a[i].imag()*b[i].imag());
		im += (a[i].real()*b[i].imag() + a[i].imag()*b[i].real());
	}
	ret.real(re); ret.imag(im);
	return;
}

void clcg_inner(lcg_complex &ret, const lcg_complex *a, const lcg_complex *b, int size)
{
	lcg_float re = 0.0, im = 0.0;
	// <a,b> = \sum{\bar{a_i} \cdot b_i}
	for (int i = 0; i < size; i++)
	{
		re += (a[i].real()*b[i].real() + a[i].imag()*b[i].imag());
		im += (a[i].real()*b[i].imag() - a[i].imag()*b[i].real());
	}
	ret.real(re); ret.imag(im);
	return;
}

void clcg_matvec(lcg_complex **A, const lcg_complex *x, lcg_complex *Ax, 
	int m_size, int n_size, lcg_matrix_e layout, clcg_complex_e conjugate)
{
	size_t i, j;
	lcg_float re, im;
	if (conjugate == Conjugate)
	{
		if (layout == MatNormal)
		{
#pragma omp parallel for private (i, j, re, im) schedule(guided)
			for (i = 0; i < m_size; i++)
			{
				re = 0.0; im = 0.0;
				for (j = 0; j < n_size; j++)
				{
					re += (A[i][j].real()*x[j].real() + A[i][j].imag()*x[j].imag());
					im += (A[i][j].real()*x[j].imag() - A[i][j].imag()*x[j].real());
				}
				Ax[i].real(re); Ax[i].imag(im);
			}
			return;
		}

#pragma omp parallel for private (i, j, re, im) schedule(guided)
		for (j = 0; j < n_size; j++)
		{
			re = 0.0; im = 0.0;
			for (i = 0; i < m_size; i++)
			{
				re += (A[i][j].real()*x[i].real() + A[i][j].imag()*x[i].imag());
				im += (A[i][j].real()*x[i].imag() - A[i][j].imag()*x[i].real());
			}
			Ax[j].real(re); Ax[j].imag(im);
		}
		return;
	}

	if (layout == MatNormal)
	{
#pragma omp parallel for private (i, j, re, im) schedule(guided)
		for (i = 0; i < m_size; i++)
		{
			re = 0.0; im = 0.0;
			for (j = 0; j < n_size; j++)
			{
				re += (A[i][j].real()*x[j].real() - A[i][j].imag()*x[j].imag());
				im += (A[i][j].real()*x[j].imag() + A[i][j].imag()*x[j].real());
			}
			Ax[i].real(re); Ax[i].imag(im);
		}
		return;
	}

#pragma omp parallel for private (i, j, re, im) schedule(guided)
	for (j = 0; j < n_size; j++)
	{
		re = 0.0; im = 0.0;
		for (i = 0; i < m_size; i++)
		{
			re += (A[i][j].real()*x[i].real() - A[i][j].imag()*x[i].imag());
			im += (A[i][j].real()*x[i].imag() + A[i][j].imag()*x[i].real());
		}
		Ax[j].real(re); Ax[j].imag(im);
	}
	return;
}

#else

lcg_complex::lcg_complex()
{
	rel = img = 0.0;
}

lcg_complex::lcg_complex(lcg_float r, lcg_float i)
{
	rel = r; img = i;
}

lcg_complex::~lcg_complex(){}

void lcg_complex::real(lcg_float a)
{
	rel = a;
	return;
}

void lcg_complex::imag(lcg_float a)
{
	img = a;
	return;
}

lcg_float lcg_complex::real()
{
	return rel;
}

lcg_float lcg_complex::imag()
{
	return img;
}

bool operator==(const lcg_complex &a, const lcg_complex &b)
{
	if (a.rel == b.rel && a.img == b.img)
		return true;
	return false;
}

bool operator!=(const lcg_complex &a, const lcg_complex &b)
{
	if (a.rel != b.rel || a.img != b.img)
		return true;
	return false;
}

lcg_complex operator+(const lcg_complex &a, const lcg_complex &b)
{
	lcg_complex ret;
	ret.rel = a.rel + b.rel;
	ret.img = a.img + b.img;
	return ret;
}

lcg_complex operator-(const lcg_complex &a, const lcg_complex &b)
{
	lcg_complex ret;
	ret.rel = a.rel - b.rel;
	ret.img = a.img - b.img;
	return ret;
}

lcg_complex operator*(const lcg_complex &a, const lcg_complex &b)
{
	lcg_complex ret;
	ret.rel = a.rel*b.rel - a.img*b.img;
	ret.img = a.rel*b.img + a.img*b.rel;
	return ret;
}

lcg_complex operator*(const lcg_float &a, const lcg_complex &b)
{
	lcg_complex ret;
	ret.rel = a*b.rel;
	ret.img = a*b.img;
	return ret;
}

lcg_complex operator/(const lcg_complex &a, const lcg_complex &b)
{
	lcg_complex ret;
	if (b.rel == 0 && b.img == 0)
	{
		ret.rel = ret.img = NAN;
		return ret;
	}

	ret.rel = (a.rel*b.rel + a.img*b.img)/(b.rel*b.rel + b.img*b.img);
	ret.img = (a.img*b.rel - a.rel*b.img)/(b.rel*b.rel + b.img*b.img);
	return ret;
}

lcg_complex operator/(const lcg_float &a, const lcg_complex &b)
{
	lcg_complex ret;
	if (b.rel == 0 && b.img == 0)
	{
		ret.rel = ret.img = NAN;
		return ret;
	}

	ret.rel = a*b.rel/(b.rel*b.rel + b.img*b.img);
	ret.img = -1.0*a*b.img/(b.rel*b.rel + b.img*b.img);
	return ret;
}

std::ostream &operator<<(std::ostream &os, const lcg_complex &a)
{
	if (a.img >= 0)
		os << a.rel << "+" << a.img << "i";
	else
		os << a.rel << a.img << "i";
	return os;
}

void clcg_set(lcg_complex *a, lcg_float r, lcg_float i)
{
	a->rel = r;
	a->img = i;
	return;
}

lcg_float clcg_square(const lcg_complex *a)
{
	return a->rel * a->rel + a->img * a->img;
}

lcg_float clcg_module(const lcg_complex *a)
{
	return sqrt(clcg_square(a));
}

lcg_complex clcg_conjugate(const lcg_complex *a)
{
	lcg_complex b;
	b.rel = a->rel;
	b.img = -1.0 * a->img;
	return b;
}

void clcg_vecrnd(lcg_complex *a, lcg_complex l, lcg_complex h, int size)
{
	srand(time(nullptr));
	for (int i = 0; i < size; i++)
	{
		a[i].rel = (h.rel-l.rel)*rand()*1.0/RAND_MAX + l.rel;
		a[i].img = (h.img-l.img)*rand()*1.0/RAND_MAX + l.img;
	}
	return;
}

void clcg_vecrnd(lcg_complex **a, lcg_complex l, lcg_complex h, int m, int n)
{
	srand(time(nullptr));
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			a[i][j].rel = (h.rel-l.rel)*rand()*1.0/RAND_MAX + l.rel;
			a[i][j].img = (h.img-l.img)*rand()*1.0/RAND_MAX + l.img;	
		}
	}
	return;
}

void clcg_dot(lcg_complex &ret, const lcg_complex *a, const lcg_complex *b, int size)
{
	clcg_set(&ret, 0.0, 0.0);
	// <a,b> = \sum{a_i \cdot b_i}
	for (int i = 0; i < size; i++)
	{
		ret.rel += (a[i].rel*b[i].rel - a[i].img*b[i].img);
		ret.img += (a[i].rel*b[i].img + a[i].img*b[i].rel);
	}
	return;
}

void clcg_inner(lcg_complex &ret, const lcg_complex *a, const lcg_complex *b, int size)
{
	clcg_set(&ret, 0.0, 0.0);
	// <a,b> = \sum{\bar{a_i} \cdot b_i}
	for (int i = 0; i < size; i++)
	{
		ret.rel += (a[i].rel*b[i].rel + a[i].img*b[i].img);
		ret.img += (a[i].rel*b[i].img - a[i].img*b[i].rel);
	}
	return;
}

void clcg_matvec(lcg_complex **A, const lcg_complex *x, lcg_complex *Ax, 
	int m_size, int n_size, lcg_matrix_e layout, clcg_complex_e conjugate)
{
	int i, j;
	lcg_float re, im;
	if (conjugate == Conjugate)
	{
		if (layout == MatNormal)
		{
#pragma omp parallel for private (i, j, re, im) schedule(guided)
			for (i = 0; i < m_size; i++)
			{
				re = 0.0; im = 0.0;
				for (j = 0; j < n_size; j++)
				{
					re += (A[i][j].rel*x[j].rel + A[i][j].img*x[j].img);
					im += (A[i][j].rel*x[j].img - A[i][j].img*x[j].rel);
				}
				clcg_set(&Ax[i], re, im);
			}
			return;
		}

#pragma omp parallel for private (i, j, re, im) schedule(guided)
		for (j = 0; j < n_size; j++)
		{
			re = 0.0; im = 0.0;
			for (i = 0; i < m_size; i++)
			{
				re += (A[i][j].rel*x[i].rel + A[i][j].img*x[i].img);
				im += (A[i][j].rel*x[i].img - A[i][j].img*x[i].rel);
			}
			clcg_set(&Ax[j], re, im);
		}
		return;
	}

	if (layout == MatNormal)
	{
#pragma omp parallel for private (i, j, re, im) schedule(guided)
		for (i = 0; i < m_size; i++)
		{
			re = 0.0; im = 0.0;
			for (j = 0; j < n_size; j++)
			{
				re += (A[i][j].rel*x[j].rel - A[i][j].img*x[j].img);
				im += (A[i][j].rel*x[j].img + A[i][j].img*x[j].rel);
			}
			clcg_set(&Ax[i], re, im);
		}
		return;
	}

#pragma omp parallel for private (i, j, re, im) schedule(guided)
	for (j = 0; j < n_size; j++)
	{
		re = 0.0; im = 0.0;
		for (i = 0; i < m_size; i++)
		{
			re+= (A[i][j].rel*x[i].rel - A[i][j].img*x[i].img);
			im += (A[i][j].rel*x[i].img + A[i][j].img*x[i].rel);
		}
		clcg_set(&Ax[j], re, im);
	}
	return;
}

#endif // LibLCG_SYSTEM_COMPLEX