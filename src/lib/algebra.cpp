#include "config.h"
#include "algebra.h"
#include "cmath"

#ifdef LCG_OPENMP
#include "omp.h"
#endif

lcg_complex::lcg_complex()
{
	rel = img = NAN;
}

lcg_complex::lcg_complex(lcg_float r, lcg_float i)
{
	set(r, i);
}

lcg_complex::~lcg_complex(){}

void lcg_complex::set(lcg_float r, lcg_float i)
{
	rel = r; img = i;
	return;
}

lcg_float lcg_complex::module()
{
	return sqrt(rel*rel + img*img);
}

lcg_complex lcg_complex::conjugate()
{
	lcg_complex ret;
	ret.rel = rel;
	ret.img = -1.0*img;
	return ret;
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

std::ostream &operator<<(std::ostream &os, const lcg_complex &a)
{
	if (a.img >= 0)
		os << a.rel << "+" << a.img << "i";
	else
		os << a.rel << a.img << "i";
	return os;
}

lcg_complex complex_dot(const lcg_complex *a, const lcg_complex *b, int x_size)
{
	lcg_complex ret;
	ret.set(0.0, 0.0);

	// <a,b> = \sum{a_i \cdot b_i}
	for (int i = 0; i < x_size; i++)
	{
		ret.rel += (a[i].rel*b[i].rel - a[i].img*b[i].img);
		ret.img += (a[i].rel*b[i].img + a[i].img*b[i].rel);
	}
	return ret;
}

lcg_complex complex_inner(const lcg_complex *a, const lcg_complex *b, int x_size)
{
	lcg_complex ret;
	ret.set(0.0, 0.0);

	// <a,b> = \sum{\bar{a_i} \cdot b_i}
	for (int i = 0; i < x_size; i++)
	{
		ret.rel += (a[i].rel*b[i].rel + a[i].img*b[i].img);
		ret.img += (a[i].rel*b[i].img - a[i].img*b[i].rel);
	}
	return ret;
}

void complex_matvec(lcg_complex **A, const lcg_complex *x, lcg_complex *Ax, 
	int m_size, int n_size, matrix_layout_e layout, complex_conjugate_e conjugate)
{
	int i, j;
	if (conjugate == Conjugate)
	{
		if (layout == Normal)
		{
#pragma omp parallel for private (i, j) schedule(guided)
			for (i = 0; i < m_size; i++)
			{
				Ax[i].set(0.0, 0.0);
				for (j = 0; j < n_size; j++)
				{
					Ax[i].rel += (A[i][j].rel*x[j].rel + A[i][j].img*x[j].img);
					Ax[i].img += (A[i][j].rel*x[j].img - A[i][j].img*x[j].rel);
				}
			}
			return;
		}

#pragma omp parallel for private (i, j) schedule(guided)
		for (j = 0; j < n_size; j++)
		{
			Ax[j].set(0.0, 0.0);
			for (i = 0; i < m_size; i++)
			{
				Ax[j].rel += (A[i][j].rel*x[i].rel + A[i][j].img*x[i].img);
				Ax[j].img += (A[i][j].rel*x[i].img - A[i][j].img*x[i].rel);
			}
		}
		return;
	}

	if (layout == Normal)
	{
#pragma omp parallel for private (i, j) schedule(guided)
		for (i = 0; i < m_size; i++)
		{
			Ax[i].set(0.0, 0.0);
			for (j = 0; j < n_size; j++)
			{
				Ax[i].rel += (A[i][j].rel*x[j].rel - A[i][j].img*x[j].img);
				Ax[i].img += (A[i][j].rel*x[j].img + A[i][j].img*x[j].rel);
			}
		}
		return;
	}

#pragma omp parallel for private (i, j) schedule(guided)
	for (j = 0; j < n_size; j++)
	{
		Ax[j].set(0.0, 0.0);
		for (i = 0; i < m_size; i++)
		{
			Ax[j].rel += (A[i][j].rel*x[i].rel - A[i][j].img*x[i].img);
			Ax[j].img += (A[i][j].rel*x[i].img + A[i][j].img*x[i].rel);
		}
	}
	return;
}
