#include "config.h"
#include "lcg.h"
#include "cmath"

#ifdef LCG_OPENMP
#include "omp.h"
#endif

bool operator==(const lcg_complex &a, const lcg_complex &b)
{
	if (a.rel == b.rel && a.img == b.img)
	{
		return true;
	}

	return false;
}

bool operator!=(const lcg_complex &a, const lcg_complex &b)
{
	if (a.rel != b.rel || a.img != b.img)
	{
		return true;
	}

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