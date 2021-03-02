/******************************************************//**
 *    C++ library of linear conjugate gradient.
 *
 * Copyright (c) 2019-2029 Yi Zhang (zhangyiss@icloud.com)
 * All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *********************************************************/

#ifndef _LCG_EIGEN_H
#define _LCG_EIGEN_H

#include "lcg.h"
#include "Eigen/Dense"

typedef void (*eigen_axfunc_ptr)(void* instance, const Eigen::VectorXd &x, Eigen::VectorXd &prod_Ax);

typedef int (*eigen_progress_ptr)(void* instance, const Eigen::VectorXd *m, const lcg_float converge, 
	const lcg_para *param, const int k);

int eigen_solver(eigen_axfunc_ptr Afp, eigen_progress_ptr Pfp, Eigen::VectorXd &m, const Eigen::VectorXd &B, 
	const lcg_para* param, void* instance, lcg_solver_enum solver_id = LCG_CG);

#endif //_LCG_EIGEN_H