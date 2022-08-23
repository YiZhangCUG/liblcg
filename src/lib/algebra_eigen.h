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

#ifndef _ALGEBRA_EIGEN_H
#define _ALGEBRA_EIGEN_H

#include "algebra.h"

#ifdef LibLCG_EIGEN

#include "Eigen/Dense"

/**
 * @brief      Set the input value within a box constraint
 *
 * @param      low_bound    Whether to include the low boundary value
 * @param      hig_bound    Whether to include the high boundary value
 * @param      m            Returned values
 */
void lcg_set2box_eigen(const Eigen::VectorXd &low, const Eigen::VectorXd &hig, Eigen::VectorXd m);

#endif // LibLCG_EIGEN

#endif // _ALGEBRA_EIGEN_H