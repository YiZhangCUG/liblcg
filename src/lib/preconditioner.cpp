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

#include "preconditioner.h"

#include "cmath"
#include "map"

void lcg_incomplete_Cholesky_half_buffsize_coo(const int *row, const int *col, int nz_size, int *lnz_size)
{
    size_t c = 0;
    for (size_t i = 0; i < nz_size; i++)
    {
        if (row[i] >= col[i])
        {
            c++;
        }
    }
    *lnz_size = c;
    return;
}

void lcg_incomplete_Cholesky_half_coo(const int *row, const int *col, const lcg_float *val, int N, int nz_size, 
    int lnz_size, int *IC_row, int *IC_col, lcg_float *IC_val)
{
    // We use this to store diagonal elements of the factorizated lower triangular matrix
    lcg_float *diagonal = new lcg_float [N];
    // A temporary row
    lcg_float *tmp_row = new lcg_float [N];
    // index of non-zero elements in tmp_row
    int *filled_idx = new int [N];
    // Begining index of each row in the input matrix
    int *row_st_idx = new int [N];

    size_t i, j, f;

    // Set initial values
    for (i = 0; i < N; i++)
    {
        diagonal[i] = 0.0;
        tmp_row[i] = 0.0;
        filled_idx[i] = -1;
        row_st_idx[i] = -1;
    }

    // copy elements in the lower triangle to the output matrix
    j = 0;
    for (i = 0; i < nz_size; i++)
    {
        if (row[i] >= col[i])
        {
            IC_row[j] = row[i];
            IC_col[j] = col[i];
            IC_val[j] = val[i];
            j++;
        }
    }

    // Get the begining index of each row in the matrix
    j = 1;
    row_st_idx[0] = IC_row[0];
    size_t old_row = IC_row[0];
    for (i = 1; i < lnz_size; i++)
    {
        if (IC_row[i] > old_row)
        {
            row_st_idx[j] = i;
            old_row = IC_row[i];
            j++;
        }
    }

    // Calculate the first element
    IC_val[0] = sqrt(IC_val[0]);
    diagonal[0] = IC_val[0];

    lcg_float dia_sum;
    dia_sum = 0.0;
    // The first one is already calculated
    for (i = 1; i < lnz_size; i++)
    {
        // Calculate the first column if there is one
        if (IC_col[i] == 0)
        {
            IC_val[i] = IC_val[i]/IC_val[0];
            dia_sum = dia_sum + IC_val[i]*IC_val[i];
            continue; // Case 1 break
        }
        
        // Calculate elements in the middle of a row
        if (IC_row[i] > IC_col[i])
        {
            // Find needed values from previous elements
            f = 0;
            j = row_st_idx[IC_col[i]];
            while (IC_col[j] < IC_col[i])
            {
                tmp_row[IC_col[j]] = IC_val[j];
                filled_idx[f]  = IC_col[j];
                f++;
                j++;
            }

            j = row_st_idx[IC_row[i]];
            while (IC_col[j] < IC_col[i])
            {
                IC_val[i] = IC_val[i] - IC_val[j]*tmp_row[IC_col[j]];
                j++;
            }
            
            IC_val[i] = IC_val[i]/diagonal[IC_col[i]];
            dia_sum = dia_sum + IC_val[i]*IC_val[i];

            // reset tmp variables
            for (j = 0; j < f; j++)
            {
                tmp_row[filled_idx[j]] = 0.0;
            }

            continue; // Case 2 break
        }
        
        // We have rearched the diagonal position
        if (IC_row[i] == IC_col[i])
        {
            IC_val[i] = sqrt(IC_val[i] - dia_sum);
            diagonal[IC_col[i]] = IC_val[i];
            dia_sum = 0.0;
        }
    }

    delete[] diagonal;
    delete[] tmp_row;
    delete[] row_st_idx;
    delete[] filled_idx;
    return;
}

void lcg_incomplete_Cholesky_full_coo(const int *row, const int *col, const lcg_float *val, int N, int nz_size, int *IC_row, int *IC_col, lcg_float *IC_val)
{
    // We use this to store diagonal elements of the factorizated lower triangular matrix
    lcg_float *diagonal = new lcg_float [N];
    // A temporary row
    lcg_float *tmp_row = new lcg_float [N];
    // index of non-zero elements in tmp_row
    int *filled_idx = new int [N];
    // Begining index of each row in the input matrix
    int *row_st_idx = new int [N];

    size_t i, j, f, l;

    // Set initial values
    for (i = 0; i < N; i++)
    {
        diagonal[i] = 0.0;
        tmp_row[i] = 0.0;
        filled_idx[i] = -1;
        row_st_idx[i] = -1;
    }

    // copy elements to the output matrix
    for (i = 0; i < nz_size; i++)
    {
        IC_row[i] = row[i];
        IC_col[i] = col[i];
        IC_val[i] = val[i];
    }

    // count element number in the lower triangular part (including the diagonal) and the upper triangular part (excluding the diagonal)
    // build map from elements' cooridnate to their index in the array
    size_t order, L_nz = 0;
    std::map<size_t, size_t> index_map;

    for (i = 0; i < nz_size; i++)
    {
        if (row[i] >= col[i]) // Count number for thr lower triangular part
        {
            L_nz++;
        }
        else // Only need to build the map for the upper triangular part
        {
            order = N*row[i] + col[i];
            index_map[order] = i;
        }
    }

    // We use to store element index in the lower triangle
    j = 0;
    size_t *low_idx = new size_t [L_nz];
    for (i = 0; i < nz_size; i++)
    {
        if (row[i] >= col[i])
        {
            low_idx[j] = i;
            j++;
        }
    }

    // Get the begining index of each row in the matrix
    j = 1;
    row_st_idx[0] = IC_row[0];
    size_t old_row = IC_row[0];
    for (i = 1; i < nz_size; i++)
    {
        if (IC_row[i] > old_row)
        {
            row_st_idx[j] = i;
            old_row = IC_row[i];
            j++;
        }
    }

    // Calculate the first element
    IC_val[0] = sqrt(IC_val[0]);
    diagonal[0] = IC_val[0];

    lcg_float dia_sum;
    dia_sum = 0.0;
    // The first one is already calculated
    for (i = 1; i < L_nz; i++)
    {
        l = low_idx[i];

        // Calculate the first column if there is one
        if (IC_col[l] == 0)
        {
            IC_val[l] = IC_val[l]/IC_val[0];
            dia_sum = dia_sum + IC_val[l]*IC_val[l];
            // Set value at the upper triangle
            order = IC_row[l];
            IC_val[index_map[order]] = IC_val[l];
            continue; // Case 1 break
        }
        
        // Calculate elements in the middle of a row
        if (IC_row[l] > IC_col[l])
        {
            // Find needed values from previous elements
            f = 0;
            j = row_st_idx[IC_col[l]];
            while (IC_col[j] < IC_col[l])
            {
                tmp_row[IC_col[j]] = IC_val[j];
                filled_idx[f]  = IC_col[j];
                f++;
                j++;
            }

            j = row_st_idx[IC_row[l]];
            while (IC_col[j] < IC_col[l])
            {
                IC_val[l] = IC_val[l] - IC_val[j]*tmp_row[IC_col[j]];
                j++;
            }
            
            IC_val[l] = IC_val[l]/diagonal[IC_col[l]];
            dia_sum = dia_sum + IC_val[l]*IC_val[l];

            // Set value at the upper triangle
            order = N*IC_col[l] + IC_row[l];
            IC_val[index_map[order]] = IC_val[l];

            // reset tmp variables
            for (j = 0; j < f; j++)
            {
                tmp_row[filled_idx[j]] = 0.0;
            }

            continue; // Case 2 break
        }
        
        // We have rearched the diagonal position
        if (IC_row[l] == IC_col[l])
        {
            IC_val[l] = sqrt(IC_val[l] - dia_sum);
            diagonal[IC_col[l]] = IC_val[l];
            dia_sum = 0.0;
        }
    }

    delete[] diagonal;
    delete[] tmp_row;
    delete[] row_st_idx;
    delete[] filled_idx;
    delete[] low_idx;
    index_map.clear();
    return;
}

void lcg_solve_upper_triangle_coo(const int *row, const int *col, const lcg_float *U, const lcg_float *B, lcg_float *x, int N, int nz_size)
{
    for (size_t i = 0; i < N; i++)
    {
        x[i] = 0.0;
    }
    
    size_t iter = nz_size - 1;
    double sum;
    for (size_t i = N-1; i >= 0; i--)
    {
        sum = 0.0;
        for (size_t j = iter; j >= 0; j--)
        {
            if (row[j] == i && col[j] > i)
            {
                sum += U[j] * x[col[j]];
            }
            else if (row[j] == i && col[j] == i)
            {
                x[i] = (B[i] - sum)/U[j];
                if (j == 0) return;
                else iter = j-1;
                break;
            }
        }
    }
    return;
}

void lcg_solve_lower_triangle_coo(const int *row, const int *col, const lcg_float *L, const lcg_float *B, lcg_float *x, int N, int nz_size)
{
    for (size_t i = 0; i < N; i++)
    {
        x[i] = 0.0;
    }
    
    size_t iter = 0;
    double sum;
    for (size_t i = 0; i < N; i++)
    {
        sum = 0.0;
        for (size_t j = iter; j < nz_size; j++)
        {
            if (row[j] == i && col[j] < i)
            {
                sum += L[j] * x[col[j]];
            }
            else if (row[j] == i && col[j] == i)
            {
                x[i] = (B[i] - sum)/L[j];
                iter = j+1;
                break;
            }
        }
    }
    return;
}

bool lcg_full_rank_coo(const int *row, const int *col, const lcg_float *M, int N, int nz_size)
{
    size_t s = 0;
    for (size_t i = 0; i < nz_size; i++)
    {
        if (row[i] == col[i] && M[i] != 0.0)
        {
            s++;
        }
    }
    
    if (s == N) return true;
    else return false;
}