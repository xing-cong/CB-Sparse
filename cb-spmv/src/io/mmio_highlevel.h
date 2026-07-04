#pragma once

#include "mmio.h"
#include "../macros.h"

using namespace std;

void exclusive_scan(int *input, int length)
{
    if (length == 0 || length == 1)
        return;

    int old_val, new_val;

    old_val = input[0];
    input[0] = 0;
    for (int i = 1; i < length; i++)
    {
        new_val = input[i];
        input[i] = old_val + input[i - 1];
        old_val = new_val;
    }
}

int read_matrix_info(string file_path, string &mm_type, int &M, int &N, int &nnz)
{
    FILE *f;
    if ((f = fopen(file_path.c_str(), "r")) == NULL)
    {
        cerr << "Could not open file " << file_path << endl;
        return -1;
    }
    MM_typecode matcode;
    if (mm_read_banner(f, &matcode) != 0)
    {
        cerr << "Could not process Matrix Market banner." << endl;
        return -1;
    }
    mm_type = "";
    for (int i = 0; i < 4; i++)
    {
        if (mm_is_valid(matcode) == 0)
        {
            cerr << "Invalid Matrix Market file." << endl;
            return -1;
        }
        mm_type += matcode[i];
    }
    // cout<<"From mmio_highlevel.cpp, the matrix type is : "<<mm_typecode_to_str(matcode)<<endl;
    if (mm_read_mtx_crd_size(f, &M, &N, &nnz) != 0)
    {
        cerr << "Could not read matrix size." << endl;
        return -1;
    }
    fclose(f);
    return 0;
}

int read_csr_matrix_allinone(int *m, int *n, int *nnz, int *isSymmetric, int **csrRowPtr, int **csrColIdx, ValType**csrVal, const char *filename)
{
    int m_tmp, n_tmp;
    int nnz_tmp;

    int ret_code;
    MM_typecode matcode;
    FILE *f;

    int nnz_mtx_report;
    int isInteger = 0, isReal = 0, isPattern = 0, isSymmetric_tmp = 0, isComplex = 0;

    // load matrix
    if ((f = fopen(filename, "r")) == NULL)
        return -1;

    if (mm_read_banner(f, &matcode) != 0)
    {
        printf("Could not process Matrix Market banner.\n");
        return -2;
    }

    if ( mm_is_pattern( matcode ) )  { isPattern = 1; /*printf("type = Pattern\n");*/ }
    if ( mm_is_real ( matcode) )     { isReal = 1; /*printf("type = real\n");*/ }
    if ( mm_is_complex( matcode ) )  { isComplex = 1; /*printf("type = real\n");*/ }
    if ( mm_is_integer ( matcode ) ) { isInteger = 1; /*printf("type = integer\n");*/ }

    /* find out size of sparse matrix .... */
    ret_code = mm_read_mtx_crd_size(f, &m_tmp, &n_tmp, &nnz_mtx_report);
    if (ret_code != 0)
        return -4;

    if ( mm_is_symmetric( matcode ) || mm_is_hermitian( matcode ) )
    {
        isSymmetric_tmp = 1;
        //printf("input matrix is symmetric = true\n");
    }
    else
    {
        //printf("input matrix is symmetric = false\n");
    }

    int *csrRowPtr_counter = (int *)malloc((m_tmp+1) * sizeof(int));
    memset(csrRowPtr_counter, 0, (m_tmp+1) * sizeof(int));

    int *csrRowIdx_tmp = (int *)malloc(nnz_mtx_report * sizeof(int));
    int *csrColIdx_tmp = (int *)malloc(nnz_mtx_report * sizeof(int));
    ValType *csrVal_tmp    = (ValType *)malloc(nnz_mtx_report * sizeof(ValType));

    /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
    /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
    /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */

    for (int i = 0; i < nnz_mtx_report; i++)
    {
        int idxi, idxj;
        double fval, fval_im;
        int ival;
        int returnvalue;

        if (isReal)
        {
            returnvalue = fscanf(f, "%d %d %lg\n", &idxi, &idxj, &fval);
        }
        else if (isComplex)
        {
            returnvalue = fscanf(f, "%d %d %lg %lg\n", &idxi, &idxj, &fval, &fval_im);
        }
        else if (isInteger)
        {
            returnvalue = fscanf(f, "%d %d %d\n", &idxi, &idxj, &ival);
            fval = ival;
        }
        else if (isPattern)
        {
            returnvalue = fscanf(f, "%d %d\n", &idxi, &idxj);
            fval = 1.0;
        }

        // adjust from 1-based to 0-based
        idxi--;
        idxj--;
        fval = 1;
        
        csrRowPtr_counter[idxi]++;
        csrRowIdx_tmp[i] = idxi;
        csrColIdx_tmp[i] = idxj;
        csrVal_tmp[i] = fval;
    }

    if (f != stdin)
        fclose(f);

    if (isSymmetric_tmp)
    {
        for (int i = 0; i < nnz_mtx_report; i++)
        {
            if (csrRowIdx_tmp[i] != csrColIdx_tmp[i])
                csrRowPtr_counter[csrColIdx_tmp[i]]++;
        }
    }

    // exclusive scan for csrRowPtr_counter
    exclusive_scan(csrRowPtr_counter, m_tmp+1);

    int *csrRowPtr_alias = (int *)malloc((m_tmp+1) * sizeof(int));
    nnz_tmp = csrRowPtr_counter[m_tmp];
    int *csrColIdx_alias = (int *)malloc(nnz_tmp * sizeof(int));
    ValType *csrVal_alias    = (ValType *)malloc(nnz_tmp * sizeof(ValType));

    memcpy(csrRowPtr_alias, csrRowPtr_counter, (m_tmp+1) * sizeof(int));
    memset(csrRowPtr_counter, 0, (m_tmp+1) * sizeof(int));

    if (isSymmetric_tmp)
    {
        for (int i = 0; i < nnz_mtx_report; i++)
        {
            if (csrRowIdx_tmp[i] != csrColIdx_tmp[i])
            {
                int offset = csrRowPtr_alias[csrRowIdx_tmp[i]] + csrRowPtr_counter[csrRowIdx_tmp[i]];
                csrColIdx_alias[offset] = csrColIdx_tmp[i];
                csrVal_alias[offset] = csrVal_tmp[i];
                csrRowPtr_counter[csrRowIdx_tmp[i]]++;

                offset = csrRowPtr_alias[csrColIdx_tmp[i]] + csrRowPtr_counter[csrColIdx_tmp[i]];
                csrColIdx_alias[offset] = csrRowIdx_tmp[i];
                csrVal_alias[offset] = csrVal_tmp[i];
                csrRowPtr_counter[csrColIdx_tmp[i]]++;
            }
            else
            {
                int offset = csrRowPtr_alias[csrRowIdx_tmp[i]] + csrRowPtr_counter[csrRowIdx_tmp[i]];
                csrColIdx_alias[offset] = csrColIdx_tmp[i];
                csrVal_alias[offset] = csrVal_tmp[i];
                csrRowPtr_counter[csrRowIdx_tmp[i]]++;
            }
        }
    }
    else
    {
        for (int i = 0; i < nnz_mtx_report; i++)
        {            
            int offset = csrRowPtr_alias[csrRowIdx_tmp[i]] + csrRowPtr_counter[csrRowIdx_tmp[i]];
            csrColIdx_alias[offset] = csrColIdx_tmp[i];
            csrVal_alias[offset] = csrVal_tmp[i];
            csrRowPtr_counter[csrRowIdx_tmp[i]]++;
        }
    }
    
    *m = m_tmp;
    *n = n_tmp;
    *nnz = nnz_tmp;
    *isSymmetric = isSymmetric_tmp;

    *csrRowPtr = csrRowPtr_alias;
    *csrColIdx = csrColIdx_alias;
    *csrVal = csrVal_alias;

    // free tmp space
    free(csrColIdx_tmp);
    free(csrVal_tmp);
    free(csrRowIdx_tmp);
    free(csrRowPtr_counter);

    return 0;
}

int read_matrix_coo_data(string file_path, vector<int> &I, vector<int> &J, vector<ValType> &val)
{
    // int block_size = 32;
    // FILE *f;
    // if ((f = fopen(file_path.c_str(), "r")) == NULL)
    // {
    //     cerr << "Could not open file " << file_path << endl;
    //     return -1;
    // }

    // MM_typecode matcode;
    // if (mm_read_banner(f, &matcode) != 0)
    // {
    //     cerr << "Could not process Matrix Market banner." << endl;
    //     return -1;
    // }

    // int m, n, nnz;
    // if (mm_read_mtx_crd_size(f, &m, &n, &nnz) != 0)
    // {
    //     cerr << "Could not read matrix size." << endl;
    //     return -1;
    // }

    // for (int i = 0; i < nnz; i++)
    // {
    //     int I_tmp, J_tmp;
    //     double val_tmp, val_imag_tmp;
    //     if (mm_read_mtx_crd_entry(f, &I_tmp, &J_tmp, &val_tmp, &val_imag_tmp, matcode) != 0)
    //     {
    //         cerr << "Could not read matrix data." << endl;
    //         return -1;
    //     }
    //     // 坐标从0开始, 默认是一般性稀疏矩阵
    //     I.push_back(I_tmp - 1);
    //     J.push_back(J_tmp - 1);
    //     // unsigned int seed = i;
    //     // mt19937 gen(seed);
    //     // uniform_real_distribution<> dis(1.00, 10000.00);
    //     // val_tmp = dis(gen);
    //     val_tmp = i % 10;
    //     val.push_back(val_tmp);
    //     if (matcode[3] == 'S' && I_tmp != J_tmp)
    //     {
    //         // 对称矩阵
    //         I.push_back(J_tmp - 1);
    //         J.push_back(I_tmp - 1);
    //         val.push_back(val_tmp);
    //     }
    //     else if (matcode[3] == 'K' && I_tmp != J_tmp)
    //     {
    //         // 反对称矩阵
    //         I.push_back(J_tmp - 1);
    //         J.push_back(I_tmp - 1);
    //         val.push_back(-val_tmp);
    //     }
    //     else if (matcode[3] == 'H' && I_tmp != J_tmp)
    //     {
    //         // Hermitian矩阵
    //         I.push_back(J_tmp - 1);
    //         J.push_back(I_tmp - 1);
    //         val.push_back(val_tmp);
    //     }
    // }
    // fclose(f);
    int *csr_row_ptr;
    int *csr_col_idx;
    ValType *csr_val;
    
    int mtx_m,mtx_n,mtx_nnz;
    int is_symmetric;
    read_csr_matrix_allinone(&mtx_m,&mtx_n,&mtx_nnz,&is_symmetric,&csr_row_ptr,&csr_col_idx,&csr_val,file_path.c_str());

    for(int i=0; i<mtx_m; i++){
        for(int j=csr_row_ptr[i]; j<csr_row_ptr[i+1]; j++){
            I.push_back(i);
            J.push_back(csr_col_idx[j]);
            val.push_back(csr_val[j]);
        }
    }
    free(csr_row_ptr);
    free(csr_col_idx);
    free(csr_val);
    return 0;
}