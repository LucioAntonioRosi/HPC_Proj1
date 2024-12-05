from mpi4py import MPI
import numpy as np
import time as t
from scipy.io import mmread
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import argparse
import os

#os.environ["OMP_NUM_THREADS"] = "1"
#os.environ["MKL_NUM_THREADS"] = "1"
#os.environ["OPENBLAS_NUM_THREADS"] = "1"
#os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
#os.environ["NUMEXPR_NUM_THREADS"] = "1"
#os.environ["BLIS_NUM_THREADS"] = "1"

########################################################################################

######################################################################
####                      HOW TO RUN THE CODE                     ####
######################################################################

# To run the code, you need to execute the following command:

### mpirun -n <number of processes> python proj.py --m <number of rows> --n <number of columns> --type <type of QR decomposition> --matrix <name of matrix>

# or, if you want the code to run only the sequential version:

### python proj.py --m <number of rows> --n <number of columns> --type <type of QR decomposition> --matrix <name of matrix>

# where:
# - <number of processes> is the number of processes you want to use to run the code.
# - <number of rows> is the number of rows of the matrix you want to build.
# - <number of columns> is the number of columns of the matrix you want to build.
# - <type of QR decomposition> is the type of QR decomposition you want to perform. It can be one of the following:
#   - "par_TSQR" for parallel TSQR.
#   - "seq_TSQR" for sequential TSQR.
#   - "par_MGS" for parallel MGS.
#   - "seq_MGS" for sequential MGS.

# - <type of matrix> is the name of matrix you want to build.
########################################################################################

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def append_to_file(filename, text):
    with open(filename, 'a') as file:
        file.write(text + '\n')

def par_TSQR(W):

    if rank==0:
        mn= np.array([W.shape[0],W.shape[1]])
    else:
        mn = np.empty(2,np.int64)

    comm.Bcast(mn,root = 0)

    [m,n] = mn

    logP = np.log2(size)

    assert logP == int(logP) and logP >= 0, f"logP is not an integer: {logP}"
    logP = int(logP)

    rows_per_proc = [m // size + (1 if i < m % size else 0) for i in range(size)]
    row_displacements = [sum(rows_per_proc[0:i]) for i in range(size)]
    local_rows = rows_per_proc[rank]
    local_W = np.zeros((local_rows, n), dtype=np.float64)
    comm.Scatterv([W, [r * n for r in rows_per_proc], [d * n for d in row_displacements], MPI.DOUBLE], local_W, root=0)
    
    if rank == 0:
        R = np.zeros((n, n), dtype=np.float64)
        Q = np.zeros((m, n), dtype=np.float64)
    else:
        R = None
        Q = None  

    [local_Q, local_R] = np.linalg.qr(local_W, mode='reduced')
    if logP > 0:
        if np.mod(rank, 2) == 0:
            local_Q_rank_above = np.zeros((rows_per_proc[rank + 1], n), dtype=np.float64)
            semi_local_R = np.zeros((2*n, n), dtype=np.float64)
            semi_local_R[:n, :] = local_R
            comm.Recv([semi_local_R[n:2*n,:], MPI.DOUBLE], source=rank+1, tag=0)
            comm.Recv([local_Q_rank_above, MPI.DOUBLE], source=rank+1, tag=1)
            
        else:
            comm.Send([local_R, MPI.DOUBLE], dest=rank-1, tag=0)
            comm.Send([local_Q, MPI.DOUBLE], dest=rank-1, tag=1)
            
        

        for j in range(1,logP):
            if np.mod(rank, 2**j) == 0:
                [local_Q1, local_R1] = np.linalg.qr(semi_local_R, mode='reduced')
                temp = np.zeros((sum(rows_per_proc[rank:rank + 2**j]), n), dtype=np.float64)
                local_m = sum(rows_per_proc[rank:rank + 2**(j-1)])
                temp[:local_m,:] = local_Q @ local_Q1[:n,:]
                temp[local_m:,:] = local_Q_rank_above @ local_Q1[n:2*n,:]
                local_Q = temp
                if np.mod(rank, 2**(j+1)) == 0:
                    local_Q_rank_above = np.zeros((sum(rows_per_proc[rank + 2**j:rank + 2**(j+1)]), n), dtype=np.float64)
                    semi_local_R = np.zeros((2*n, n), dtype=np.float64)
                    semi_local_R[:n, :] = local_R1
                    comm.Recv([semi_local_R[n:2*n,:], MPI.DOUBLE], source=rank+2**j, tag=0)
                    comm.Recv([local_Q_rank_above, MPI.DOUBLE], source=rank+2**j, tag=1)
                else:
                    comm.Send([local_R1, MPI.DOUBLE], dest=rank-2**j, tag=0)
                    comm.Send([local_Q, MPI.DOUBLE], dest=rank-2**j, tag=1)


        if rank == 0:
            [local_Q1, local_R1] = np.linalg.qr(semi_local_R, mode='reduced')
            Q[:sum(rows_per_proc[rank:rank + 2**(logP-1)]),:] = local_Q @ local_Q1[:n,:]
            Q[sum(rows_per_proc[rank:rank + 2**(logP-1)]):,:] = local_Q_rank_above @ local_Q1[n:2*n,:]
            R = local_R1
    
    else:
        Q = local_Q
        R = local_R

    return Q,R

def seq_TSQR(W, n_proc):

    m = W.shape[0]
    n = W.shape[1]

    logP = np.log2(n_proc)

    assert logP == int(logP), f"logP is not an integer: {logP}"
    logP = int(logP)

    R = np.zeros((n, n), dtype=np.float64)
    Q = np.zeros((m, n), dtype=np.float64)

    if logP > 0:
        rows_per_proc = [m // n_proc + (1 if i < m % n_proc else 0) for i in range(n_proc)]
        row_displacements = [sum(rows_per_proc[0:i]) for i in range(n_proc)]

        local_Q_temp = [None] * (n_proc)
        semi_local_R = [None] * (n_proc // 2)
        for j in range(n_proc):
            local_Q_temp[j] = np.zeros((rows_per_proc[j], n), dtype=np.float64)
        for j in range(n_proc // 2):
            semi_local_R[j] = np.zeros((2 * n, n), dtype=np.float64)

        for i in range(n_proc):
            [local_Q , local_R] = np.linalg.qr(W[row_displacements[i]:row_displacements[i] + rows_per_proc[i], :], mode='reduced')
            semi_local_R[i // 2][(1 if i % 2 == 1 else 0) * n:(2 if i % 2 == 1 else 1) * n, :] = local_R
            local_Q_temp[i] = local_Q

        for j in range(1, logP):
            aux = [np.zeros((2 * n, n), dtype=np.float64)] * (n_proc // 2**(j + 1))
            for i in range(0,n_proc,2**j):
                [local_Q1, local_R1] = np.linalg.qr(semi_local_R[i // 2**j])
                aux[i // 2**(j + 1)][(0 if i % 2**(j + 1) == 0 else 1) * n:(1 if i % 2**(j + 1) == 0 else 2) * n, :] = local_R1

                temp = np.zeros((sum(rows_per_proc[i:i + 2**j]), n), dtype=np.float64)
                local_m = sum(rows_per_proc[i:i + 2**(j-1)])
                temp[:local_m,:] = local_Q_temp[i] @ local_Q1[:n,:]
                temp[local_m:,:] = local_Q_temp[i + 2**(j-1)] @ local_Q1[n:2*n,:]

                local_Q_temp[i] = temp
            semi_local_R = aux



        [local_Q1, local_R1] = np.linalg.qr(semi_local_R[0], mode='reduced')
        Q[:sum(rows_per_proc[rank:rank + 2**(logP-1)]),:] = local_Q_temp[0] @ local_Q1[:n,:]
        Q[sum(rows_per_proc[rank:rank + 2**(logP-1)]):,:] = local_Q_temp[n_proc // 2] @ local_Q1[n:2*n,:]
        R = local_R1
    else:
        [Q, R] = np.linalg.qr(W, mode='reduced')
    return Q, R

def seq_MGS(W,speed):
    m = W.shape[0]
    n = W.shape[1]
    Q = W.copy()
    R = np.zeros((n, n))
    condition_numbers = []
    loss = []
    if speed == 0:
        for j in range(n):
            for i in range(j):
                R[i, j] = np.dot(Q[:, i].T, Q[:, j])
                Q[:, j] = Q[:, j] - R[i, j] * Q[:, i]
            R[j, j] = np.linalg.norm(Q[:, j], ord=2)
            Q[:, j] = Q[:, j] / R[j, j]
            condition_numbers.append(np.linalg.cond(Q[:, :j + 1]))
            loss.append(np.linalg.norm(np.eye(j + 1) - Q[:, :j + 1].T @ Q[:, :j + 1], ord=2))
        return Q, R, condition_numbers, loss
    else:
        for j in range(n):
            for i in range(j):
                R[i, j] = np.dot(Q[:, i].T, Q[:, j])
                Q[:, j] = Q[:, j] - R[i, j] * Q[:, i]
            R[j, j] = np.linalg.norm(Q[:, j], ord=2)
            Q[:, j] = Q[:, j] / R[j, j]        
        return Q, R
    
def par_MGS(W,speed):
    if rank == 0:
        mn = np.array([W.shape[0], W.shape[1]])
    else:
        mn = np.empty(2, np.int64)
    comm.Bcast(mn, root=0)
    [m, n] = mn
    R = np.zeros((n, n))
    rows_per_proc = [m // size + (1 if i < m % size else 0) for i in range(size)]
    row_displacements = [sum(rows_per_proc[0:i]) for i in range(size)]
    local_rows = rows_per_proc[rank] 
    local_Q = np.zeros((local_rows, n), dtype=np.float64)
    comm.Scatterv([W, [r *n for r in rows_per_proc], [d*n for d in row_displacements], MPI.DOUBLE], local_Q, root=0)
    Q = None

    if speed == 0:

        if rank == 0:
            Q = np.zeros((m, n), dtype=np.float64)
            

        condition_numbers = []
        loss = []

        for j in range(n):
            for i in range(j):
                local_rho = local_Q[:, i].T @ (local_Q[:,j])
                R[i,j] = comm.allreduce(local_rho, op=MPI.SUM)
                local_Q[:,j] = local_Q[:,j] - local_Q[:,i] * R[i, j]
            local_beta = np.sum(local_Q[:,j] ** 2)
            R[j,j] = np.sqrt(comm.allreduce(local_beta, op=MPI.SUM))
            local_Q[:,j] = local_Q[:,j] / R[j, j]
            Q_j = np.zeros((m,1), dtype=np.float64)
            Q_aux = np.ascontiguousarray(local_Q[:,j].T)
            comm.Gatherv(Q_aux, [Q_j if rank == 0 else None, rows_per_proc, row_displacements, MPI.DOUBLE], root=0)
            
            if rank == 0:
                Q[:,j] = Q_j[:,0]
                condition_numbers.append(np.linalg.cond(Q[:,:j + 1]))
                loss.append(np.linalg.norm(np.eye(j + 1) - Q[:,:j + 1].T @ Q[:,:j + 1], ord=2))
            
        
    
        return Q, R, condition_numbers, loss

    else:
        if rank == 0:
            Q = np.zeros((m, n), dtype=np.float64)
        for j in range(n):
            for i in range(j):
                local_rho = local_Q[:,i].T @ local_Q[:,j]
                R[i,j] = comm.allreduce(local_rho, op=MPI.SUM)
                local_Q[:,j] = local_Q[:, j] - local_Q[:,i] * R[i, j]
            local_beta = np.sum(local_Q[:,j] ** 2)
            R[j,j] = np.sqrt(comm.allreduce(local_beta, op=MPI.SUM))
            local_Q[:,j] = local_Q[:,j] / R[j, j]
        
        comm.Gatherv(local_Q, [Q, [r*n for r in rows_per_proc], [d*n for d in row_displacements], MPI.DOUBLE], root=0)

        return Q, R

def save(matrix_name, m, n, condition_numbers, bool_cond, dir, loss, bool_loss, alg, cond_C):
    if bool_cond == 1 or bool_loss == 1:
        if matrix_name == 'None':
            base_filename = 'combined_C(' + str(m) + 'x' + str(n) + ')_' + 'np' + str(size) + '_'
        else:
            base_filename = 'combined_' + matrix_name + '_'
        extension = '.png'

        i = 0
        while os.path.exists(f'{dir}/{base_filename}{i:03d}{extension}'):
            i += 1
        filename = f'{dir}/{base_filename}{i:03d}{extension}'

        fig, axs = plt.subplots(2, 1, figsize=(10, 8))  # Create 2 subplots in one figure

        if bool_cond == 1:
            axs[0].plot(range(1, n + 1), condition_numbers)
            y_min = min(condition_numbers)
            y_max = max(condition_numbers)
            y_range = y_max - y_min
            axs[0].set_ylim(y_min - 0.2 * y_range - 1e-16, y_max + 0.20 * y_range + 1e-16)
            axs[0].grid(True, which='both', linestyle='--', linewidth=0.5)
            formatter = ScalarFormatter(useMathText=True)
            formatter.set_scientific(True)
            formatter.set_powerlimits((-3, 3))
            axs[0].yaxis.set_major_formatter(formatter)
            axs[0].set_xlabel('Iteration')
            axs[0].set_ylabel('Condition Number')
            axs[0].set_title('Condition Number of Q at Each Iteration')

        if bool_loss == 1:
            axs[1].plot(range(1, n + 1), loss)
            y_min = min(loss)
            y_max = max(loss)
            y_range = y_max - y_min
            axs[1].set_ylim(y_min - 0.2 * y_range, y_max + 0.20 * y_range)
            axs[1].grid(True, which='both', linestyle='--', linewidth=0.5)
            formatter = ScalarFormatter(useMathText=True)
            formatter.set_scientific(True)
            formatter.set_powerlimits((-3, 3))
            axs[1].yaxis.set_major_formatter(formatter)
            axs[1].set_xlabel('Iteration')
            axs[1].set_ylabel('Loss of Orthogonality')
            axs[1].set_title('Loss of Orthogonality at Each Iteration')

        # Add a big title on top of the two subplots with the value of cond_C
        fig.suptitle(f'cond(C) = {cond_C:.4e}', fontsize=16)

        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to make room for the big title
        plt.savefig(filename)
        plt.close()

    if alg == 'T':
        if bool_cond == 1:
            if matrix_name == 'None':
                base_filename = 'bar_cond_C(' + str(m) + 'x' + str(n) + ')_'
            else:
                base_filename = 'bar_cond_' + matrix_name + '_'
            extension = '.png'

            i = 0
            while os.path.exists(f'{dir}/{base_filename}{i:03d}{extension}'):
                i += 1
            filename = f'{dir}/{base_filename}{i:03d}{extension}'

            plt.bar(range(1, n + 1), condition_numbers)
            plt.grid(True, which='both', linestyle='--', linewidth=0.5)
            formatter = ScalarFormatter(useMathText=True)
            formatter.set_scientific(True)
            formatter.set_powerlimits((-3, 3))
            plt.gca().yaxis.set_major_formatter(formatter)
            plt.xlabel('Iteration')
            plt.ylabel('Condition Number')
            plt.title('Condition Number of Q at Each Iteration (Bar Chart)')
            plt.savefig(filename)
            plt.close()

        if bool_loss == 1:
            if matrix_name == 'None':
                base_filename = 'bar_loss_C(' + str(m) + 'x' + str(n) + ')_'
            else:
                base_filename = 'bar_loss_' + matrix_name + '_'
            extension = '.png'

            i = 0
            while os.path.exists(f'{dir}/{base_filename}{i:03d}{extension}'):
                i += 1
            filename = f'{dir}/{base_filename}{i:03d}{extension}'

            plt.bar(range(1, n + 1), loss)
            plt.grid(True, which='both', linestyle='--', linewidth=0.5)
            formatter = ScalarFormatter(useMathText=True)
            formatter.set_scientific(True)
            formatter.set_powerlimits((-3, 3))
            plt.gca().yaxis.set_major_formatter(formatter)
            plt.xlabel('Iteration')
            plt.ylabel('Loss of Orthogonality')
            plt.title('Loss of Orthogonality at Each Iteration (Bar Chart)')
            plt.savefig(filename)
            plt.close()

def f(x, mu):
    return np.sin(10 * (mu + x)) / (np.cos(100 * (mu - x)) + 1.1)

def main():

    parser = argparse.ArgumentParser(description='Parallel QR decomposition')
    parser.add_argument("--m", type=int, default=50000, help='Number of rows of the matrix')
    parser.add_argument("--n", type=int, default=600, help='Number of columns of the matrix')
    parser.add_argument("--type", type=str, default='All', help='Type of QR decomposition to perform')
    parser.add_argument("--matrix", type=str, default='None', help='Type of matrix to build')
    args = parser.parse_args()

    if rank == 0:
        filename_val = "values.txt"
        spaces = "#"*30
        append_to_file(filename_val, spaces)

    type = args.type
    matrix = args.matrix



    C = None
        
    if matrix == 'None':

        m = args.m
        n = args.n
    
        rows_per_proc = [m // size + (1 if i < m % size else 0) for i in range(size)]
        row_displacements = [sum(rows_per_proc[:i]) for i in range(size)]
        local_rows = rows_per_proc[rank]

        local_start = row_displacements[rank]
        local_end = local_start + local_rows

        if rank == 0:
            print("Building matrix C...")

        start_time = t.time()

        local_C = np.zeros((local_rows, n), np.float64)

        for i in range(local_start, local_end):
            for j in range(0, n):
                local_C[i - local_start, j] = f((i) / (m - 1), (j) / (n - 1))

        if rank == 0:
            C = np.zeros((m, n), np.float64)

        comm.Gatherv(local_C, [C if rank == 0 else None, [r * n for r in rows_per_proc], [d * n for d in row_displacements], MPI.DOUBLE], root=0)

        built_time = t.time()

        if rank == 0:
            print("Building completed. Time to build the matrix: ", built_time - start_time)
        comm.Barrier()

    elif not(os.path.exists(matrix + '.mtx')):
        if rank == 0:
            print('Matrix not found... Download it from sparse.temu.edu')
        # break the programm
        return


    else:
        if rank == 0:
            sparse_matrix = mmread(matrix + '.mtx')
            C = sparse_matrix.toarray().astype(np.float64)
            m = C.shape[0]
            n = C.shape[1]
            print("Matrix read from file. Matrix shape: ", C.shape)

    if rank == 0:
        cond_C = np.linalg.cond(C)

    if matrix == 'None':
        if rank == 0:
            append_to_file(filename_val, f"Matrix C({m}x{n}), with condition number {cond_C}, and number of processors {size}")
    else:
        if rank == 0:
            append_to_file(filename_val, f"Matrix {matrix}, with condition number {cond_C}, and number of processors {size}")

    if type != 'All':

        dir = 'images/' + type
        if rank == 0:
            append_to_file(filename_val, f"Type of QR decomposition: {type}")        
        
        built_time = t.time()

        if type == "par_TSQR":
            if rank == 0:
                print("Starting parallel TSQR...")

            [Q, R] = par_TSQR(C)

        elif type == "par_MGS":

            if rank == 0:
                print("Starting parallel MGS...")

            [Q, R,condition_numbers,loss] = par_MGS(C, 0)

            if rank == 0:
                save(matrix, m, n, condition_numbers, 1, dir, loss, 1, 'M', cond_C)

        elif type == "seq_MGS":

            if rank == 0:
                print("Starting sequential MGS...")
                [Q, R, condition_numbers, loss] = seq_MGS(C,0)
                save(matrix, m, n, condition_numbers, 1, dir, loss, 1, 'M', cond_C)


        elif type == "seq_TSQR":

            if rank == 0:
                print("Starting sequential TSQR...")
                [Q, R] = seq_TSQR(C,4)
                
        if rank == 0:
                mgs_time = t.time()
                append_to_file(filename_val, f"Time to build the QR decomposition: {mgs_time - built_time}")
                print("Construction of QR matrices completed. Time to build them: ", mgs_time - built_time)
                append_to_file(filename_val, f"Reconstruction error norm: {np.linalg.norm(C - Q @ R, ord=2)}")
                print("Reconstruction error norm:", np.linalg.norm(C - Q @ R, ord=2))
                append_to_file(filename_val, f"Condition number of Q: {np.linalg.cond(Q)}")
                append_to_file(filename_val, f"Loss of orthogonality: {np.linalg.norm(np.eye(n) - Q.T @ Q, ord=2)}")
                print("||I - Q^T*Q||:", np.linalg.norm(np.eye(n) - Q.T @ Q, ord=2))
                print("Condition number of C:", np.linalg.cond(C))

    else:
        start_time = t.time()

        if rank == 0:
            append_to_file(filename_val, f"Type of QR decomposition: All")
            logP = np.log2(size)
            if logP == int(logP) and logP > 0:
                print("Starting sequential TSQR...")
                [Q1, R1] = seq_TSQR(C,size)
            else:
                print("Sequential TSQR was not run because of the number of processors")
            seq_TSQR_time = t.time()

            print("Starting sequential MGS...")
            [Q2, R2] = seq_MGS(C, 1)
            seq_MGS_time = t.time()
       

        logP = np.log2(size)
        if logP == int(logP) and logP > 0:
            if rank == 0:
                print("Starting parallel TSQR...")
            [Q3, R3] = par_TSQR(C)
        else:
            if rank == 0:
                print("Parallel TSQR was not run because of the number of processors")
        par_TSQR_time = t.time()
        if rank == 0:
            print("Starting parallel MGS...")
        [Q4, R4] = par_MGS(C, 1)
        par_MGS_time = t.time()
        if rank == 0:
            print("Starting numpy QR decomposition...")
            [Q5, R5] = np.linalg.qr(C, mode='reduced')
            numpy_time = t.time() 

        if rank == 0:
            dir = 'images/All'

            if matrix == 'None':
                base_filename = 'time_C(' + str(m) + 'x' + str(n) + ')_' + 'np' + str(size) + '_'
            else:
                base_filename = 'time_' + matrix + '_'
            extension = '.png'

            i = 0
            while os.path.exists(f'{dir}/{base_filename}{i:03d}{extension}'):
                i += 1
            filename = f'{dir}/{base_filename}{i:03d}{extension}'

            #plot of the times
            fig, ax = plt.subplots(2,2, figsize=(10, 8))

            
            formatter = ScalarFormatter(useMathText=True)
            formatter.set_scientific(True)
            formatter.set_powerlimits((-3, 3))
            if logP == int(logP) and logP > 0:
                print("All algorithms were run")
                names = ['seq_TSQR', 'seq_MGS', 'par_TSQR' ,'par_MGS', 'numpy']
                times = [seq_TSQR_time - start_time, seq_MGS_time - seq_TSQR_time, par_TSQR_time - seq_MGS_time, par_MGS_time - par_TSQR_time, numpy_time - par_MGS_time]
                condition_number_total = [np.linalg.cond(Q1), np.linalg.cond(Q2), np.linalg.cond(Q3), np.linalg.cond(Q4), np.linalg.cond(Q5)]
                losses = [np.linalg.norm(np.eye(n) - Q1.T @ Q1, ord=2), np.linalg.norm(np.eye(n) - Q2.T @ Q2, ord=2), np.linalg.norm(np.eye(n) - Q3.T @ Q3, ord=2), np.linalg.norm(np.eye(n) - Q4.T @ Q4, ord=2), np.linalg.norm(np.eye(n) - Q5.T @ Q5, ord=2)]
                errors = [np.linalg.norm(C - Q1 @ R1, ord=2), np.linalg.norm(C - Q2 @ R2, ord=2), np.linalg.norm(C - Q3 @ R3, ord=2), np.linalg.norm(C - Q4 @ R4, ord=2), np.linalg.norm(C - Q5 @ R5, ord=2)]
            else:
                print("Two of the algorithms were not run")
                names = ['seq_MGS', 'par_MGS', 'numpy']
                times = [seq_MGS_time - seq_TSQR_time, par_MGS_time - par_TSQR_time, numpy_time - par_MGS_time]
                condition_number_total = [np.linalg.cond(Q2), np.linalg.cond(Q4), np.linalg.cond(Q5)]
                losses = [np.linalg.norm(np.eye(n) - Q2.T @ Q2, ord=2), np.linalg.norm(np.eye(n) - Q4.T @ Q4, ord=2), np.linalg.norm(np.eye(n) - Q5.T @ Q5, ord=2)]
                errors = [np.linalg.norm(C - Q2 @ R2, ord=2), np.linalg.norm(C - Q4 @ R4, ord=2), np.linalg.norm(C - Q5 @ R5, ord=2)]

            
           
            append_to_file(filename_val, f"In order the algorithms are: {names}")
            append_to_file(filename_val, f"The times are: {times}")
            cond_as_float = [float(cond) for cond in condition_number_total]
            append_to_file(filename_val, f"The condition numbers are: {cond_as_float}")
            losses_as_float = [float(loss) for loss in losses]
            append_to_file(filename_val, f"The losses are: {losses_as_float}")
            errors_as_float = [float(error) for error in errors]
            append_to_file(filename_val, f"The errors are: {errors_as_float}")
            
            ax[0,0].bar(names, times)
            ax[0,0].set_ylabel('Time (s)')
            ax[0,0].set_title('Time taken by each algorithm')
            ax[0,0].yaxis.set_major_formatter(formatter)
            y_min = min(times)
            y_max = max(times)
            if y_max/y_min > 20:
                ax[0,0].set_yscale('log')
            ax[0,0].grid(True, which='both', linestyle='--', linewidth=0.5)

            #plot of the condition numbers last condition number comparison

            ax[1,0].bar(names, condition_number_total)
            ax[1,0].set_ylabel('Condition Number')
            ax[1,0].set_title('Condition number of Q at the last iteration')
            y_min = min(condition_number_total)
            y_max = max(condition_number_total)
            if y_max/y_min > 20:
                ax[1,0].set_yscale('log')
            ax[1,0].yaxis.set_major_formatter(formatter)
            ax[1,0].grid(True, which='both', linestyle='--', linewidth=0.5)

            #plot of the loss of orthogonality last loss comparison
            
            ax[0,1].bar(names, losses)
            ax[0,1].set_ylabel('Loss of Orthogonality')
            ax[0,1].set_title('Loss of Orthogonality at the last iteration')
            ax[0,1].yaxis.set_major_formatter(formatter)
            y_min = min(losses)
            y_max = max(losses)
            if y_max/y_min > 20:
                ax[0,1].set_yscale('log')
            ax[0,1].grid(True, which='both', linestyle='--', linewidth=0.5)

            #plot of the error norm

            ax[1,1].bar(names, errors)
            ax[1,1].set_ylabel('Error Norm')
            ax[1,1].set_title('Error Norm of the QR decomposition')
            ax[1,1].yaxis.set_major_formatter(formatter)
            y_min = min(errors)
            y_max = max(errors)
           
            if y_max/y_min > 20:
                ax[1,1].set_yscale('log')
            ax[1,1].grid(True, which='both', linestyle='--', linewidth=0.5)


            fig.suptitle(f'cond(C) = {cond_C:.4e}', fontsize=16)

            plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to make room for the big title
            plt.savefig(filename)
            plt.close()

            
            

    if rank == 0:
        print("Process completed")
        append_to_file(filename_val, spaces)
        append_to_file(filename_val, ' ')
        append_to_file(filename_val, ' ')

if __name__ == "__main__":
    main()
