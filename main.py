import numpy as np
import prettytable

def max_row_norms (matrix):
    return np.max(np.sum(np.abs(matrix), axis=1))
print("Введите размерность квадратной матрицы:")
n = int(input())
delta = 10**(-8)
rtol = 10**(-6)
proper_number_list = list(np.random.randn(1, n) * 100)[0]
proper_number_list = sorted(proper_number_list, key=abs, reverse=True)
matrix_Proper = np.diag(proper_number_list) ## создание диагональной матрицы
while True:
    random_matrix = np.random.randn(n, n) * 100
    if np.linalg.det(random_matrix) != 0:
        break
A = np.linalg.inv(random_matrix) @ matrix_Proper @ random_matrix ## создание рабочей матрицы A
##A = [[ 75.3291527, -5.82315345,  -9.31919342], [  1.78970067,  35.9243854,  -37.51167932], [  6.26172701, -11.16629923,  95.59824119]]

def Power_law_method(A, rtol, delta):
    n = A.shape[0]
    y_0 = [1] * n  ## задаём начальный вектор
    z_0 = [y_0[i] / max(y_0) for i in range(n)]  ## нормировка (хотя для выбранного вектора это ненужно)
    z_old = np.array(z_0)
    q_old = np.array([y_0[i] / z_old[i] for i in range(n) if abs(z_old[i]) > delta])
    k = 0
    ## Быть внимательным с оператором @.
    # Я проверил, он очень умный
    # (умеет сам транспонировать матрицы (как минимум одномерные вектора),
    # поэтому надо быть аккуратным)
    while True:
        k += 1
        y = A @ z_old  ## Шаг 2
        z_k = np.array([y[i] / max(y) for i in range(n)])
        q_k = []
        # q_k = np.array([y[i]/z_old[i] for i in range(n) if abs(z_old[i]) > delta])      # Шаг 3
        for i in range(n):
            if abs(z_old[i]) > delta:
                q_k.append(y[i] / z_old[i])
            else:
                q_k.append(0)
        q_k = np.array(q_k)
        if max(abs(q_k - q_old)) <= rtol * max(max(abs(q_k)), max(abs(q_old))):  # Проверка на сходимость
            break
        z_old = z_k
        q_old = q_k

    prop_num = sum(q_k) / (np.count_nonzero(q_k))
    return [prop_num, z_k]

def pribl_of_prop_number(A, proper_number_list, k = 20):
    n = A.shape[0]
    step_aprox_pr = np.linspace(-max_row_norms(A), max_row_norms(A), k)

    list_of_approx_prop_numb = [[] for i in range(n)]
    temp_list = [[] for i in range(n)]
    proper_number_list = np.array(proper_number_list)

    temp_matrix = [abs(proper_number_list - step_aprox_pr[i]) for i in range(len(step_aprox_pr))]
    min_index_ap = np.argmin(temp_matrix, axis=1)
    temp_matrix = np.sort(temp_matrix, axis=1)
    for i in range(len(step_aprox_pr)):
        if temp_matrix[i][0] != temp_matrix[i][1]:
            list_of_approx_prop_numb[min_index_ap[i]].append(step_aprox_pr[i])
            temp_list[min_index_ap[i]].append(abs(proper_number_list[min_index_ap[i]] - step_aprox_pr[i]) /
                                              max(abs(proper_number_list - step_aprox_pr[i])))
    best_approx = [list_of_approx_prop_numb[i][np.argmin(temp_list[i])] for i in range(n)]
    if len(set(min_index_ap)) < 3:
        return pribl_of_prop_number(A, proper_number_list, k + 100)
    else:
        return best_approx


def Inverse_power_method(A, rtol, delta):
    n = A.shape[0]
    k = 200
    initial_shift = pribl_of_prop_number(A, proper_number_list, k)
    print(f"\nсобственные числа:{proper_number_list}")
    print(f"приближения:{initial_shift}")
    answer = []
    ## Это обратный степенной метод
    for i in range(n):
        z_old = [1] * n
        sig_old = initial_shift[i]
        w = 0
        while w != 10000:
            w += 1
            if np.linalg.det(A - sig_old * np.eye(n)) == 0:
                    sig_old += 1e-10
            y_k = np.linalg.solve(A - sig_old * np.eye(n), z_old)
            z_k = y_k/max(abs(y_k))
            mu_k = []
            for i in range(n):
                if abs(y_k[i]) > delta:
                    mu_k.append(z_old[i]/y_k[i])
                else:
                    mu_k.append(0)
            mu_k = np.array(mu_k)
            sig_k = sig_old + sum(mu_k)/np.count_nonzero(mu_k)
            if abs(sig_k - sig_old) < rtol and max(abs(z_k-z_old)) < rtol:
                break
            z_old = z_k
            sig_old = sig_k

        answer.append([sig_k, z_k])
    ##

    return answer

def Hess(A):
    n = A.shape[0]
    for i in range(n-2):
        s = np.sign(A[i + 1, i]) * np.sqrt(np.sum(A[i + 1:, i] ** 2))
        mu = 1 / np.sqrt(2 * s * (s - A[i + 1, i]))
        ## Процесс построения вектора v
        v = np.zeros((1, n))
        v[0, i+1] = A[i + 1, i] - s
        v[0, i+2:] = A[i + 2:, i]
        v = mu * np.transpose(v)
        H = np.eye(n) - (2*v) @ np.transpose(v)
        A = H @ A @ H
    return A

def QR_decomposition(B, delta):
    C = B
    answer = []
    n = B.shape[0]
    n_ch = n
    while n_ch != 0:
        B = B[:n_ch, :n_ch]
        w = 0
        while w != 10000:
            w += 1
            b_old = B[n_ch - 1, n_ch - 1]
            Q, R = np.linalg.qr(B - B[n_ch - 1, n_ch - 1] * np.eye(n_ch))
            B = R @ Q + B[n_ch - 1, n_ch - 1] * np.eye(n_ch)
            b_new = B[n_ch - 1, n_ch - 1]
            if abs(b_new - b_old) < 1/3*abs(b_old):
                if n_ch == 1:
                    break
                elif abs(B[n_ch - 1, n_ch-2]) < delta:
                    break
        answer.append(b_new)
        # if n_ch != n:
        #     C[n_ch-1, :] = np.array(list(B[n_ch-1, :]) + list(np.zeros((1, n-n_ch))))
        #         #np.concatenate((B[n_ch-1, :], np.zeros((1, n-n_ch))), axis=1)
        #     C[:, n_ch-1] = np.array(list(B[:, n_ch-1]) + list(np.zeros((1, n-n_ch))))
        #         # np.concatenate((B[:, n_ch-1], np.zeros((n-n_ch, 1))), axis=0)
        # else:
        #     C[n_ch - 1, :] = B[n_ch - 1, :]
        #     C[:, n_ch - 1] = B[:, n_ch-1]
        n_ch -= 1
    return C, answer



result_power_law_method = Power_law_method(A, rtol, delta)
result_inv_pow = Inverse_power_method(A, rtol, delta)
print("Результат степенного метода:")
print(f"Наибольшое собственное число: {result_power_law_method[0]}")
print(f"Собственный вектор: {result_power_law_method[1]}")
print(f"Проверка A * x == lam * x: {A @ result_power_law_method[1]} == {result_power_law_method[0] * result_power_law_method[1]}")

print("\nРезультат обратно степенного метода:")
for i in range(n):
    print(f"Собственное число_{i+1}: {result_inv_pow[i][0]}")
    print(f"Собственный вектор_{i+1}: {result_inv_pow[i][1]}")
    print(f"Проверка A * x == lam * x:{A @ result_inv_pow[i][1]} == {result_inv_pow[i][0] * result_inv_pow[i][1]}")

#np.set_printoptions(precision=2, suppress=True)
print(Hess(A))
B = Hess(A)
U, result_QR = QR_decomposition(B, delta)
# print(f"\n Матрица, которая получилась после применений QR:\n{U}")
print(f"\n Собственные числа, полученные QR: {sorted(result_QR, key=abs, reverse=True)}")
print(f"\n Изначальные собственные числа: {proper_number_list}")
