import umatrix
import ulinalg

# 定义一个函数来交换两行
def swap_row(m, i, j):
    m[i], m[j] = m[j], m[i]

# 定义一个函数来将某一行乘以一个常数
def multiply_row(m, i, k):
    for j in range(len(m[i])):
        m[i][j] *= k

# 定义一个函数来将某一行加上另一行乘以一个常数
def add_row(m, i, j, k):
    for l in range(len(m[i])):
        m[i][l] += k * m[j][l]

# 定义一个函数来求逆矩阵
def inverse_matrix(m):
    # 获取矩阵的大小
    n = len(m)
    # 构造增广矩阵
    aug = [[0 for _ in range(2 * n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            aug[i][j] = m[i][j]
        aug[i][i + n] = 1
    # 对增广矩阵进行高斯消元法
    for i in range(n):
        # 如果主对角线上的元素为0，则交换该列的不为0的行到主对角线上
        if aug[i][i] == 0:
            for j in range(i + 1, n):
                if aug[j][i] != 0:
                    swap_row(aug, i, j)
                    break
            else:
                return None # 如果没有找到不为0的元素，则说明没有逆矩阵
        # 将该行除以主元，使主对角线上的元素为1
        multiply_row(aug, i, 1 / aug[i][i])
        # 将其他行加上该行乘以相应的系数，使其他列在该行的位置为0
        for j in range(n):
            if j != i:
                add_row(aug, j, i, -aug[j][i])
        #print(aug)
    # 返回右半部分作为逆矩阵
    return [row[n:] for row in aug]

# 测试代码

m = [[2,-3,-8],[5,-4,-7],[6,-5,-9]] # 给定一个3x3的方阵
m = [[60.0    , 48.0    , 1.0     , 0.0     , 0.0     , 0.0     , -0.0    , -0.0    ],
     [0.0     , 0.0     , 0.0     , 60.0    , 48.0    , 1.0     , -0.0    , -0.0    ],
     [288.0   , 48.0    , 1.0     , 0.0     , 0.0     , 0.0     , -40320.0, -6720.0 ],
     [0.0     , 0.0     , 0.0     , 288.0   , 48.0    , 1.0     , -0.0    , -0.0    ],
     [288.0   , 206.0   , 1.0     , 0.0     , 0.0     , 0.0     , -40320.0, -28840.0],
     [0.0     , 0.0     , 0.0     , 288.0   , 206.0   , 1.0     , -28800.0, -20600.0],
     [60.0    , 206.0   , 1.0     , 0.0     , 0.0     , 0.0     , -0.0    , -0.0    ],
     [0.0     , 0.0     , 0.0     , 60.0    , 206.0   , 1.0     , -6000.0 , -20600.0]]

inv_m = inverse_matrix(m) # 求其逆矩阵
for row in inv_m:
    print(row)
# if inv_m: # 如果有逆矩阵，则打印出来
#     print("The inverse matrix is:")
#     for row in inv_m:
#         print(row)
# else: # 如果没有逆矩阵，则提示无解
#     print("The matrix has no inverse.")








# A = umatrix.matrix([[0,0],[140,0],[140,100],[0,100]])
# B = umatrix.matrix([[60,48],[288,48],[288,206],[60,206]])
#
# C = ulinalg.inverse_matrix(A)
# print(A)
# print(C)
# det,A = ulinalg.det_inv(B)
# print(A)
# print(ulinalg.dot(A,B))
#x,y = A.shape
#print(x,y)
#x,y = ulinalg.det_inv(B)
#print(y*A)
#print(x)
#print(y)