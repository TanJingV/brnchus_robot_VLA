import numpy as np


def compute_rigid_transform(A, B):
    """
    计算从点集 A 到点集 B 的刚性变换。
    A: N x 3 的 numpy 数组，真实模型中的3个点
    B: N x 3 的 numpy 数组，虚拟模型中的对应3个点
    返回值：
      R: 3x3 旋转矩阵
      t: 3x1 平移向量
    """
    assert A.shape == B.shape, "两个点集的维度必须一致"
    N = A.shape[0]

    # 计算质心
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    # 去中心化
    AA = A - centroid_A
    BB = B - centroid_B

    # 计算协方差矩阵
    H = np.dot(AA.T, BB)

    # 奇异值分解
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # 确保 R 为正交矩阵（行列式为1）
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = np.dot(Vt.T, U.T)

    # 计算平移向量 t
    t = centroid_B - np.dot(R, centroid_A)

    return R, t

'''# 示例数据：假设A和B分别为真实模型和虚拟模型中的3个对应点
# A: 真实模型中的3个点
A = np.array([[30, 40, 50],
              [35, 45, 55],
              [40, 50, 60]], dtype=np.float64)

# B: 虚拟模型中的3个对应点（假设B是经过某个旋转和平移后的A）
# 这里我们构造一个已知旋转和平移作为示例
theta = np.radians(20)
R_true = np.array([[np.cos(theta), -np.sin(theta), 0],
                   [np.sin(theta), np.cos(theta), 0],
                   [0, 0, 1]])
t_true = np.array([5, -3, 2])

B = np.dot(A, R_true.T) + t_true

# 计算刚性变换
R_est, t_est = compute_rigid_transform(A, B)
print("估计的旋转矩阵 R:")
print(R_est)
print("估计的平移向量 t:")
print(t_est)

# 验证变换效果
A_transformed = np.dot(A, R_est.T) + t_est
print("变换后的A:")
print(A_transformed)
print("B:")
print(B)'''
