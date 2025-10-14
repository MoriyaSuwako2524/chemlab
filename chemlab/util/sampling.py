# sampling.py
# 集成常见采样方法
import numpy as np
from scipy import stats

# ==============================
# 1. 通用采样方法
# ==============================

def inverse_transform_sampling(cdf_inv, size=1, random_state=None):
    """
    逆变换采样
    cdf_inv: 累积分布函数的反函数 (callable)
    size: 样本数
    """
    rng = np.random.default_rng(random_state)
    u = rng.uniform(0, 1, size)
    return cdf_inv(u)


def accept_reject_sampling(pdf, proposal_sampler, proposal_pdf, n_samples, M, random_state=None):
    """
    接受-拒绝采样
    pdf: 目标分布的概率密度函数
    proposal_sampler: 可采样的建议分布采样函数
    proposal_pdf: 建议分布的概率密度函数
    n_samples: 需要采样的数量
    M: 常数，使 pdf(x) <= M * proposal_pdf(x)
    """
    rng = np.random.default_rng(random_state)
    samples = []
    while len(samples) < n_samples:
        x = proposal_sampler(1)
        u = rng.uniform(0, 1)
        if u < pdf(x) / (M * proposal_pdf(x)):
            samples.append(x[0])
    return np.array(samples)


# ==============================
# 2. 均匀分布采样
# ==============================

def lcg(seed, a=1664525, c=1013904223, m=2**32, n=10):
    """线性同余法 (LCG)"""
    x = seed
    result = []
    for _ in range(n):
        x = (a * x + c) % m
        result.append(x / m)
    return np.array(result)


def lagged_fibonacci(n, j=24, k=55, seed=1234):
    """滞后Fibonacci发生器 (LFG)"""
    rng = np.random.default_rng(seed)
    s = rng.random(k)
    out = []
    for i in range(n):
        new_val = (s[-j] + s[-k]) % 1.0
        out.append(new_val)
        s = np.append(s[1:], new_val)
    return np.array(out)


# ==============================
# 3. 特定分布采样
# ==============================

def exponential_inverse(lam=1.0, size=1, random_state=None):
    """逆变换采样 - 指数分布"""
    rng = np.random.default_rng(random_state)
    u = rng.random(size)
    return -np.log(1 - u) / lam


def laplace_inverse(mu=0.0, b=1.0, size=1, random_state=None):
    """逆变换采样 - 拉普拉斯分布"""
    rng = np.random.default_rng(random_state)
    u = rng.random(size) - 0.5
    return mu - b * np.sign(u) * np.log(1 - 2 * abs(u))


# ==============================
# 4. 正态分布采样
# ==============================

def normal_box_muller(size=1, random_state=None):
    """Box-Muller 方法"""
    rng = np.random.default_rng(random_state)
    u1 = rng.random(size)
    u2 = rng.random(size)
    z0 = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)
    z1 = np.sqrt(-2 * np.log(u1)) * np.sin(2 * np.pi * u2)
    return np.concatenate([z0, z1])[:size]


def normal_central_limit(size=1, random_state=None):
    """中心极限定理法 (12个均匀分布相加)"""
    rng = np.random.default_rng(random_state)
    return np.sum(rng.random((size, 12)), axis=1) - 6


def normal_inverse(size=1, random_state=None):
    """逆变换采样 - 正态分布 (用 SciPy 的 ppf)"""
    rng = np.random.default_rng(random_state)
    u = rng.random(size)
    return stats.norm.ppf(u)


def normal_ziggurat(size=1, random_state=None):
    """Ziggurat 算法（NumPy 内置实现）"""
    rng = np.random.default_rng(random_state)
    return rng.normal(size=size)


def split_samples(samples, n1, n2, n3, seed=None):
    """
    从给定的样本数组中划分为三组（不重叠）
    """
    N = len(samples)
    assert n1 + n2 + n3 <= N, "划分总数超过样本数！"

    rng = np.random.default_rng(seed)
    indices = rng.permutation(N)

    idx1 = indices[:n1]
    idx2 = indices[n1:n1 + n2]
    idx3 = indices[n1 + n2:n1 + n2 + n3]

    return samples[idx1], samples[idx2], samples[idx3]
