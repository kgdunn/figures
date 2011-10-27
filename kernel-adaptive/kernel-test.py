import numpy as np

N, K, A = 100, 10, 3

X = np.random.randn(N, K);
#X = np.arange(N*K).reshape(N, K)
X[:,3] = X[:,0] + X[:,1]
X[:,4] = X[:,2] + X[:,0]
X[:,5] = X[:,3] + X[:,1]
X[:,6] = X[:,4] + X[:,2]
X[:,7] = X[:,5] + X[:,3]
X[:,8] = X[:,6] + X[:,4]
X[:,9] = X[:,7] + X[:,5]

X = (X - np.mean(X, axis=0)) / np.std(X, axis=0, ddof=0)

U, S, V = np.linalg.svd(X)

P = V[0:A, :].T
T = U[:, 0:A] * S[0:A]

# Now compare the kernel method
# -----------------------------
covXTX = np.dot(X.T, X)
val, vec = np.linalg.eig(covXTX)
vec = np.real(vec)
vec = vec[:, np.argsort(val)[::-1]]  # sort eigenvectors from largest to smallest eigenvalue
P_kern = vec[:, range(0, A)]
T_kern = np.dot(X, P_kern)

np.linalg.norm(np.abs(P_kern) - np.abs(P))
np.linalg.norm(np.abs(T_kern) - np.abs(T))

# Now compare the kernel method
# -----------------------------
covXXT = np.dot(X, X.T)
val, vec = np.linalg.eig(covXXT)
val, vec = np.real(val), np.real(vec)
vec = vec[:, np.argsort(val)[::-1]]  # sort eigenvectors from largest to smallest eigenvalue
val = np.sort(val)[::-1]
T_wide = vec[:, 0:A] * np.sqrt(val[0:3])
P_wide = np.dot(np.linalg.inv(np.dot(T.T, T)),  np.dot(T_wide.T, X))
P_wide = P_wide.T
np.linalg.norm(np.abs(P_wide) - np.abs(P))
np.linalg.norm(np.abs(T_wide) - np.abs(T))