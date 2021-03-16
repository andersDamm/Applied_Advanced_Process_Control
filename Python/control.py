# The control package for 2021 Special course: Applied Advanced Process Control


import numpy as np
from scipy import linalg


def lqr(A, B, R1, R2, N=None):
	# Python implementation of the lqr() in MATLAB

	# Solve the continuous-time Riccati equation
	P = linalg.solve_continuous_are(A, B, R1, R2, None, N)
	#Compute the continuous-time LQR gain
	K = np.dot(inv(R2), (np.dot(B.T, P)))
	#Compute eigenvalues
	ev, _ = linalg.eig(A-B*K)
	return K, P, ev

def dlqr(F, G, R1, R2, N=None):
	# Python implementation of the dlqr() in MATLAB

	# Solve the discrete-time Riccati equation
	P = linalg.solve_discrete_are(F, G, R1, R2, None, N)
	print(P)
	#Compute the discrete-time LQR gain
	K = np.dot(np.dot(np.dot(inv(R2+np.dot(np.dot(G.T, P),G)), G.T), P), F)
	#Compute eigenvalues
	ev, _ = linalg.eig(F-G*K)
	return K, P, ev

def inv(M):
	# Function to calculate the inverse of M for either
	# scalar of matrices
	if isinstance(M, list):
		return linalg.inv(M)
	return np.reciprocal(M)

def main():
	a = np.array([[4, 3], [-4.5, -3.5]])
	b = np.array([[1], [-1]])
	q = np.array([[9, 6], [6, 4]])
	r = np.array([10.0])
	[K, P, ev] = dlqr(a,b,q,r)
	print(K)
	print(ev)
	


if __name__ == "__main__":
	main()
