"""

Linear Model Predictive Control

% A function to design an Model Predictive Controller (MPC)
% The matrices represent the offline calculations used in the online
% optimization. Disturbance is included but can be left unused.
% Date: 29-03-2020
% Author: Anders Hilmar Damm Andersen

% The state-space matrices (without the D) is A, B, Cz and G.
% N: Horizon
% Qz: State weight matrix
% S: input rate weight matrix

"""

import numpy as np
import math
from scipy import linalg
from qpsolvers import solve_qp

class linearMPC():

	def __init__(self, A, B, C, G, N, Qz, S, umin = None, umax = None, dumin = None, dumax = None):

		self.A, self.B, self.C, self.G = A, B, C, G
		self.N, self.Qz, self.S = N, Qz, S

		# Input bounds and rate contraints
		self.umin, self.umax, self.dumin, self.dumax = umin, umax, dumin, dumax

		# Dimentions
		self.DimA, self.DimB, self.DimC, self.DimG = np.shape(A)[0], np.shape(B)[1], np.shape(C)[0], np.shape(G)[1]
		# Prediction matrices
		self.Phi, self.Phi_d, self.Gamma, self.Gamma_d = self.predict()


		# Weight matrices
		self.Q = np.kron(np.eye(N),Qz); # Weight matrix for horizon: ||z-r||^2_Q

		# Creating cost function for ||delta_u||^2_S
		self.Hs = np.kron(np.eye(N),2*S)+np.kron(np.diag(np.ones(N-1), k=-1), -S )+np.kron(np.diag(np.ones(N-1), k=1), -S )

		H = self.Gamma.T.dot(self.Q).dot(self.Gamma) + self.Hs
		self.H = 0.5*(H+H.T)

		self.M_x0 = self.Gamma.T.dot(self.Q).dot(self.Phi)
		self.M_R = -self.Gamma.T.dot(self.Q)
		self.M_u = -np.vstack((self.S,np.zeros((2*(self.N-1),self.DimB))))
		self.M_D = self.Gamma.T.dot(self.Q).dot(self.Gamma_d)
	
		L_x0 = -self.inv(self.H).dot(self.M_x0)
		L_R = -self.inv(self.H).dot(self.M_R)
		L_u = -self.inv(self.H).dot(self.M_u)
		L_D = -self.inv(self.H).dot(self.M_D)

		# --- Unconstrained MPC Control gains ---

		self.K_x0 = L_x0[0:self.DimB]
		self.K_R = L_R[0:self.DimB]
		self.K_u = L_u[0:self.DimB]
		self.K_D = L_D[0:self.DimG]

		# --- Input constrained variables --- 
		# Creating the intput rate constraint matrix from slide 71 "Lecture 07 MPC"
		self.Lambda = np.kron(np.eye(N),np.eye(self.DimB))+np.kron(np.diag(np.ones(N-1), k=-1), -np.eye(self.DimB));
		self.I0 = np.vstack((np.eye(self.DimB), np.zeros(((N-1)*2,self.DimB))));

	def predict(self):
		"""
		 ------- State-space prediction --------
		 Function to compute the prediction matrices Phi and Gamma for no
		 disturbance with a prediction horizon of N
		"""

		Gamma = np.zeros((self.DimC*self.N, self.DimB*self.N))   # Init Gamma to zero matrix
		Gamma_d = np.zeros((self.DimC*self.N, self.DimG*self.N)) # Init Gamma_d to zero matrix

		Phi = np.zeros((self.DimC*self.N, self.DimA))    # Init Phi to zero matrix
		Phi_d = np.zeros((self.DimC*self.N, self.DimG))  # Init Phi_d to zero matrix


		# --- Creating Gamma and Gamma_d ---
		a = 1
		for n in range(1, self.N+1):
			for k in range(1, self.N+1):
				Gamma[self.DimC*(n-1):self.DimC*n, self.DimB*(k-1):self.DimB*k] = self.Markov(self.A, self.B, self.C, a-k+1)
			a+=1
		a = 1
		for n in range(1, self.N+1):
			for k in range(1, self.N+1):
				Gamma_d[self.DimC*(n-1):self.DimC*n, self.DimG*(k-1):self.DimG*k] = self.Markov(self.A, self.G, self.C, a-k+1)
			a+=1

		# --- Creating Phi and Phi_d ---

		for n in range(1, self.N+1):
			Phi[self.DimC*(n-1):self.DimC*n] = self.C.dot(np.linalg.matrix_power(self.A, n))

		for k in range(1, self.N+1):
			Phi_d[self.DimC*(k-1):self.DimC*k] = self.C.dot(np.linalg.matrix_power(self.A, k-1)).dot(self.G)

		return Phi, Phi_d, Gamma, Gamma_d

	def Markov(self, A, B, C, i):
		if (i > 0):
			H = C.dot(np.linalg.matrix_power(A, i-1)).dot(B)
		else:
			H = np.zeros((np.shape(C)[0], np.shape(B)[1]))
		return H


	def show_gains(self):
		print("K_x0 = {}, K_R ={}, K_u ={}, K_D = {}".format(self.K_x0, self.K_R, self.K_u, self.K_D))


	def u_prim(self, r, x, u_old, d):

		# Input constrained MPC
		if all(v is not None for v in [self.umin, self.umax, self.dumin, self.dumax]):
			g = self.M_x0.dot(x)+self.M_R.dot(r)+self.M_u.dot(u_old)+self.M_D.dot(d)

			u = self.qpsolver(self.H, g, np.tile(self.umin,(self.N,1)), np.tile(self.umax,(self.N,1)), self.Lambda, np.tile(self.dumin,(self.N,1))+self.I0.dot(u_old), np.tile(self.dumax,(self.N,1))+self.I0.dot(u_old))
			u_prim =np.array([[u[0],u[1]]]).T
			return u_prim
		# Unconstrained MPC
		return self.K_x0.dot(x)+self.K_R.dot(r)+self.K_u.dot(u_old)+self.K_D.dot(d)

	def ref_con(self, ref):
		ref_con = np.zeros((self.DimC*self.N, 1))
		
		for n in range(1, self.N+1):
			ref_con[(n-1)*self.DimC:n*self.DimC,:] = ref
		return ref_con

	def dist_con(self, dist):
		dist_con = np.zeros((self.DimG*self.N, 1))
		
		for n in range(1, self.N+1):
			dist_con[(n-1)*self.DimG:n*self.DimG,:] = dist
		return dist_con

	@staticmethod
	def inv(M):
		# Function to calculate the inverse of M for either
		# scalar of matrices
		if isinstance(M, np.ndarray):
			return linalg.inv(M)
		return np.reciprocal(M)

	@staticmethod
	def qpsolver(H, g, l, u, A, bl, bu):
		"""
		% -------- QP -------------
		% min_x    phi = 1/2 x'Hx + g'x
		% s.t.  
		% l<=x<=u
		% bl <=Ax<=bu
		% -------------------------
		% Convert bl <=Ax<=bu to [A;-A] <= [bu;bl]
		"""
		
		b = np.squeeze(np.vstack((bu,-bl)))
		At = np.vstack((A, -A))
		return solve_qp(H, np.squeeze(g), At, b, None, None, np.squeeze(l), np.squeeze(u))

if __name__ == "__main__":


	# Four tank system linear discrete-time State-space model with sampling time ts = 1s
	A = np.array([[0.976720390386899 ,                  0.0 ,  0.041767337660069 ,                  0.0],
		[0.0,    0.979210750341296 ,        0.0 ,   0.049969586167812],
		[0.0     ,    0.0  ,  0.957735236905049      ,   0.0],
		[0.0  ,   0.0  ,      0.0   , 0.949499104511771]])

	B = np.array([[ 0.494157249977014 ,        0.010558870576226],
				[0.011380653647353  , 0.544262938845320],
				[0.0   , 0.489357763351457],
				[0.438539165731771,         0.0]])
	C = np.array([[ 0.002630660293103 , 0.0,0.0,0.0],[0.0, 0.002630660293103  ,0.0,0.0]])
	G = np.array([[0.021117741152452   ,        0.0],
				[0.0  ,  0.025290341438563],
				[0.978715526702914   , 0.0],
				[0.0,         0.974531479403935]])
	N = 2

	Qz = np.diag([1000,1000])
	S = np.diag([0.1,0.1])

	umin = np.array([[-2,-2]]).T
	umax = np.array([[2,2]]).T

	dumin = np.array([[-2,-2]]).T
	dumax = np.array([[2,2]]).T
	MPC = linearMPC(A, B, C, G, N, Qz, S, umin, umax, dumin, dumax)
	MPC.show_gains()
	r = np.array([[2,3]]).T
	dist = np.array([[0,0]]).T
	MPC.ref_con(r)
	ref = MPC.ref_con(r)
	dist = MPC.dist_con(dist)
	x_hat = np.array([[1,2,3,4]]).T
	u_old = np.array([[1,2]]).T
	F_hat = MPC.u_prim(ref, x_hat, u_old, dist)