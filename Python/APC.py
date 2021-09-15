"""

APC module


"""


import numpy as np
import math
from scipy import linalg
from qpsolvers import solve_qp
import OPCUA_SQL as db
from Parameters import *
from casadi import *
import scipy.linalg as sci
import time

class PID():

	def __init__(self, Kp, tau_i, tau_d, N, Ts, umin = None, umax = None):

		self.Kp, self.tau_i, self.tau_d, self.N, self.Ts = Kp, tau_i, tau_d, N, Ts

		# Input bounds and rate contraints
		self.umin, self.umax = umin, umax

		# Initiate variables
		# Integrator init
		self.y1, self.y2 = 0.0, 0.0
		self.I1, self.I2 = 0.0, 0.0
		self.D1, self.D2 = 0.0, 0.0




	def show_gains(self):
		print("Kp = {}, tau_i ={}, tau_d ={}, N = {}".format(self.Kp, self.tau_i, self.tau_d, self.N))

	def run(self, r, y):

		e1, e2 = r[0]-y[0], r[1]-y[1]
		P1, P2 = self.Kp[0]*e1, self.Kp[1]*e2
		self.D1 = self.tau_d[0]/(self.tau_d[0]+self.N*self.Ts)*self.D1-self.Kp[0]*self.tau_d[0]*self.N/(self.tau_d[0]+self.N*self.Ts)*(y[0]-self.y1)
		self.D2 = self.tau_d[1]/(self.tau_d[1]+self.N*self.Ts)*self.D2-self.Kp[1]*self.tau_d[1]*self.N/(self.tau_d[1]+self.N*self.Ts)*(y[1]-self.y2)

		F1 = P1 + self.I1 + self.D1
		F2 = P2 + self.I2 + self.D2

		satF1, satF2 = F1, F2
		if all(v is not None for v in [self.umin, self.umax]):
			satF1 = self.sat(F1, self.umin, self.umax)
			satF2 = self.sat(F2, self.umin, self.umax)
		#Update prev output
		self.y1, self.y2 = y[0], y[1]

		# Update Integrator w. anti integrator windup
		self.I1 = self.I1+self.Ts*self.Kp[0]/(self.tau_i[0])*e1+self.Ts*self.windup_cal(satF1, F1, 0.5*self.tau_i[0])
		self.I2 = self.I2+self.Ts*self.Kp[1]/(self.tau_i[1])*e2+self.Ts*self.windup_cal(satF2, F2, 0.5*self.tau_i[1])

		return satF1, satF2
	@staticmethod
	def sat(x, xmin, xmax):
		if (x > xmax):
			x = xmax
		if (x < xmin):
			x = xmin
		return x

	@staticmethod
	def windup_cal(satu, u, tau_t):
		return (satu-u)/tau_t
		

class LMPC():
	"""

	Linear Model Predictive Control

	% A function to design a linear Model Predictive Controller (MPC)
	% The matrices represent the offline calculations used in the online
	% optimization. Disturbance is included but can be left unused.
	% Date: 29-03-2020
	% Author: Anders Hilmar Damm Andersen

	% The state-space matrices (without the D) is A, B, Cz and G.
	% N: Horizon
	% Qz: State weight matrix
	% S: input rate weight matrix

	"""
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
			u_prim = np.array([[u[0],u[1]]]).T
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


class NMPC():
	"""

	Nonlinear MPC based on Direct Multiple Shooting

	This NMPC toolbox is based on CasADi for the MFTS

	Date:08/04/2021
	Author: Anders Hilmar Damm Andersen


	"""
	def __init__(self, C, N, Q, S, Ts, umin = None, umax = None):

		self.C, self.N, self.Q, self.S, self.Ts= C, N, Q, S, Ts

		# Input bounds
		self.umin, self.umax = umin, umax

		# States in symbolic form
		x = SX.sym('x',4)
		self.n_states = x.numel()

		# Input signals in symbolic variables
		F= SX.sym('F',2)
		self.n_controls = F.numel()

		# Dimentions of reference signals
		self.n_ref = np.shape(C)[0]

		# Four tank system model
		qin1 = gamma1*F[0]         # Valve 1 to tank 1 [cm3/s]
		qin2 = gamma2*F[1]        # Valve 2 to tank 2 [cm3/s]
		qin3 = (1-gamma2)*F[1]    #+F_d(2) # Valve 2 to tank 3 [cm3/s] Disturbances are not added so far
		qin4 = (1-gamma1)*F[0]     #+F_d(1) # Valve 1 to tank 4 [cm3/s]

		h1 = x[0]/(rho*A1) # Liquid level in each tank 1 [cm]
		h2 = x[1]/(rho*A2) # Liquid level in each tank 2 [cm]
		h3 = x[2]/(rho*A3) # Liquid level in each tank 3 [cm]
		h4 = x[3]/(rho*A4) # Liquid level in each tank 1 [cm]


		q1 = a1*sqrt(2*g*h1) # Outflow from each tank 1 [cm3/s]
		q2 = a2*sqrt(2*g*h2) # Outflow from each tank 1 [cm3/s]
		q3 = a3*sqrt(2*g*h3) # Outflow from each tank 1 [cm3/s]
		q4 = a4*sqrt(2*g*h4) # Outflow from each tank 1 [cm3/s]

		# Differential equations, mass balances

		xdot1 = rho*(qin1+q3-q1) # Tank 1
		xdot2 = rho*(qin2+q4-q2) # Tank 2
		xdot3 = rho*(qin3-q3)    # Tank 3
		xdot4 = rho*(qin4-q4)    # Tank 4

		x_dot = vertcat(
			xdot1,
			xdot2,
			xdot3,
			xdot4
		)

		# Symbolic Matrix used to store all predictions in the states over the horizon
		X = SX.sym('X', self.n_states, self.N + 1)

		# Symbolic Matrix to store all inputs over the horizon
		U = SX.sym('U', self.n_controls, self.N)

		# Matrix to store initial state and reference signals
		P = SX.sym('P', self.n_states + self.n_ref)

		# Weight matrix
		Q = diagcat(self.Q[0,0], self.Q[1,1])

		# controls weights matrix
		S = diagcat(self.S[0,0], self.S[1,1])

		# Create a functions to calculate the xdot from symbolic states and inputs
		f = Function('f', [x, F], [x_dot])


		# Cost function J
		J = 0
		MScont = X[:, 0] - P[:self.n_states]
		for k in range(0, self.N):
			x_cur, u_cur = X[:, k], U[:, k]
			J = J + (self.C@x_cur-P[self.n_states:self.n_states+2]).T@Q@(self.C@x_cur-P[self.n_states:self.n_states+2]) + u_cur.T@S@u_cur
			x_dot = f(x_cur, u_cur)
			x_next = x_cur + x_dot*self.Ts
			MScont = vertcat(MScont, X[:,k+1]-x_next)
				
		# Reshaping the optimisation variables into a vector
		OPTvar_MS = vertcat(X.reshape((-1,1)), U.reshape((-1, 1)))

		nlp_prob = {
			'f': J,
			'x': OPTvar_MS,
			'g': MScont,
			'p': P
		}

		opts = {
			'ipopt': {
				'max_iter': 100,
				'print_level': 0,
			},
			'print_time': 0
		}


		# Constraints on optimisation variables. This is constraints on states and inputs
		lbx = DM.zeros((self.n_states*(self.N+1)+self.n_controls*(self.N), 1))
		ubx = DM.zeros((self.n_states*(self.N+1)+self.n_controls*(self.N), 1))

		# Constraints on the states. No upper constraint is set 
		# but lower constraints are used to prevent invalid solution due to sqrt()
		for i in range(self.n_states*(self.N+1)):
			lbx[i] = 0.15
		for i in range(self.n_states*(self.N+1)):
			ubx[i] = inf
		
		# Input constraints
		for i in range(self.n_states*(self.N+1), (self.n_states*(self.N+1)+self.n_controls*(self.N))):
			lbx[i] = 0.0
		for i in range(self.n_states*(self.N+1), (self.n_states*(self.N+1)+self.n_controls*(self.N))):
			ubx[i] = 400


		# Equality constraints for the state trajectory. Used to preserve continuity.
		lbg = DM.zeros((self.n_states*(N+1), 1))
		ubg = DM.zeros((self.n_states*(N+1), 1))

		
		self.solver = nlpsol('solver', 'ipopt', nlp_prob, opts)

		self.args = {
			'lbx': lbx,
			'ubx': ubx,
			'lbg': DM.zeros((self.n_states*(N+1), 1)),
			'ubg': DM.zeros((self.n_states*(N+1), 1))
		}


	def u_prim(self, r, x, opt_var_0):
		self.args['p'] = vertcat(
			x,    # initial state
			r,    # reference
		)
		# optimization variable current state
		self.args['x0'] = reshape(opt_var_0, self.n_states*(self.N+1) + self.n_controls*self.N, 1)
		#print("self.args['x0'] = ",self.args['x0'])
		sol = self.solver(
			x0=self.args['x0'],
			lbx=self.args['lbx'],
			ubx=self.args['ubx'],
			lbg = self.args['lbg'],
			ubg = self.args['ubg'],
			p=self.args['p']
		)
		u = reshape(sol['x'][self.n_states * (self.N + 1):],self.n_controls, self.N).T
		X = reshape(sol['x'][:self.n_states * (self.N + 1)],self.n_states, self.N+1).T
		u_mpc = u[0,:]
		u1, u2 = float(str(u_mpc[0]).lstrip('[').rstrip(']')), float(str(u_mpc[1]).lstrip('[').rstrip(']')), 
		return np.array([[u1, u2]]).T, u, X

	def opt_var_shift(self, u0, X0):
		u_shift = vertcat(u0[1:,:], u0[-1,:])
		X_shift = vertcat(X0[1:,:], X0[-1,:])
		opt_var_0 = vertcat(reshape(X_shift.T, self.n_states*(self.N+1),1), reshape(u_shift.T, self.n_controls*self.N,1))
		return opt_var_0


# ------------------ State Estimators --------------------

class DKF():

	def __init__(self, Ts, x_0, u_0, d_0, dist_est):

		self.C = np.array([[1/(rho*A1), 0, 0, 0], [0, 1/(rho*A2),0 ,0]])
		self.Ts = Ts
		self.x_0 = x_0
		self.x_hat = np.array([[0,0,0,0]]).T
		self.u_0   = u_0
		self.y_0 = self.C@x_0
		self.e = np.array([[0.0, 0.0]]) # Innovation
		self.d_0 = d_0
		# Symbolic state transistion:
		self.f, x, F, Fd = self.sys()
		self.ff = Function('ff',[x, F, Fd],[self.f])

		# Initialised symbolic Jacobian: df(x,u,t)/dx
		self.jacA = jacobian(self.f,x)
		A_k = Function('Jr',[x, F, Fd],[self.jacA])
		self.Ac = A_k(self.x_0, u_0, d_0) # Creating A matrix at operating point

		self.jacB = jacobian(self.f,F)
		B_k = Function('Jr',[x, F, Fd],[self.jacB])
		self.Bc = B_k(self.x_0, u_0, d_0)

		self.jacG = jacobian(self.f,Fd)
		G_k = Function('Jr',[x, F, Fd],[self.jacG])
		self.Gc = G_k(self.x_0, u_0, d_0)

		# C2d with zoh:
		self.Ad, self.Bd = self.c2dzoh(self.Ac, self.Bc, self.Ts)
		_, self.Gd = self.c2dzoh(self.Ac, self.Gc, self.Ts)
		self.Gd_kf = np.diag([1, 1, 1, 1])
		# Covariance matrix P 
		self.P = SX.eye(4)*100 # initialisation of P

		self.I = SX.eye(4) # Identity

		# Process noise and measurement noise matrices
		self.Q, self.R = SX.eye(4)*1, SX.eye(2)*0.05

		if dist_est:
			self.Ad = np.vstack((np.hstack((self.Ad, self.Gd)), np.hstack((np.zeros((2,4)),np.eye(2)))))
			self.Bd = np.vstack((self.Bd, np.zeros((2,2))))
			self.C  = np.hstack((self.C, np.zeros((2,2))))
			self.Gd_kf = np.vstack((np.hstack((self.Gd_kf, np.zeros((4,2)))), np.hstack((np.zeros((2,4)),np.eye(2)))))
			self.Q, self.R = diag(SX([1, 1, 1, 1, 0.1, 0.1])), SX.eye(2)*0.05
			self.P = SX.eye(6)*100 # initialisation of P
			self.I = SX.eye(6) # Identity
			self.x_hat = np.array([[0,0,0,0,0,0]]).T
			self.x_0 = np.vstack((x_0, d_0))


	def predict(self, F):
		# --- Time-update ---

		# Mean update
		F_bar = np.subtract(F, self.u_0)
		self.x_hat = self.Ad@self.x_hat+self.Bd@F_bar

		# Covariance update
		self.P =self.Ad@self.P@self.Ad.T+self.Gd_kf@self.Q@self.Gd_kf.T
	def correct(self, y):
		# --- Measurement-update ---
		y_bar = np.subtract(y, self.y_0)
		R_ek = self.R+self.C@self.P@self.C.T
		K    = self.P@self.C.T@inv(R_ek) # Kalman gain
		self.e    = y_bar - self.C@self.x_hat # Innovation

		self.x_hat = self.x_hat + K@self.e
		self.P = self.P-K@R_ek@K.T

	def x_real(self, dist):
		if dist == 1:
			np.add(self.x_hat, self.x_0)
		return np.add(self.x_hat, self.x_0)

	def get_inno(self):
		return self.e

	def get_y_hat(self):
		return self.C@self.x_hat

	def get_x_hat(self, dist):

		x_hat = self.x_real(dist)

		if dist == 1:
			return str(x_hat[0]).lstrip('[').rstrip(']'), \
			str(x_hat[1]).lstrip('[').rstrip(']'), str(x_hat[2]).lstrip('[').rstrip(']'), \
				 str(x_hat[3]).lstrip('[').rstrip(']'), str(x_hat[4]).lstrip('[').rstrip(']'), \
				  str(x_hat[5]).lstrip('[').rstrip(']')
		return str(x_hat[0]).lstrip('[').rstrip(']'), \
			str(x_hat[1]).lstrip('[').rstrip(']'), str(x_hat[2]).lstrip('[').rstrip(']'), \
				 str(x_hat[3]).lstrip('[').rstrip(']')

	def c2dzoh(self, Ac, Bc,Ts):
		# continuous to discrete time with zero order hold method
		nx, nu = Bc.shape
		M = np.vstack((np.hstack((Ac, Bc)), np.hstack((np.zeros((nu, nx)),np.zeros((nu, nu))))))
		Phi = sci.expm(M*Ts)
		A, B = Phi[0:nx,0:nx], Phi[0:nx, nx:nx+nu]
		return A, B

	def sys(self):
		x = SX.sym('x',4)
		# Controls symbolic variables
		F = SX.sym('F',2)
		Fd = SX.sym('Fd',2)
		# Four tank system model
		qin1 = gamma1*F[0]         # Valve 1 to tank 1 [cm3/s]
		qin2 = gamma2*F[1]        # Valve 2 to tank 2 [cm3/s]
		qin3 = (1-gamma2)*F[1]+Fd[1]    #+F_d(2) # Valve 2 to tank 3 [cm3/s] Disturbances are not added so far
		qin4 = (1-gamma1)*F[0]+Fd[0]     #+F_d(1) # Valve 1 to tank 4 [cm3/s]

		h1 = x[0]/(rho*A1) # Liquid level in each tank 1 [cm]
		h2 = x[1]/(rho*A2) # Liquid level in each tank 2 [cm]
		h3 = x[2]/(rho*A3) # Liquid level in each tank 3 [cm]
		h4 = x[3]/(rho*A4) # Liquid level in each tank 1 [cm]


		q1 = a1*sqrt(2*g*h1) # Outflow from each tank 1 [cm3/s]
		q2 = a2*sqrt(2*g*h2) # Outflow from each tank 1 [cm3/s]
		q3 = a3*sqrt(2*g*h3) # Outflow from each tank 1 [cm3/s]
		q4 = a4*sqrt(2*g*h4) # Outflow from each tank 1 [cm3/s]

		# Differential equations, mass balances

		xdot1 = rho*(qin1+q3-q1) # Tank 1
		xdot2 = rho*(qin2+q4-q2) # Tank 2
		xdot3 = rho*(qin3-q3)    # Tank 3
		xdot4 = rho*(qin4-q4)    # Tank 4

		x_dot = vertcat(
			xdot1,
			xdot2,
			xdot3,
			xdot4
		)
		return x_dot, x, F, Fd

class CDEKF():
	# Continuous-Discrete Time Extended Kalman Filter

	def __init__(self, Ts, x_hat_0):
		self.C = np.array([[1/(rho*A1), 0, 0, 0], [0, 1/(rho*A2),0 ,0]])
		self.G = np.diag([1, 1, 1, 1])
		self.Ts = Ts
		self.x_hat = x_hat_0 
		self.e = np.array([[0.0, 0.0]])

		# Symbolic state transistion:
		self.f, x, F = self.sys()
		self.ff = Function('ff',[x, F],[self.f])

		# Initialised symbolic Jacobian: df(x,u,t)/dx
		self.jac = jacobian(self.f,x)
		self.A_k = Function('Jr',[x, F],[self.jac])

		# Covariance matrix P 
		self.P = SX.eye(4)*100 # initialisation of P

		self.I = SX.eye(4) # Identity


		# Process noise and measurement noise matrices
		self.Q, self.R = SX.eye(4)*1, SX.eye(2)*0.05

	def predict(self, F):
		# --- Time-update ---

		# Mean update
		self.x_hat = self.x_hat + self.Ts*self.ff(self.x_hat, F)

		# Covariance update
		self.P = (self.I+self.Ts*self.A_k(self.x_hat,F))@self.P@(self.I+self.Ts*self.A_k(self.x_hat,F)).T+self.Ts*self.G@self.Q@self.G

	def correct(self, y):
		# --- Measurement-update ---

		R_ek = self.R+self.C@self.P@self.C.T
		K    = self.P@self.C.T@inv(R_ek) # Kalman gain
		self.e    = y - self.C@self.x_hat # Innovation


		self.x_hat = self.x_hat + K@self.e
		self.P = (self.I-K@self.C)@self.P@(self.I-K@self.C).T+K@self.R@K.T

	def get_y_hat(self):
		return self.C@self.x_hat

	def get_x_hat(self):
		return str(self.x_hat[0]).lstrip('[').rstrip(']'), \
			str(self.x_hat[1]).lstrip('[').rstrip(']'), str(self.x_hat[2]).lstrip('[').rstrip(']'), \
				 str(self.x_hat[3]).lstrip('[').rstrip(']')

	def get_inno(self):
		return self.e

	def sys(self):
		x = SX.sym('x',4)
		# Controls symbolic variables
		F = SX.sym('F',2)

		# Four tank system model
		qin1 = gamma1*F[0]         # Valve 1 to tank 1 [cm3/s]
		qin2 = gamma2*F[1]        # Valve 2 to tank 2 [cm3/s]
		qin3 = (1-gamma2)*F[1]    #+F_d(2) # Valve 2 to tank 3 [cm3/s] Disturbances are not added so far
		qin4 = (1-gamma1)*F[0]     #+F_d(1) # Valve 1 to tank 4 [cm3/s]

		h1 = x[0]/(rho*A1) # Liquid level in each tank 1 [cm]
		h2 = x[1]/(rho*A2) # Liquid level in each tank 2 [cm]
		h3 = x[2]/(rho*A3) # Liquid level in each tank 3 [cm]
		h4 = x[3]/(rho*A4) # Liquid level in each tank 1 [cm]


		q1 = a1*sqrt(2*g*h1) # Outflow from each tank 1 [cm3/s]
		q2 = a2*sqrt(2*g*h2) # Outflow from each tank 1 [cm3/s]
		q3 = a3*sqrt(2*g*h3) # Outflow from each tank 1 [cm3/s]
		q4 = a4*sqrt(2*g*h4) # Outflow from each tank 1 [cm3/s]

		# Differential equations, mass balances

		xdot1 = rho*(qin1+q3-q1) # Tank 1
		xdot2 = rho*(qin2+q4-q2) # Tank 2
		xdot3 = rho*(qin3-q3)    # Tank 3
		xdot4 = rho*(qin4-q4)    # Tank 4

		x_dot = vertcat(
			xdot1,
			xdot2,
			xdot3,
			xdot4
		)
		return x_dot, x, F

class getSSmatrices():

	def __init__(self, Ts, x_hat_0, u_0, d_0):

		self.C = np.array([[1/(rho*A1), 0, 0, 0], [0, 1/(rho*A2),0 ,0]])
		self.Ts = Ts
		self.x_hat_0 = x_hat_0
		self.u_0   = u_0

		# Symbolic state transistion:
		self.f, x, F, Fd = self.sys()
		self.ff = Function('ff',[x, F, Fd],[self.f])

		# Initialised symbolic Jacobian: df(x,u,t)/dx
		self.jacA = jacobian(self.f,x)
		A_k = Function('Jr',[x, F, Fd],[self.jacA])
		self.Ac = A_k(self.x_hat_0, u_0, d_0) # Creating A matrix at operating point

		self.jacB = jacobian(self.f,F)
		B_k = Function('Jr',[x, F, Fd],[self.jacB])
		self.Bc = B_k(self.x_hat_0, u_0, d_0)

		self.jacG = jacobian(self.f,Fd)
		G_k = Function('Jr',[x, F, Fd],[self.jacG])
		self.Gc = G_k(self.x_hat_0, u_0, d_0)


		#print(self.Ac, self.Bc)

		# C2d with zoh:

		self.Ad, self.Bd = self.c2dzoh(self.Ac, self.Bc, self.Ts)
		# C2d with zoh:
		_, self.Gd = self.c2dzoh(self.Ac, self.Gc, self.Ts)
  
	def getABG(self):
		return self.Ad, self.Bd, self.Gd

	def getys(self):
		return self.C@self.x_hat_0

	def c2dzoh(self, Ac, Bc,Ts):
		# continuous to discrete time with zero order hold method
		nx, nu = Bc.shape
		M = np.vstack((np.hstack((Ac, Bc)), np.hstack((np.zeros((nu, nx)),np.zeros((nu, nu))))))
		Phi = sci.expm(M*Ts)
		A, B = Phi[0:nx,0:nx], Phi[0:nx, nx:nx+nu]
		return A, B
	def sys(self):
		x = SX.sym('x',4)
		# Controls symbolic variables
		F = SX.sym('F',2)

		Fd = SX.sym('Fd',2)

		# Four tank system model
		qin1 = gamma1*F[0]         # Valve 1 to tank 1 [cm3/s]
		qin2 = gamma2*F[1]        # Valve 2 to tank 2 [cm3/s]
		qin3 = (1-gamma2)*F[1]+Fd[1]    #+F_d(2) # Valve 2 to tank 3 [cm3/s] Disturbances are not added so far
		qin4 = (1-gamma1)*F[0]+Fd[0]     #+F_d(1) # Valve 1 to tank 4 [cm3/s]

		h1 = x[0]/(rho*A1) # Liquid level in each tank 1 [cm]
		h2 = x[1]/(rho*A2) # Liquid level in each tank 2 [cm]
		h3 = x[2]/(rho*A3) # Liquid level in each tank 3 [cm]
		h4 = x[3]/(rho*A4) # Liquid level in each tank 1 [cm]


		q1 = a1*sqrt(2*g*h1) # Outflow from each tank 1 [cm3/s]
		q2 = a2*sqrt(2*g*h2) # Outflow from each tank 1 [cm3/s]
		q3 = a3*sqrt(2*g*h3) # Outflow from each tank 1 [cm3/s]
		q4 = a4*sqrt(2*g*h4) # Outflow from each tank 1 [cm3/s]

		# Differential equations, mass balances

		xdot1 = rho*(qin1+q3-q1) # Tank 1
		xdot2 = rho*(qin2+q4-q2) # Tank 2
		xdot3 = rho*(qin3-q3)    # Tank 3
		xdot4 = rho*(qin4-q4)    # Tank 4

		x_dot = vertcat(
			xdot1,
			xdot2,
			xdot3,
			xdot4
		)
		return x_dot, x, F, Fd

class APC():

	def __init__(self, conn, rsid, Ts, x0 = [2568.8, 3837.4, 285.4, 507.4], u0 = [200.0, 200.0], d0 = [0.0, 0.0]):

		# Parameters for mode and database
		self.conn, self.rsid = conn, rsid
		self.Controlmode = db.getControlmode(conn, rsid)[0]
		self.Estimationmode = 1

		# System Parameters
		self.Ts = Ts
		self.x0 = np.array([[2568.8, 3837.4, 285.4, 507.4]]).T
		self.u0 = np.array([[200, 200]]).T
		self.u = self.u0
		self.u_old = self.u0
		self.d0 = np.array([[0.0, 0.0]]).T
		self.C = np.array([[1/(rho*A1), 0, 0, 0], [0, 1/(rho*A2),0 ,0]])
		ss = getSSmatrices(self.Ts, self.x0, self.u0, self.d0)

		self.A, self.B, self.G = ss.getABG()
		self.y0 = ss.getys()
		print("Operating point: u0 = {}, y0 = {}".format(self.u0, self.y0))
		# Contraints on the input
		umin = np.array([[-150,-200]]).T
		umax = np.array([[250,200]]).T

		dumin = np.array([[-100,-100]]).T
		dumax = np.array([[100,100]]).T

		# Tuning parameters for MPC/NMPC
		self.N = 200

		self.Qz = np.diag([1000,1000])
		self.S = np.diag([0.001, 0.001])


		## ---- Control Init ----
		self.u_old_nmpc = np.vstack((np.full(self.N,db.getu1(conn, rsid)[0]),np.full(self.N, db.getu1(conn, rsid)[0])))

		# Init Controllers
		self.PID = PID([45.0,40.0], [30,35], [10, 8], 5, self.Ts, 0, 400)
		self.LMPC = LMPC(self.A, self.B, self.C, self.G, self.N, self.Qz, self.S, umin, umax, dumin, dumax)
		self.NMPC = NMPC(self.C, self.N, self.Qz, self.S, self.Ts)
		u0 = np.full((2, self.N), 200)
		X0 = repmat(self.x0, 1, self.N+1)         # initial state full
		self.opt_var_0 = vertcat(reshape(X0, 4*(self.N+1),1), reshape(u0, 2*self.N,1))

		# Init Estimator
		if(self.Estimationmode == 0):
			# Discrete time Kalman Filter
			self.kf  = DKF(self.Ts, self.x0, self.u0, self.d0, 0)
		elif(self.Estimationmode == 1):
			# Continuous-Discrete time Extended Kalman FIlter
			self.kf = CDEKF(self.Ts, self.x0)

		print("APC Module initialised")

	def control(self):
	# Read output

		r1, r2 = db.getr1(self.conn, self.rsid)[0], db.getr2(self.conn, self.rsid)[0]
		y1, y2 = db.gety1(self.conn, self.rsid)[0], db.gety2(self.conn, self.rsid)[0]

		r = [r1, r2]
		# Read time from Process
		ts = db.gety1(self.conn, self.rsid)
		cm = db.getControlmode(conn, rsid)[0]
		
		# Measurement update of Kalman Filter
		self.kf.correct(np.array([[y1, y2]]).T)
		# Init Estimator
		if(self.Estimationmode == 0):
			# Discrete time Kalman Filter
			x_hat1, x_hat2, x_hat3, x_hat4 = self.kf.get_x_hat(0)
		elif(self.Estimationmode == 1):
			# Continuous-Discrete time Extended Kalman FIlter
			x_hat1, x_hat2, x_hat3, x_hat4 = self.kf.get_x_hat()
		

		# Insert x_hat in database

		# Create the x_hat vector used by the controllers
		x_hat_KF = np.array([[float(x_hat1), float(x_hat2), float(x_hat3), float(x_hat4)]]).T

		if(db.getControlmode(self.conn, self.rsid)[0] == 0):
			u1, u2 = db.getu1(self.conn, self.rsid)[0], db.getu2(self.conn, self.rsid)[0]
			self.u = [u1,u2]
			db.insertu1(self.conn, self.rsid, ts[1], u1, "PID", cm)
			db.insertu2(self.conn, self.rsid, ts[1], u2, "PID", cm)
			print("Controlmode = Manual Control", end='\r')

		if(db.getControlmode(self.conn, self.rsid)[0] == 1):
			u1, u2 = self.PID.run(r, [y1,y2])
			self.u = [u1,u2]
			db.insertu1(self.conn, self.rsid, ts[1], u1, "PID", cm)
			db.insertu2(self.conn, self.rsid, ts[1], u2, "PID", cm)
			print("Controlmode = PID", end='\r')

		if(db.getControlmode(self.conn, self.rsid)[0] == 2):
			# Linear MPC
			ref = np.array([[r[0]-self.y0[0], r[1]-self.y0[1]]]).T # Reference signals in deviations from operating point
			ref = self.LMPC.ref_con(ref)   # Reshape reference vector
			dist = np.array([[0.0, 0.0]]).T
			dist = self.LMPC.dist_con(dist) # Reshape disturbance vector
			x_hat = np.subtract(x_hat_KF, self.x0) # Estimates in deviations from operating point
			u_hat = self.LMPC.u_prim(ref, x_hat, self.u_old, dist)
			self.u_old = u_hat
			self.u = np.add(u_hat, self.u0)
			db.insertu1(self.conn, self.rsid, ts[1], self.u[0][0], "LMPC", cm)
			db.insertu2(self.conn, self.rsid, ts[1], self.u[1][0], "LMPC", cm)
			print("Controlmode = LMPC", end='\r')

		if(db.getControlmode(self.conn, self.rsid)[0] == 3): 
			# Direct Multiple Shooting MPC (NMPC)
			self.u, u, X = self.NMPC.u_prim(r, x_hat_KF, self.opt_var_0)
			db.insertu1(self.conn, self.rsid, ts[1], self.u[0][0], "NMPC", cm)
			db.insertu2(self.conn, self.rsid, ts[1], self.u[1][0], "NMPC", cm)
			self.opt_var_0 = self.NMPC.opt_var_shift(u, X)

			print("Controlmode = NMPC", end='\r')
		
		# KF Time-update
		self.kf.predict(self.u)
		
if __name__ == "__main__":

	Ts = 1.0 # Sampling time of Controller
	conn = db.connect()
	rsid = db.getRSID(conn)
	print("APC RSID = {}".format(rsid))
	apc = APC(conn, rsid, Ts)

	print("Running APC Module")
	try:
		while True:
			apc.control()
			time.sleep(Ts-(time.time()%Ts))
	finally:
		print("Stopping APC module")