"""

Simulation Server for FTS inspired by ESF


"""
import time
import numpy as np
from opcua import Server
from scipy.integrate import odeint


class FourTankSystem():

	def __init__(self, x0 = [0.0, 0.0, 0.0, 0.0]):

		self.x = np.array(x0)
		self.t = 0.0

		self.gamma = [0.6, 0.7]
		self.A = 380.0
		self.a = 1.2
		self.g = 981.0
		self.rho = 1.0

	def ode(self, x, t, u):

		qin = np.array([
			self.gamma[0]*u[0],
			self.gamma[1]*u[1],
			(1-self.gamma[1])*u[1],
			(1-self.gamma[0])*u[0],
			])

		h = x/(self.rho*self.A)
		qout = self.a*np.sqrt(2*self.g*h)

		qoutin = np.array([
			qout[2],
			qout[3],
			0,
			0
			])

		dxdt = self.rho*(qin+qoutin-qout)
		return dxdt

	def y(self, x):
		#Measurements
		return x/(self.rho*self.A)

	def step(self, dt, u):
		sol = odeint(self.ode, self.x, [0, dt], args=(u, ))
		self.x = sol[1]
		self.t += dt
		return self.x

if __name__ == "__main__":

	server = Server()
	url = "opc.tcp://localhost:4840/FTS"
	server.set_endpoint(url)
	addspace = server.register_namespace('http://dtu.dk')
	objects = server.get_objects_node()

	LT1 = objects.add_object(addspace,"LT1 (y1)")
	y1_meas = LT1.add_variable(addspace, "Height [cm]", 0.0)

	LT2 = objects.add_object(addspace,"LT2 (y2)")
	y2_meas = LT2.add_variable(addspace, "Height [cm]", 0.0)

	# Pump actuators
	P1 = objects.add_object(addspace,"P1 (F1)")
	F1_val = P1.add_variable(addspace, "Flow rate cm^3\s^{-1}", 200)
	P2 = objects.add_object(addspace,"P2 (F2)")
	F2_val = P2.add_variable(addspace, "Flow rate cm^3\s^{-1}", 200)

	y1_meas.set_writable()
	y2_meas.set_writable()
	F1_val.set_writable()
	F2_val.set_writable()


	#Init FTS
	Ts = 0.1
	FTS = FourTankSystem()
	FTS.x = [2704.8, 5406.5, 205.8, 260.5]
	u = [100, 100]
	print("Start OPC UA Server")
	server.start()
	print("Server is started")
	try:
		while True:
			u[0], u[1] = F1_val.get_value(), F2_val.get_value()
			#print(u[0], u[1])
			u = [u[0], u[1]]
			y = (FTS.y(FTS.step(Ts,u)))
			y1_meas.set_value(y[0])
			y2_meas.set_value(y[1])
			time.sleep(Ts-(time.time()%Ts))
	finally:
		print("Stopping Server")
		server.stop()
		print("Server Offline")