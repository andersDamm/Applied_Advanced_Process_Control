"""

Parameters of Four tank system

"""

a1 = 1.2 #[cm2] Area of outlet pipe 1
a2 = 1.2 #[cm2] Area of outlet pipe 2
a3 = 1.2 #[cm2] Area of outlet pipe 3
a4 = 1.2 #[cm2] Area of outlet pipe 4

A1 = 380.0 #[cm2] Cross sectional area of tank 1
A2 = 380.0 #[cm2] Cross sectional area of tank 2
A3 = 380.0 #[cm2] Cross sectional area of tank 3
A4 = 380.0 #[cm2] Cross sectional area of tank 4


gamma1 = 0.6 #Flow distribution constant. Valve 1
gamma2 = 0.7 #Flow distribution constant. Valve 2

g = 981 #[cm/s2] The acceleration of gravity
rho = 1.00 #[g/cm3] Density of water

p = [a1, a2, a3, a4, A1, A2, A3, A4, gamma1, gamma2,  g,  rho]#


if __name__ == "__main__":
	print("Parameters: ", p)
	print(p[:-2])