import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

class calc_data():
	def __init__(self, t_data, h_data, T, P_data):
		
		#############Experiment data############
		self.t_data = t_data		#s
		self.h_data = h_data		#mm
		self.P_data = P_data
		self.T = T	#K
		self.P_av = np.average(P_data)		#atm
		#############Experiment data############
		
		############Substance properties############
		self.M_acetone = 58.08 #g/mol
		self.M_air = 28.851 #g/mol
		self.sigma_acetone = 4.6			#A
		self.sigma_air = 3.711				#A
		self.sigma = (self.sigma_acetone+self.sigma_air)/2
		self.epsilon_acetone = 560.2	#K, note that epsilon/k
		self.epsilon_air = 78.6				#K, note that epsilon/k
		self.epsilon = (self.epsilon_acetone*self.epsilon_air)**0.5
		self.collision_integral = self.calc_collision_integral(self.T)
		self.P_set = 10**(4.42448 -(1312.253/(self.T-32.445)))*0.9869 #atm
		#reference by (1)
		
		self.Tb_acetone =	329.4						#nomal B.P in K
		self.Tb_air = 80									#nomal B.P in K
		self.Tc_acetone =	508.2						#critlcal T in K
		self.Tc_air =	132.2								#critlcal T in K
		self.Pc_acetone = 47.01 * 0.9869	#critlcal P in atm
		self.Pc_air =	37.45 * 0.9869			#critlcal P in atm
		self.Vc_acetone = 209							#critical molar volume in cm^3/mol
		self.Vc_air =	84.8								#critical molar volume in cm^3/mol 
		self.Vb_acetone, self.Vb_air = self.liquid_density()	#cm^3/mol at normal BP
		#reference by (2), (3)
		
		self.Vd_acetone = 16.5*3 + 1.98*6 + 5.48*1		#diffusion volume
		self.Vd_air = 20.1
		#reference by (7)
		
		self.Acetone_dens = 57.6214/(0.233955**(1+(1-self.T/507.803)**0.254167))/1000 #g/cm^3
		#reference by (4)
		############Substance properties############
		
		#############Result############
		self.D_exp = None
		self.D_exp_new = None
		self.D_chap = None
		self.D_brokaw = None
		self.D_chen = None
		self.D_fuller = None
		self.D_reff = None
		#############Result############
	
	def liquid_density(self):
		#reference by (5)
		Tb_N = 77.3
		Tb_O = 90.2
		Tb_acetone = self.Tb_acetone
		
		C1_N, C2_N, C3_N, C4_N = (3.2091, 0.2861, 126.2, 0.2966)
		d_N = C1_N/C2_N**(1+(1-Tb_N/C3_N)**C4_N) * 0.001		#d in mol/cm^3, T in K
		
		C1_O, C2_O, C3_O, C4_O = (3.9143, 0.28772, 154.58, 0.2924)
		d_O = C1_O/C2_O**(1+(1-Tb_O/C3_N)**C4_N) * 0.001
		
		d_air = d_N*0.79 + d_O*0.21
		
		C1_acetone, C2_acetone, C3_acetone, C4_acetone = (1.2332, 0.25886, 508.2, 0.2913)
		d_acetone = C1_acetone/C2_acetone**(1+(1-Tb_acetone/C3_acetone)**C4_acetone) * 0.001
		
		return (1/d_acetone, 1/d_air)
		
	def calc_collision_integral(self, T):
		Dimensionless_T = T/self.epsilon

		if Dimensionless_T < 0.3 or Dimensionless_T > 400:
			print('dimensionless temperature is out of range')
			return 0
		
		DT_data = np.load('Dimensionless_T.npy')
		C_data = np.load('Collision_integral.npy')
		for i in range(0,len(DT_data)):
			det = Dimensionless_T - DT_data[i]
			if det<0:
				result = (C_data[i]-C_data[i-1])/(DT_data[i]-DT_data[i-1])*(Dimensionless_T-DT_data[i])+C_data[i]
				return result
	
	def poly_regression(self, x, y, n, title, Figure = True):
		plt.close()
		plt.plot(x, y, 'bo', label = 'Data')
		
		result = np.polyfit(x, y, n)
		if Figure == True:
			if n == 1:
				fit_x = [x[0],x[len(x)-1]]
				fit_y = [result[0]*fit_x[0]+result[1], result[0]*fit_x[1]+result[1]]
		
				if result[1]<0:
					plt.plot(fit_x, fit_y, 'r', label = 'y={}x{}'.format(np.round_(result[0], 4), np.round_(result[1], 4)))
				else:
					plt.plot(fit_x, fit_y, 'r', label = 'y={}x+{}'.format(np.round_(result[0], 4), np.round_(result[1], 4)))
			elif n == 2:
				x_fit = np.arange(x[0], x[len(x)-1]+1, 1)
				x = sp.Symbol('x')
				y_fit = result[0]*x_fit**2 + result[1]*x_fit + result[2]
			
				plt.plot(x_fit, y_fit, 'r', label = 'y={}x^2 + {}x + {}'.format(result[0], result[1], np.round_(result[2], 4)))
			
			plt.grid()
			plt.legend()
			plt.title(title)
			plt.xlabel('x axis')
			plt.ylabel('y axis')
			plt.show()
		
		return result
		
	def refference_D(self):
		#0'C 1atm -> D = 0.109 cm^2/s
		#reference by (6)
		P = self.P_av
		T = self.T
		D = 0.109 * (1/P)*(T/273.15)**1.75
		self.D_reff = D
	
	def exp_D(self):
		P_1 = self.P_set
		P_2 = 0
		P = self.P_av
		R = 82.06				# cm^3 atm / (mol K)
		rho = self.Acetone_dens
		T = self.T
		M_A = self.M_acetone

		x_data = self.t_data
		y_data = self.h_data**2-self.h_data[0]**2
		slope = self.poly_regression(x_data, y_data, 1, 't vs hf^2 - hi^2')[0]
		
		Plm = (P-P_2-(P-P_1))/(np.log(P-P_2)-np.log(P-P_1))
		D = slope*rho*R*T*Plm/(2*M_A*P*P_1)
		self.D_exp = np.round_(D, 4)
	
	def exp_D_new(self):	
		x_data = self.t_data
		y_data = self.P_data
		
		P_1 = self.P_set
		M_A = self.M_acetone
		T = self.T
		rho = self.Acetone_dens
		R = 82.06			# cm^3 atm / (mol K)
		
		a, b, c = self.poly_regression(x_data, y_data, 2, 't vs P')
		
		t = sp.Symbol('t')
		
		P = a*t**2 + b*t + c
		pi = P/(P_1/(sp.log(P/(P-P_1))))
		
		X_data = np.array([])
		delt = 1
		ds = 0
		t_cord = 0
		for i in range(len(x_data)):
			while True:
				if t_cord != x_data[i]:
					t_cord += delt
					ds += float(pi.subs(t, t_cord)*delt)
				else:
					X_data = np.hstack(( X_data, np.round(np.array([ds]), 4) ))
					break
		
		Y_data = self.h_data**2-self.h_data[0]**2

		slope = self.poly_regression(X_data, Y_data, 1, 'pi(t)-pi(0) vs hf^2-hi^2')[0]
		D = slope/2*R*rho*T/(M_A*P_1)
		
		self.D_exp_new = np.round_(D, 4)
		
		
	def chap_D(self):
		T = self.T
		M_A = self.M_acetone
		M_B = self.M_air
		P = self.P_av
		sigma = self.sigma
		C_I = self.collision_integral
		
		def func(T, MA, MB, P, sigma, omega):
			return (0.001858*T**1.5*(1/MA+1/MB)**0.5)/(P*sigma**2*omega)
		D = func(T, M_A, M_B, P, sigma, C_I)
		
		self.D_chap = np.round_(D, 4)
		
	def brokaw_D(self):
		delta = 0 # Because dipole moment of air = 0, delta of air = 0
		Tb_A = self.Tb_acetone
		Tb_B = self.Tb_air
		Vb_A = self.Vb_acetone
		Vb_B = self.Vb_air
		M_A = self.M_acetone
		M_B = self.M_air
		P = self.P_av
		T = self.T
		
		#Calculate epsilon value
		def func_epsilon(delta, Tb):
			return 1.18*(1+1.3*delta**2)*Tb
		epsilon_acetone = func_epsilon(delta, Tb_A)
		epsilon_air = func_epsilon(delta, Tb_B)
		epsilon = (epsilon_air * epsilon_acetone)**0.5
		
		D_T = T/epsilon			#dimensionless temperature
		
		#Calculate collision integral by dimensionless temperature
		def func_col_int(delta, D_T):
			A, B, C, D, E, F, G, H = (1.06036, 0.15610, 0.19300, 0.47635, 1.03587, 1.52996, 1.76474, 3.89411)
			return A/D_T**B + C/np.exp(D*D_T)+E/np.exp(F*D_T)+G/np.exp(H*D_T)+0.196*delta**2/D_T
		
		collision_integral = func_col_int(delta, D_T)
		
		#Calculate sigma value
		def func_sigma(delta, Vb):
			return (1.585 * Vb/(1+1.3*delta**2))**(1/3)
		
		sigma_acetone = func_sigma(delta, Vb_A)
		sigma_air = func_sigma(delta, Vb_B)
		sigma = (sigma_acetone*sigma_air)**0.5		
		
		#Calculate diffusivity by brokaw corrected factors
		def func(T, MA, MB, P, sigma, omega):
			return (0.001858*T**1.5*(1/MA+1/MB)**0.5)/(P*sigma**2*omega)
		
		D = func(T, M_A, M_B, P, sigma, collision_integral)
		self.D_brokaw = np.round_(D, 4)
		
	def chen_D(self):
		M_A = self.M_acetone
		M_B = self.M_air
		P = self.P_av
		T = self.T
		Tc_A = self.Tc_acetone
		Tc_B = self.Tc_air
		Vc_A = self.Vc_acetone
		Vc_B = self.Vc_air
		
		def func(T, MA, MB, P, TcA, TcB, VcA, VcB):
			return 0.01498*T**1.81*(1/MA+1/MB)**0.5/(P*(TcA*TcB)**0.1405*(VcA**0.4+VcB**0.4)**2)
		
		D = func(T, M_A, M_B, P, Tc_A, Tc_B, Vc_A, Vc_B)
		
		self.D_chen = np.round_(D, 4)
	
	def fuller_D(self):
		Vd_A = self.Vd_acetone
		Vd_B = self.Vd_air
		T = self.T
		P = self.P_av
		M_A = self.M_acetone
		M_B = self.M_air		
		
		def func(T, MA, MB, P, VA, VB):
			return 0.001*T**1.75*(1/MA+1/MB)**0.5/(P*(VA**(1/3)+VB**(1/3))**2)
			
		D = func(T, M_A, M_B, P, Vd_A, Vd_B)
		
		self.D_fuller = np.round_(D, 4)
	
	def find_T(self):
		#########exp vs cahp#########
		
		#exp value calc
		P_1 = self.P_set
		P_2 = 0
		P = self.P_av
		R = 82.06				# cm^3 atm / (mol K)
		rho = self.Acetone_dens
		T = sp.Symbol('T')
		M_A = self.M_acetone
		M_B = self.M_air
		
		T_array = np.arange(293.15, 318.15+0.01, 0.01)
		
		#exp value calc
		x_data = self.t_data
		y_data = self.h_data**2-self.h_data[0]**2
		slope = self.poly_regression(x_data, y_data, 1, 't vs hf^2 - hi^2', Figure = False)[0]
		
		Plm = (P-P_2-(P-P_1))/(np.log(P-P_2)-np.log(P-P_1))
		D_exp = slope*rho*R*T*Plm/(2*M_A*P*P_1)					#function of T
		
		
		D_exp_array = np.array([])
		for i in range(len(T_array)):
			temp = D_exp.subs(T, T_array[i])
			D_exp_array = np.hstack([D_exp_array, np.array([temp])])
		
		
		#chap value calc
		sigma = self.sigma
		def func(T, MA, MB, P, sigma, omega):
			return (0.001858*T**1.5*(1/MA+1/MB)**0.5)/(P*sigma**2*omega)
		
		D_chap_array = np.array([])
		for i in range(len(T_array)):
			C_I = self.calc_collision_integral(T_array[i])
			temp = func(T_array[i], M_A, M_B, P, sigma, C_I)
			D_chap_array = np.hstack([D_chap_array, np.array([temp])])
		
		#brokaw calc
		delta = 0 # Because dipole moment of air = 0, delta of air = 0
		Tb_A = self.Tb_acetone
		Tb_B = self.Tb_air
		Vb_A = self.Vb_acetone
		Vb_B = self.Vb_air
		
		def func_epsilon(delta, Tb):
			return 1.18*(1+1.3*delta**2)*Tb
		epsilon_acetone = func_epsilon(delta, Tb_A)
		epsilon_air = func_epsilon(delta, Tb_B)
		epsilon = (epsilon_air * epsilon_acetone)**0.5
		
		D_brokaw_array = np.array([])
		
		for i in range(len(T_array)):
			D_T = T_array[i]/epsilon			#dimensionless temperature
		
			def func_col_int(delta, D_T):
				A, B, C, D, E, F, G, H = (1.06036, 0.15610, 0.19300, 0.47635, 1.03587, 1.52996, 1.76474, 3.89411)
				return A/D_T**B + C/np.exp(D*D_T)+E/np.exp(F*D_T)+G/np.exp(H*D_T)+0.196*delta**2/D_T
		
			collision_integral = func_col_int(delta, D_T)
		
			def func_sigma(delta, Vb):
				return (1.585 * Vb/(1+1.3*delta**2))**(1/3)
		
			sigma_acetone = func_sigma(delta, Vb_A)
			sigma_air = func_sigma(delta, Vb_B)
			sigma = (sigma_acetone*sigma_air)**0.5		
		
			def func(T, MA, MB, P, sigma, omega):
				return (0.001858*T**1.5*(1/MA+1/MB)**0.5)/(P*sigma**2*omega)
			
			temp = func(T_array[i], M_A, M_B, P, sigma, collision_integral)
			
			D_brokaw_array = np.hstack([D_brokaw_array, np.array([temp])])
		
		#Chen calc
		Tc_A = self.Tc_acetone
		Tc_B = self.Tc_air
		Vc_A = self.Vc_acetone
		Vc_B = self.Vc_air
		
		def func(T, MA, MB, P, TcA, TcB, VcA, VcB):
			return 0.01498*T**1.81*(1/MA+1/MB)**0.5/(P*(TcA*TcB)**0.1405*(VcA**0.4+VcB**0.4)**2)
		
		D_chen_array = np.array([])
		for i in range(len(T_array)):
			temp = func(T_array[i], M_A, M_B, P, Tc_A, Tc_B, Vc_A, Vc_B)
			D_chen_array = np.hstack([D_chen_array, np.array([temp])])
			
		#Fuller calc
		Vd_A = self.Vd_acetone
		Vd_B = self.Vd_air
		
		def func(T, MA, MB, P, VA, VB):
			return 0.001*T**1.75*(1/MA+1/MB)**0.5/(P*(VA**(1/3)+VB**(1/3))**2)
		
		D_fuller_array = np.array([])
		for i in range(len(T_array)):
			temp = func(T_array[i], M_A, M_B, P, Vd_A, Vd_B)
			D_fuller_array = np.hstack([D_fuller_array, np.array([temp])])
			
		#Refferecne calc
		D_reff_array = np.array([])
		for i in range(len(T_array)):
			temp = 0.109 * (1/P)*(T_array[i]/273.15)**1.75
			D_reff_array = np.hstack([D_reff_array, np.array([temp])])
		
		plt.close()
		plt.plot(T_array, D_exp_array, label = 'Exp')
		plt.plot(T_array, D_chap_array, label = 'Chapman')
		plt.plot(T_array, D_brokaw_array, label = 'Brokaw')
		plt.plot(T_array, D_chen_array, label = 'Chen')
		plt.plot(T_array, D_fuller_array, label = 'Fuller')
		plt.plot(T_array, D_reff_array, label = 'Refference')
		plt.title('Diffusivity If different diffusion temperature')
		plt.xlabel('Diffusion temperature in K')
		plt.ylabel('Diffusivity in cm^2/s')
		plt.grid()
		plt.legend()
		plt.show()
		
	def find_h(self):
		del_h_start = -1
		del_h_end = 3
		del_h_del = 0.1
		del_h = np.arange(del_h_start, del_h_end + del_h_del, del_h_del)
		h_data = self.h_data
		
		P_1 = self.P_set
		P_2 = 0
		P = self.P_av
		R = 82.06				# cm^3 atm / (mol K)
		rho = self.Acetone_dens
		T = self.T
		M_A = self.M_acetone
		
		Plm = (P-P_2-(P-P_1))/(np.log(P-P_2)-np.log(P-P_1))
		
		D_array = np.array([])
		for i in range(len(del_h)):
			x_data = self.t_data
			y_data = (h_data - del_h[i])**2 - (h_data[0] - del_h[i])**2
			slope = self.poly_regression(x_data, y_data, 1, 't vs hf^2 - hi^2', Figure = False)[0]
			temp = slope*rho*R*T*Plm/(2*M_A*P*P_1)
			D_array = np.hstack([D_array, np.array([temp])])

		plt.close()
		plt.plot(del_h, D_array)
		plt.xlabel('Variation of h')
		plt.ylabel('Diffusivity in cm^2/s')
		plt.title('Diffusivity change If h-variation exist')
		plt.grid()
		plt.show()
				
	def calc_all(self):
		self.refference_D()
		self.exp_D()
		self.chap_D()
		self.brokaw_D()
		self.chen_D()
		self.fuller_D()
		self.exp_D_new()
	
	def print_result(self):
		print('Refference result : ' + str(self.D_reff) + ' cm^2/s')
		print('Experimental result : ' + str(self.D_exp) + ' cm^2/s')
		print('Experimental result(new) : ' + str(self.D_exp_new) + ' cm^2/s')
		print('Chapman-Enskog Equation result : ' + str(self.D_chap) + ' cm^2/s')
		print('Brokaw method : ' + str(self.D_brokaw) + 'cm^2/s')
		print('Chen and othmer equation : ' + str(self.D_chen) + 'cm^2/s')
		print('Fuller, Schettler and gidding equation : ' + str(self.D_fuller) + 'cm^2/s')
		
if __name__ == '__main__':
	
	t_data = np.arange(0,3900,300)	#s
	h_data = np.array([5.24, 5.3, 5.31, 5.35, 5.39, 5.42, 5.44, 5.47, 5.48, 5.53, 5.56, 5.59, 5.64]) 	#cm
	P_data = np.array([121, 126, 127, 128, 128, 129, 129, 130, 130, 128, 130, 129, 128])*9.678411*10**(-5)	#atm
	first_team = calc_data(t_data, h_data, 45+273.15, 1-P_data)
	
	first_team.calc_all()
	first_team.print_result()
#reference
#(1). https://webbook.nist.gov/cgi/cbook.cgi?ID=C67641&Mask=4&Type=ANTOINE&Plot=on
#(2). introduction to chemical engineering thermodynamics eighth edition, J.M. smith, Appendix B, Table B.1
#(3). https://www.bnl.gov/magnets/staff/gupta/cryogenic-data-handbook/Section6.pdf
#(4). http://ddbonline.ddbst.de/DIPPR105DensityCalculation/DIPPR105CalculationCGI.exe?component=Acetone
#(5). Perry's Handbook 7th edition Table 2-30
#(6). Perry’s chemical engineers’ handbook seventh edition, 374p, table2-371 Diffusivities of pairs of gases and vapors (1atm)
#(7). Fundamentals of momentum, heat and mass transfer fifth edition, 410p, Table 24.3
