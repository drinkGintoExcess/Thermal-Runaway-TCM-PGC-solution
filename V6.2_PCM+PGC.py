# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 13:18:31 2023

@author: fanlu and kyle
"""
import numpy as np
from numpy import arange
from pylab import meshgrid,cm,imshow,contour,clabel,colorbar,axis,title
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pickle
import waterfall_chart
import time

start = time.time()

flname = 'Design6_PCM,PGC_combined_sur_rad_water_21unit_0.75mmHB_testing'

#%% The most important parameter !!!!!
timestep = 5000
n = 5 # number of column in one unit 
T_evp = 100 # evaooration temperature
total_t = 13 #s
h_cp_bot = 7e-3 # m
h_cp_cha = 3e-3 # m
UnitOn1Side = 10 # units on each side
w_hb = 0.75e-3 # m, choose not to depend too much on the hb
NTU = 1.7
#%% constant of 2D mesh 

numOfUnitReal = 20 # number of unit in a real package
numOfChannel = 45
m = 13 # is the number of rows
numOfUnit = 2*UnitOn1Side + 1
N = 10 # number of division of battery
divOfBatt = 3
numOfelement = n*m*(numOfUnit)
#%% time Constants
# times
t_div = total_t + 1
t_duration_TR = 13
tnp = 0 # sec nail penetration
t_rampUP = tnp + 3
#t_TR = tnp + 13
t_TR = t_rampUP + t_duration_TR - 3
t_rampDown = t_TR + 3

#%% temperatures
T_surr = 25
T_exhaust = 30
T_hp = 25
T0 = 25
T_hg = 1000
E_TR = 1.64e6/4 # J no symmetry any more
E_PG = 322e3 # J heat source of product gas
S_dot_EVP = 4341 # W
S_dot_prod_gas = 2.48e4 # W
epsilon_emit = 0.9
stef_boltz = 5.67e-8 # W/m^2K^4

#%% porosity
eps_coolant = 0.25
eps_steelWool = 0.9
eps_gasTube = 0.1
eps_hs_wall = 0.16
eps_cha2all = h_cp_cha/(h_cp_bot+h_cp_cha)

#%% density
rho_al = 2710 # kg/m^3
rho_coolant = 1000 # kg/m^3
rho_batt = 2100 # kg/m^3
rho_tcm = 1680 # kg/m^3
rho_tim = 3255 # kg/m^3
rho_hs = rho_al # kg/m^3
rho_hs_w = rho_al*(1-eps_hs_wall) # kg/m^3
rho_cp_cha = (1-eps_coolant)*((1-eps_steelWool)*rho_al+eps_steelWool*rho_coolant) # kg/m^3
rho_hb = 160 # kg/m^3
rho_cp_bot = rho_al*(1-eps_gasTube) # kg/m^3
rho_co2 = 0.6226 # kg/m^3
#%% dimension
# heights
h_batt = 53e-3 # m

h_void = 10e-3 # m
h_b2cha = 2e-3 # m
h_bot2cha = 1e-3 #m
# widths
w_batt = 9e-3 # m
w_hs = 1e-3 # m
w_tim = 0.1e-3 # m
w_unit = w_batt + w_hb + w_hb #m

depth = 544e-3 # m
ratio_cp_wall2all = 1/2
V_cp_cha = h_cp_cha*(w_batt+w_hb+w_hs)*numOfUnitReal*depth # m^3
V_cp_bot = h_cp_bot*(w_batt+w_hb+w_hs)*numOfUnitReal*depth # m^3

#%% thermal conductivities
k_batt_v = 25 # W/mK
k_batt_h = 1  # W/mK
k_tcm = 5 # W/mK
k_tim = 3 # W/mK
k_air = 0.025 # W/mK
# k_eff_tcm = k_air*(k_tcm/k_air)**(0.280-0.757*math.log10(epsilon)-0.057*math.log(k_tcm/k_air))
k_eff_tcm = 5.0
k_hb = 0.02 # W/m-K
k_al = 200 # W/m-K
k_hs = k_al # W/m-K
k_hs_w = k_al*0.8353 # W/mK
k_cp_bot = k_al* (1-eps_gasTube) # W/m-K
k_cp_cha = (1-eps_coolant)*((1-eps_steelWool)*k_al+eps_steelWool*rho_coolant) # W/m-K
k_b2cha = 3 # W/m-K
#%% specific heat
cp_batt = 1300 # J/kgK
cp_hs = 900 # J/kgK
cp_hs_w = 900 # J/kgK
cp_hb = 1100 # J/kgK
cp_tcm = 1325 # J/kgK
cp_tim = 820 # J/kgK
cp_cp_bot = 900 # J/kgK
cp_cp_cha = 2360 # J/kgK
cp_co2 = 2408 # J/kgK
q_lat = 2.26e6 # J/kgK

q_evp_stored = q_lat*rho_coolant*eps_steelWool*(1-eps_coolant) # J/m^3

#%% Dictionaries
den_list = [rho_batt, rho_hs, rho_hb, rho_hs_w, rho_cp_bot, np.nan, rho_cp_cha]
cp_list = [cp_batt, cp_hs, cp_hb, cp_hs_w, cp_cp_bot, np.nan, cp_cp_cha]
k_list_h = [k_batt_h, k_hs, k_hb, k_hs_w, k_cp_bot, np.nan, k_cp_cha]
k_list_v = [k_batt_v, k_hs, k_hb, k_hs_w, k_cp_bot, np.nan, k_cp_cha]

#%% create material matrix
materialUnit = [[0] * n for i in range(m)] # 
tcm_count = 0
for i in range(m): # i is the row iterator
    for j in range(n): # j is the column iterator
        if 1 <= j < divOfBatt+1 and i<N :
            materialUnit[i][j] = 0 # battery 1 
        elif j == divOfBatt+1 and i< N:
            materialUnit[i][j] = 1 # HS
        elif j == divOfBatt+1 and i == N:
            materialUnit[i][j] = 3 # HS with Windows
        elif j == 0 and i < N:
            materialUnit[i][j] = 2 # HB
        elif i == N + 1:
            materialUnit[i][j] = 4 # the bot of the cold plate
        elif i == N + 2:
            materialUnit[i][j] = 6 # the channel of the cold plate
        else:
            materialUnit[i][j] = 5 # vod
    

#%% create dimension matrix
widthUnit = [[0] * n for i in range(m)]
for i in range(m): # i is the row iterator
    for j in range(n): # j is the column iterator
        if 1 <= j < divOfBatt+1:
            widthUnit[i][j] = w_batt/divOfBatt # battery
        elif j == 0:
            widthUnit[i][j] = w_hb # HB
        else:
            widthUnit[i][j] = w_hs # HS
             
heightUnit = [[0] * n for i in range(m)] # 
for i in range(m): # i is the row iterator
    for j in range(n): # j is the column iterator
        if i < N : # first 10 cell is battery section
            heightUnit[i][j] = h_batt/N # battery section 
        elif i == N:
            heightUnit[i][j] = h_void
        elif i == N+2:
            heightUnit[i][j] = h_cp_cha
        else:
            heightUnit[i][j] = h_cp_bot  # TCM

#%% patch materialUnit matrix into a large material matrix
material = materialUnit
for i  in range (numOfUnit-1):
    material = np.concatenate((material,materialUnit),axis=1)
    
width = widthUnit
for i  in range (numOfUnit-1):
    width = np.concatenate((width,widthUnit),axis=1)
    
height = heightUnit
for i  in range (numOfUnit-1):
    height = np.concatenate((height,heightUnit),axis=1)

#%% plot material matrix
xp = arange(0,n*numOfUnit,1)
yp = arange(0,m,1)
X,Y = meshgrid(xp, yp) # grid of point
plot = plt.pcolormesh(xp, yp, material, cmap='RdBu')
plt.xlim(0, n*numOfUnit)
plt.ylim(0, m-1)
plt.colorbar(plot, location ='bottom')
plt.axis('scaled')
plt.show(block=False)

#%% getter
def getDensity(i,j): # i is the row number and j is the column number
    return den_list[material[i][j]]

def getCp(i,j):
    return cp_list[material[i][j]]

def getWidth(i,j):
    return width[i][j]

def getHeight(i,j):
    return height[i][j]

def getK_h(i,j, T): # get thermal conductivity
    if material[i][j] == 2:
        return 0.000161*T+0.0281
    return k_list_h[material[i][j]]

def getK_v(i,j,T):
    if material[i][j] == 2:
        return 0.000161*T+0.0281
    return k_list_v[material[i][j]]

def getV(i,j): # get volume
    return getWidth(i,j)*getHeight(i,j)*depth


widthOfAUnit = 0
for j in range(n): # j is x
    widthOfAUnit = widthOfAUnit + getWidth(0,j)
heightOfAUnit = 0 # i is y
for i in range(m):
    heightOfAUnit = heightOfAUnit + getHeight(i,0)
    
#%% Heat Generation
S_dot_SS = 0.0 # W

def S_dot_tr(t): # heat generated by thermal runaway
    
    if t < tnp:
        S_dot = S_dot_SS # J/s (W)
    elif t >= tnp and t <= t_rampUP:
        S_dot = S_dot_SS + ((E_TR/N/divOfBatt)/t_duration_TR-S_dot_SS)/(t_rampUP - tnp)*(t-tnp) # J/s (W)
    elif t >= t_rampUP and t <= t_TR:
        S_dot = ((E_TR/N/divOfBatt)/t_duration_TR-S_dot_SS)
    elif t >= t_TR and t <= t_rampDown:
        S_dot = (E_TR/N/divOfBatt)/t_duration_TR + (S_dot_SS - (E_TR/N/divOfBatt)/t_duration_TR)/(t_rampDown - t_TR)*(t-t_TR) # J/s (W)        
    else:
        S_dot = S_dot_SS # J/s (W) 
    return S_dot

def S_dot_2(t):
    S_dot = S_dot_SS # J/s (W)     
    return S_dot

def q_evp(T): # heat dissipated by PCM  
    areaOf1Wall = w_unit*numOfUnitReal*h_cp_cha
    totalWallArea = 4*areaOf1Wall*numOfChannel
    q_evp = (T-T_evp)/(h_b2cha/(k_b2cha*totalWallArea))/V_cp_cha # W/m^3     
    return q_evp 

def q_prod_gas2wall(t,T):
    if t < t_TR:
        q_prod_gas2wall = (T_hg-T)/0.0363/V_cp_bot
    else:
        q_prod_gas2wall = 0
    return q_prod_gas2wall

def q_sur_rad(T):
    q_rad = epsilon_emit*stef_boltz*((T+273.15)**4-(T_surr+273.15)**4)
    return q_rad # W/m^2

material = np.array(material)

def getR(i1, j1, i2, j2,T1,T2):
    if i2 < 0 or j2<0 or i2>= material.shape[0] or j2 >= material.shape[1]: #check the boundary
        return 100000
    if i1 == i2: # in the same row, heat transfers horizontally
        # if the adjacent cells are battery and copper, there is a tim between them
        if (material[i1][j1] == 0 and material[i2][j2]==1) or (material[i1][j1] == 1 and material[i2][j2]==0): #
            return  w_tim/(k_tim*getHeight(i1,j1)*depth) + 0.5*getWidth(i1, j1)/(getK_h(i1,j1,T1)*getHeight(i1,j1)*depth)+0.5*getWidth(i2, j2)/(getK_h(i2,j2,T2)*getHeight(i1,j1)*depth)
        elif (material[i1][j1] == 5 or material[i2][j2] == 5): #either the first cell or the second one is a void
            return 100000
        return  0.5*getWidth(i1, j1)/(getK_h(i1,j1,T1)*getHeight(i1,j1)*depth)+0.5*getWidth(i2, j2)/(getK_h(i2,j2,T2)*getHeight(i1,j1)*depth)
        
    elif j1 == j2: # in the same column, heat transfer vertically
        # if the adjacent cells are battery and tcm, block the heat transfer
        # if (material[i1][j1] == 0 or material[i2][j2]==4) or (material[i1][j1] == 4 and material[i2][j2]==0): # << for V4
        #     return 100000
        if (material[i1][j1] ==5 or material[i2][j2]==5) : 
             return 100000
        elif (material[i1][j1] == 4 and material[i2][j2]==6) or (material[i1][j1] == 6 and material[i2][j2]==4):
            R_cp_w = h_cp_cha/k_cp_bot/(0.5*getWidth(i1, j1)*depth*ratio_cp_wall2all)
            R_cp_t_h = 0.5*(1-ratio_cp_wall2all)*(getWidth(i1,j1))/(k_cp_bot*h_cp_bot*getWidth(i1,j1))
            R1 = R_cp_w + 2*R_cp_t_h
            if material[i1][j1] == 4:
                R_cp_b_v = 0.5*getHeight(i1, j1)/(getK_v(i1,j1,T1)*getWidth(i1,j1)*depth)
                R_cp_cha = 0.5*getHeight(i2,j2)/(getK_v(i2,j2,T2)*getWidth(i2,j2)*depth)
            else:
                R_cp_b_v = 0.5*h_bot2cha/(getK_v(i2,j2,T2)*getWidth(i2,j2)*depth)
                R_cp_cha = 0.5*getHeight(i1, j1)/(getK_v(i1,j1,T1)*getWidth(i1,j1)*depth)
            R2 = h_b2cha/(k_b2cha*getWidth(i1,j1)*depth) + R_cp_cha # interface resistance
            return R_cp_b_v + 1/(1/(R1/2+R2)+1/R2)
        
        return 0.5*getHeight(i1, j1)/(getK_v(i1,j1,T1)*getWidth(i1,j1)*depth)+0.5*getHeight(i2, j2)/(getK_v(i2,j2,T2)*getWidth(i1,j1)*depth)
    else: 
        raise Exception("these two cells need to be adjacent") 

Q = lambda T_h, T_l, Rk : (T_h - T_l)/(Rk)

# D_array = np.zeros(n*m*numOfUnit*2) << V4
D_array = np.zeros(n*m*numOfUnit)


#%% solve set of DEs
# set times
t_span = np.array([0, total_t])
#t_span = np.array([0, 40000])
times = np.linspace(t_span[0], t_span[1], t_div)
timeUnit = times[1]-times[0]

#%% create alpha matrix
# the first index is position index and the second is the time 
alpha_PCM = [[0] * len(times)   for i in range(numOfelement)] # for PCM
for i in range(numOfelement): # i is the row iterator
    for j in range(len(times)): # j is the column iterator
            alpha_PCM[i][j] = 0
            

#%% Define differential equation
def f(t,y):
    T = y
    # traverse the 2D mesh
    for ix in range(0,n*numOfUnit):
        for iy in range(0,m):
            #change the index into a 1D list index
            i = iy*(n*numOfUnit)+ix
            # find the 1D index in four direction of the current cell
            iT = i+(n*numOfUnit); iB = i-(n*numOfUnit); iL = i-1; iR = i+1 # iT is iTop
            # get the current material of the current cell
            curMat = material[iy][ix]
            # if on the corner, calculate the heat transfer
            if curMat == 5: # suggest this is a void
                D_array[i] = 0
            else:
                if ix == 0 and iy == 0:
                    RR = getR(iy,ix,iy,ix+1,T[i],T[iR])
                    RT = getR(iy,ix,iy+1,ix,T[i],T[iT])
    
                    QR = Q(T[i],T[iR],RR)
                    QT = Q(T[i],T[iT],RT)
                    QL = 0
                    QB = 0
                # of on the bot edge
                elif iy == 0:
                    RR = getR(iy,ix,iy,ix+1,T[i],T[iR])
                    RT = getR(iy,ix,iy+1,ix,T[i],T[iT])
                    RL = getR(iy,ix,iy,ix-1,T[i],T[iL])
                    
                    QR = Q(T[i],T[iR],RR)
                    QT = Q(T[i],T[iT],RT)
                    QL = Q(T[i],T[iL],RL)
                    QB = 0
                # if on the top right corner
                elif iy == m-1 and ix == n*numOfUnit-1:
                    RL = getR(iy,ix,iy,ix-1,T[i],T[iL])
                    RB = getR(iy,ix,iy-1,ix,T[i],T[iB])
                    
                    QR = 0
                    QT = 0
                    QL = Q(T[i],T[iL],RL)
                    QB = Q(T[i],T[iB],RB)
                # if on the top edge
                elif iy == m-1:
                    RR = getR(iy,ix,iy,ix+1,T[i],T[iR])
                    RL = getR(iy,ix,iy,ix-1,T[i],T[iL])
                    RB = getR(iy,ix,iy-1,ix,T[i],T[iB])
                    
                    QR = Q(T[i],T[iR],RR)
                    QT = 0
                    QL = Q(T[i],T[iL],RL)
                    QB = Q(T[i],T[iB],RB)
                # on the other cell
                else:
                    RR = getR(iy,ix,iy,ix+1,T[i],T[iR])
                    RT = getR(iy,ix,iy+1,ix,T[i],T[iT])
                    RL = getR(iy,ix,iy,ix-1,T[i],T[iL])
                    RB = getR(iy,ix,iy-1,ix,T[i],T[iB])
                    
                    QR = Q(T[i],T[iR],RR)
                    QT = Q(T[i],T[iT],RT)
                    QL = Q(T[i],T[iL],RL)
                    QB = Q(T[i],T[iB],RB)
                    
                Qsum = QR + QT + QL + QB
                # apply the thermal runaway 
                print(t)
                if n*UnitOn1Side + 1 <= ix < n*UnitOn1Side + divOfBatt + 1 and iy < N:
                    Sdot = S_dot_tr(t)
                elif curMat == 6 and T[i] > T_evp and alpha_PCM[i][int(t)] < 1: 
                    # if it is channel and the temperature of the channel is above the evaporation temperature
                    # and there is still coolant left in that cell
                    Sdot = - q_evp(T[i])*getV(iy,ix)
                    if int(t) == 0:
                        alpha_PCM[i][int(t)] = 0
                    else:
                        alpha_PCM[i][int(t)] = alpha_PCM[i][int(t)-1] + q_evp(T[i])/q_evp_stored
                elif curMat == 6:
                    alpha_PCM[i][int(t)] = alpha_PCM[i][int(t)-1]
                    Sdot = 0
                elif curMat == 4: # the product gas heat - surface radiation
                    Sdot = q_prod_gas2wall(t, T[i]) * getV(iy,ix) - q_sur_rad(T[i])*getWidth(iy,ix)*depth
                else:
                    Sdot = 0
                # build the differential equation for the current cell
                D_array[i] = (Sdot - Qsum)/(getDensity(iy,ix)*getCp(iy, ix)*getV(iy, ix))
    return D_array



# set initial conditions

y0 = T0*np.ones(n*m*numOfUnit)
# solve
soln = solve_ivp(f, t_span, y0, t_eval=times, max_step=t_duration_TR/timestep)

#%% plot gas outlet temperature distribution
f = plt.figure(figsize=(8,6))
for tc in range(0,t_duration_TR+1):
    if ((tc-1) % 4 != 0 and tc != 0):
        continue
    T_cp_bot_list = []
    xposition_list = []
    T = soln.y[:,tc].reshape(m,n*numOfUnit)
    for ix in range(0,numOfUnit*n):
        if(ix == 0):
            xposition_list.append(getWidth(m-2,ix))
        else:
            xposition_list.append(getWidth(m-2,ix) + xposition_list[len(xposition_list)-1])
        T_cp_bot_list.append(T[m-2][ix] + (T_hg - T0)*np.exp(-NTU))
    plt.plot(xposition_list,T_cp_bot_list, label='t='+str(tc)+'s')
plt.rc('xtick', labelsize=12) 
plt.rc('ytick', labelsize=12) 
plt.xlabel(r'$x$ (m)', fontsize=12)
plt.ylabel(r'$T$ ($^o$C)', fontsize=12)
plt.legend(fontsize=12)
plt.grid()
plt.savefig('C:/Users/fanlu/OneDrive - Umich/0 GM LiB pack TR solutions/8 plot/'+flname+'/hot gas out1.png')
