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