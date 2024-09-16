import numpy as np


def standard(a,b,c):
    
    x_plus = ( -b + np.sqrt(b**2 - 4*a*c) ) / (2*a)
    x_minus = ( -b - np.sqrt(b**2 - 4*a*c) ) / (2*a)
    return x_minus,x_plus



print(standard(0.001,1000,0.001))



def modified(a,b,c):
    
    x_plus = (2*c) / ( -b - np.sqrt(b**2 - 4*a*c) )
    x_minus = (2*c) / ( -b + np.sqrt(b**2 - 4*a*c) )
    return x_minus,x_plus

print(modified(0.001,1000,0.001))



def new(a,b,c):
    x_plus = ( -b + np.sqrt(b**2 - 4*a*c) ) / (2*a)
    x_minus = (2*c) / ( -b + np.sqrt(b**2 - 4*a*c) )
    return x_minus,x_plus

print(new(0.001,1000,0.001))