import autograd.numpy as np

def deformation(P1,P2,x,y):
    
    return ((P1[0] - P1[1]*np.abs(x) - P1[3]*x**2 - 
            P1[2]*np.abs(y) - P1[5]*np.abs(x*y) - P1[4]*y**2),
           (P2[0] - P2[1]*np.abs(x) - P2[3]*x**2 - 
            P2[2]*np.abs(y) - P2[5]*np.abs(x*y) - P2[4]*y**2))

