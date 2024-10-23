import random

NAME = 'Game of Life - Python script'
LIVE = 1
DEAD = 0
width    = 100      # Width of mesh
height   = 60       # height of mesh
nStates  = 2        # number of states of CA
timeStep = 150      # number of timesteps
S0        = []       # mesh cell state, changing in initial() procedure
S1        = []       # mesh cell state, changing in initial() procedure
executing = True    # indicates if the simulation still be run

'''
 nw | n | ne
----|---|----
 w  | c |  e
----|---|----
 sw | s | se
 ''' 

nw = -1
n  = -1
ne = -1
w  = -1
e  = -1
sw = -1
s  = -1
se = -1
c  = -1
     

def initial():
    
    #Must initialize the state matrix. In this example, they are S0 and S1
    global S #Set global to change global variable. Otherwise, a local copy will be created 
    for j in range(0, height):
        W = []

        for i in range(0, width):
            if random.random() < 0.750:
                W.append(1)
                
            else:
                W.append(0)
                
                
        S0.append(W)
        S1.append(W)
        
    
    #print(S)
def update(ts, states):
    global S0
    global S1
    global width
    global height
    global timeStep    
    global executing
    global nw
    global n 
    global ne
    global w
    global e 
    global sw
    global s
    global se 
    global c
    S0 = states
    for j in range(0, height):
        for i in range(0, width):
            nw = n = ne = w = e = sw = s = se = c = -1
            sum = 0;
            
            c = S0[j][i]
            periodicBoundary(i, j)
            
            sum = nw + n + ne + w + e + sw + s + se
                        
            if ((sum == 3) and (c == 0)):
                S1[j][i] = LIVE
            elif ((sum >= 2) and (sum <= 3) and (c == 1)):
                S1[j][i] = LIVE
            else:
                S1[j][i] = DEAD
            
    executing = ts < executing

def periodicBoundary(i, j):
    global S0
    global width
    global height
    global nw
    global n 
    global ne
    global w
    global e 
    global sw
    global s
    global se 
    
    if (i + 1 == width and j + 1 == height):
        se = S0[0][0]
    elif (i + 1 == width):
        se = S0[j+1][0]
    elif (j + 1 == height):
        se = S0[0][i+1]
    else:
        se = S0[j + 1][i + 1]
     
    if (i - 1 < 0 and j + 1 == height):
        sw = S0[0][width - 1]
    elif (i - 1 < 0):
        sw = S0[j + 1][width - 1]
    elif (j + 1 == height):
        sw = S0[0][i-1]
    else:
        sw = S0[j + 1][i - 1]
    
    if (i + 1 == width and j - 1 < 0):
        ne = S0[height - 1][0]
    elif (j - 1 < 0):
        ne = S0[height - 1][i + 1]
    elif (i + 1 == width):
        ne = S0[j - 1][0]
    else:
        ne = S0[j - 1][i + 1];
    
    if (i - 1 < 0 and j - 1 < 0): 
        nw = S0[height - 1][width - 1]
    elif (i - 1 < 0):
        nw = S0[j - 1][width - 1]
    elif (j - 1 < 0):
        nw = S0[height - 1][i - 1]
    else:
        nw = S0[j - 1][i - 1];
    
    if (i - 1 < 0):
        w = S0[j][width - 1]
        e = S0[j][i + 1]
    elif (i + 1 == width):
        w = S0[j][i - 1]
        e = S0[j][0]
    else:
        w = S0[j][i - 1]
        e = S0[j][i + 1]
    
    if (j - 1 < 0):
        n = S0[height - 1][i]
        s = S0[j + 1][i]
    elif (j + 1 == height):
        n = S0[j - 1][i]
        s = S0[0][i]
    else:
        n = S0[j - 1][i]
        s = S0[j + 1][i]
     
'''
if __name__ == '__main__':
    print(width)
    initial()
    update(1, S)
'''
