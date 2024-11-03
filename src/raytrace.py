import math
import numpy
def trace(phi,g):
    done=False
    path = [[0,0]]
    x=[0,0]
    alpha=phi
    v = [math.cos(alpha),-math.sin(alpha)]
    g.reset_current_region()
    try:
        c = g.get_regions()[g.get_current_region(x,v)][-1]
    except:
        c=1
    count = 0
    while c>=1:
        x,alpha,n,c_1=g.get_next(x,v)
        # if new region is not outside boundary and it tries to TIR we stop.
        # if new region is outside boundary and it tries to TIR we stop.
        # if new region is in
        if c_1!= -1 and abs((c*math.sin(alpha))/c_1) >= 1:
            c=-1
            continue
        elif c_1==-1 and abs((c*math.sin(alpha))/c_1) >= 1:
            path.append(x)
            c=-1
            continue
        elif count == 5:
            c=-1
            continue
        elif c_1==c:
            count+=1
        else:
            alpha = math.asin((c*math.sin(alpha)/c_1))
            count = 0
        c=c_1
        v = [n[0]*math.cos(alpha)+n[1]*math.sin(alpha),n[1]*math.cos(alpha)-n[0]*math.sin(alpha)]
        path.append(x)
    return path
def d(path,g):
    return math.sqrt((g.get_target()[0]-path[-1][0])**2+(g.get_target()[1]-path[-1][1])**2),path
