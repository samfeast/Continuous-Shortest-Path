import math
import raytrace
import numpy
import display

def evol(init,gen,n,g):
    new = []
    for theta in init:
        for i in range(-n,n+1):
            new.append(theta+((math.pi*i)/(4*(n**(gen)+1))))
    distances = []
    paths = []
    for phi in new:
        dist, path = raytrace.d(raytrace.trace(phi,g),g)
        distances.append(dist)
        paths.append(path)
    optim=[]
    optimd=[]
    optimp=[]
    for i in range(n//3):
        optimp.append(paths.pop(distances.index(min(distances))))
        optim.append(new.pop(distances.index(min(distances))))
        optimd.append(distances.pop(distances.index(min(distances))))
    return optim,optimp,optimd

def findPath(g):
    theta = [math.pi/4,]
    j = 0
    while True:
        theta,ps,ds=evol(theta,j+1,20,g)
        j += 1
        display.draw(g,ps)
        difs = []
        for i in range(len(ds)-1):
            difs.append(abs(ds[i]-ds[i+1]))
        if max(difs)<=0.00001:
            break
