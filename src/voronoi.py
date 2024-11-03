import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt
import generate_lines
from random import randint
from display import draw
def f(x):
    if x[0][1]==0:
        return x[0][0]
    elif x[0][0]==1:
        return 1-x[0][1]
    elif x[0][1]==-1:
        return 3-x[0][0]
    else:
        return 4+x[0][1]

def mod_bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if f(arr[j]) > f(arr[j+1]):
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr

def voronoi_to_graph():
    points = np.asmatrix(np.random.rand(5,2))*np.matrix([[1,0],[0,-1]])

    vor = Voronoi(points)
    outside = [-1]
    verts = vor.vertices
    pts = vor.points
    ridge_verts=vor.ridge_vertices
    ridge_pts=vor.ridge_points
    regions=vor.regions
    boundary = generate_lines.Graph([[(0,0), (1,0), (1,-1), (0,-1), 1]],[1,-1])
    boundaryterms = [[[0,0],None],[[1,0],None],[[1,-1],None],[[0,-1],None]]
    graph = []
    for i in range(len(verts)):
        if vor.vertices[i][0] >= 1 or vor.vertices[i][0] <= 0 or vor.vertices[i][1] <= -1 or vor.vertices[i][1] >= 0:
            outside.append(i)
    for i in range(len(ridge_verts)):
        if ridge_verts[i][0] in outside and ridge_verts[i][1] in outside:
            pass
        elif ridge_verts[i][0] in outside:
            m = ((pts[ridge_pts[i][0]][0]-pts[ridge_pts[i][1]][0])/2,(pts[ridge_pts[i][0]][1]-pts[ridge_pts[i][1]][1])/2)
            v = (m[0]-verts[ridge_verts[i][1]][0],m[1]-verts[ridge_verts[i][1]][1])
            x = verts[ridge_verts[i][1]]
            new = boundary.get_next(x,v)[0]
            boundaryterms.append([new,i])
        elif ridge_verts[i][1] in outside:
            m = ((pts[ridge_pts[i][0]][0]-pts[ridge_pts[i][1]][0])/2,(pts[ridge_pts[i][0]][1]-pts[ridge_pts[i][1]][1])/2)
            v = (m[0]-verts[ridge_verts[i][0]][0],m[1]-verts[ridge_verts[i][0]][1])
            x = verts[ridge_verts[i][0]]
            new = boundary.get_next(x,v)[0]
            boundaryterms.append([new,i])
    boundaryterms = mod_bubble_sort(boundaryterms)
    for i in regions:
        outsides = False
        if i ==[]:
            continue
        for j in i:
            if j in outside:
                outsides = True
        if outsides:
            continue
        a=[]
        for j in i:
            a.append([verts[j][0],verts[j][1]])
        a.append(randint(1,4))
        graph.append(a)
    for i in range(len(verts)):
        if i in outside:
            continue
        a=[verts[i]]
        for j in range(len(boundaryterms)):
            try:
                if boundaryterms[j][1]==i:
                    a.append([boundaryterms[j][0][0],boundaryterms[j][0][1]])
                else:
                    break
            except:
                a.append([boundaryterms[j][0][0],boundaryterms[j][0][1]])
        if len(a)<3:
            continue
        a.append(randint(1,4))
        graph.append(a)
    return generate_lines.Graph(graph,[1,-1])
                    
