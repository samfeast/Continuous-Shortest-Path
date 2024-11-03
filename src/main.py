import generate_lines
import evolution
import voronoi
import perlin_noisemap

r1 = [(0,0),(1,0),(2/3,-1/3),(0,-1/3),3]
r2=[(0,-1/3),(1/3,-1/3),(1/3,-1),(0,-1),1.5]
r3=[(1/3,-1/3),(2/3,-1/3),(1/2,-1/2),(1/2,-1),(1/3,-1),4]
r4=[(1,0),(1,-1/2),(1/2,-1/2),1]
r5=[(1/2,-1/2),(1,-1/2),(1,-1),(1/2,-1),2]
graph = generate_lines.Graph([r2,r3,r4,r5],(1,-1))
graph2=generate_lines.Graph(perlin_noisemap.generate_noise_graph(2),(1,-1))
evolution.findPath(graph2)
