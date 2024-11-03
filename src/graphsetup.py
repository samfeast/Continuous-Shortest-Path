import generate_lines

def fix_micheal(d):
    lst = list(d.values())
    g = generate_lines.Graph(lst,(1,-1))
    for shape1 in lst:
        for shape2 in lst:
            shared = False
            within = False
            for i in shape2[:-1]:
                for j in shape2[:-1]:
                    if i != j and i in shape1 and j in shape1 and shape1!=shape2:
                        shared = True
                    if g.is_strictly_in_region(((i[0]+j[0])/2,(i[1]+j[1])/2),shape1):
                        within = True
        if within and shared:
            lst.remove(shape1)
                

