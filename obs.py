import numpy as np

balls = [
    {"x": 10, "y": 10, "r": 5},
    {"x": 20, "y": 20, "r": 2},
    {"x": 30, "y": 30, "r": 3},
]


sigma=0.5
abc = np.random.normal(0, sigma*sigma)

for b in balls :
    x = b["x"]
    y = b["y"]
    r = b["r"]
    dist = np.sqrt(x*x + y*y)
    dist_near = dist - r 
    norm = dist_near + abc
    print("ball dist", dist , "dist_near", dist_near, "norm" , norm)   




dist = np.sqrt(x*x + y*y)

dist_near = dist - r 
