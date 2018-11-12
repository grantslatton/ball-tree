A `BallTree` is a space-partitioning data-structure that allows for finding 
nearest neighbors in logarithmic time. 

It does this by partitioning data into a series of nested bounding spheres
("balls" in the literature). Spheres are used because it is trivial to 
compute the distance between a point and a sphere (distance to the sphere's
center minus thte radius). The key observation is that a potential neighbor
is necessarily closer than all neighbors that are located inside of a 
bounding sphere that is farther than the aforementioned neighbor.

Graphically:
```

   A -  
   |  ----         distance(A, B) = 4
   |      - B      distance(A, S) = 6
    |       
     |
     |    S
       --------
     /        G \ 
    /   C        \
   |           D |
   |       F     |
    \ E         /
     \_________/
```

In the diagram, `A` is closer to `B` than to `S`, and because `S` bounds
`C`, `D`, `E`, `F`, and `G`, it can be determined that `A` it is necessarily 
closer to `B` than the other points without even computing exact distances
to them.

Ball trees are most commonly used as a form of predictive model where the
points are features and each point is associated with a value or label. Thus,
This implementation allows the user to associate a value with each point. If
this functionality is unneeded, `()` can be used as a value.

This implementation returns the nearest neighbors, their distances, and their
associated values. Returning the distances allows the user to perform some 
sort of weighted interpolation of the neighbors for predictive purposes.
