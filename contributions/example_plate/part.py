import cadquery as cq

# A simple rectangular plate. `result` must hold the final solid.
result = cq.Workplane("XY").box(10, 20, 5)
