import cadquery as cq

result = (
    cq.Workplane("XY")
    .circle(8.0)
    .extrude(110.0)
    .faces(">Z").workplane()
    .polygon(6, 27.71)
    .extrude(10.0)
    .edges(">Z")
    .chamfer(1.5)
)

# Export
show_object(result)