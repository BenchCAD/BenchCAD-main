import cadquery as cq

result = (
    cq.Workplane("XZ")
    .cylinder(15.6, 91.15)
    .faces(">Y").workplane()
    .hole(50.8)
    .faces(">Y").workplane()
    .circle(53.25)
    .extrude(10.4)
    .edges(">Z")
    .chamfer(1.5)
    .edges("<Z")
    .fillet(1.2)
    .faces(">Y").workplane()
    .polarArray(59.4, 0, 360, 4)
    .hole(8.2)
    .faces(">Y").workplane()
    .polarArray(73.36, 0, 360, 4)
    .hole(7.8)
)

# Export
show_object(result)