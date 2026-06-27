import cadquery as cq

result = (
    cq.Workplane("XY")
    .circle(12.0)
    .extrude(0.8)
    .faces(">Z").workplane()
    .hole(15.0)
    .cut(
        cq.Workplane("XY")
            .transformed(offset=cq.Vector(9.75, 0.0, 0.0), rotate=cq.Vector(0, 0, 0))
            .box(10.0, 2.25, 1.8)
    )
)

# Export
show_object(result)