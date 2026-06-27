import cadquery as cq

result = (
    cq.Workplane("XY")
    .polygon(6, 27.713)
    .extrude(12.0)
    .union(
        cq.Workplane("XY")
            .transformed(offset=cq.Vector(0, 0, 14.0), rotate=cq.Vector(0, 0, 0))
            .cylinder(4.0, 11.7)
    )
    .union(
        cq.Workplane("XY")
            .transformed(offset=cq.Vector(0, 0, 28.15), rotate=cq.Vector(0, 0, 0))
            .cylinder(24.3, 10.65)
    )
    .edges(">Z")
    .chamfer(0.8)
    .faces("<Z")
    .chamfer(0.8)
    .faces(">Z").workplane()
    .hole(13.8)
)

# Export
show_object(result)