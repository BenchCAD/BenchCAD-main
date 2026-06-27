import cadquery as cq

result = (
    cq.Workplane("XY")
    .circle(11.0)
    .extrude(1.5)
    .union(
        cq.Workplane("XY")
            .transformed(offset=cq.Vector(0, 0, -5.5), rotate=cq.Vector(0, 0, 0))
            .cylinder(11.0, 4.5)
    )
    .cut(
        cq.Workplane("XY")
            .transformed(offset=cq.Vector(0, 0, -4.75), rotate=cq.Vector(0, 0, 0))
            .cylinder(14.5, 3.0)
    )
)

# Export
show_object(result)