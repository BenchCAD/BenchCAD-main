import cadquery as cq

result = (
    cq.Workplane("YZ")
    .rect(35.9, 59.5)
    .extrude(1.7)
    .cut(
        cq.Workplane("YZ")
            .transformed(offset=cq.Vector(0, 17.33, 0.85), rotate=cq.Vector(0, 0, 0))
            .cylinder(3.7, 6.9)
    )
    .cut(
        cq.Workplane("YZ")
            .transformed(offset=cq.Vector(0, 6.88, 0.85), rotate=cq.Vector(0, 0, 0))
            .slot2D(20.9, 7.7, 90)
            .extrude(3.7, both=True)
    )
    .cut(
        cq.Workplane("YZ")
            .transformed(offset=cq.Vector(12.55, -20.09, 0.85), rotate=cq.Vector(0, 0, 0))
            .cylinder(3.7, 2.0)
    )
    .cut(
        cq.Workplane("YZ")
            .transformed(offset=cq.Vector(-12.55, -20.09, 0.85), rotate=cq.Vector(0, 0, 0))
            .cylinder(3.7, 2.0)
    )
    .cut(
        cq.Workplane("YZ")
            .transformed(offset=cq.Vector(12.55, 21.47, 0.85), rotate=cq.Vector(0, 0, 0))
            .cylinder(3.7, 2.0)
    )
    .cut(
        cq.Workplane("YZ")
            .transformed(offset=cq.Vector(-12.55, 21.47, 0.85), rotate=cq.Vector(0, 0, 0))
            .cylinder(3.7, 2.0)
    )
    .edges("|Z")
    .chamfer(0.4)
)

# Export
show_object(result)