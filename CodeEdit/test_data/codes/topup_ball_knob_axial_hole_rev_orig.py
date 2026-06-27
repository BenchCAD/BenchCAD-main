import cadquery as cq

# --- parameters ---
# ball_diameter
# thread_m
# ball_radius
# stem_radius
# stem_height

result = (
    cq.Workplane("XY")
    .sphere(12.5)
    .union(
        cq.Workplane("XY")
            .transformed(offset=cq.Vector(0.0, 0.0, -37.5), rotate=cq.Vector(0, 0, 0))
            .cylinder(50.0, 4.5)
    )
)

# Export
result = result.cut(cq.Workplane('XY').cylinder(225.0,3.00))

show_object(result)