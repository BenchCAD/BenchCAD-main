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
show_object(result)