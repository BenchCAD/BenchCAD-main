import cadquery as cq

# --- parameters ---
# ball_diameter = 25.0
# thread_m = 6.0
# ball_radius = 12.5
# stem_radius = 4.5
# stem_height = 50.0

result = (
    cq.Workplane("XY")
    .sphere(12.5)
    .union(
        cq.Workplane("XY")
            .transformed(offset=cq.Vector(0.0, 0.0, -37.5), rotate=cq.Vector(0, 0, 0))
            .cylinder(50.0, 5.85)
    )
)

# Export
show_object(result)