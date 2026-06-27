import cadquery as cq

# --- parameters ---
# cell_count = 2
# cell_d = 10.5
# cell_L = 44.5
# pitch = 14.0
# block_L = 30.0
# block_W = 14.5
# block_T = 8.8
# wall_t = 2.0

result = (
    cq.Workplane("XY")
    .box(48.5, 39.0, 8.8)
    .cut(
        cq.Workplane("XY")
            .transformed(offset=cq.Vector(0, -7.0, 1.35), rotate=cq.Vector(0, 90, 0))
            .cylinder(46.5, 5.25)
    )
    .cut(
        cq.Workplane("XY")
            .transformed(offset=cq.Vector(0, 7.0, 1.35), rotate=cq.Vector(0, 90, 0))
            .cylinder(46.5, 5.25)
    )
)

# Export
show_object(result)