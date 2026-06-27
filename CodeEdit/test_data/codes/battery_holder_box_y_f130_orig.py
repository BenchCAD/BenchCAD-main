import cadquery as cq

# --- parameters ---
# cell_count
# cell_d
# cell_L
# pitch
# block_L
# block_W
# block_T
# wall_t

result = (
    cq.Workplane("XY")
    .box(48.5, 30.0, 8.8)
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