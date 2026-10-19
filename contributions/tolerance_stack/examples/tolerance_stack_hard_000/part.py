import cadquery as cq

P = {'feature_count': 3, 'seed': 0, 'length_a': 25.0, 'length_b': 25.0, 'width': 30.0, 'min_hole_a': 5.8, 'min_hole_b': 5.7}

# Independently designed axial two-block clearance specimen; not source drawings.
# A and B meet at x=0; hole axes are parallel to X and share y,z locations.
# The drawing tolerance values are supplied explicitly in the QA text.
length_a = P['length_a']
length_b = P['length_b']
width = P['width']
height = 16.0
centers = [(y, 0) for y in [-width / 4, width / 4]]
part_a = cq.Workplane('YZ', origin=(-length_a, 0, 0)).rect(width, height).extrude(length_a)
part_b = cq.Workplane('YZ', origin=(0, 0, 0)).rect(width, height).extrude(length_b)
for y, z in centers:
    part_a = part_a.cut(cq.Workplane('YZ', origin=(-length_a - 1, y, z)).circle(P['min_hole_a'] / 2).extrude(length_a + 2))
    part_b = part_b.cut(cq.Workplane('YZ', origin=(-1, y, z)).circle(P['min_hole_b'] / 2).extrude(length_b + 2))
for index in range(P['feature_count']):
    # Blind relief pockets, remote from the fastening bores.
    x = -length_a + 4 + index * 5
    part_a = part_a.cut(cq.Workplane('XY', origin=(x, 0, 6)).circle(1).extrude(3))
result = cq.Compound.makeCompound([part_a.val(), part_b.val()])
