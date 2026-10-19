import cadquery as cq

P = {'feature_count': 0, 'seed': 0, 'diameter': 12.0, 'shaft_length': 45.0}

# Independently designed stepped plunger, not an orthosis replica.
shaft_radius = P['diameter'] / 2
shaft_length = P['shaft_length']
head_radius = shaft_radius * 1.7
head_thickness = 5.0
result = cq.Workplane('XY').circle(head_radius).extrude(head_thickness)
result = result.union(cq.Workplane('XY', origin=(0, 0, head_thickness)).circle(shaft_radius).extrude(shaft_length))
result = result.union(cq.Workplane('XY', origin=(0, 0, head_thickness + shaft_length)).circle(shaft_radius * .65).extrude(6))
for index in range(P['feature_count']):
    z = head_thickness + 8 + index * 6
    cutter = cq.Workplane('XY', origin=(0, 0, z)).circle(shaft_radius + .5).circle(shaft_radius - .6).extrude(1)
    result = result.cut(cutter)
