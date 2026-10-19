import cadquery as cq
import math

P = {'feature_count': 3, 'seed': 0, 'link_a': 125.0, 'link_b': 100.0, 'teeth_a': 16, 'teeth_b': 8, 'thickness': 5.0}

# Independent educational two-link arm and angular transmission analogue.
# Not a reconstruction of EduSCARA manufacturing geometry.
l1 = P['link_a']
l2 = P['link_b']
teeth_a = P['teeth_a']
teeth_b = P['teeth_b']
thickness = P['thickness']
def link(length, z):
    p = cq.Workplane('XY', origin=(length / 2, 0, z)).rect(length, 14).extrude(thickness)
    for x in [0, length]:
        p = p.union(cq.Workplane('XY', origin=(x, 0, z)).circle(10).extrude(thickness))
        p = p.cut(cq.Workplane('XY', origin=(x, 0, z - 1)).circle(3).extrude(thickness + 2))
    for index in range(P['feature_count']):
        p = p.cut(cq.Workplane('XY', origin=(length * (index + 1) / (P['feature_count'] + 1), 0, z - 1)).circle(2).extrude(thickness + 2))
    return p
arm_a = link(l1, 0)
arm_b = link(l2, thickness + 2).translate((l1, 0, 0))
def gear(n, x, y):
    # Countable simplified tooth envelope; ideal nominal ratio, not contact analysis.
    radius = n * 1.0
    p = cq.Workplane('XY', origin=(x, y, 0)).circle(radius - 1.3).extrude(4)
    for index in range(n):
        a = 2 * math.pi * index / n
        tooth = cq.Workplane('XY', origin=(x + (radius - 1) * math.cos(a), y + (radius - 1) * math.sin(a), 0)).rect(2, 2).extrude(4)
        p = p.union(tooth)
    return p.cut(cq.Workplane('XY', origin=(x, y, -1)).circle(2).extrude(6))
# Separate exploded view; QA explicitly stipulates ideal external meshing.
gear_a = gear(teeth_a, 0, -35)
gear_b = gear(teeth_b, teeth_a + teeth_b + 5, -35)
result = cq.Compound.makeCompound([arm_a.val(), arm_b.val(), gear_a.val(), gear_b.val()])
