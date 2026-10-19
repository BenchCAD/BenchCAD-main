TEMPLATES = {'planar_arm_geartrain': "\n# Independent educational two-link arm and angular transmission analogue.\n# Not a reconstruction of EduSCARA manufacturing geometry.\nl1 = P['link_a']\nl2 = P['link_b']\nteeth_a = P['teeth_a']\nteeth_b = P['teeth_b']\nthickness = P['thickness']\ndef link(length, z):\n    p = cq.Workplane('XY', origin=(length / 2, 0, z)).rect(length, 14).extrude(thickness)\n    for x in [0, length]:\n        p = p.union(cq.Workplane('XY', origin=(x, 0, z)).circle(10).extrude(thickness))\n        p = p.cut(cq.Workplane('XY', origin=(x, 0, z - 1)).circle(3).extrude(thickness + 2))\n    for index in range(P['feature_count']):\n        p = p.cut(cq.Workplane('XY', origin=(length * (index + 1) / (P['feature_count'] + 1), 0, z - 1)).circle(2).extrude(thickness + 2))\n    return p\narm_a = link(l1, 0)\narm_b = link(l2, thickness + 2).translate((l1, 0, 0))\ndef gear(n, x, y):\n    # Countable simplified tooth envelope; ideal nominal ratio, not contact analysis.\n    radius = n * 1.0\n    p = cq.Workplane('XY', origin=(x, y, 0)).circle(radius - 1.3).extrude(4)\n    for index in range(n):\n        a = 2 * math.pi * index / n\n        tooth = cq.Workplane('XY', origin=(x + (radius - 1) * math.cos(a), y + (radius - 1) * math.sin(a), 0)).rect(2, 2).extrude(4)\n        p = p.union(tooth)\n    return p.cut(cq.Workplane('XY', origin=(x, y, -1)).circle(2).extrude(6))\n# Separate exploded view; QA explicitly stipulates ideal external meshing.\ngear_a = gear(teeth_a, 0, -35)\ngear_b = gear(teeth_b, teeth_a + teeth_b + 5, -35)\nresult = cq.Compound.makeCompound([arm_a.val(), arm_b.val(), gear_a.val(), gear_b.val()])\n"}

def parameters(family, difficulty, seed):
    if difficulty not in ('easy', 'medium', 'hard') or not isinstance(seed, int) or seed < 0:
        raise ValueError('Expected easy/medium/hard and a nonnegative integer seed')
    n = {'easy': 0, 'medium': 1, 'hard': 3}[difficulty]
    # Fifty non-proportional variants per difficulty. Seed zero is the anchor.
    k = seed % 50
    p = {'feature_count': n, 'seed': seed}
    if family == 'bearing_plunger':
        p.update(diameter=12 + .04 * k, shaft_length=45 + .7 * k)
    elif family == 'tolerance_stack':
        p.update(length_a=25 + .08 * k, length_b=25 + .03 * k, width=30 + .3 * k,
                 min_hole_a=5.8 + .01 * k, min_hole_b=5.7 + .01 * k)
    elif family == 'cycloidal_cartridge':
        p.update(rollers=12 + 2 * (k % 3), pitch_radius=41.5 + .18 * k,
                 eccentricity=2.025 + .004 * k, thickness=4 + .02 * k)
    elif family == 'profiled_support_plate':
        p.update(scale_x=1 + .004 * k, scale_y=1 + .001 * k, thickness=2 + .025 * k, hole_centers=[[-40, 0], [60, -20], [-30, 40]])
    elif family == 'dual_drive_metering_head':
        p.update(lead=8 + .02 * k, bore=4.61 + .004 * k, large_teeth=60 + k % 3)
    elif family == 'planar_arm_geartrain':
        p.update(link_a=125 + .4 * k, link_b=100 + .15 * k, teeth_a=16 + k % 3,
                 teeth_b=8 + k % 2, thickness=5 + .03 * k)
    else:
        raise ValueError(family)
    return p


def code_for(family, difficulty, seed, override=None):
    p = parameters(family, difficulty, seed)
    p.update(override or {})
    imports = 'import cadquery as cq\n'
    if family in ('cycloidal_cartridge', 'dual_drive_metering_head', 'planar_arm_geartrain'):
        imports += 'import math\n'
    return imports + '\nP = ' + repr(p) + '\n' + TEMPLATES[family]
