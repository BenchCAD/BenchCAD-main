TEMPLATES = {'bearing_plunger': "\n# Independently designed stepped plunger, not an orthosis replica.\nshaft_radius = P['diameter'] / 2\nshaft_length = P['shaft_length']\nhead_radius = shaft_radius * 1.7\nhead_thickness = 5.0\nresult = cq.Workplane('XY').circle(head_radius).extrude(head_thickness)\nresult = result.union(cq.Workplane('XY', origin=(0, 0, head_thickness)).circle(shaft_radius).extrude(shaft_length))\nresult = result.union(cq.Workplane('XY', origin=(0, 0, head_thickness + shaft_length)).circle(shaft_radius * .65).extrude(6))\nfor index in range(P['feature_count']):\n    z = head_thickness + 8 + index * 6\n    cutter = cq.Workplane('XY', origin=(0, 0, z)).circle(shaft_radius + .5).circle(shaft_radius - .6).extrude(1)\n    result = result.cut(cutter)\n"}

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
