TEMPLATES = {'tolerance_stack': "\n# Independently designed axial two-block clearance specimen; not source drawings.\n# A and B meet at x=0; hole axes are parallel to X and share y,z locations.\n# The drawing tolerance values are supplied explicitly in the QA text.\nlength_a = P['length_a']\nlength_b = P['length_b']\nwidth = P['width']\nheight = 16.0\ncenters = [(y, 0) for y in [-width / 4, width / 4]]\npart_a = cq.Workplane('YZ', origin=(-length_a, 0, 0)).rect(width, height).extrude(length_a)\npart_b = cq.Workplane('YZ', origin=(0, 0, 0)).rect(width, height).extrude(length_b)\nfor y, z in centers:\n    part_a = part_a.cut(cq.Workplane('YZ', origin=(-length_a - 1, y, z)).circle(P['min_hole_a'] / 2).extrude(length_a + 2))\n    part_b = part_b.cut(cq.Workplane('YZ', origin=(-1, y, z)).circle(P['min_hole_b'] / 2).extrude(length_b + 2))\nfor index in range(P['feature_count']):\n    # Blind relief pockets, remote from the fastening bores.\n    x = -length_a + 4 + index * 5\n    part_a = part_a.cut(cq.Workplane('XY', origin=(x, 0, 6)).circle(1).extrude(3))\nresult = cq.Compound.makeCompound([part_a.val(), part_b.val()])\n"}

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
