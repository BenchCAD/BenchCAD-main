TEMPLATES = {'dual_drive_metering_head': "\n# Independent nominal mechanism; no MULA CAD or manual image is copied.\n# Functional connections are explicit construction constraints, not fit claims.\nlead = P['lead']\nbore = P['bore']\nlarge_teeth = P['large_teeth']\nmotor_teeth = 20\noutput_teeth = 20\nbelt_pitch = 2.0\n# A rigid frame carries both the barrel cradle and the screw thrust bearing.\nframe = cq.Workplane('XY').box(70, 45, 4)\nframe = frame.union(cq.Workplane('XY', origin=(-25, 0, 2)).box(4, 35, 50, centered=(True, True, False)))\n# Lower and upper thrust seats attach the simplified screw to the rigid frame.\nseat = cq.Workplane('XY', origin=(-15, 0, 2)).circle(5.5).circle(3.1).extrude(6)\nframe = frame.union(seat)\nupper = cq.Workplane('XY', origin=(-20, 0, 52)).box(20, 12, 4, centered=(True, True, False))\nupper = upper.cut(cq.Workplane('XY', origin=(-15, 0, 51)).circle(3.1).extrude(6))\nframe = frame.union(upper)\nbarrel = cq.Workplane('XY', origin=(0, 0, 8)).circle(bore / 2 + 1.5).circle(bore / 2).extrude(30)\ncradle = cq.Workplane('XY', origin=(-5, 0, 2)).box(30, 14, 6, centered=(True, True, False)).cut(cq.Workplane('XY', origin=(0, 0, 0)).circle(bore / 2 + 1.6).extrude(50))\n# Screw A connects a fixed thrust seat to the plunger's translating nut.\nscrew = cq.Workplane('XY', origin=(-15, 0, 8)).circle(3).extrude(45)\nnut = cq.Workplane('XY', origin=(-15, 0, 44)).rect(10, 10).extrude(5).cut(cq.Workplane('XY', origin=(-15, 0, 43)).circle(3.1).extrude(7))\nplunger_bridge = cq.Workplane('XY', origin=(-7.5, 0, 49)).box(25, 7, 3)\nplunger = cq.Workplane('XY', origin=(0, 0, 22)).circle(bore / 2 - .1).extrude(26)\n# Lead is a design parameter; the purchased screw is a simplified cylinder.\ndef pulley(teeth, x, z):\n    radius = teeth * belt_pitch / (2 * math.pi)\n    p = cq.Workplane('XY', origin=(x, 0, z)).circle(radius - 1).extrude(4)\n    for index in range(teeth):\n        a = 2 * math.pi * index / teeth\n        tooth = cq.Workplane('XY', origin=(x + (radius - .6) * math.cos(a), (radius - .6) * math.sin(a), z)).rect(1.3, 1.3).extrude(4)\n        p = p.union(tooth)\n    return p.cut(cq.Workplane('XY', origin=(x, 0, z - 1)).circle(2).extrude(6))\n# Drive B's large and linear-output pulleys share one rigid shaft.\ngear_axis_x = 75\nlarge = pulley(large_teeth, gear_axis_x, 0)\nlinear = pulley(output_teeth, gear_axis_x, 7)\nmotor = pulley(motor_teeth, gear_axis_x + (large_teeth + motor_teeth) * belt_pitch / (2 * math.pi) + 8, 0)\nshared_shaft = cq.Workplane('XY', origin=(gear_axis_x, 0, -3)).circle(1.9).extrude(16)\nsolids = [frame.val(), barrel.val(), cradle.val(), screw.val(), nut.val(), plunger_bridge.val(), plunger.val(), large.val(), linear.val(), motor.val(), shared_shaft.val()]\nfor index in range(P['feature_count']):\n    solids.append(cq.Workplane('XY', origin=(-20 + index * 12, -18, 2)).circle(3).circle(1).extrude(4).val())\nresult = cq.Compound.makeCompound(solids)\n"}

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
        p.update(scale_x=1 + .004 * k, scale_y=1 + .001 * k, thickness=2 + .025 * k, hole_centers=[[-45, -25], [45, -25], [35, 35]])
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
