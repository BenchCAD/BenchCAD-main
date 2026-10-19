TEMPLATES = {'cycloidal_cartridge': "\n# Analytic profile formula: Wesley Roozing and Glenn Roozing, MIT (2022).\n# New CAD construction and dimensions; not the authors' complete assembly.\n# Finite polyline approximation, 48 samples per lobe; nominal kinematics only.\nrollers = P['rollers']\nlobes = rollers - 1\npitch_radius = P['pitch_radius']\neccentricity = P['eccentricity']\nroller_radius = 4.5\noutput_bearing_radius = 4.5\noutput_count = 6\noutput_pitch_radius = pitch_radius * .60\njournal_radius = pitch_radius * .28\nthickness = P['thickness']\nprofile = []\nfor i in range(lobes * 48):\n    t = 2 * math.pi * i / (lobes * 48)\n    psi = math.atan2(math.sin((1 - rollers) * t), pitch_radius / (eccentricity * rollers) - math.cos((1 - rollers) * t))\n    x = pitch_radius * math.cos(t) - roller_radius * math.cos(t + psi) - eccentricity * math.cos(rollers * t)\n    y = -pitch_radius * math.sin(t) + roller_radius * math.sin(t + psi) + eccentricity * math.sin(rollers * t)\n    profile.append((x, y))\ndisk = cq.Workplane('XY').polyline(profile).close().extrude(thickness)\ndisk = disk.cut(cq.Workplane('XY', origin=(0, 0, -1)).circle(journal_radius + .1).extrude(thickness + 2))\nfor index in range(output_count):\n    a = 2 * math.pi * index / output_count\n    disk = disk.cut(cq.Workplane('XY', origin=(output_pitch_radius * math.cos(a), output_pitch_radius * math.sin(a), -1)).circle(output_bearing_radius + eccentricity).extrude(thickness + 2))\n# Exploded axial arrangement makes the two identical-profile disks visible.\ndisk_a = disk.translate((eccentricity, 0, 0))\ndisk_b = disk.rotate((0, 0, 0), (0, 0, 1), 180 / lobes).translate((-eccentricity, 0, thickness + 7))\nshaft = cq.Workplane('XY', origin=(0, 0, -5)).circle(5).extrude(2 * thickness + 19)\nshaft = shaft.union(cq.Workplane('XY', origin=(eccentricity, 0, 0)).circle(journal_radius).extrude(thickness))\nshaft = shaft.union(cq.Workplane('XY', origin=(-eccentricity, 0, thickness + 7)).circle(journal_radius).extrude(thickness))\nsolids = [disk_a.val(), disk_b.val(), shaft.val()]\nfor index in range(rollers):\n    a = 2 * math.pi * index / rollers\n    solids.append(cq.Workplane('XY', origin=(pitch_radius * math.cos(a), pitch_radius * math.sin(a), -1)).circle(roller_radius).extrude(2 * thickness + 9).val())\n# Reference output bearings are shown above the exploded disk pair.\nfor index in range(output_count):\n    a = 2 * math.pi * index / output_count\n    solids.append(cq.Workplane('XY', origin=(output_pitch_radius * math.cos(a), output_pitch_radius * math.sin(a), 2 * thickness + 10)).circle(output_bearing_radius).circle(1.5).extrude(3).val())\nfor index in range(P['feature_count']):\n    # Axial guide rings do not alter the nominal transmission.\n    solids.append(cq.Workplane('XY', origin=(0, 0, -3 - index * 2)).circle(9 + index).circle(5.1).extrude(1).val())\nresult = cq.Compound.makeCompound(solids)\n"}

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
