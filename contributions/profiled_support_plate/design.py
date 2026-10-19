TEMPLATES = {'profiled_support_plate': "# Independently authored support side-plate. No source DXF coordinates copied.\n# Exact cubic Bezier segments, two enclosed openings and one edge-connected slot.\nfrom OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge\nfrom OCP.Geom import Geom_BezierCurve\nfrom OCP.TColgp import TColgp_Array1OfPnt\nfrom OCP.gp import gp_Pnt\nsegments = [\n [(-70,-30),(-70,-40),(-65,-45),(-55,-45)],\n [(-55,-45),(50,-45)],\n [(50,-45),(65,-45),(70,-35),(70,-25)],\n [(70,-25),(70,15)],\n [(70,15),(70,35),(58,55),(45,55)],\n [(45,55),(8,55)],[(8,55),(8,35)],\n [(8,35),(8,25),(-8,25),(-8,35)],\n [(-8,35),(-8,55)],[(-8,55),(-35,55)],\n [(-35,55),(-50,55),(-60,45),(-60,30)],\n [(-60,30),(-60,0)],\n [(-60,0),(-60,-15),(-70,-15),(-70,-30)]]\nedges = []\nfor points in segments:\n    if len(points) == 2:\n        edges.append(cq.Edge.makeLine(cq.Vector(*points[0], 0), cq.Vector(*points[1], 0)))\n    else:\n        poles = TColgp_Array1OfPnt(1, len(points))\n        for i, (x, y) in enumerate(points, 1):\n            poles.SetValue(i, gp_Pnt(x, y, 0))\n        edges.append(cq.Edge(BRepBuilderAPI_MakeEdge(Geom_BezierCurve(poles)).Edge()))\nwire = cq.Wire.assembleEdges(edges)\nplate = cq.Workplane('XY').newObject([cq.Solid.extrudeLinear(wire, [], cq.Vector(0, 0, P['thickness']))])\nplate = plate.cut(cq.Workplane('XY', origin=(-25, 5, -1)).ellipse(8, 12).extrude(P['thickness'] + 2))\nopening = cq.Workplane('XY', origin=(25, 5, -1)).rect(25, 30).extrude(P['thickness'] + 2).edges('|Z').fillet(5)\nplate = plate.cut(opening)\nplate = plate.val().transformGeometry(cq.Matrix([[P['scale_x'], 0, 0, 0], [0, P['scale_y'], 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]))\nresult = cq.Workplane('XY').newObject([plate])\nfor x, y in P['hole_centers'][:P['feature_count']]:\n    cutter = cq.Workplane('XY', origin=(x * P['scale_x'], y * P['scale_y'], -1)).circle(2).extrude(P['thickness'] + 2)\n    result = result.cut(cutter)\n"}

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
