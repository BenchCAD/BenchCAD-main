import cadquery as cq

P = {'feature_count': 3, 'seed': 0, 'scale_x': 1.0, 'scale_y': 1.0, 'thickness': 2.0, 'hole_centers': [[-45, -25], [45, -25], [35, 35]]}
# Independently authored support side-plate. No source DXF coordinates copied.
# Exact cubic Bezier segments, two enclosed openings and one edge-connected slot.
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge
from OCP.Geom import Geom_BezierCurve
from OCP.TColgp import TColgp_Array1OfPnt
from OCP.gp import gp_Pnt
segments = [
 [(-70,-30),(-70,-40),(-65,-45),(-55,-45)],
 [(-55,-45),(50,-45)],
 [(50,-45),(65,-45),(70,-35),(70,-25)],
 [(70,-25),(70,15)],
 [(70,15),(70,35),(58,55),(45,55)],
 [(45,55),(8,55)],[(8,55),(8,35)],
 [(8,35),(8,25),(-8,25),(-8,35)],
 [(-8,35),(-8,55)],[(-8,55),(-35,55)],
 [(-35,55),(-50,55),(-60,45),(-60,30)],
 [(-60,30),(-60,0)],
 [(-60,0),(-60,-15),(-70,-15),(-70,-30)]]
edges = []
for points in segments:
    if len(points) == 2:
        edges.append(cq.Edge.makeLine(cq.Vector(*points[0], 0), cq.Vector(*points[1], 0)))
    else:
        poles = TColgp_Array1OfPnt(1, len(points))
        for i, (x, y) in enumerate(points, 1):
            poles.SetValue(i, gp_Pnt(x, y, 0))
        edges.append(cq.Edge(BRepBuilderAPI_MakeEdge(Geom_BezierCurve(poles)).Edge()))
wire = cq.Wire.assembleEdges(edges)
plate = cq.Workplane('XY').newObject([cq.Solid.extrudeLinear(wire, [], cq.Vector(0, 0, P['thickness']))])
plate = plate.cut(cq.Workplane('XY', origin=(-25, 5, -1)).ellipse(8, 12).extrude(P['thickness'] + 2))
opening = cq.Workplane('XY', origin=(25, 5, -1)).rect(25, 30).extrude(P['thickness'] + 2).edges('|Z').fillet(5)
plate = plate.cut(opening)
plate = plate.val().transformGeometry(cq.Matrix([[P['scale_x'], 0, 0, 0], [0, P['scale_y'], 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]))
result = cq.Workplane('XY').newObject([plate])
for x, y in P['hole_centers'][:P['feature_count']]:
    cutter = cq.Workplane('XY', origin=(x * P['scale_x'], y * P['scale_y'], -1)).circle(2).extrude(P['thickness'] + 2)
    result = result.cut(cutter)
