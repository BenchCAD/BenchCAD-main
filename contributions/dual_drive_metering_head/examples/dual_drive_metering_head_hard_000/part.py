import cadquery as cq
import math

P = {'feature_count': 3, 'seed': 0, 'lead': 8.0, 'bore': 4.61, 'large_teeth': 60}

# Independent nominal mechanism; no MULA CAD or manual image is copied.
# Functional connections are explicit construction constraints, not fit claims.
lead = P['lead']
bore = P['bore']
large_teeth = P['large_teeth']
motor_teeth = 20
output_teeth = 20
belt_pitch = 2.0
# A rigid frame carries both the barrel cradle and the screw thrust bearing.
frame = cq.Workplane('XY').box(70, 45, 4)
frame = frame.union(cq.Workplane('XY', origin=(-25, 0, 2)).box(4, 35, 50, centered=(True, True, False)))
# Lower and upper thrust seats attach the simplified screw to the rigid frame.
seat = cq.Workplane('XY', origin=(-15, 0, 2)).circle(5.5).circle(3.1).extrude(6)
frame = frame.union(seat)
upper = cq.Workplane('XY', origin=(-20, 0, 52)).box(20, 12, 4, centered=(True, True, False))
upper = upper.cut(cq.Workplane('XY', origin=(-15, 0, 51)).circle(3.1).extrude(6))
frame = frame.union(upper)
barrel = cq.Workplane('XY', origin=(0, 0, 8)).circle(bore / 2 + 1.5).circle(bore / 2).extrude(30)
cradle = cq.Workplane('XY', origin=(-5, 0, 2)).box(30, 14, 6, centered=(True, True, False)).cut(cq.Workplane('XY', origin=(0, 0, 0)).circle(bore / 2 + 1.6).extrude(50))
# Screw A connects a fixed thrust seat to the plunger's translating nut.
screw = cq.Workplane('XY', origin=(-15, 0, 8)).circle(3).extrude(45)
nut = cq.Workplane('XY', origin=(-15, 0, 44)).rect(10, 10).extrude(5).cut(cq.Workplane('XY', origin=(-15, 0, 43)).circle(3.1).extrude(7))
plunger_bridge = cq.Workplane('XY', origin=(-7.5, 0, 49)).box(25, 7, 3)
plunger = cq.Workplane('XY', origin=(0, 0, 22)).circle(bore / 2 - .1).extrude(26)
# Lead is a design parameter; the purchased screw is a simplified cylinder.
def pulley(teeth, x, z):
    radius = teeth * belt_pitch / (2 * math.pi)
    p = cq.Workplane('XY', origin=(x, 0, z)).circle(radius - 1).extrude(4)
    for index in range(teeth):
        a = 2 * math.pi * index / teeth
        tooth = cq.Workplane('XY', origin=(x + (radius - .6) * math.cos(a), (radius - .6) * math.sin(a), z)).rect(1.3, 1.3).extrude(4)
        p = p.union(tooth)
    return p.cut(cq.Workplane('XY', origin=(x, 0, z - 1)).circle(2).extrude(6))
# Drive B's large and linear-output pulleys share one rigid shaft.
gear_axis_x = 75
large = pulley(large_teeth, gear_axis_x, 0)
linear = pulley(output_teeth, gear_axis_x, 7)
motor = pulley(motor_teeth, gear_axis_x + (large_teeth + motor_teeth) * belt_pitch / (2 * math.pi) + 8, 0)
shared_shaft = cq.Workplane('XY', origin=(gear_axis_x, 0, -3)).circle(1.9).extrude(16)
solids = [frame.val(), barrel.val(), cradle.val(), screw.val(), nut.val(), plunger_bridge.val(), plunger.val(), large.val(), linear.val(), motor.val(), shared_shaft.val()]
for index in range(P['feature_count']):
    solids.append(cq.Workplane('XY', origin=(-20 + index * 12, -18, 2)).circle(3).circle(1).extrude(4).val())
result = cq.Compound.makeCompound(solids)
