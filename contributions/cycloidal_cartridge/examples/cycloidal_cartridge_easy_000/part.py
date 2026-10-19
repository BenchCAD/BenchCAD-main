import cadquery as cq
import math

P = {'feature_count': 0, 'seed': 0, 'rollers': 12, 'pitch_radius': 41.5, 'eccentricity': 2.025, 'thickness': 4.0}

# Analytic profile formula: Wesley Roozing and Glenn Roozing, MIT (2022).
# New CAD construction and dimensions; not the authors' complete assembly.
# Finite polyline approximation, 48 samples per lobe; nominal kinematics only.
rollers = P['rollers']
lobes = rollers - 1
pitch_radius = P['pitch_radius']
eccentricity = P['eccentricity']
roller_radius = 4.5
output_bearing_radius = 4.5
output_count = 6
output_pitch_radius = pitch_radius * .60
journal_radius = pitch_radius * .28
thickness = P['thickness']
profile = []
for i in range(lobes * 48):
    t = 2 * math.pi * i / (lobes * 48)
    psi = math.atan2(math.sin((1 - rollers) * t), pitch_radius / (eccentricity * rollers) - math.cos((1 - rollers) * t))
    x = pitch_radius * math.cos(t) - roller_radius * math.cos(t + psi) - eccentricity * math.cos(rollers * t)
    y = -pitch_radius * math.sin(t) + roller_radius * math.sin(t + psi) + eccentricity * math.sin(rollers * t)
    profile.append((x, y))
disk = cq.Workplane('XY').polyline(profile).close().extrude(thickness)
disk = disk.cut(cq.Workplane('XY', origin=(0, 0, -1)).circle(journal_radius + .1).extrude(thickness + 2))
for index in range(output_count):
    a = 2 * math.pi * index / output_count
    disk = disk.cut(cq.Workplane('XY', origin=(output_pitch_radius * math.cos(a), output_pitch_radius * math.sin(a), -1)).circle(output_bearing_radius + eccentricity).extrude(thickness + 2))
# Exploded axial arrangement makes the two identical-profile disks visible.
disk_a = disk.translate((eccentricity, 0, 0))
disk_b = disk.rotate((0, 0, 0), (0, 0, 1), 180 / lobes).translate((-eccentricity, 0, thickness + 7))
shaft = cq.Workplane('XY', origin=(0, 0, -5)).circle(5).extrude(2 * thickness + 19)
shaft = shaft.union(cq.Workplane('XY', origin=(eccentricity, 0, 0)).circle(journal_radius).extrude(thickness))
shaft = shaft.union(cq.Workplane('XY', origin=(-eccentricity, 0, thickness + 7)).circle(journal_radius).extrude(thickness))
solids = [disk_a.val(), disk_b.val(), shaft.val()]
for index in range(rollers):
    a = 2 * math.pi * index / rollers
    solids.append(cq.Workplane('XY', origin=(pitch_radius * math.cos(a), pitch_radius * math.sin(a), -1)).circle(roller_radius).extrude(2 * thickness + 9).val())
# Reference output bearings are shown above the exploded disk pair.
for index in range(output_count):
    a = 2 * math.pi * index / output_count
    solids.append(cq.Workplane('XY', origin=(output_pitch_radius * math.cos(a), output_pitch_radius * math.sin(a), 2 * thickness + 10)).circle(output_bearing_radius).circle(1.5).extrude(3).val())
for index in range(P['feature_count']):
    # Axial guide rings do not alter the nominal transmission.
    solids.append(cq.Workplane('XY', origin=(0, 0, -3 - index * 2)).circle(9 + index).circle(5.1).extrude(1).val())
result = cq.Compound.makeCompound(solids)
