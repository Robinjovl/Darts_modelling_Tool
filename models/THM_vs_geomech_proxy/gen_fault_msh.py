import gmsh
import math
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
def generate_3d_fault_mesh(
    input_data,
    msh_filename=os.path.join(BASE_DIR, 'meshes', 'case_5', "mesh.msh"),
    fault_dip_degrees=45.0,
    reservoir_block_offset=180.0,
    damage_width_left=100,
    damage_width_right=200,
    well_mesh_size=10.0,
    well_cylinder_radius=20.0,
    well_transition_radius=250.0,
):
    gmsh.initialize()
    gmsh.model.add("fault_1")
    geo = gmsh.model.occ

    # ---- Parameters ----
    #W = H =  4500.0
    W = H = 10000.0  # [m]
    a, b = 30.0, 230.0
    Lplus = b + 150
    phi = math.radians(fault_dip_degrees)
    lc = 300.0
    mult, mult1 = 0.7, 0.3
    fault_offset = 500.0
    fault_thickness = 20.0  # true thickness normal to the fault [m]

    # Keep the block containing both wells fixed at z_old=[-b, -a].
    # Shift only the opposite block by this rigid vertical offset.
    right_reservoir_min = -b + reservoir_block_offset
    right_reservoir_max = -a + reservoir_block_offset
    aligned_blocks = abs(reservoir_block_offset) < 1.0e-9
    overlapping_blocks = (
        0.0 < reservoir_block_offset < (b - a)
    )

    if abs(math.sin(phi)) < 1.0e-8:
        raise ValueError("fault_dip_degrees must not be 0 or 180 degrees")
    if right_reservoir_min <= -400.0 or right_reservoir_max >= 400.0:
        raise ValueError(
            "reservoir_block_offset places the movable reservoir outside "
            "the central model interval (-400, 400 m)"
        )
    if not (0.0 < well_mesh_size <= lc):
        raise ValueError("well_mesh_size must be in the interval (0, lc]")
    if well_cylinder_radius <= 0.0:
        raise ValueError("well_cylinder_radius must be positive")
    if well_transition_radius <= well_cylinder_radius:
        raise ValueError(
            "well_transition_radius must be larger than well_cylinder_radius"
        )
    prod_well_coords = input_data.other.prod_well_coords
    inj_well_coords = input_data.other.inj_well_coords
    if prod_well_coords is None or inj_well_coords is None:
        raise ValueError(
            "input_data.other.prod_well_coords and inj_well_coords are required"
        )
    well_coords = (prod_well_coords, inj_well_coords)
    if any(len(coords) != 4 for coords in well_coords):
        raise ValueError(
            "prod_well_coords and inj_well_coords must be [X, Y, Z1, Z2]"
        )
    if any(coords[2] == coords[3] for coords in well_coords):
        raise ValueError("Each well must have different Z1 and Z2 coordinates")

    # ======================================================
    # NEW: asymmetric damage-zone thickness (meters)
    # left  side  = shift in -y direction
    # right side  = shift in +y direction
    # ======================================================
    damage_left = damage_width_left     # m
    damage_right = damage_width_right   # m

    # Material and boundary physical tags
    RES = 1
    OVERBURDEN = 2
    UNDERBURDEN = 3
    DAMAGEZONE_LEFT = 4
    DAMAGEZONE_RIGHT = 5
    FAULT = 99991

    LEFTBOUNDARY = 991
    RIGHTBOUNDARY = 992
    FRONTBOUNDARY = 993
    BACKBOUNDARY = 994
    BOTTOMBOUNDARY = 995
    TOPBOUNDARY = 996

    # ---- Coordinate transforms ----
    def z_new(z_old): return 2800.0 - z_old
    def x_new(x_old): return x_old + W / 2
    def y_new(y_old): return y_old + H / 2

    addP, addL, addCL, addPS, addSL, addV = (
        geo.addPoint, geo.addLine, geo.addCurveLoop,
        geo.addPlaneSurface, geo.addSurfaceLoop, geo.addVolume
    )

    # ======================================================
    # Fault plane scalar function (parallel planes):
    # In OLD coords:  y_old - z_old/tan(phi) = fault_y
    # In NEW coords:  y_new + z_new/tan(phi) = const
    # Translating in y only changes the constant -> parallel planes.
    # ======================================================
    tanphi = math.tan(phi)
    # Translating a plane by dy changes its normal separation by
    # dy*sin(phi), hence this conversion from true thickness to y offset.
    fault_dy = fault_thickness / abs(math.sin(phi))
    c0 = (fault_offset + H / 2.0) + (2800.0 / tanphi)
    c_left = c0 - damage_left                     # shift -y
    c_fault_right = c0 + fault_dy
    c_right = c_fault_right + damage_right

    def fault_s(yN, zN):
        return yN + zN / tanphi

    # --------- outer box points (transformed) ---------
    addP(x_new(W / 2), y_new(-H / 2), z_new(-400.0), lc, tag=1)
    addP(x_new(W / 2), y_new(H / 2), z_new(-400.0), lc, tag=2)
    addP(x_new(-W / 2), y_new(H / 2), z_new(-400.0), lc, tag=3)
    addP(x_new(-W / 2), y_new(-H / 2), z_new(-400.0), lc, tag=4)
    addP(x_new(W / 2), y_new(-H / 2), z_new(400.0), lc, tag=5)
    addP(x_new(W / 2), y_new(H / 2), z_new(400.0), lc, tag=6)
    addP(x_new(-W / 2), y_new(H / 2), z_new(400.0), lc, tag=7)
    addP(x_new(-W / 2), y_new(-H / 2), z_new(400.0), lc, tag=8)

    # --------- fault front (x=+W/2) ----------
    addP(x_new(W / 2), y_new(-b / tanphi + fault_offset), z_new(-b), mult1 * lc, tag=16)
    addP(x_new(W / 2), y_new(-a / tanphi + fault_offset), z_new(-a), mult1 * lc, tag=17)
    if not aligned_blocks:
        addP(x_new(W / 2), y_new(right_reservoir_min / tanphi + fault_offset), z_new(right_reservoir_min), mult1 * lc, tag=18)
        addP(x_new(W / 2), y_new(right_reservoir_max / tanphi + fault_offset), z_new(right_reservoir_max), mult1 * lc, tag=19)
    addP(x_new(W / 2), y_new(-Lplus / tanphi + fault_offset), z_new(-400.0), mult1 * lc, tag=15)
    addP(x_new(W / 2), y_new(Lplus / tanphi + fault_offset), z_new(400.0), mult1 * lc, tag=20)

    # --------- fault back (x=-W/2) ----------
    addP(x_new(-W / 2), y_new(-b / tanphi + fault_offset), z_new(-b), mult1 * lc, tag=116)
    addP(x_new(-W / 2), y_new(-a / tanphi + fault_offset), z_new(-a), mult1 * lc, tag=117)
    if not aligned_blocks:
        addP(x_new(-W / 2), y_new(right_reservoir_min / tanphi + fault_offset), z_new(right_reservoir_min), mult1 * lc, tag=118)
        addP(x_new(-W / 2), y_new(right_reservoir_max / tanphi + fault_offset), z_new(right_reservoir_max), mult1 * lc, tag=119)
    addP(x_new(-W / 2), y_new(-Lplus / tanphi + fault_offset), z_new(-400.0), mult1 * lc, tag=115)
    addP(x_new(-W / 2), y_new(Lplus / tanphi + fault_offset), z_new(400.0), mult1 * lc, tag=120)

    # --------- frame split points ----------
    addP(x_new(-W / 2), y_new(-H / 2), z_new(-b), mult * lc, tag=203)
    addP(x_new(W / 2), y_new(-H / 2), z_new(-b), mult * lc, tag=204)
    addP(x_new(W / 2), y_new(-H / 2), z_new(-a), mult * lc, tag=201)
    addP(x_new(-W / 2), y_new(-H / 2), z_new(-a), mult * lc, tag=202)
    addP(x_new(W / 2), y_new(H / 2), z_new(right_reservoir_max), mult * lc, tag=301)
    addP(x_new(-W / 2), y_new(H / 2), z_new(right_reservoir_max), mult * lc, tag=302)
    addP(x_new(W / 2), y_new(H / 2), z_new(right_reservoir_min), mult * lc, tag=303)
    addP(x_new(-W / 2), y_new(H / 2), z_new(right_reservoir_min), mult * lc, tag=304)

    # --------- bottom and top edges ----------
    addL(1, 15, tag=1)
    addL(15, 2, tag=209)
    addL(2, 3, tag=2)
    addL(3, 115, tag=3)
    addL(115, 4, tag=210)
    addL(4, 1, tag=4)

    addL(5, 20, tag=5)
    addL(20, 6, tag=211)
    addL(6, 7, tag=6)
    addL(7, 120, tag=7)
    addL(120, 8, tag=212)
    addL(8, 5, tag=8)

    # --------- vertical frame lines ----------
    addL(1, 204, tag=11)
    addL(201, 204, tag=12)
    addL(201, 5, tag=13)
    addL(2, 303, tag=14)
    addL(301, 303, tag=15)
    addL(301, 6, tag=16)
    addL(3, 304, tag=17)
    addL(302, 304, tag=18)
    addL(302, 7, tag=19)
    addL(4, 203, tag=20)
    addL(202, 203, tag=21)
    addL(202, 8, tag=22)

    # --------- fault polylines ----------
    if aligned_blocks:
        # With zero throw the left and right contacts are the same points.
        addL(20, 17, tag=122)
        addL(17, 16, tag=123)
        addL(16, 15, tag=126)
        addL(120, 117, tag=127)
        addL(117, 116, tag=128)
        addL(116, 115, tag=131)
        right_lower_front, right_upper_front = 16, 17
        right_lower_back, right_upper_back = 116, 117
    elif overlapping_blocks:
        # Contact order from top to bottom is: right top, fixed top,
        # right bottom, fixed bottom.
        addL(20, 19, tag=122)
        addL(19, 17, tag=123)
        addL(17, 18, tag=124)
        addL(18, 16, tag=125)
        addL(16, 15, tag=126)
        addL(120, 119, tag=127)
        addL(119, 117, tag=128)
        addL(117, 118, tag=129)
        addL(118, 116, tag=130)
        addL(116, 115, tag=131)
        right_lower_front, right_upper_front = 18, 19
        right_lower_back, right_upper_back = 118, 119
    else:
        addL(20, 19, tag=122)
        addL(19, 18, tag=123)
        addL(18, 17, tag=124)
        addL(17, 16, tag=125)
        addL(16, 15, tag=126)
        addL(120, 119, tag=127)
        addL(119, 118, tag=128)
        addL(118, 117, tag=129)
        addL(117, 116, tag=130)
        addL(116, 115, tag=131)
        right_lower_front, right_upper_front = 18, 19
        right_lower_back, right_upper_back = 118, 119
    addL(115, 15, tag=133)
    addL(20, 120, tag=134)

    # --------- connectors ----------
    addL(201, 17, tag=33)
    addL(17, 117, tag=34)
    addL(117, 202, tag=35)
    addL(202, 201, tag=36)
    addL(204, 16, tag=37)
    addL(16, 116, tag=38)
    addL(116, 203, tag=39)
    addL(203, 204, tag=40)
    addL(right_upper_front, 301, tag=41)
    addL(301, 302, tag=42)
    addL(302, right_upper_back, tag=43)
    addL(right_upper_back, right_upper_front, tag=44)
    addL(right_lower_front, 303, tag=45)
    addL(303, 304, tag=46)
    addL(304, right_lower_back, tag=47)
    addL(right_lower_back, right_lower_front, tag=48)

    # --------- surfaces ----------
    # bottom & top
    addCL([-1, -4, -210, 133], tag=1)
    addPS([1], tag=1)
    addCL([-209, -133, -3, -2], tag=101)
    addPS([101], tag=101)

    addCL([-5, -8, -212, -134], tag=202)
    addPS([202], tag=202)
    addCL([-211, 134, -7, -6], tag=201)
    addPS([201], tag=201)

    # FRONT
    if aligned_blocks:
        addCL([-5, -13, 33, -122], tag=3)
        addPS([3], tag=3)
        addCL([-33, 12, 37, -123], tag=4)
        addPS([4], tag=4)
    elif overlapping_blocks:
        addCL([-5, -13, 33, -123, -122], tag=3)
        addPS([3], tag=3)
        addCL([-33, 12, 37, -125, -124], tag=4)
        addPS([4], tag=4)
    else:
        addCL([-5, -13, 33, -124, -123, -122], tag=3)
        addPS([3], tag=3)
        addCL([-33, 12, 37, -125], tag=4)
        addPS([4], tag=4)
    addCL([-37, -11, 1, -126], tag=905)
    addPS([905], tag=905)

    if aligned_blocks:
        addCL([-45, 126, 209, 14], tag=6)
        addPS([6], tag=6)
        addCL([-41, 123, 45, -15], tag=7)
        addPS([7], tag=7)
    elif overlapping_blocks:
        addCL([-45, 125, 126, 209, 14], tag=6)
        addPS([6], tag=6)
        addCL([-41, 123, 124, 45, -15], tag=7)
        addPS([7], tag=7)
    else:
        addCL([-45, 124, 125, 126, 209, 14], tag=6)
        addPS([6], tag=6)
        addCL([-41, 123, 45, -15], tag=7)
        addPS([7], tag=7)
    addCL([-211, 122, 41, 16], tag=8)
    addPS([8], tag=8)

    # BACK
    if aligned_blocks:
        addCL([212, -22, -35, -127], tag=9)
        addPS([9], tag=9)
        addCL([35, 21, -39, -128], tag=10)
        addPS([10], tag=10)
    elif overlapping_blocks:
        addCL([212, -22, -35, -128, -127], tag=9)
        addPS([9], tag=9)
        addCL([35, 21, -39, -130, -129], tag=10)
        addPS([10], tag=10)
    else:
        addCL([212, -22, -35, -129, -128, -127], tag=9)
        addPS([9], tag=9)
        addCL([35, 21, -39, -130], tag=10)
        addPS([10], tag=10)
    addCL([39, -20, -210, -131], tag=11)
    addPS([11], tag=11)

    if aligned_blocks:
        addCL([47, 131, -3, 17], tag=12)
        addPS([12], tag=12)
        addCL([43, 128, -47, -18], tag=13)
        addPS([13], tag=13)
    elif overlapping_blocks:
        addCL([47, 130, 131, -3, 17], tag=12)
        addPS([12], tag=12)
        addCL([43, 128, 129, -47, -18], tag=13)
        addPS([13], tag=13)
    else:
        addCL([47, 129, 130, 131, -3, 17], tag=12)
        addPS([12], tag=12)
        addCL([43, 128, -47, -18], tag=13)
        addPS([13], tag=13)
    addCL([7, 127, -43, 19], tag=14)
    addPS([14], tag=14)

    # LEFT
    addCL([6, -19, -42, 16], tag=15)
    addPS([15], tag=15)
    addCL([42, 18, -46, -15], tag=16)
    addPS([16], tag=16)
    addCL([46, -17, -2, 14], tag=17)
    addPS([17], tag=17)

    # RIGHT
    addCL([-8, -22, 36, 13], tag=18)
    addPS([18], tag=18)
    addCL([-36, 21, 40, -12], tag=19)
    addPS([19], tag=19)
    addCL([-40, -20, 4, 11], tag=20)
    addPS([20], tag=20)

    # FAULT walls (store surface IDs)
    addCL([-134, 122, -44, -127], tag=21); s21 = addPS([21], tag=21)
    if aligned_blocks:
        addCL([44, 123, -38, -128], tag=24); s24 = addPS([24], tag=24)
        addCL([38, 131, 133, -126], tag=25); s25 = addPS([25], tag=25)
        fault_surfs = [s21, s24, s25]
    elif overlapping_blocks:
        addCL([44, 123, 34, -128], tag=22);    s22 = addPS([22], tag=22)
        addCL([-34, 124, 48, -129], tag=23);   s23 = addPS([23], tag=23)
        addCL([-48, 125, 38, -130], tag=24);   s24 = addPS([24], tag=24)
        addCL([-38, 126, -133, -131], tag=25); s25 = addPS([25], tag=25)
        fault_surfs = [s21, s22, s23, s24, s25]
    else:
        addCL([44, 123, -48, -128], tag=22);   s22 = addPS([22], tag=22)
        addCL([48, 124, 34, -129], tag=23);    s23 = addPS([23], tag=23)
        addCL([-34, 125, 38, -130], tag=24);   s24 = addPS([24], tag=24)
        addCL([-38, 126, -133, -131], tag=25); s25 = addPS([25], tag=25)
        fault_surfs = [s21, s22, s23, s24, s25]

    # Reservoir interface surfaces
    addCL([33, 34, 35, 36], tag=310); addPS([310], tag=310)
    addCL([37, 38, 39, 40], tag=311); addPS([311], tag=311)
    addCL([41, 42, 43, 44], tag=410); addPS([410], tag=410)
    addCL([45, 46, 47, 48], tag=411); addPS([411], tag=411)

    # --------- volumes ----------
    reservoir1_fault_surfaces = [23, 24] if overlapping_blocks else [24]
    addSL(
        reservoir1_fault_surfaces + [10, 19, 4, 310, 311],
        tag=1,
        sewing=True,
    ); addV([1], tag=1)
    if aligned_blocks:
        addSL([7, 16, 13, 24, 410, 411], tag=2, sewing=True); addV([2], tag=2)
        addSL([8, 15, 14, 21, 201, 410], tag=3); addV([3], tag=3)
        addSL([21, 9, 18, 3, 202, 310], tag=5, sewing=True); addV([5], tag=5)
        addSL([6, 17, 12, 25, 101, 411], tag=4, sewing=True); addV([4], tag=4)
    elif overlapping_blocks:
        addSL([7, 16, 13, 22, 23, 410, 411], tag=2, sewing=True); addV([2], tag=2)
        addSL([8, 15, 14, 21, 201, 410], tag=3); addV([3], tag=3)
        addSL([21, 22, 9, 18, 3, 202, 310], tag=5, sewing=True); addV([5], tag=5)
        addSL([6, 17, 12, 24, 25, 101, 411], tag=4, sewing=True); addV([4], tag=4)
    else:
        addSL([7, 16, 13, 22, 410, 411], tag=2, sewing=True); addV([2], tag=2)
        addSL([8, 15, 14, 21, 201, 410], tag=3); addV([3], tag=3)
        addSL([21, 22, 23, 9, 18, 3, 202, 310], tag=5, sewing=True); addV([5], tag=5)
        addSL([6, 17, 12, 23, 24, 25, 101, 411], tag=4, sewing=True); addV([4], tag=4)
    addSL([25, 11, 20, 905, 1, 311], tag=6); addV([6], tag=6)

    # ======================================================
    # NEW: two extra boxes above and below the current cube
    # ======================================================
    x_min = x_new(-W / 2); x_max = x_new(W / 2)
    y_min = y_new(-H / 2); y_max = y_new(H / 2)

    # Top box (shallower): z_old from 400 to 800
    z_top0 = z_new(400.0)
    z_top1 = 5000.
    #z_top1 = z_new(800.0)
    zmin_top = min(z_top0, z_top1); zmax_top = max(z_top0, z_top1)
    top_box = geo.addBox(
        x_min, y_min, zmin_top,
        x_max - x_min, y_max - y_min, zmax_top - zmin_top
    )

    # Bottom box (deeper): z_old from -800 to -400
    z_bot0 = z_new(-400.0)
    z_bot1 = 0.
    #z_bot1 = z_new(-800.0)
    zmin_bot = min(z_bot0, z_bot1); zmax_bot = max(z_bot0, z_bot1)
    bottom_box = geo.addBox(
        x_min, y_min, zmin_bot,
        x_max - x_min, y_max - y_min, zmax_bot - zmin_bot
    )

    gmsh.model.occ.synchronize()

    # Material identity starts from the explicitly constructed volumes. It is
    # propagated through every Boolean operation using OCC's output maps.
    material_by_volume = {
        1: RES,
        2: RES,
        3: OVERBURDEN,
        5: OVERBURDEN,
        4: UNDERBURDEN,
        6: UNDERBURDEN,
        top_box: OVERBURDEN,
        bottom_box: UNDERBURDEN,
    }

    material_priority = {
        FAULT: 5,
        DAMAGEZONE_LEFT: 4,
        DAMAGEZONE_RIGHT: 4,
        RES: 3,
        OVERBURDEN: 2,
        UNDERBURDEN: 2,
    }

    def propagate_boolean_materials(inputs, output_map, input_materials):
        """Propagate material tags from Boolean inputs to their descendants."""
        descendants = {}
        for source, mapped_entities in zip(inputs, output_map):
            if source[0] != 3 or source[1] not in input_materials:
                continue
            material = input_materials[source[1]]
            for dim, tag in mapped_entities:
                if dim != 3:
                    continue
                current = descendants.get(tag)
                if (
                    current is None
                    or material_priority[material] > material_priority[current]
                ):
                    descendants[tag] = material
        return descendants

    # ======================================================
    # Damage-zone boundaries and thin 3-D fault
    # ======================================================
    left_copy = geo.copy([(2, s) for s in fault_surfs])
    right_copy = geo.copy([(2, s) for s in fault_surfs])

    # The original fault surfaces are the fixed left face of the fault slab.
    # Only the right face/right damage boundary is displaced in +y.
    geo.translate(left_copy, 0.0, -damage_left, 0.0)
    geo.translate(right_copy, 0.0, fault_dy + damage_right, 0.0)
    gmsh.model.occ.synchronize()

    fault_left_surfs = [e[1] for e in left_copy if e[0] == 2]
    fault_right_surfs = [e[1] for e in right_copy if e[0] == 2]

    # Extruding each conformal fault patch creates five adjacent volumes that
    # together form one thin fault slab spanning the complete model width.
    host_volumes = geo.getEntities(3)

    fault_extrusion = geo.extrude(
        [(2, s) for s in fault_surfs], 0.0, fault_dy, 0.0
    )
    fault_seed_volumes = [entity for entity in fault_extrusion if entity[0] == 3]
    gmsh.model.occ.synchronize()

    for _, volume in fault_seed_volumes:
        material_by_volume[volume] = FAULT

    # Fragmenting removes overlap between host rock and fault material and
    # makes both fault faces and both damage-zone boundaries conformal.
    first_fragment_inputs = host_volumes + fault_seed_volumes
    _, first_fragment_map = geo.fragment(
        host_volumes,
        fault_seed_volumes,
        removeObject=True,
        removeTool=True,
    )
    gmsh.model.occ.synchronize()

    material_by_volume = propagate_boolean_materials(
        first_fragment_inputs, first_fragment_map, material_by_volume
    )

    volumes_before_damage_split = geo.getEntities(3)
    damage_split_tools = [
        (2, s) for s in (fault_left_surfs + fault_right_surfs)
    ]
    second_fragment_inputs = volumes_before_damage_split + damage_split_tools
    materials_before_damage_split = dict(material_by_volume)
    _, second_fragment_map = geo.fragment(
        volumes_before_damage_split,
        damage_split_tools,
        removeObject=True,
        removeTool=True,
    )
    gmsh.model.occ.synchronize()
    material_by_volume = propagate_boolean_materials(
        second_fragment_inputs, second_fragment_map, material_by_volume
    )

    # Only descendants actually produced by a damage-boundary split can be
    # damage-zone material. This prevents an unsplit host volume (notably an
    # underburden volume) from being relabeled wholesale just because its
    # center of mass falls inside the fault-normal interval.
    for source, descendants in zip(
        volumes_before_damage_split,
        second_fragment_map[:len(volumes_before_damage_split)],
    ):
        source_material = materials_before_damage_split.get(source[1])
        child_volumes = sorted({
            tag for dim, tag in descendants if dim == 3
        })
        if source_material == FAULT or len(child_volumes) <= 1:
            continue
        for volume in child_volumes:
            _, y_center, z_center = geo.getCenterOfMass(3, volume)
            scalar = fault_s(y_center, z_center)
            if c_left < scalar < c0:
                material_by_volume[volume] = DAMAGEZONE_LEFT
            elif c_fault_right < scalar < c_right:
                material_by_volume[volume] = DAMAGEZONE_RIGHT

    # ======================================================
    # Solid, visible wells (reservoir inclusions -- no holes)
    # ======================================================
    # Coordinates are [X, Y, Z1, Z2] and are supplied by main.py. Fragmenting
    # the cylinders into the host creates conformal cylindrical interfaces
    # without subtracting any material. The cylinder descendants retain the
    # RES material tag and are therefore part of the reservoir physical volume.
    volumes_before_wells = geo.getEntities(3)
    well_solids = [
        (
            3,
            geo.addCylinder(
                coords[0],
                coords[1],
                min(coords[2], coords[3]),
                0.0,
                0.0,
                abs(coords[3] - coords[2]),
                well_cylinder_radius,
            ),
        )
        for coords in well_coords
    ]
    for _, volume in well_solids:
        material_by_volume[volume] = RES

    gmsh.model.occ.synchronize()
    well_fragment_inputs = volumes_before_wells + well_solids
    _, well_fragment_map = geo.fragment(
        volumes_before_wells,
        well_solids,
        removeObject=True,
        removeTool=True,
    )
    gmsh.model.occ.synchronize()
    material_by_volume = propagate_boolean_materials(
        well_fragment_inputs, well_fragment_map, material_by_volume
    )
    well_volumes = sorted({
        tag
        for descendants in well_fragment_map[-len(well_solids):]
        for dim, tag in descendants
        if dim == 3
    })

    # OCC can retain an inverse (negative-mass) solid when the two reservoir
    # contacts are exactly aligned. It represents the exterior complement,
    # not model material, and must not be sent to the mesh generator.
    invalid_volumes = [
        (3, volume)
        for _, volume in geo.getEntities(3)
        if geo.getMass(3, volume) <= 0.0
    ]
    if invalid_volumes:
        geo.remove(invalid_volumes, recursive=True)
        gmsh.model.occ.synchronize()
        for _, volume in invalid_volumes:
            material_by_volume.pop(volume, None)

    # ======================================================
    # Physical tagging after all Boolean operations
    # ======================================================
    groups = {
        RES: [],
        OVERBURDEN: [],
        UNDERBURDEN: [],
        DAMAGEZONE_LEFT: [],
        DAMAGEZONE_RIGHT: [],
        FAULT: [],
    }

    for volume, material in material_by_volume.items():
        groups[material].append(volume)

    # All material groups, including both damage zones, form one complete and
    # mutually exclusive partition of the model volumes.
    model_volumes = {volume for _, volume in gmsh.model.getEntities(3)}
    tagged_volumes = [
        volume for volumes in groups.values() for volume in volumes
    ]
    duplicate_volumes = {
        volume for volume in tagged_volumes if tagged_volumes.count(volume) > 1
    }
    missing_volumes = model_volumes - set(tagged_volumes)
    if duplicate_volumes or missing_volumes:
        raise RuntimeError(
            "Invalid material tagging: "
            f"missing={sorted(missing_volumes)}, "
            f"duplicated={sorted(duplicate_volumes)}"
        )

    physical_names = {
        RES: "RESERVOIR",
        OVERBURDEN: "OVERBURDEN",
        UNDERBURDEN: "UNDERBURDEN",
        DAMAGEZONE_LEFT: "DAMAGEZONE_LEFT",
        DAMAGEZONE_RIGHT: "DAMAGEZONE_RIGHT",
        FAULT: "FAULT",
    }
    for physical_tag, volumes in groups.items():
        if volumes:
            gmsh.model.addPhysicalGroup(
                3, volumes, tag=physical_tag, name=physical_names[physical_tag]
            )

    # External faces are selected by their bounding boxes. This remains valid
    # when the fault dip, offset, thickness, or well Boolean topology changes.
    boundary_groups = {
        LEFTBOUNDARY: [], RIGHTBOUNDARY: [],
        FRONTBOUNDARY: [], BACKBOUNDARY: [],
        BOTTOMBOUNDARY: [], TOPBOUNDARY: [],
    }
    z_domain_top = zmin_top #z_new(800.0)
    z_domain_bottom = zmin_bot #z_new(-800.0)
    bbox_tol = 1.0e-5

    for _, surface in gmsh.model.getEntities(2):
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(2, surface)
        if abs(xmin - x_min) < bbox_tol and abs(xmax - x_min) < bbox_tol:
            boundary_groups[LEFTBOUNDARY].append(surface)
        elif abs(xmin - x_max) < bbox_tol and abs(xmax - x_max) < bbox_tol:
            boundary_groups[RIGHTBOUNDARY].append(surface)
        elif abs(ymin - y_min) < bbox_tol and abs(ymax - y_min) < bbox_tol:
            boundary_groups[FRONTBOUNDARY].append(surface)
        elif abs(ymin - y_max) < bbox_tol and abs(ymax - y_max) < bbox_tol:
            boundary_groups[BACKBOUNDARY].append(surface)
        elif abs(zmin - z_domain_bottom) < bbox_tol and abs(zmax - z_domain_bottom) < bbox_tol:
            boundary_groups[BOTTOMBOUNDARY].append(surface)
        elif abs(zmin - z_domain_top) < bbox_tol and abs(zmax - z_domain_top) < bbox_tol:
            boundary_groups[TOPBOUNDARY].append(surface)

    boundary_names = {
        LEFTBOUNDARY: "LEFTBOUNDARY", RIGHTBOUNDARY: "RIGHTBOUNDARY",
        FRONTBOUNDARY: "FRONTBOUNDARY", BACKBOUNDARY: "BACKBOUNDARY",
        BOTTOMBOUNDARY: "BOTTOMBOUNDARY", TOPBOUNDARY: "TOPBOUNDARY",
    }
    for physical_tag, surfaces in boundary_groups.items():
        if surfaces:
            gmsh.model.addPhysicalGroup(
                2, surfaces, tag=physical_tag, name=boundary_names[physical_tag]
            )

    # Smoothly refine the reservoir mesh around the conformal well surfaces.
    # This field changes only element sizes; it does not define well material.
    well_surfaces = sorted({
        surface
        for dim, surface in gmsh.model.getBoundary(
            [(3, volume) for volume in well_volumes],
            combined=False,
            oriented=False,
        )
        if dim == 2
    })
    well_distance = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(
        well_distance, "SurfacesList", well_surfaces
    )
    gmsh.model.mesh.field.setNumber(well_distance, "Sampling", 100)

    well_threshold = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(
        well_threshold, "InField", well_distance
    )
    gmsh.model.mesh.field.setNumber(
        well_threshold, "SizeMin", well_mesh_size
    )
    gmsh.model.mesh.field.setNumber(well_threshold, "SizeMax", lc)
    gmsh.model.mesh.field.setNumber(
        well_threshold, "DistMin", well_cylinder_radius
    )
    gmsh.model.mesh.field.setNumber(
        well_threshold, "DistMax", well_transition_radius
    )
    gmsh.model.mesh.field.setAsBackgroundMesh(well_threshold)

    # ======================================================
    # Final meshing setup
    # ======================================================
    gmsh.option.setNumber("Mesh.ToleranceInitialDelaunay", 1e-12)
    gmsh.option.setNumber("Mesh.AngleToleranceFacetOverlap", 0.1)
    gmsh.option.setNumber("Mesh.ToleranceEdgeLength", 1e-12)

    gmsh.model.occ.synchronize()
    gmsh.model.mesh.removeDuplicateNodes()
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.1)
    gmsh.model.mesh.generate(3)
    gmsh.write(msh_filename)
    gmsh.finalize()


if __name__ == "__main__":
    from set_case import set_input_data
    idata = set_input_data('case_5', physics_type='single_phase_thermal', wells_type='doublet')
    generate_3d_fault_mesh(idata)
