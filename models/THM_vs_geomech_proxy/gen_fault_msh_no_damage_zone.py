import gmsh
import math
import os


def gen_fault_msh_no_damage_zone():
    gmsh.initialize()
    gmsh.model.add("test_3D_fault")
    geo = gmsh.model.geo

    # ---- Parameters ----
    W = 10000.0          # x extent
    H = 10000.0          # y extent (extrusion length)
    T = 5000.0           # total thickness (Z2 - Z1)
    D = 2500.0           # mid-depth
    Z1 = D - T / 2.0     # top    = 0    (shallowest z)
    Z2 = D + T / 2.0     # bottom = 5000 (deepest z)
    a = 50.0             # inner (damage) half-thickness
    b = 250.0            # outer (damage) half-thickness
    # Fault half-length beyond the damage zone. This SETS THE FAULT Z EXTENT:
    # the buried fault spans z in [D - Lplus, D + Lplus], symmetric about D.
    Lplus = b + 200.0
    phi = -60.0 * math.pi / 180.0   # dip angle (rad)
    lc = 400.0
    mult = 0.5
    mult1 = 0.1
    tol = 1e-6
    n_extrude = 20       # layers along y

    # --------- 2D section in the y = 0 plane (x-z), extruded in y ----------
    # Outer box points (x from 0 to W, z from Z1=top to Z2=bottom)
    p1 = geo.addPoint(0, 0, Z2, lc, tag=1)
    p2 = geo.addPoint(W, 0, Z2, lc, tag=2)
    p3 = geo.addPoint(W, 0, D + a, mult * lc, tag=3)
    p4 = geo.addPoint(W, 0, D - b, mult * lc, tag=4)
    p5 = geo.addPoint(W, 0, Z1, lc, tag=5)
    p6 = geo.addPoint(0, 0, Z1, lc, tag=6)
    p7 = geo.addPoint(0, 0, D - a, mult * lc, tag=7)
    p8 = geo.addPoint(0, 0, D + b, mult * lc, tag=8)

    # Inner fault-related points. p19/p20 are the buried fault tips at D +- Lplus.
    p15 = geo.addPoint(W / 2 + b / math.tan(phi), 0, D + b, mult1 * lc, tag=15)
    p16 = geo.addPoint(W / 2 + a / math.tan(phi), 0, D + a, mult1 * lc, tag=16)
    p17 = geo.addPoint(W / 2 - a / math.tan(phi), 0, D - a, mult1 * lc, tag=17)
    p18 = geo.addPoint(W / 2 - b / math.tan(phi), 0, D - b, mult1 * lc, tag=18)
    p19 = geo.addPoint(W / 2 - Lplus / math.tan(phi), 0, D - Lplus, mult1 * lc, tag=19)  # shallow tip
    p20 = geo.addPoint(W / 2 + Lplus / math.tan(phi), 0, D + Lplus, mult1 * lc, tag=20)  # deep tip

    # Lines
    l1 = geo.addLine(p1, p2, tag=1)
    l2 = geo.addLine(p2, p3, tag=2)
    l3 = geo.addLine(p3, p4, tag=3)
    l4 = geo.addLine(p4, p5, tag=4)
    l5 = geo.addLine(p5, p6, tag=5)
    l6 = geo.addLine(p6, p7, tag=6)
    l7 = geo.addLine(p7, p8, tag=7)
    l8 = geo.addLine(p8, p1, tag=8)

    l9 = geo.addLine(p3, p16, tag=9)
    l10 = geo.addLine(p4, p18, tag=10)
    l11 = geo.addLine(p8, p15, tag=11)
    l12 = geo.addLine(p7, p17, tag=12)
    l13 = geo.addLine(p18, p17, tag=13)
    l14 = geo.addLine(p17, p16, tag=14)
    l15 = geo.addLine(p16, p15, tag=15)
    l16 = geo.addLine(p15, p20, tag=16)   # fault extension into deep (underburden)
    l17 = geo.addLine(p18, p19, tag=17)   # fault extension into shallow (overburden)

    # Curve loops. cl5/cl6 are degenerate ([line, -line]) so the fault tip
    # lines l16/l17 get embedded INSIDE surfaces s1/s4 (buried fault tips).
    cl1 = geo.addCurveLoop([l1, l2, l9, l15, -l11, l8], tag=1)
    cl2 = geo.addCurveLoop([l3, l10, l13, l14, -l9], tag=2)
    cl3 = geo.addCurveLoop([-l12, l7, l11, -l15, -l14], tag=3)
    cl4 = geo.addCurveLoop([l4, l5, l6, l12, -l13, -l10], tag=4)
    cl5 = geo.addCurveLoop([l16, -l16], tag=5)
    cl6 = geo.addCurveLoop([l17, -l17], tag=6)

    s1 = geo.addPlaneSurface([cl1, -cl5], tag=1)   # deep  -> underburden
    s2 = geo.addPlaneSurface([cl2], tag=2)         # reservoir (deep side of fault)
    s3 = geo.addPlaneSurface([cl3], tag=3)         # reservoir (shallow side of fault)
    s4 = geo.addPlaneSurface([cl4, -cl6], tag=4)   # shallow -> overburden

    geo.synchronize()

    # --------- Extrude the section in +y with n_extrude layers ----------
    surfaces_to_extrude = [s1, s2, s3, s4]
    surface_extrusion_map = {}
    for surf in surfaces_to_extrude:
        extruded = geo.extrude([(2, surf)], 0, H, 0, [n_extrude], recombine=True)
        surface_extrusion_map[surf] = extruded
    geo.synchronize()

    all_surfaces = gmsh.model.getEntities(2)

    # ------------------------------------------------------------
    # Physical group tags
    # ------------------------------------------------------------
    LEFT = 991     # x = 0
    RIGHT = 992    # x = W
    FRONT = 993    # y = 0
    BACK = 994     # y = H
    TOP = 995      # z = Z1 (shallowest)
    BOT = 996      # z = Z2 (deepest)

    #FAULT = 9991  # is not used for now
    
    RESERVOIR = 99991
    OVERBURDEN = 99992
    UNDERBURDEN = 99993

    # ------------------------------------------------------------
    # Tag outer boundary faces by bounding box (a surface belongs to a
    # boundary plane only if it is flat against that plane).
    # ------------------------------------------------------------
    left_surfaces, right_surfaces = [], []
    front_surfaces, back_surfaces = [], []
    top_surfaces, bottom_surfaces = [], []

    for dim, tag in all_surfaces:
        x_min, y_min, z_min, x_max, y_max, z_max = gmsh.model.getBoundingBox(dim, tag)
        if abs(x_min) < tol and abs(x_max) < tol:
            left_surfaces.append(tag)
        elif abs(x_min - W) < tol and abs(x_max - W) < tol:
            right_surfaces.append(tag)
        elif abs(y_min) < tol and abs(y_max) < tol:
            front_surfaces.append(tag)
        elif abs(y_min - H) < tol and abs(y_max - H) < tol:
            back_surfaces.append(tag)
        elif abs(z_min - Z1) < tol and abs(z_max - Z1) < tol:
            top_surfaces.append(tag)
        elif abs(z_min - Z2) < tol and abs(z_max - Z2) < tol:
            bottom_surfaces.append(tag)

    gmsh.model.addPhysicalGroup(2, left_surfaces, tag=LEFT, name="LEFT")
    gmsh.model.addPhysicalGroup(2, right_surfaces, tag=RIGHT, name="RIGHT")
    gmsh.model.addPhysicalGroup(2, front_surfaces, tag=FRONT, name="FRONT")
    gmsh.model.addPhysicalGroup(2, back_surfaces, tag=BACK, name="BACK")
    gmsh.model.addPhysicalGroup(2, top_surfaces, tag=TOP, name="TOP")
    gmsh.model.addPhysicalGroup(2, bottom_surfaces, tag=BOT, name="BOT")

    # ------------------------------------------------------------
    # Identify internal reservoir horizontal surfaces (constant z at the
    # damage-zone boundaries) so they are NOT mistaken for the fault.
    # ------------------------------------------------------------
    boundary_surfaces = set(left_surfaces + right_surfaces + front_surfaces +
                            back_surfaces + top_surfaces + bottom_surfaces)
    expected_z_levels = [D + a, D - a, D + b, D - b]
    reservoir_hz_surfaces = []
    for dim, tag in all_surfaces:
        if tag in boundary_surfaces:
            continue
        _, _, z_min, _, _, z_max = gmsh.model.getBoundingBox(dim, tag)
        if abs(z_min - z_max) < tol and any(abs(z_min - z) < tol for z in expected_z_levels):
            reservoir_hz_surfaces.append(tag)

    # Fault surfaces = internal, non-boundary, non-horizontal-reservoir surfaces.
    fault_surfaces = [tag for dim, tag in all_surfaces
                      if tag not in boundary_surfaces and tag not in reservoir_hz_surfaces]
    if fault_surfaces:
        pass #gmsh.model.addPhysicalGroup(2, fault_surfaces, tag=FAULT, name="FAULT")

    # ------------------------------------------------------------
    # Volume physical groups (robust: read volumes from the extrusion map).
    #   s1 -> deep   (underburden),  s4 -> shallow (overburden)
    #   s2, s3       -> reservoir
    # ------------------------------------------------------------
    def vol_of(surf):
        for e in surface_extrusion_map[surf]:
            if e[0] == 3:
                return e[1]
        return None

    under_vol = vol_of(s1)
    over_vol = vol_of(s4)
    res_vols = [vol_of(s2), vol_of(s3)]
    gmsh.model.addPhysicalGroup(3, res_vols, tag=RESERVOIR, name="RESERVOIR")
    gmsh.model.addPhysicalGroup(3, [over_vol], tag=OVERBURDEN, name="OVERBURDEN")
    gmsh.model.addPhysicalGroup(3, [under_vol], tag=UNDERBURDEN, name="UNDERBURDEN")

    print(f"LEFT={left_surfaces} RIGHT={right_surfaces} "
          f"FRONT={front_surfaces} BACK={back_surfaces} "
          f"TOP={top_surfaces} BOT={bottom_surfaces}")
    print(f"FAULT={fault_surfaces} (excluded reservoir hz={reservoir_hz_surfaces})")
    print(f"RESERVOIR={res_vols} OVERBURDEN={over_vol} UNDERBURDEN={under_vol}")

    # ------------------------------------------------------------
    # Vertical well centerlines + local refinement (kept away from the
    # fault, which sits near x = W/2).
    # ------------------------------------------------------------
    if False:
        try:
            wells = [
                (0.225 * W, 0.18 * H),
                (0.75 * W, 0.80 * H),
            ]
            well_lines = []
            for (wx, wy) in wells:
                p_bot = geo.addPoint(wx, wy, Z2, mult1 * lc)
                p_top = geo.addPoint(wx, wy, Z1, mult1 * lc)
                well_lines.append(geo.addLine(p_bot, p_top))
    
            geo.synchronize()
            vol_tags = [v for (dim, v) in gmsh.model.getEntities(3)]
            for l in well_lines:
                gmsh.model.mesh.embed(1, [l], 3, vol_tags)
    
            thr_fields = []
            for l in well_lines:
                fdist = gmsh.model.mesh.field.add("Distance")
                gmsh.model.mesh.field.setNumbers(fdist, "CurvesList", [l])
                gmsh.model.mesh.field.setNumber(fdist, "Sampling", 10000)
    
                fthr = gmsh.model.mesh.field.add("Threshold")
                gmsh.model.mesh.field.setNumber(fthr, "InField", fdist)
                gmsh.model.mesh.field.setNumber(fthr, "SizeMin", max(1.0, 0.05 * lc))
                gmsh.model.mesh.field.setNumber(fthr, "SizeMax", 1e12)
                gmsh.model.mesh.field.setNumber(fthr, "DistMin", 0.08)
                gmsh.model.mesh.field.setNumber(fthr, "DistMax", 10.0)
                gmsh.model.mesh.field.setNumber(fthr, "StopAtDistMax", 1)
                thr_fields.append(fthr)
    
            f_all = thr_fields[0] if len(thr_fields) == 1 else gmsh.model.mesh.field.add("Min")
            if len(thr_fields) > 1:
                gmsh.model.mesh.field.setNumbers(f_all, "FieldsList", thr_fields)
            gmsh.model.mesh.field.setAsBackgroundMesh(f_all)
            gmsh.option.setNumber("Mesh.MeshSizeFromFields", 1)
            gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
            gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        except Exception as e:
            print(f"[WARN] Well refinement setup failed: {e}")

    # ------------------------------------------------------------
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.1)
    gmsh.write("test_3d_fault_shifted.geo_unrolled")
    gmsh.model.mesh.generate(3)
    gmsh.model.mesh.removeDuplicateNodes()

    gmsh.write(os.path.join('meshes', 'no_damage_zone', "mesh.msh"))
    gmsh.finalize()
    print('mesh generation is completed')


if __name__ == '__main__':
    gen_fault_msh_no_damage_zone()
