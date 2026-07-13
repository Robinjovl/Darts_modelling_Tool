import gmsh
import math

def create_geo_with_api():
    gmsh.initialize()
    gmsh.model.add("geo_model")
    geo = gmsh.model.geo
    
    # Parameters
    W = 4500.0
    T = 1000.0          # Total thickness (3000 - 2000)
    a = 50.0
    D = 2500.0          # Mid-depth (average of 2000 and 3000)
    Z1 = D - T / 2      # Top = 2000
    Z2 = D + T / 2      # Bottom = 3000
    b = 140.0
    Lplus = b + 200.0
    phi = -60 * math.pi / 180
    lc = 300.0
    mult = 0.5
    mult1 = 0.1
    tol = 1e-6

    # Points (x from 0 to W, z from 2000 to 3000)
    p1 = geo.addPoint(0,    0, Z2, lc, tag=1)
    p2 = geo.addPoint(W,    0, Z2, lc, tag=2)
    p3 = geo.addPoint(W,    0, D + a, mult * lc, tag=3)
    p4 = geo.addPoint(W,    0, D - b , mult * lc, tag=4)
    p5 = geo.addPoint(W,    0, Z1, lc, tag=5)
    p6 = geo.addPoint(0,    0, Z1, lc, tag=6)
    p7 = geo.addPoint(0,    0, D - a, mult * lc, tag=7)
    p8 = geo.addPoint(0,    0, D + b, mult * lc, tag=8)

    # Inner fault-related points
    p15 = geo.addPoint(W/2 + b / math.tan(phi), 0, D + b, mult1 * lc, tag=15)
    p16 = geo.addPoint(W/2 + a / math.tan(phi), 0, D + a, mult1 * lc, tag=16)
    p17 = geo.addPoint(W/2 - a / math.tan(phi), 0, D - a, mult1 * lc, tag=17)
    p18 = geo.addPoint(W/2 - b / math.tan(phi), 0, D - b, mult1 * lc, tag=18)
    p19 = geo.addPoint(W/2 - Lplus / math.tan(phi), 0, D - Lplus, mult1 * lc, tag=19)
    p20 = geo.addPoint(W/2 + Lplus / math.tan(phi), 0, D + Lplus, mult1 * lc, tag=20)



    # Lines with explicit tags
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
    l16 = geo.addLine(p15, p20, tag=16)
    l17 = geo.addLine(p18, p19, tag=17)

    # Curve loops with explicit tags
    cl1 = geo.addCurveLoop([l1, l2, l9, l15, -l11, l8], tag=1)
    cl2 = geo.addCurveLoop([l3, l10, l13, l14, -l9], tag=2)
    cl3 = geo.addCurveLoop([-l12, l7, l11, -l15, -l14], tag=3)
    cl4 = geo.addCurveLoop([l4, l5, l6, l12, -l13, -l10], tag=4)
    cl5 = geo.addCurveLoop([l16, -l16], tag=5)
    cl6 = geo.addCurveLoop([l17, -l17], tag=6)

    # Surfaces with explicit tags
    s1 = geo.addPlaneSurface([cl1, -cl5], tag=1)
    s2 = geo.addPlaneSurface([cl2], tag=2)
    s3 = geo.addPlaneSurface([cl3], tag=3)
    s4 = geo.addPlaneSurface([cl4, -cl6], tag=4)

    # Synchronize geometry
    geo.synchronize()
    
    # Extrude in Y direction with 15 layers
    all_extruded_entities = []
    surfaces_to_extrude = [s1, s2, s3, s4]
    
    # Store mapping of original surfaces to their extruded volumes and surfaces
    surface_extrusion_map = {}
    
    for surf in surfaces_to_extrude:
        # Extrude in Y direction: {0, 4500, 0} with 15 layers
        extruded = geo.extrude([(2, surf)], 0, 4500, 0, [15], recombine=True)
        all_extruded_entities.extend(extruded)
        surface_extrusion_map[surf] = extruded
        print(f"Surface {surf} extruded to: {extruded}")
    
    geo.synchronize()
    
    all_volumes = gmsh.model.getEntities(3)
    all_surfaces = gmsh.model.getEntities(2)

    # Physical groups
    LEFT = 991
    RIGHT = 992
    BOT = 993
    TOP = 994
    FRONT = 995
    BACK = 996
    RES = 9991
    OUTER = 9992
    FRAC = 99991
    FRAC_BOUND_STICK = 1
    FRAC_BOUND_FREE = 2

    # ZM surfaces (original surfaces at y=0)
    original_surfaces = [s1, s2, s3, s4]
    gmsh.model.addPhysicalGroup(2, original_surfaces, tag=FRONT, name="FRONT")
 
    zp_surfaces = []
    for surf, extruded in surface_extrusion_map.items():
        # The first entity in extruded list is the top surface
        if len(extruded) > 0 and extruded[0][0] == 2:  # 2 = surface
            zp_surfaces.append(extruded[0][1])
    
    gmsh.model.addPhysicalGroup(2, zp_surfaces, tag=BACK, name="BACK")

    # Classify side surfaces using bounding boxes (no OCC calls)
    left_surfaces = []
    right_surfaces = []
    top_surfaces = []      # z = Z1 (smallest z value = top)
    bottom_surfaces = []   # z = Z2 (largest z value = bottom)

    for dim, tag in all_surfaces:
        # Skip original and top surfaces
        if tag in original_surfaces or tag in zp_surfaces:
            continue
            
        try:
            bbox = gmsh.model.getBoundingBox(dim, tag)
            x_min, _, z_min, x_max, _, z_max = bbox
            
            if abs(x_min - 0) < tol and abs(x_max - 0) < tol:
                left_surfaces.append(tag)
            # Right surfaces (x ≈ W)
            elif abs(x_min - W) < tol and abs(x_max - W) < tol:
                right_surfaces.append(tag)
            # Top surfaces (z ≈ Z1) - smallest z value
            elif abs(z_min - Z1) < tol and abs(z_max - Z1) < tol:
                top_surfaces.append(tag)
            # Bottom surfaces (z ≈ Z2) - largest z value
            elif abs(z_min - Z2) < tol and abs(z_max - Z2) < tol:
                bottom_surfaces.append(tag)
        except:
            continue  # Skip if we can't get bounding box

    # Assign physical groups for sides
    gmsh.model.addPhysicalGroup(2, left_surfaces, tag=LEFT, name="LEFT")
    gmsh.model.addPhysicalGroup(2, right_surfaces, tag=RIGHT, name="RIGHT")
    gmsh.model.addPhysicalGroup(2, top_surfaces, tag=TOP, name="TOP")      # Changed from BOT to TOP
    gmsh.model.addPhysicalGroup(2, bottom_surfaces, tag=BOT, name="BOT")   # Changed from TOP to BOT
    

    if len(all_volumes) == 4:
        # Adjust these based on your geometry needs
        res_volumes = [all_volumes[1][1], all_volumes[2][1]]  # Example: volumes 2 and 3 as RES
        outer_volumes = [all_volumes[0][1], all_volumes[3][1]]  # Example: volumes 1 and 4 as OUTER
        
        gmsh.model.addPhysicalGroup(3, res_volumes, tag=RES, name="RES")
        gmsh.model.addPhysicalGroup(3, outer_volumes, tag=OUTER, name="OUTER")

    # Identify reservoir top and bottom surfaces
    reservoir_top_bottom_surfaces = []

    # Get surfaces that are internal and likely part of reservoir boundaries
    for dim, tag in all_surfaces:
        if (tag not in original_surfaces and tag not in zp_surfaces and 
            tag not in left_surfaces and tag not in right_surfaces and
            tag not in top_surfaces and tag not in bottom_surfaces):
            
            # Check if this is a reservoir top/bottom surface
            try:
                bbox = gmsh.model.getBoundingBox(dim, tag)
                _, _, z_min, _, _, z_max = bbox
                
                # Reservoir surfaces are typically horizontal (constant z) and internal
                if abs(z_min - z_max) < tol:  # Horizontal surface
                    # Check if it's likely a reservoir boundary by its z-position
                    z_pos = z_min
                    expected_z_levels = [D + a, D - a, D + b, D - b]
                    
                    for expected_z in expected_z_levels:
                        if abs(z_pos - expected_z) < tol:
                            reservoir_top_bottom_surfaces.append(tag)
                            break
                            
            except:
                continue

    # Fracture surfaces (internal surfaces along fault) - EXCLUDE reservoir top/bottom
    fracture_surfaces = []
    for dim, tag in all_surfaces:
        if (tag not in original_surfaces and tag not in zp_surfaces and 
            tag not in left_surfaces and tag not in right_surfaces and
            tag not in top_surfaces and tag not in bottom_surfaces and
            tag not in reservoir_top_bottom_surfaces):  # EXCLUDE reservoir surfaces
            
            fracture_surfaces.append(tag)
    
    if fracture_surfaces:
        gmsh.model.addPhysicalGroup(2, fracture_surfaces, tag=FRAC, name="FRAC")
        print(f"FRAC surfaces (excluding reservoir): {fracture_surfaces}")
        print(f"Excluded reservoir surfaces: {reservoir_top_bottom_surfaces}")

    # Fracture boundary curves - using the explicit line tags we defined
    frac_bound_stick = [149, 53] 
    frac_bound_free = [121, 17, 13, 14, 15, 16, 25, 22, 64, 63]  
    # gmsh.model.addPhysicalGroup(1, frac_bound_stick, tag=FRAC_BOUND_STICK, name="FRAC_BOUND_STICK")
    # gmsh.model.addPhysicalGroup(1, frac_bound_free, tag=FRAC_BOUND_FREE, name="FRAC_BOUND_FREE")
        
    # Final synchronization
    geo.synchronize()
    
        # ------------------------------------------------------------
        # ------------------------------------------------------------
    # OCC well + proper identification of cylinder side faces
    # ------------------------------------------------------------
    print("Adding OCC well (isolated kernel) and refinement field...")

    occ = gmsh.model.occ
    xw, yw = 2250.0, 1800.0
    r_well = 0.10
    Z_top, Z_bot = 2000.0, 3000.0
    lc_fine, lc_coarse = 50.0, 300.0

    # --- create cylinder in OCC space ---
    well_cyl = occ.addCylinder(xw, yw, Z_top, 0, 0, (Z_bot - Z_top), r_well)

    # retrieve the OCC surfaces explicitly before any merge
    occ_surfaces = gmsh.model.getEntities(2)
    well_side_faces = []
    for (dim, tag) in occ_surfaces:
        x0, y0, z0, x1, y1, z1 = gmsh.model.getBoundingBox(dim, tag)
        x_c = 0.5 * (x0 + x1)
        y_c = 0.5 * (y0 + y1)
        z_extent = abs(z1 - z0)
        # keep only tall, vertical surfaces near the well axis
        if abs(x_c - xw) < 2*r_well and abs(y_c - yw) < 2*r_well and z_extent > 0.8*(Z_bot - Z_top):
            well_side_faces.append(tag)

    print(f"OCC well side faces identified: {len(well_side_faces)}")
    

    # ------------------------------------------------------------
    # Local mesh refinement field (distance + box)
    # ------------------------------------------------------------
    if well_side_faces:
        gmsh.model.mesh.field.add("Distance", 201)
        gmsh.model.mesh.field.setNumbers(201, "FacesList", well_side_faces)

        gmsh.model.mesh.field.add("Threshold", 202)
        gmsh.model.mesh.field.setNumber(202, "InField", 201)
        gmsh.model.mesh.field.setNumber(202, "SizeMin", lc_fine)
        gmsh.model.mesh.field.setNumber(202, "SizeMax", lc_coarse)
        gmsh.model.mesh.field.setNumber(202, "DistMin", 2.0 * r_well)
        gmsh.model.mesh.field.setNumber(202, "DistMax", 20.0 * r_well)

        # confine refinement spatially around the well
        pad_xy = 10.0 * r_well
        pad_z = 20.0 * r_well
        gmsh.model.mesh.field.add("Box", 203)
        gmsh.model.mesh.field.setNumber(203, "XMin", xw - pad_xy)
        gmsh.model.mesh.field.setNumber(203, "XMax", xw + pad_xy)
        gmsh.model.mesh.field.setNumber(203, "YMin", yw - pad_xy)
        gmsh.model.mesh.field.setNumber(203, "YMax", yw + pad_xy)
        gmsh.model.mesh.field.setNumber(203, "ZMin", Z_top - pad_z)
        gmsh.model.mesh.field.setNumber(203, "ZMax", Z_bot + pad_z)
        gmsh.model.mesh.field.setNumber(203, "VIn", 0)
        gmsh.model.mesh.field.setNumber(203, "VOut", lc_coarse)

        gmsh.model.mesh.field.add("Min", 204)
        gmsh.model.mesh.field.setNumbers(204, "FieldsList", [202, 203])
        gmsh.model.mesh.field.setAsBackgroundMesh(204)

    # physical tags (isolated, not touching your main model)
    WELL_VOLUME  = 19001
    WELL_SURFACE = 19002
    gmsh.model.addPhysicalGroup(3, [well_cyl], WELL_VOLUME)
    gmsh.model.setPhysicalName(3, WELL_VOLUME, "WELL_VOLUME")
    if well_side_faces:
        gmsh.model.addPhysicalGroup(2, well_side_faces, WELL_SURFACE)
        gmsh.model.setPhysicalName(2, WELL_SURFACE, "WELL_SURFACE")

    print("Well and refinement successfully defined (geometry untouched).")
    # occ.synchronize()
    # geo.synchronize()


    # Generate mesh
    gmsh.model.mesh.generate(3)
    gmsh.model.mesh.removeDuplicateNodes()

    
    # Set MSH file version
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.1)
    # Save mesh
    gmsh.write("2d_to_3d_without_damage_zone.msh")
    
    gmsh.finalize()

if __name__ == "__main__":
    create_geo_with_api()