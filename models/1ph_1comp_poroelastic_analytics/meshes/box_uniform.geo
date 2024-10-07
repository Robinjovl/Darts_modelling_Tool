avert = 0.1;
ahor = 0.05;
lc = 0.2 * ahor;
lc_frac = 0.03 * ahor;
phi = Pi / 3;
l = ahor / Cos(phi);
h = Pi * ahor / 4;
Nz = 3;

Point(1) = {0, 0, 0, lc};
Point(2) = {ahor, 0,  0, lc};
Point(3) = {ahor, avert, 0, lc};
Point(4) = {0, avert, 0, lc};
Point(5) = {ahor/2 - l / 2 * Cos(phi), avert/2 + l / 2 * Sin(phi), 0, lc_frac};
Point(6) = {ahor/2 + l / 2 * Cos(phi), avert/2 - l / 2 * Sin(phi), 0, lc_frac};

// Bot & centerline
Line(1) = {1,2};
Line(2) = {2,6};
Line(4) = {6,5};
Line(6) = {5,1};
// Top
Line(7) = {6,3};
Line(8) = {3,4};
Line(9) = {4,5};

//Transfinite Curve{1} = Nx_pt;
//Transfinite Curve{2} = (Ny_pt - 1) / 2 + 1;
//Transfinite Curve{4} = N_frac;
//Transfinite Curve{6} = (Ny_pt - 1) / 2 + 1;
//Transfinite Curve{7} = (Ny_pt - 1) / 2 + 1;
//Transfinite Curve{8} = Nx_pt;
//Transfinite Curve{9} = (Ny_pt - 1) / 2 + 1;

Line Loop(1) = {1,2,4,6};
Line Loop(2) = {7, 8, 9, -4};
//Line Loop(2) = {5, -5};

//Plane Surface(1) = {1};
Plane Surface(1) = {1};
Plane Surface(2) = {2};
//Transfinite Surface{1} = {1,2,6,5};
//Transfinite Surface{2} = {5,6,3,4};


//Recombine Surface{:};
out[] = Extrude {0, 0, h} { 
		Surface{1:2};
		Layers{Nz};
		Recombine;
	};

Physical Volume("matrix", 99991) = {out[1], out[7]};
Physical Surface("boundary_xm0", 991) = {out[5]};
Physical Surface("boundary_xm1", 981) = {out[10]};
Physical Surface("boundary_xp", 992) = {out[3], out[8]};
Physical Surface("boundary_ym", 993) = {out[2]};
Physical Surface("boundary_yp", 994) = {out[9]};
Physical Surface("boundary_zm", 995) = {1, 2};
Physical Surface("boundary_zp", 996) = {out[0], out[6]};
//Physical Surface("fracture", 91) = {out[11]};
//Physical Curve("fracture_tips", 1) = {4, 21, 13, 25};
Mesh 3;
Coherence Mesh;
Mesh.MshFileVersion = 2.1;