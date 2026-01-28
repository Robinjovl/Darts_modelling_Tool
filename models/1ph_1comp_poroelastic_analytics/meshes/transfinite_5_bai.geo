a = 1;
b = 7;
lc = 25;
h = 1;
Nx_pt = 2 + 1;
Ny_pt = 5 + 1;
Nz = 1;

Point(1) = {0, 0, 0, lc};
Point(2) = {a, 0, 0, lc};
Point(3) = {a, b, 0, lc};
Point(4) = {0, b, 0, lc};


Line(1) = {1,2};
Line(2) = {3,2};
Line(3) = {3,4};
Line(4) = {4,1};

Transfinite Curve{1} = Nx_pt;
Transfinite Curve{2} = Ny_pt;
Transfinite Curve{3} = Nx_pt;
Transfinite Curve{4} = Ny_pt;

Line Loop(1) = {4,1,-2,3};

Plane Surface(1) = {1};
Transfinite Surface{1} = {1, 2, 3, 4};
Recombine Surface{1};

out[] = Extrude {0, 0, h} { 
		Surface{1};
		Layers{Nz};
		Recombine;
	};

Mesh.Smoothing = 0;
Physical Volume("matrix", 99991) = {out[1]};
Physical Surface("boundary_xm", 991) = {out[2]};
Physical Surface("boundary_xp", 992) = {out[4]};
Physical Surface("boundary_ym", 993) = {out[3]};
Physical Surface("boundary_yp", 994) = {out[5]};
Physical Surface("boundary_zm", 995) = {1};
Physical Surface("boundary_zp", 996) = {out[0]};
Mesh 3;
Coherence Mesh;
Mesh.MshFileVersion = 2.1;