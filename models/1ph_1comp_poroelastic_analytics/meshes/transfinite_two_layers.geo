ax = 100;
ay = 100;
x_ratio = 0.75;
ax1 = x_ratio * ax;
lc = 25;
lc_frac = 600;
l = 100;
phi = Pi / 4;
h = 10;
Nx_pt = 40;
Ny_pt = 40;
Nz = 1;

Point(1) = {0, 0, 0, lc};
Point(2) = {ax1, 0, 0, lc};
Point(3) = {ax, 0,  0, lc};
Point(4) = {ax, ay, 0, lc};
Point(5) = {ax1, ay, 0, lc};
Point(6) = {0, ay, 0, lc};
//Point(5) = {a/2, a/2 + 1, 0, lc};
//Point(5) = {a/2 - l * Cos(phi) / 2, a/2 - l * Sin(phi) / 2, 0, lc_frac};
//Point(6) = {a/2 + l * Cos(phi) / 2, a/2 + l * Sin(phi) / 2, 0, lc_frac};

Line(1) = {1,2};
Line(2) = {2,3};
Line(3) = {3,4};
Line(4) = {4,5};
Line(5) = {5,6};
Line(6) = {6,1};
Line(7) = {2,5};

Transfinite Curve{1} = x_ratio * Nx_pt + 1;
Transfinite Curve{2} = (1.0 - x_ratio) * Nx_pt + 1;
Transfinite Curve{3} = Ny_pt + 1;
Transfinite Curve{4} = (1.0 - x_ratio) * Nx_pt + 1;
Transfinite Curve{5} = x_ratio * Nx_pt + 1;
Transfinite Curve{6} = Ny_pt + 1;
Transfinite Curve{7} = Ny_pt + 1;

Line Loop(1) = {1, 7, 5, 6};
Line Loop(2) = {2, 3, 4, -7};

Plane Surface(1) = {1};
Transfinite Surface{1} = {1, 2, 5, 6};
Plane Surface(2) = {2};
Transfinite Surface{2} = {2, 3, 4, 5};
Recombine Surface{:};

out[] = Extrude {0, 0, h} { 
		Surface{1,2};
		Layers{Nz};
		Recombine;
	};

Mesh.Smoothing = 0;
Physical Volume("matrix2", 99992) = {out[1]};
Physical Volume("matrix1", 99991) = {out[7]};
Physical Surface("boundary_xm", 991) = {out[5]};
Physical Surface("boundary_xp", 992) = {out[9]};
Physical Surface("boundary_ym", 993) = {out[2], out[8]};
Physical Surface("boundary_yp", 994) = {out[4], out[10]};
Physical Surface("boundary_zm", 995) = {1, 2};
Physical Surface("boundary_zp", 996) = {out[0], out[6]};
Mesh 3;
Coherence Mesh;
Mesh.MshFileVersion = 2.1;