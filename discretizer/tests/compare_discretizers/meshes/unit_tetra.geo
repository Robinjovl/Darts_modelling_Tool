a = 1;
lc = 0.2;
lc_frac = 300;
l = 200;
phi = Pi / 4;
h = 1;
Nz = 2;

Point(1) = {0, 0, 0, lc};
Point(2) = {a, 0, 0, lc};
Point(3) = {a, a, 0, lc};
Point(4) = {0, a, 0, lc};

Point(5) = {0, 0, a, lc};
Point(6) = {a, 0, a, lc};
Point(7) = {a, a, a, lc};
Point(8) = {0, a, a, lc};
//Point(5) = {a/2 - l * Cos(phi) / 2, a/2 - l * Sin(phi) / 2, 0, lc_frac};
//Point(6) = {a/2 + l * Cos(phi) / 2, a/2 + l * Sin(phi) / 2, 0, lc_frac};

Line(1) = {1,2};
Line(2) = {3,2};
Line(3) = {3,4};
Line(4) = {4,1};

Line(5) = {5,6};
Line(6) = {7,6};
Line(7) = {7,8};
Line(8) = {8,5};

Line(9) = {1,5};
Line(10) = {2,6};
Line(11) = {3,7};
Line(12) = {4,8};
//Line(5) = {5, 6};

Line Loop(1) = {4,1,-2,3};
Line Loop(2) = {8,5,-6,7};
Line Loop(3) = {4, 9, -8, -12};
Line Loop(4) = {2, 10, -6, -11};
Line Loop(5) = {1, 10, -5, -9};
Line Loop(6) = {-3, 11, 7, -12};

Plane Surface(1) = {1};
Plane Surface(2) = {2};
Plane Surface(3) = {3};
Plane Surface(4) = {4};
Plane Surface(5) = {5};
Plane Surface(6) = {6};
Surface Loop(100) = {1, 2, 3, 4, 5, 6};
Volume(101) = {100};

Physical Volume("matrix", 99991) = {101};
Physical Surface("boundary_xm", 991) = {3};
Physical Surface("boundary_xp", 992) = {4};
Physical Surface("boundary_ym", 993) = {5};
Physical Surface("boundary_yp", 994) = {6};
Physical Surface("boundary_zm", 995) = {1};
Physical Surface("boundary_zp", 996) = {2};

Mesh 3;
Coherence Mesh;
Mesh.MshFileVersion = 2.1;