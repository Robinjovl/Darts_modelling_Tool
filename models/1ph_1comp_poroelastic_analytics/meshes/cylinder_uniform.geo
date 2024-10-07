a = 0.05;
a1 = a / 2;
h = 0.05;
k = Sqrt(2);
Phi = -Pi / 3;
lc = 0.01;
lc_frac = 0.01;
rot = Pi / 10;

Point(1) = {0, 0, 0, lc};
Point(2) = {a / 2 * Cos(rot), a / 2 * Sin(rot), 0, lc};
Point(3) = {a / 2 * Cos(rot + 2 * Pi / 3), a / 2 * Sin(rot + 2 * Pi / 3), 0, lc};
Point(4) = {a / 2 * Cos(rot + 4 * Pi / 3), a / 2 * Sin(rot + 4 * Pi / 3), 0, lc};


Point(101) = {0, 0, h, lc_frac};
Point(102) = {a / 2 * Cos(rot), a / 2 * Sin(rot), h + a / 2 * Cos(rot) * Tan(Phi), lc_frac};
Point(103) = {a / 2 * Cos(rot + 2 * Pi / 3), a / 2 * Sin(rot + 2 * Pi / 3), h + a / 2 * Cos(rot + 2 * Pi / 3) * Tan(Phi), lc_frac};
Point(104) = {a / 2 * Cos(rot + 4 * Pi / 3), a / 2 * Sin(rot + 4 * Pi / 3), h + a / 2 * Cos(rot + 4 * Pi / 3) * Tan(Phi), lc_frac};
Point(105) = {a / 2, 0, h + a / 2 * Tan(Phi), lc_frac};

Point(201) = {0, 0, 2 * h, lc};
Point(202) = {a / 2 * Cos(rot), a / 2 * Sin(rot), 2 * h, lc};
Point(203) = {a / 2 * Cos(rot + 2 * Pi / 3), a / 2 * Sin(rot + 2 * Pi / 3), 2 * h, lc};
Point(204) = {a / 2 * Cos(rot + 4 * Pi / 3), a / 2 * Sin(rot + 4 * Pi / 3), 2 * h, lc};

Circle(1) = {2, 1, 3};
Circle(2) = {3, 1, 4};
Circle(3) = {4, 1, 2};

Ellipse(101) = {102, 101, 105, 103};
Ellipse(102) = {103, 101, 105, 104};
Ellipse(103) = {104, 101, 105, 102};

Circle(201) = {202, 201, 203};
Circle(202) = {203, 201, 204};
Circle(203) = {204, 201, 202};

Line(302) = {2, 102};
Line(303) = {3, 103};
Line(304) = {4, 104};

Line(402) = {102, 202};
Line(403) = {103, 203};
Line(404) = {104, 204};

Curve Loop(1) = {1, 2, 3};
Curve Loop(2) = {101, 102, 103};
Curve Loop(3) = {201, 202, 203};
Plane Surface(1) = {1};
Plane Surface(2) = {2};
Plane Surface(3) = {3};

Curve Loop(101) = {1, 303, -101, -302};
Curve Loop(102) = {2, 304, -102, -303};
Curve Loop(103) = {3, 302, -103, -304};
Surface(101) = {101};
Surface(102) = {102};
Surface(103) = {103};

Curve Loop(201) = {201, -403, -101, 402};
Curve Loop(202) = {202, -404, -102, 403};
Curve Loop(203) = {203, -402, -103, 404};
Surface(201) = {201};
Surface(202) = {202};
Surface(203) = {203};

Surface Loop(1) = {1, 2, 101, 102, 103};
Surface Loop(2) = {2, 3, 201, 202, 203};
Volume(1) = {1};
Volume(2) = {2};

//Recombine Surface{:};

Physical Volume("matrix", 99991) = {1, 2};
//Physical Surface("fracture", 91) = {2};
Physical Surface("side", 991) = {101:103, 201:203};
Physical Surface("boundary_zm", 992) = {1};
Physical Surface("boundary_zp", 993) = {3};
//Physical Curve("fracture_shape", 1) = {101, 102, 103};
Mesh.OptimizeMesh = "HighOrderElastic";
//Smoother Surface{304} = 40; 
Mesh 3;
Coherence Mesh;
Mesh.MshFileVersion = 2.1;