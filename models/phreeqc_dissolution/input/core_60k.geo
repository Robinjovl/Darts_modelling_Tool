a = 0.07621;
a1 = a / 2;
h = 0.07519 / 2;
k = Sqrt(2);
Phi = -Pi / 3;
lc = 0.003;
rot = Pi / 10;

Point(1) = {0, 0, 0, lc};
Point(2) = {a / 2 * Cos(rot), a / 2 * Sin(rot), 0, lc};
Point(3) = {a / 2 * Cos(rot + 2 * Pi / 3), a / 2 * Sin(rot + 2 * Pi / 3), 0, lc};
Point(4) = {a / 2 * Cos(rot + 4 * Pi / 3), a / 2 * Sin(rot + 4 * Pi / 3), 0, lc};

Point(201) = {0, 0, 2 * h, lc};
Point(202) = {a / 2 * Cos(rot), a / 2 * Sin(rot), 2 * h, lc};
Point(203) = {a / 2 * Cos(rot + 2 * Pi / 3), a / 2 * Sin(rot + 2 * Pi / 3), 2 * h, lc};
Point(204) = {a / 2 * Cos(rot + 4 * Pi / 3), a / 2 * Sin(rot + 4 * Pi / 3), 2 * h, lc};

Circle(1) = {2, 1, 3};
Circle(2) = {3, 1, 4};
Circle(3) = {4, 1, 2};

Circle(201) = {202, 201, 203};
Circle(202) = {203, 201, 204};
Circle(203) = {204, 201, 202};

Line(302) = {2, 202};
Line(303) = {3, 203};
Line(304) = {4, 204};

Curve Loop(1) = {1, 2, 3};
Curve Loop(2) = {201, 202, 203};
Plane Surface(1) = {1};
Plane Surface(2) = {2};

Curve Loop(101) = {1, 303, -201, -302};
Curve Loop(102) = {2, 304, -202, -303};
Curve Loop(103) = {3, 302, -203, -304};
Surface(101) = {101};
Surface(102) = {102};
Surface(103) = {103};

Surface Loop(1) = {1, 2, 101, 102, 103};
Volume(1) = {1};

//Recombine Surface{:};

Physical Volume("matrix", 99991) = {1};
Physical Surface("side", 991) = {101:103};
Physical Surface("boundary_zm", 992) = {1};
Physical Surface("boundary_zp", 993) = {2};
Mesh.OptimizeMesh = "HighOrderElastic";
//Smoother Surface{304} = 40; 
Mesh 3;
Coherence Mesh;
Mesh.MshFileVersion = 2.1;