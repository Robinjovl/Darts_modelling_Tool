// Gmsh project created on Wed Apr 05 16:32:39 2017
Mesh.Algorithm = 6;

T = 1.0;
lc_mult = 0.5;
L = 105*T;
S = 0.38*L;
M = L*0.5;
XS= S;
h = 100.0;

Point(1) = {0, 150*T, 0, lc_mult * S};

Point(3) = {400*T, 100*T, 0, lc_mult * XS};
Point(5) = {800*T, 150*T, 0, lc_mult * S};
Point(6) = {1200*T, 100*T, 0, lc_mult * XS};
Point(9) = {1600*T, 150*T, 0, lc_mult * S};
Point(10) = {1600*T, -1000*T, 0, lc_mult * 1.1*L};
Point(11) = {1500*T, -1000*T, 0, lc_mult * L};
Point(13) = {1000*T, -1000*T, 0, lc_mult * L};
Point(15) = {0, -1000*T, 0, lc_mult * 1.1*L};
Point(16) = {1076.92*T, -576.92*T, 0, lc_mult * M};
Point(20) = {0, -400*T, 0, lc_mult * L};
Point(21) = {1600*T, -400*T, 0, lc_mult * L};


Line(11) = {15, 13};
Line(21) = {13, 16};
Line(31) = {16, 3};
Line(41) = {3, 1};
Line(61) = {1, 20};
Line(611)= {20, 15};


Line(72) = {13, 11};
Line(82) = {11, 16};

Line(93) = {11, 10};
Line(103) ={10, 21};
Line(1013)={21, 9};
Line(113) = {9, 6};

Line(133) = {6, 16};


Line(144) = {6, 5};
Line(154) = {5, 3};

Line Loop(1) = {11, 21, 31, 41, 61, 611};
Line Loop(2) = {72, 82, -21};
Line Loop(3) = {-82, 93, 103, 1013, 113, 133};
Line Loop(4) = {-31, -133, 144, 154};

Plane Surface(11) = {1};
Plane Surface(12) = {2};
Plane Surface(13) = {3};
Plane Surface(14) = {4};

out[] = Extrude{ 0.0, 0.0, h }
	{ Surface{ 11:14 }; Layers{ 1 }; Recombine; };

Physical Volume("Volume", 99991) = { out[1], out[9], out[14], out[22] };
Physical Surface("left", 991) = {out[6], out[7]};
Physical Surface("right", 992) = {out[17], out[18]};
Physical Surface("bottom", 993) = {out[2], out[10], out[16]};
Physical Surface("top", 994) = {out[5], out[19], out[25], out[26]};
Physical Surface("z_minus", 995) = {11:14};
Physical Surface("z_plus", 996) = {out[0], out[8], out[13], out[21]};
Physical Surface("frac1", 9991) = {out[3], out[20]};
Physical Surface("frac2", 9992) = {out[23], out[15]};

Mesh 3;  // Generate 3D mesh
Coherence Mesh;  // Remove duplicate entities
Mesh.MshFileVersion = 2.1;
