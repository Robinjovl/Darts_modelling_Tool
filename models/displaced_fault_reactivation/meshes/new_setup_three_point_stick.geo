W = 4500.0;
H = 4500.0;
a = 75.0;
D = H / 2;
Y1 = D - H;
Y2 = D;
b = 150.0;
delta = 0.01 * W;
phi = 90 * Pi / 180;
Lplus = b + 200;

lc = 1000;
mult = 0.5;

Point(1) = {-W/2, Y1, 0, lc};
Point(2) = {W/2, Y1, 0, lc};
Point(3) = {W/2, -a, 0, mult * lc};
Point(4) = {W/2, b, 0, mult * lc};
Point(5) = {W/2, Y2, 0, lc};
Point(6) = {-W/2, Y2, 0, lc};
Point(7) = {-W/2, a, 0, mult * lc};
Point(8) = {-W/2, -b, 0, mult * lc};

Point(9) = {-delta, Y1, 0, lc};
Point(10) = {delta, Y1, 0, lc};
Point(11) = {W/2, -mult * delta, 0, mult * lc};
Point(12) = {W/2, mult * delta, 0, mult * lc};
Point(13) = {-W/2, mult * delta, 0, mult * lc};
Point(14) = {-W/2, -mult * delta, 0, mult * lc};

Point(15) = {-b * Cos(phi), -b * Sin(phi), 0, mult * lc};
Point(16) = {-a * Cos(phi), -a * Sin(phi), 0, mult * lc};
Point(17) = {a * Cos(phi), a * Sin(phi), 0, mult * lc};
Point(18) = {b * Cos(phi), b * Sin(phi), 0, mult * lc};
Point(19) = {Lplus * Cos(phi), Lplus * Sin(phi), 0, mult * lc};
Point(20) = {-Lplus * Cos(phi), -Lplus * Sin(phi), 0, mult * lc};

Line(1) = {1,9};
Line(2) = {9,10};
Line(3) = {10,2};
Line(4) = {2,3};
Line(5) = {3,11};
Line(6) = {11,12};
Line(7) = {12,4};
Line(8) = {4,5};
Line(9) = {5,6};
Line(10) = {6,7};
Line(11) = {7,13};
Line(12) = {13,14};
Line(13) = {14,8};
Line(14) = {8,1};

Line(15) = {3,16};
Line(16) = {16,15};
Line(17) = {15,8};

Line(18) = {4,18};
Line(19) = {18,17};
Line(20) = {17,7};
Line(21) = {17, 16};

Line(22) = {19, 18};
Line(23) = {20, 15};

Line Loop(1) = {1, 2, 3, 4, 15, 16, 17, 14};
Line Loop(2) = {5, 6, 7, 18, 19, 21, -15};
Line Loop(3) = {20, 11, 12, 13, -17, -16, -21};
Line Loop(4) = {8, 9, 10, -20, -19, -18};

Line Loop(5) = {22,-22};
Line Loop(6) = {23,-23};

Plane Surface(1) = {1,-6};
//Line{23} In Surface{1};
Plane Surface(2) = {2};
Plane Surface(3) = {3};
Plane Surface(4) = {4,-5};
//Line{22} In Surface{4};

out[] = Extrude {0, 0, 500} {
	 Surface{1:4};
	 Layers{1};
	 Recombine;
	};

LEFT = 991;
RIGHT = 992;
BOT = 993;
TOP = 994;
ZM = 995;
ZP = 996;
BOT_STICK = 998;
RIGHT_STICK = 999;
RES = 99991;
OUTER = 99992;
FRAC = 9991;
FRAC_BOUND = 1;

Physical Volume(RES) = {out[13], out[22]};
Physical Volume(OUTER) = {out[1], out[31]};
Physical Surface(ZM) = {1,2,3,4};
Physical Surface(ZP) = {out[0], out[12], out[21], out[30]};
Physical Surface(BOT) = {out[2], out[4]};
Physical Surface(BOT_STICK) = {out[3]};
Physical Surface(RIGHT) = {out[5], out[14], out[16], out[32]};
Physical Surface(TOP) = {out[33]};
Physical Surface(LEFT) = {out[9], out[24], out[26], out[34]};
Physical Surface(RIGHT_STICK) = {out[15], out[25]};
Physical Surface(FRAC) = {out[18], out[19], out[28], out[10], out[38]};
Physical Curve(FRAC_BOUND) = {157, 30, 82, 81, 33, 68, 22, 19, 21, 16, 23, 184};

Mesh 3;
Coherence Mesh;
Mesh.MshFileVersion = 2.1;