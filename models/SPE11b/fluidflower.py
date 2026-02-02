# from darts.reservoirs.mesh.geometry.shapes import *
from shapes import *
class Point:
    def __init__(self, tag, xyz):
        self.tag = tag
        self.xyz = xyz

class Curve:
    def __init__(self, tag, curve_type, points):
        self.tag = tag
        self.curve_type = curve_type
        self.points = points

class Surface:
    def __init__(self, tag, curves):
        self.tag = tag
        self.curves = curves

class FluidFlower(Shape):
    def __init__(self, lc: list):
        self.lc = lc

        """Define points, curves, surfaces, volumes"""
        self.points = [Point(1, [0.0, 0.0, 1.05887623]),
                       Point(2, [0.05852737, 0.0, 1.06647472]),
                       Point(3, [0.141867, 0.0, 1.06030826]),
                       Point(4, [0.20139539, 0.0, 1.0652414]),
                       Point(5, [0.26886089, 0.0, 1.06277485]),
                       Point(6, [0.3588148, 0.0, 1.06770799]),
                       Point(7, [0.39453181, 0.0, 1.07264116]),
                       Point(8, [0.55856543, 0.0, 1.07264116]),
                       Point(9, [0.6022195400000001, 0.0, 1.08004091]),
                       Point(10, [0.72524478, 0.0, 1.08004091]),
                       Point(11, [0.79667878, 0.0, 1.0825075]),
                       Point(12, [0.89456983, 0.0, 1.07634102]),
                       Point(13, [0.96335839, 0.0, 1.07825002]),
                       Point(14, [1.0461686, 0.0, 0.8153513599999999]),
                       Point(15, [1.0107815, 0.0, 0.9351043600000001]),
                       Point(16, [0.97782129, 0.0, 1.07328838]),
                       Point(17, [1.073684, 0.0, 0.7778846700000001]),
                       Point(18, [1.0511073, 0.0, 1.04633095]),
                       Point(19, [1.1719278, 0.0, 1.02906486]),
                       Point(20, [1.2989216, 0.0, 1.0274205]),
                       Point(21, [1.3875526, 0.0, 1.02865381]),
                       Point(22, [1.4589862, 0.0, 1.04327831]),
                       Point(23, [1.4788295, 0.0, 1.05414181]),
                       Point(24, [1.5705472, 0.0, 1.06606361]),
                       Point(25, [1.7045961, 0.0, 1.07264116]),
                       Point(26, [1.893764, 0.0, 1.08579626]),
                       Point(27, [2.0441281, 0.0, 1.08661848]),
                       Point(28, [2.1534839, 0.0, 1.08579626]),
                       Point(29, [2.2615169, 0.0, 1.07880761]),
                       Point(30, [2.4136449, 0.0, 1.0652414]),
                       Point(31, [2.5525443, 0.0, 1.04468659]),
                       Point(32, [2.6583725, 0.0, 1.03070927]),
                       Point(33, [2.7708148, 0.0, 1.01426538]),
                       Point(34, [2.8, 0.0, 1.00768784]),
                       Point(35, [1.1096647, 0.0, 0.65577094]),
                       Point(36, [1.1041106, 0.0, 0.69950843]),
                       Point(37, [1.461015, 0.0, 1.00481018]),
                       Point(38, [1.4087074, 0.0, 0.9940537500000001]),
                       Point(39, [1.3241453, 0.0, 0.99046352]),
                       Point(40, [1.2678722, 0.0, 0.98014258]),
                       Point(41, [1.4097158, 0.0, 0.98394664]),
                       Point(42, [1.4567819, 0.0, 0.99001067]),
                       Point(43, [1.4340289, 0.0, 0.9288394600000001]),
                       Point(44, [1.3723841, 0.0, 0.91176628]),
                       Point(45, [1.2440658, 0.0, 0.90611105]),
                       Point(46, [1.1837462, 0.0, 0.89256276]),
                       Point(47, [1.3797921, 0.0, 0.8999266699999999]),
                       Point(48, [1.4250335, 0.0, 0.9105867500000001]),
                       Point(49, [1.3808506, 0.0, 0.7707495200000001]),
                       Point(50, [1.3681507, 0.0, 0.66470516]),
                       Point(51, [0.0, 0.0, 0.99767306]),
                       Point(52, [0.046092579, 0.0, 1.00431686]),
                       Point(53, [0.11170597, 0.0, 1.00185028]),
                       Point(54, [0.19742678, 0.0, 1.00777005]),
                       Point(55, [0.28208935, 0.0, 1.00530347]),
                       Point(56, [0.34981939, 0.0, 1.01072995]),
                       Point(57, [0.43183622, 0.0, 1.01368986]),
                       Point(58, [0.46834694, 0.0, 1.01862302]),
                       Point(59, [0.54930548, 0.0, 1.01023667]),
                       Point(60, [0.6016904200000001, 0.0, 1.01714308]),
                       Point(61, [0.63978856, 0.0, 1.01171661]),
                       Point(62, [0.66518728, 0.0, 1.01862302]),
                       Point(63, [0.71439739, 0.0, 1.01418317]),
                       Point(64, [0.88266416, 0.0, 1.01664976]),
                       Point(65, [0.9917945208685106, 0.0, 0.9924161571213459]),
                       Point(66, [0.0, 0.0, 0.91650654]),
                       Point(67, [0.048738249, 0.0, 0.92439961]),
                       Point(68, [0.17520287, 0.0, 0.92439961]),
                       Point(69, [0.26039461, 0.0, 0.9288394600000001]),
                       Point(70, [0.41331627, 0.0, 0.9387057600000001]),
                       Point(71, [0.5350186800000001, 0.0, 0.9456122]),
                       Point(72, [0.6794741000000001, 0.0, 0.94758547]),
                       Point(73, [0.8731396300000001, 0.0, 0.94413225]),
                       Point(74, [1.014106629856621, 0.0, 0.9238518336353093]),
                       Point(75, [0.0, 0.0, 0.8331361100000001]),
                       Point(76, [0.15562473, 0.0, 0.8449756900000001]),
                       Point(77, [0.3588148, 0.0, 0.8528687300000001]),
                       Point(78, [0.49903708, 0.0, 0.85632195]),
                       Point(79, [0.65142963, 0.0, 0.85829521]),
                       Point(80, [0.8260461, 0.0, 0.8523754100000001]),
                       Point(81, [0.99590029, 0.0, 0.8430024100000001]),
                       Point(82, [1.041774608229471, 0.0, 0.830221007314873]),
                       Point(83, [0.0, 0.0, 0.72756639]),
                       Point(84, [0.10218144, 0.0, 0.7428592100000001]),
                       Point(85, [0.23658327, 0.0, 0.7428592100000001]),
                       Point(86, [0.31701265, 0.0, 0.74137924]),
                       Point(87, [0.42442823, 0.0, 0.7443391699999999]),
                       Point(88, [0.4746966300000001, 0.0, 0.74581909]),
                       Point(89, [0.50221193, 0.0, 0.7517389000000001]),
                       Point(90, [0.72445107, 0.0, 0.7477923599999999]),
                       Point(91, [0.9551564499999999, 0.0, 0.7492723100000001]),
                       Point(92, [1.0435231, 0.0, 0.7492723100000001]),
                       Point(93, [1.074567513716102, 0.0, 0.7439783059327543]),
                       Point(94, [1.009479502287156, 0.0, 0.9757326843432286]),
                       Point(95, [1.1064907, 0.0, 0.96041171]),
                       Point(96, [1.1980322, 0.0, 0.9470921600000001]),
                       Point(97, [1.3446041, 0.0, 0.95005205]),
                       Point(98, [1.444296941553982, 0.0, 0.9564449856971537]),
                       Point(99, [1.478414672353937, 0.0, 0.9929772396773882]),
                       Point(100, [1.6509766, 0.0, 0.9954371599999999]),
                       Point(101, [1.9282463, 0.0, 1.01566312]),
                       Point(102, [2.1145039, 0.0, 1.01763639]),
                       Point(103, [2.2520805, 0.0, 1.00727676]),
                       Point(104, [2.3605544, 0.0, 0.99790374]),
                       Point(105, [2.5589822, 0.0, 0.97471788]),
                       Point(106, [2.7267197, 0.0, 0.94906542]),
                       Point(107, [2.8, 0.0, 0.93179936]),
                       Point(108, [1.047106550023635, 0.0, 0.859783849830798]),
                       Point(109, [1.1567591, 0.0, 0.8518821000000001]),
                       Point(110, [1.3340213, 0.0, 0.8607618]),
                       Point(111, [1.413252806866091, 0.0, 0.8733012957784394]),
                       Point(112, [1.083446950660515, 0.0, 0.7527361695406424]),
                       Point(113, [1.331905310551492, 0.0, 0.6629419797006704]),
                       Point(114, [1.447109345500682, 0.0, 0.8997766726044782]),
                       Point(115, [1.5582006, 0.0, 0.91601321]),
                       Point(116, [1.7319351, 0.0, 0.92916833]),
                       Point(117, [1.965639, 0.0, 0.93245711]),
                       Point(118, [2.1543659, 0.0, 0.93081272]),
                       Point(119, [2.3448565, 0.0, 0.9151910400000001]),
                       Point(120, [2.5177092, 0.0, 0.89381401]),
                       Point(121, [2.6764516, 0.0, 0.8691481599999999]),
                       Point(122, [2.8, 0.0, 0.83708262]),
                       Point(123, [1.419721755310721, 0.0, 0.8182397843799947]),
                       Point(124, [1.5352711, 0.0, 0.8313272700000001]),
                       Point(125, [1.7486912, 0.0, 0.84201578]),
                       Point(126, [1.9303629, 0.0, 0.84777115]),
                       Point(127, [2.1120346, 0.0, 0.8485933400000001]),
                       Point(128, [2.3677859, 0.0, 0.8272163]),
                       Point(129, [2.54593, 0.0, 0.8000839000000001]),
                       Point(130, [2.8, 0.0, 0.73513058]),
                       Point(131, [1.38287688852451, 0.0, 0.7085471913495027]),
                       Point(132, [1.5564368, 0.0, 0.6940209000000001]),
                       Point(133, [1.8289443, 0.0, 0.70717599]),
                       Point(134, [2.0079703, 0.0, 0.71868671]),
                       Point(135, [2.2257999, 0.0, 0.71128696]),
                       Point(136, [2.3774868, 0.0, 0.69484307]),
                       Point(137, [2.6552857, 0.0, 0.64715581]),
                       Point(138, [2.8, 0.0, 0.62824537]),
                       Point(139, [2.8, 0.0, 0.68908772]),
                       Point(140, [2.5344653, 0.0, 0.74417471]),
                       Point(141, [2.3069348, 0.0, 0.7721293]),
                       Point(142, [2.1817042, 0.0, 0.7634265]),
                       Point(143, [2.3064935, 0.0, 0.7571790100000001]),
                       Point(144, [2.5324382, 0.0, 0.7281660799999999]),
                       Point(145, [2.7011447, 0.0, 0.68990993]),
                       Point(146, [2.8, 0.0, 0.6603109400000001]),
                       Point(147, [0.48669045, 0.0, 0.7203311100000001]),
                       Point(148, [0.38527181, 0.0, 0.47038413]),
                       Point(149, [0.3475265, 0.0, 0.286212752]),
                       Point(150, [0.29778718, 0.0, 0.04613213000000001]),
                       Point(151, [0.35264148, 0.0, 0.13780677]),
                       Point(152, [0.5453278377588584, 0.0, 0.7509732443436353]),
                       Point(153, [0.50785609, 0.0, 0.72690864]),
                       Point(154, [0.50080088, 0.0, 0.70059845]),
                       Point(155, [0.5025646699999999, 0.0, 0.65702215]),
                       Point(156, [0.47875335, 0.0, 0.6101571]),
                       Point(157, [0.47610764, 0.0, 0.57480278]),
                       Point(158, [0.46111533, 0.0, 0.5320486800000001]),
                       Point(159, [0.44347729, 0.0, 0.5312265]),
                       Point(160, [0.4364220700000001, 0.0, 0.45476249]),
                       Point(161, [0.41437458, 0.0, 0.43338544]),
                       Point(162, [0.43994972, 0.0, 0.42680792]),
                       Point(163, [0.42848496, 0.0, 0.386520413]),
                       Point(164, [0.38615374, 0.0, 0.269768809]),
                       Point(165, [0.48404474, 0.0, 0.66195532]),
                       Point(166, [0.46376106, 0.0, 0.62084564]),
                       Point(167, [0.40731933, 0.0, 0.4564069]),
                       Point(168, [0.42583929, 0.0, 0.5287599199999999]),
                       Point(169, [0.44171351, 0.0, 0.5526035400000001]),
                       Point(170, [0.45406014, 0.0, 0.5854912799999999]),
                       Point(171, [0.373979068020963, 0.0, 0.4152832485480341]),
                       Point(172, [0.4352564109737261, 0.0, 0.4103154927490871]),
                       Point(173, [0.3432710443942921, 0.0, 0.2656726153694278]),
                       Point(174, [0.3602231398669452, 0.0, 0.2672918947668426]),
                       Point(175, [0.3817442, 0.0, 0.2952568764]),
                       Point(176, [0.36234239, 0.0, 0.2952568764]),
                       Point(177, [0.38615374, 0.0, 0.312522921]),
                       Point(178, [0.38879941, 0.0, 0.326500257]),
                       Point(179, [0.39938222, 0.0, 0.339655291]),
                       Point(180, [0.3790985, 0.0, 0.353632628]),
                       Point(181, [0.41349265, 0.0, 0.375009683]),
                       Point(182, [0.4170223212318815, 0.0, 0.4117937308822809]),
                       Point(183, [0.0, 0.0, 0.69730968]),
                       Point(184, [0.10570907, 0.0, 0.71128696]),
                       Point(185, [0.20360018, 0.0, 0.70799818]),
                       Point(186, [0.36146046, 0.0, 0.68086578]),
                       Point(187, [0.4723414619281126, 0.0, 0.6849679239562484]),
                       Point(188, [0.0, 0.0, 0.59700201]),
                       Point(189, [0.11452814, 0.0, 0.6126236900000001]),
                       Point(190, [0.27238842, 0.0, 0.6035795399999999]),
                       Point(191, [0.40379174, 0.0, 0.5912466500000001]),
                       Point(192, [0.4393585027572717, 0.0, 0.6036811798605378]),
                       Point(193, [0.0, 0.0, 0.457229]),
                       Point(194, [0.027219901, 0.0, 0.4588734]),
                       Point(195, [0.09071680000000001, 0.0, 0.47860606]),
                       Point(196, [0.12422906, 0.0, 0.48189486]),
                       Point(197, [0.27150656, 0.0, 0.45476249]),
                       Point(198, [0.36234239, 0.0, 0.4564069]),
                       Point(199, [0.0, 0.0, 0.3034787977999999]),
                       Point(200, [0.09336247, 0.0, 0.320744842]),
                       Point(201, [0.17273359, 0.0, 0.30841191]),
                       Point(202, [0.0, 0.0, 0.09793036000000011]),
                       Point(203, [0.09071680000000001, 0.0, 0.10533008]),
                       Point(204, [0.20977348, 0.0, 0.06668698000000001]),
                       Point(205, [0.24504955, 0.0, 0.04037671999999998]),
                       Point(206, [0.5122655899999999, 0.0, 0.5780915600000001]),
                       Point(207, [0.6048652600000001, 0.0, 0.5698696299999999]),
                       Point(208, [0.79094641, 0.0, 0.58795787]),
                       Point(209, [0.96115342, 0.0, 0.61591246]),
                       Point(210, [1.0943205, 0.0, 0.63153416]),
                       Point(211, [1.1992667, 0.0, 0.62988978]),
                       Point(212, [1.285693, 0.0, 0.6381116999999999]),
                       Point(213, [1.5158693, 0.0, 0.6389339199999999]),
                       Point(214, [1.7072419, 0.0, 0.64386707]),
                       Point(215, [1.964757, 0.0, 0.6389339199999999]),
                       Point(216, [2.2116895, 0.0, 0.6167346600000001]),
                       Point(217, [2.4700865, 0.0, 0.5682252200000001]),
                       Point(218, [2.6976172, 0.0, 0.52711553]),
                       Point(219, [2.8, 0.0, 0.5197158]),
                       Point(220, [0.50785609, 0.0, 0.42680792]),
                       Point(221, [0.55724259, 0.0, 0.41283058]),
                       Point(222, [0.6330861099999999, 0.0, 0.41118618]),
                       Point(223, [0.7080476999999999, 0.0, 0.42351911]),
                       Point(224, [0.80505685, 0.0, 0.43256324]),
                       Point(225, [0.88530989, 0.0, 0.45229589]),
                       Point(226, [0.98672854, 0.0, 0.4580512]),
                       Point(227, [1.0149494, 0.0, 0.4720285400000001]),
                       Point(228, [1.0387607, 0.0, 0.4646288200000001]),
                       Point(229, [1.1269508, 0.0, 0.48518367]),
                       Point(230, [1.1586993, 0.0, 0.47778395]),
                       Point(231, [1.4497268, 0.0, 0.4942278]),
                       Point(232, [1.5776024, 0.0, 0.50594405]),
                       Point(233, [1.7010685, 0.0, 0.5172492200000001]),
                       Point(234, [1.9678685, 0.0, 0.51642702]),
                       Point(235, [2.2840293, 0.0, 0.48879387]),
                       Point(236, [2.4282208, 0.0, 0.45143662]),
                       Point(237, [2.52611, 0.0, 0.44160736]),
                       Point(238, [2.6901437, 0.0, 0.3912743]),
                       Point(239, [2.8, 0.0, 0.379942796]),
                       Point(240, [0.6053062300000001, 0.0, 0.05887616]),
                       Point(241, [0.72348095, 0.0, 0.06463148000000001]),
                       Point(242, [0.84827, 0.0, 0.03626580000000001]),
                       Point(243, [0.96953144, 0.0, 0.04448773]),
                       Point(244, [1.1534079, 0.0, 0.09957467]),
                       Point(245, [1.3024492, 0.0, 0.11478531]),
                       Point(246, [1.4805932, 0.0, 0.12506274]),
                       Point(247, [1.6410992, 0.0, 0.13287356]),
                       Point(248, [1.8487871, 0.0, 0.12259613]),
                       Point(249, [2.0379548, 0.0, 0.13410686]),
                       Point(250, [2.3069348, 0.0, 0.10574119]),
                       Point(251, [2.5441662, 0.0, 0.06257597000000001]),
                       Point(252, [2.7205465, 0.0, 0.0]),
                       Point(253, [1.499878366836212, 0.0, 1.056877818214859]),
                       Point(254, [0.74091829, 0.0, 0.71219448]),
                       Point(255, [0.78880314, 0.0, 0.72239944]),
                       Point(256, [0.87972384, 0.0, 0.72516497]),
                       Point(257, [1.077695011272843, 0.0, 0.7361181795052441]),
                       Point(258, [0.0, 0.0, 0.0]),
                       Point(259, [2.8, 0.0, 0.0]),
                       Point(260, [2.8, 0.0, 1.2]),
                       Point(261, [0.0, 0.0, 1.2]),
                       Point(262, [1.06279958947309, 0.0, 0.7735538277872919]),
                       Point(263, [0.9985240475666676, 0.0, 0.9721032223371173]),
                       Point(264, [1.002896349862664, 0.0, 0.9960188587033136]),
                       Point(265, [1.093881310647157, 0.0, 0.6954382566060642]),
                       Point(266, [1.088299992695025, 0.0, 0.7402351609748854]),
                       Point(267, [1.324381450280446, 0.0, 0.9816580842499051]),
                       Point(268, [1.244489000172595, 0.0, 0.8948443915649497]),
                       Point(269, [1.373535216140802, 0.0, 0.7096659524519872]),
                       Point(270, [1.021531857361188, 0.0, 0.9385930040326425]),
                       Point(271, [1.025152357130446, 0.0, 0.9274363294102262]),
                       Point(272, [0.4100857431308563, 0.0, 0.3357744604426111]),
                       Point(273, [0.4230544550183695, 0.0, 0.3715428109574566]),
                       Point(274, [0.375058247046978, 0.0, 0.3044773764934574]),
                       Point(275, [1.068683551594596, 0.0, 0.7587660668600231]),
                       Point(276, [1.08578816096, 0.0, 0.7157782180556541]),
                       Point(277, [1.36642635, 0.0, 0.992258635]),
                       Point(278, [1.29600875, 0.0, 0.98530305]),
                       Point(279, [1.296126825140223, 0.0, 0.9809003321249525]),
                       Point(280, [1.367048625140223, 0.0, 0.9828023621249525]),
                       Point(281, [1.370842958070401, 0.0, 0.6871855562259936]),
                       Point(282, [1.377192908070401, 0.0, 0.7402077362259936]),
                       Point(283, [1.059045306858051, 0.0, 0.7466253079663772]),
                       Point(284, [1.469656844262255, 0.0, 0.7012840456747513]),
                       Point(285, [1.375513794262255, 0.0, 0.6866261756747514]),
                       Point(286, [1.401299321917615, 0.0, 0.7633934878647487]),
                       Point(287, [0.9787094256364217, 0.0, 0.730641574752622]),
                       ]

        # spe11a line curves
        self.curves = [Curve(1, curve_type='line', points=[252, 258]),
                       Curve(2, curve_type='line', points=[252, 259]),
                       Curve(3, curve_type='line', points=[34, 107]),
                       Curve(4, curve_type='line', points=[34, 260]),
                       Curve(5, curve_type='line', points=[107, 122]),
                       Curve(6, curve_type='line', points=[122, 130]),
                       Curve(7, curve_type='line', points=[130, 139]),
                       Curve(8, curve_type='line', points=[138, 146]),
                       Curve(9, curve_type='line', points=[138, 219]),
                       Curve(10, curve_type='line', points=[139, 146]),
                       Curve(11, curve_type='line', points=[219, 239]),
                       Curve(12, curve_type='line', points=[239, 259]),
                       Curve(13, curve_type='line', points=[260, 261]),
                       Curve(14, curve_type='line', points=[1, 51]),
                       Curve(15, curve_type='line', points=[1, 261]),
                       Curve(16, curve_type='line', points=[51, 66]),
                       Curve(17, curve_type='line', points=[66, 75]),
                       Curve(18, curve_type='line', points=[75, 83]),
                       Curve(19, curve_type='line', points=[83, 183]),
                       Curve(20, curve_type='line', points=[183, 188]),
                       Curve(21, curve_type='line', points=[188, 193]),
                       Curve(22, curve_type='line', points=[193, 199]),
                       Curve(23, curve_type='line', points=[199, 202]),
                       Curve(24, curve_type='line', points=[202, 258]),
                       Curve(25, curve_type='line', points=[1, 2]),
                       Curve(26, curve_type='line', points=[2, 3]),
                       Curve(27, curve_type='line', points=[3, 4]),
                       Curve(28, curve_type='line', points=[4, 5]),
                       Curve(29, curve_type='line', points=[5, 6]),
                       Curve(30, curve_type='line', points=[6, 7]),
                       Curve(31, curve_type='line', points=[7, 8]),
                       Curve(32, curve_type='line', points=[8, 9]),
                       Curve(33, curve_type='line', points=[9, 10]),
                       Curve(34, curve_type='line', points=[10, 11]),
                       Curve(35, curve_type='line', points=[11, 12]),
                       Curve(36, curve_type='line', points=[12, 13]),
                       Curve(37, curve_type='line', points=[14, 82]),
                       Curve(38, curve_type='line', points=[15, 74]),
                       Curve(39, curve_type='line', points=[74, 82]),
                       Curve(40, curve_type='line', points=[15, 263]),
                       Curve(41, curve_type='line', points=[263, 65]),
                       Curve(42, curve_type='line', points=[65, 13]),
                       Curve(43, curve_type='line', points=[13, 16]),
                       Curve(44, curve_type='line', points=[16, 264]),
                       Curve(45, curve_type='line', points=[264, 94]),
                       Curve(46, curve_type='line', points=[94, 270]),
                       Curve(47, curve_type='line', points=[270, 271]),
                       Curve(48, curve_type='line', points=[271, 108]),
                       Curve(49, curve_type='line', points=[108, 17]),
                       Curve(50, curve_type='line', points=[16, 18]),
                       Curve(51, curve_type='line', points=[18, 19]),
                       Curve(52, curve_type='line', points=[19, 20]),
                       Curve(53, curve_type='line', points=[20, 21]),
                       Curve(54, curve_type='line', points=[21, 22]),
                       Curve(55, curve_type='line', points=[22, 23]),
                       Curve(56, curve_type='line', points=[23, 253]),
                       Curve(57, curve_type='line', points=[24, 253]),
                       Curve(58, curve_type='line', points=[24, 25]),
                       Curve(59, curve_type='line', points=[25, 26]),
                       Curve(60, curve_type='line', points=[26, 27]),
                       Curve(61, curve_type='line', points=[27, 28]),
                       Curve(62, curve_type='line', points=[28, 29]),
                       Curve(63, curve_type='line', points=[29, 30]),
                       Curve(64, curve_type='line', points=[30, 31]),
                       Curve(65, curve_type='line', points=[31, 32]),
                       Curve(66, curve_type='line', points=[32, 33]),
                       Curve(67, curve_type='line', points=[33, 34]),
                       Curve(68, curve_type='line', points=[14, 262]),
                       Curve(69, curve_type='line', points=[262, 275]),
                       Curve(70, curve_type='line', points=[275, 93]),
                       Curve(71, curve_type='line', points=[93, 257]),
                       Curve(72, curve_type='line', points=[257, 276]),
                       Curve(73, curve_type='line', points=[276, 265]),
                       Curve(74, curve_type='line', points=[265, 35]),
                       Curve(75, curve_type='line', points=[35, 36]),
                       Curve(76, curve_type='line', points=[17, 112]),
                       Curve(77, curve_type='line', points=[112, 266]),
                       Curve(78, curve_type='line', points=[266, 36]),
                       Curve(79, curve_type='line', points=[23, 37]),
                       Curve(80, curve_type='line', points=[37, 38]),
                       Curve(81, curve_type='line', points=[38, 277]),
                       Curve(82, curve_type='line', points=[277, 39]),
                       Curve(83, curve_type='line', points=[39, 278]),
                       Curve(84, curve_type='line', points=[278, 40]),
                       Curve(85, curve_type='line', points=[40, 279]),
                       Curve(86, curve_type='line', points=[279, 267]),
                       Curve(87, curve_type='line', points=[267, 280]),
                       Curve(88, curve_type='line', points=[280, 41]),
                       Curve(89, curve_type='line', points=[41, 42]),
                       Curve(90, curve_type='line', points=[42, 98]),
                       Curve(91, curve_type='line', points=[43, 98]),
                       Curve(92, curve_type='line', points=[43, 44]),
                       Curve(93, curve_type='line', points=[44, 45]),
                       Curve(94, curve_type='line', points=[45, 46]),
                       Curve(95, curve_type='line', points=[46, 268]),
                       Curve(96, curve_type='line', points=[268, 47]),
                       Curve(97, curve_type='line', points=[47, 48]),
                       Curve(98, curve_type='line', points=[48, 111]),
                       Curve(99, curve_type='line', points=[49, 111]),
                       Curve(100, curve_type='line', points=[50, 281]),
                       Curve(101, curve_type='line', points=[281, 269]),
                       Curve(102, curve_type='line', points=[269, 282]),
                       Curve(103, curve_type='line', points=[282, 49]),
                       Curve(104, curve_type='line', points=[51, 52]),
                       Curve(105, curve_type='line', points=[52, 53]),
                       Curve(106, curve_type='line', points=[53, 54]),
                       Curve(107, curve_type='line', points=[54, 55]),
                       Curve(108, curve_type='line', points=[55, 56]),
                       Curve(109, curve_type='line', points=[56, 57]),
                       Curve(110, curve_type='line', points=[57, 58]),
                       Curve(111, curve_type='line', points=[58, 59]),
                       Curve(112, curve_type='line', points=[59, 60]),
                       Curve(113, curve_type='line', points=[60, 61]),
                       Curve(114, curve_type='line', points=[61, 62]),
                       Curve(115, curve_type='line', points=[62, 63]),
                       Curve(116, curve_type='line', points=[63, 64]),
                       Curve(117, curve_type='line', points=[64, 65]),
                       Curve(118, curve_type='line', points=[66, 67]),
                       Curve(119, curve_type='line', points=[67, 68]),
                       Curve(120, curve_type='line', points=[68, 69]),
                       Curve(121, curve_type='line', points=[69, 70]),
                       Curve(122, curve_type='line', points=[70, 71]),
                       Curve(123, curve_type='line', points=[71, 72]),
                       Curve(124, curve_type='line', points=[72, 73]),
                       Curve(125, curve_type='line', points=[73, 74]),
                       Curve(126, curve_type='line', points=[75, 76]),
                       Curve(127, curve_type='line', points=[76, 77]),
                       Curve(128, curve_type='line', points=[77, 78]),
                       Curve(129, curve_type='line', points=[78, 79]),
                       Curve(130, curve_type='line', points=[79, 80]),
                       Curve(131, curve_type='line', points=[80, 81]),
                       Curve(132, curve_type='line', points=[81, 82]),
                       Curve(133, curve_type='line', points=[83, 84]),
                       Curve(134, curve_type='line', points=[84, 85]),
                       Curve(135, curve_type='line', points=[85, 86]),
                       Curve(136, curve_type='line', points=[86, 87]),
                       Curve(137, curve_type='line', points=[87, 88]),
                       Curve(138, curve_type='line', points=[88, 89]),
                       Curve(139, curve_type='line', points=[89, 152]),
                       Curve(140, curve_type='line', points=[90, 152]),
                       Curve(141, curve_type='line', points=[90, 91]),
                       Curve(142, curve_type='line', points=[91, 92]),
                       Curve(143, curve_type='line', points=[92, 283]),
                       Curve(144, curve_type='line', points=[283, 93]),
                       Curve(145, curve_type='line', points=[94, 95]),
                       Curve(146, curve_type='line', points=[95, 96]),
                       Curve(147, curve_type='line', points=[96, 97]),
                       Curve(148, curve_type='line', points=[97, 98]),
                       Curve(149, curve_type='line', points=[99, 100]),
                       Curve(150, curve_type='line', points=[100, 101]),
                       Curve(151, curve_type='line', points=[101, 102]),
                       Curve(152, curve_type='line', points=[102, 103]),
                       Curve(153, curve_type='line', points=[103, 104]),
                       Curve(154, curve_type='line', points=[104, 105]),
                       Curve(155, curve_type='line', points=[105, 106]),
                       Curve(156, curve_type='line', points=[106, 107]),
                       Curve(157, curve_type='line', points=[108, 109]),
                       Curve(158, curve_type='line', points=[109, 110]),
                       Curve(159, curve_type='line', points=[110, 111]),
                       Curve(160, curve_type='line', points=[49, 112]),
                       Curve(161, curve_type='line', points=[35, 113]),
                       Curve(162, curve_type='line', points=[114, 115]),
                       Curve(163, curve_type='line', points=[115, 116]),
                       Curve(164, curve_type='line', points=[116, 117]),
                       Curve(165, curve_type='line', points=[117, 118]),
                       Curve(166, curve_type='line', points=[118, 119]),
                       Curve(167, curve_type='line', points=[119, 120]),
                       Curve(168, curve_type='line', points=[120, 121]),
                       Curve(169, curve_type='line', points=[121, 122]),
                       Curve(170, curve_type='line', points=[123, 124]),
                       Curve(171, curve_type='line', points=[124, 125]),
                       Curve(172, curve_type='line', points=[125, 126]),
                       Curve(173, curve_type='line', points=[126, 127]),
                       Curve(174, curve_type='line', points=[127, 128]),
                       Curve(175, curve_type='line', points=[128, 129]),
                       Curve(176, curve_type='line', points=[129, 130]),
                       Curve(177, curve_type='line', points=[131, 284]),
                       Curve(178, curve_type='line', points=[284, 132]),
                       Curve(179, curve_type='line', points=[132, 133]),
                       Curve(180, curve_type='line', points=[133, 134]),
                       Curve(181, curve_type='line', points=[134, 135]),
                       Curve(182, curve_type='line', points=[135, 136]),
                       Curve(183, curve_type='line', points=[136, 137]),
                       Curve(184, curve_type='line', points=[137, 138]),
                       Curve(185, curve_type='line', points=[139, 140]),
                       Curve(186, curve_type='line', points=[140, 141]),
                       Curve(187, curve_type='line', points=[141, 142]),
                       Curve(188, curve_type='line', points=[142, 143]),
                       Curve(189, curve_type='line', points=[143, 144]),
                       Curve(190, curve_type='line', points=[144, 145]),
                       Curve(191, curve_type='line', points=[145, 146]),
                       Curve(192, curve_type='line', points=[88, 147]),
                       Curve(193, curve_type='line', points=[147, 187]),
                       Curve(194, curve_type='line', points=[148, 192]),
                       Curve(195, curve_type='line', points=[187, 192]),
                       Curve(196, curve_type='line', points=[148, 171]),
                       Curve(197, curve_type='line', points=[149, 171]),
                       Curve(198, curve_type='line', points=[149, 173]),
                       Curve(199, curve_type='line', points=[150, 173]),
                       Curve(200, curve_type='line', points=[150, 151]),
                       Curve(201, curve_type='line', points=[152, 153]),
                       Curve(202, curve_type='line', points=[153, 154]),
                       Curve(203, curve_type='line', points=[154, 155]),
                       Curve(204, curve_type='line', points=[155, 156]),
                       Curve(205, curve_type='line', points=[156, 157]),
                       Curve(206, curve_type='line', points=[157, 158]),
                       Curve(207, curve_type='line', points=[158, 159]),
                       Curve(208, curve_type='line', points=[159, 160]),
                       Curve(209, curve_type='line', points=[160, 161]),
                       Curve(210, curve_type='line', points=[161, 162]),
                       Curve(211, curve_type='line', points=[162, 172]),
                       Curve(212, curve_type='line', points=[163, 172]),
                       Curve(213, curve_type='line', points=[163, 273]),
                       Curve(214, curve_type='line', points=[273, 272]),
                       Curve(215, curve_type='line', points=[272, 164]),
                       Curve(216, curve_type='line', points=[151, 164]),
                       Curve(217, curve_type='line', points=[155, 165]),
                       Curve(218, curve_type='line', points=[165, 166]),
                       Curve(219, curve_type='line', points=[156, 166]),
                       Curve(220, curve_type='line', points=[148, 167]),
                       Curve(221, curve_type='line', points=[167, 168]),
                       Curve(222, curve_type='line', points=[168, 169]),
                       Curve(223, curve_type='line', points=[169, 170]),
                       Curve(224, curve_type='line', points=[166, 170]),
                       Curve(225, curve_type='line', points=[171, 182]),
                       Curve(226, curve_type='line', points=[172, 182]),
                       Curve(227, curve_type='line', points=[164, 174]),
                       Curve(228, curve_type='line', points=[173, 174]),
                       Curve(229, curve_type='line', points=[174, 175]),
                       Curve(230, curve_type='line', points=[175, 176]),
                       Curve(231, curve_type='line', points=[177, 274]),
                       Curve(232, curve_type='line', points=[274, 176]),
                       Curve(233, curve_type='line', points=[177, 178]),
                       Curve(234, curve_type='line', points=[178, 179]),
                       Curve(235, curve_type='line', points=[179, 180]),
                       Curve(236, curve_type='line', points=[180, 181]),
                       Curve(237, curve_type='line', points=[181, 182]),
                       Curve(238, curve_type='line', points=[183, 184]),
                       Curve(239, curve_type='line', points=[184, 185]),
                       Curve(240, curve_type='line', points=[185, 186]),
                       Curve(241, curve_type='line', points=[186, 187]),
                       Curve(242, curve_type='line', points=[188, 189]),
                       Curve(243, curve_type='line', points=[189, 190]),
                       Curve(244, curve_type='line', points=[190, 191]),
                       Curve(245, curve_type='line', points=[191, 192]),
                       Curve(246, curve_type='line', points=[193, 194]),
                       Curve(247, curve_type='line', points=[194, 195]),
                       Curve(248, curve_type='line', points=[195, 196]),
                       Curve(249, curve_type='line', points=[196, 197]),
                       Curve(250, curve_type='line', points=[197, 198]),
                       Curve(251, curve_type='line', points=[148, 198]),
                       Curve(252, curve_type='line', points=[199, 200]),
                       Curve(253, curve_type='line', points=[200, 201]),
                       Curve(254, curve_type='line', points=[149, 201]),
                       Curve(255, curve_type='line', points=[202, 203]),
                       Curve(256, curve_type='line', points=[203, 204]),
                       Curve(257, curve_type='line', points=[204, 205]),
                       Curve(258, curve_type='line', points=[150, 205]),
                       Curve(259, curve_type='line', points=[157, 206]),
                       Curve(260, curve_type='line', points=[206, 207]),
                       Curve(261, curve_type='line', points=[207, 208]),
                       Curve(262, curve_type='line', points=[208, 209]),
                       Curve(263, curve_type='line', points=[209, 210]),
                       Curve(264, curve_type='line', points=[210, 211]),
                       Curve(265, curve_type='line', points=[211, 212]),
                       Curve(266, curve_type='line', points=[212, 213]),
                       Curve(267, curve_type='line', points=[213, 214]),
                       Curve(268, curve_type='line', points=[214, 215]),
                       Curve(269, curve_type='line', points=[215, 216]),
                       Curve(270, curve_type='line', points=[216, 217]),
                       Curve(271, curve_type='line', points=[217, 218]),
                       Curve(272, curve_type='line', points=[218, 219]),
                       Curve(273, curve_type='line', points=[162, 220]),
                       Curve(274, curve_type='line', points=[220, 221]),
                       Curve(275, curve_type='line', points=[221, 222]),
                       Curve(276, curve_type='line', points=[222, 223]),
                       Curve(277, curve_type='line', points=[223, 224]),
                       Curve(278, curve_type='line', points=[224, 225]),
                       Curve(279, curve_type='line', points=[225, 226]),
                       Curve(280, curve_type='line', points=[226, 227]),
                       Curve(281, curve_type='line', points=[227, 228]),
                       Curve(282, curve_type='line', points=[228, 229]),
                       Curve(283, curve_type='line', points=[229, 230]),
                       Curve(284, curve_type='line', points=[230, 231]),
                       Curve(285, curve_type='line', points=[231, 232]),
                       Curve(286, curve_type='line', points=[232, 233]),
                       Curve(287, curve_type='line', points=[233, 234]),
                       Curve(288, curve_type='line', points=[234, 235]),
                       Curve(289, curve_type='line', points=[235, 236]),
                       Curve(290, curve_type='line', points=[236, 237]),
                       Curve(291, curve_type='line', points=[237, 238]),
                       Curve(292, curve_type='line', points=[238, 239]),
                       Curve(293, curve_type='line', points=[151, 240]),
                       Curve(294, curve_type='line', points=[240, 241]),
                       Curve(295, curve_type='line', points=[241, 242]),
                       Curve(296, curve_type='line', points=[242, 243]),
                       Curve(297, curve_type='line', points=[243, 244]),
                       Curve(298, curve_type='line', points=[244, 245]),
                       Curve(299, curve_type='line', points=[245, 246]),
                       Curve(300, curve_type='line', points=[246, 247]),
                       Curve(301, curve_type='line', points=[247, 248]),
                       Curve(302, curve_type='line', points=[248, 249]),
                       Curve(303, curve_type='line', points=[249, 250]),
                       Curve(304, curve_type='line', points=[250, 251]),
                       Curve(305, curve_type='line', points=[251, 252]),
                       Curve(306, curve_type='line', points=[50, 113]),
                       Curve(307, curve_type='line', points=[50, 285]),
                       Curve(308, curve_type='line', points=[285, 131]),
                       Curve(309, curve_type='line', points=[99, 114]),
                       Curve(310, curve_type='line', points=[99, 253]),
                       Curve(311, curve_type='line', points=[114, 123]),
                       Curve(312, curve_type='line', points=[123, 286]),
                       Curve(313, curve_type='line', points=[286, 131]),
                       Curve(314, curve_type='line', points=[154, 254]),
                       Curve(315, curve_type='line', points=[254, 255]),
                       Curve(316, curve_type='line', points=[255, 256]),
                       Curve(317, curve_type='line', points=[256, 287]),
                       Curve(318, curve_type='line', points=[287, 257]),
                       Curve(319, curve_type='line', points=[199, 202]),
                       Curve(320, curve_type='line', points=[193, 1]),
                       Curve(321, curve_type='line', points=[34, 219])
                      ]

        # spe11a curve surfaces
        self.surfaces = [Surface(1, curves=[305, 1, -24, 255, 256, 257, -258, 200, 293, 294, 295, 296, 297, 298, 299,
                                            300, 301, 302, 303, 304]),
                         Surface(2, curves=[12, -2, -305, -304, -303, -302, -301, -300, -299, -298, -297, -296, -295,
                                            -294, -293, 216, -215, -214, -213, 212, -211, 273, 274, 275, 276, 277, 278,
                                            279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292]),
                         Surface(3, curves=[199, -198, 254, -253, -252, 23, 255, 256, 257, -258]),
                         Surface(4, curves=[160, 77, 78, -75, 161, -306, 100, 101, 102, 103]),
                         Surface(5, curves=[177, 178, 179, 180, 181, 182, 183, 184, 8, -191, -190, -189, -188, -187,
                                            -186, -185, -7, -176, -175, -174, -173, -172, -171, -170, 312, 313]),
                         Surface(6, curves=[131, 132, -37, 68, 69, 70, -144, -143, -142, -141, 140, -139, -138, -137,
                                            -136, -135, -134, -133, -18, 126, 127, 128, 129, 130]),
                         Surface(7, curves=[11, -292, -291, -290, -289, -288, -287, -286, -285, -284, -283, -282, -281,
                                            -280, -279, -278, -277, -276, -275, -274, -273, -210, -209, -208, -207,
                                            -206, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272]),
                         Surface(8, curves=[196, -197, 254, -253, -252, -22, 246, 247, 248, 249, 250, -251]),
                         Surface(9, curves=[204, 219, -218, -217]),
                         Surface(10, curves=[194, -245, -244, -243, -242, 21, 246, 247, 248, 249, 250, -251]),
                         Surface(11, curves=[203, 204, 205, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270,
                                             271, 272, -9, -184, -183, -182, -181, -180, -179, -178, -177, -308, -307,
                                             306, -161, -74, -73, -72, -318, -317, -316, -315, -314]),
                         Surface(12, curves=[10, -191, -190, -189, -188, -187, -186, -185]),
                         Surface(13, curves=[6, -176, -175, -174, -173, -172, -171, -170, -311, 162, 163, 164, 165, 166,
                                             167, 168, 169]),
                         Surface(14, curves=[99, -159, -158, -157, 49, 76, -160]),
                         Surface(15, curves=[39, -132, -131, -130, -129, -128, -127, -126, -17, 118, 119, 120, 121, 122,
                                             123, 124, 125]),
                         Surface(22, curves=[212, 226, -237, -236, -235, -234, -233, 231, 232, -230, -229, -227, -215,
                                             -214, -213]),
                         Surface(16, curves=[20, 242, 243, 244, 245, -195, -241, -240, -239, -238]),
                         Surface(17, curves=[201, 202, 314, 315, 316, 317, 318, -71, -144, -143, -142, -141, 140]),
                         Surface(18, curves=[5, -169, -168, -167, -166, -165, -164, -163, -162, -309, 149, 150, 151,
                                             152, 153, 154, 155, 156]),
                         Surface(19, curves=[91, -148, -147, -146, -145, 46, 47, 48, 157, 158, 159, -98, -97, -96, -95,
                                             -94, -93, -92]),
                         Surface(20, curves=[117, -41, -40, 38, -125, -124, -123, -122, -121, -120, -119, -118, -16,
                                             104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116]),
                         Surface(21, curves=[205, 206, 207, 208, 209, 210, 211, 226, -225, -196, 220, 221, 222, 223,
                                             -224, -219]),
                         Surface(23, curves=[199, 228, -227, -216, -200]),
                         Surface(24, curves=[220, 221, 222, 223, -224, -218, -217, -203, -202, -201, -139, -138, 192,
                                             193, 195, -194]),
                         Surface(25, curves=[103, 99, -98, -97, -96, -95, -94, -93, -92, 91, -90, -89, -88, -87, -86,
                                             -85, -84, -83, -82, -81, -80, -79, 56, -310, 309, 311, 312, 313, -308,
                                             -307, 100, 101, 102]),
                         Surface(26, curves=[225, -237, -236, -235, -234, -233, 231, 232, -230, -229, -228, -198, 197]),
                         Surface(27, curves=[193, -241, -240, -239, -238, -19, 133, 134, 135, 136, 137, 192]),
                         Surface(28, curves=[14, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117,
                                             42, -36, -35, -34, -33, -32, -31, -30, -29, -28, -27, -26, -25]),
                         Surface(29, curves=[44, 45, 145, 146, 147, 148, -90, -89, -88, -87, -86, -85, -84, -83, -82,
                                             -81, -80, -79, -55, -54, -53, -52, -51, -50]),
                         Surface(30, curves=[57, -310, 149, 150, 151, 152, 153, 154, 155, 156, -3, -67, -66, -65, -64,
                                             -63, -62, -61, -60, -59, -58]),
                         Surface(31, curves=[72, 73, 74, 75, -78, -77, -76, -49, -48, -47, -46, -45, -44, -43, -42, -41,
                                             -40, 38, 39, -37, 68, 69, 70, 71]),
                         Surface(32, curves=[4, 13, -15, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 43, 50, 51, 52,
                                             53, 54, 55, 56, -57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67])
                         ]


        self.connect_points()

        self.volumes = []

        """Define Physical Points, Curves, Surfaces, Volumes"""
        self.physical_points = {}

        self.physical_curves = {
            # 'lower left': [319],
            # 'upper left': [320],
            # 'lower right': [12],
            # 'upper right': [321]
        }

        self.physical_surfaces = {}

        self.physical_volumes = {}


    def convert_to_spe11b(self):
        """
        Function to convert spe11a lab-scale geometry to spe11b field-scale geometry
        """
        lab_x_scale = 0
        lab_y_scale = 0
        for point in self.points:
            if point.xyz[0] > lab_x_scale:
                lab_x_scale = point.xyz[0]
            if point.xyz[2] > lab_y_scale:
                lab_y_scale = point.xyz[2]
        scale_factor_x = 8400.0 / lab_x_scale
        scale_factor_y = 1200.0 / lab_y_scale

        for point in self.points:
            point.xyz[0] *= scale_factor_x
            point.xyz[2] *= scale_factor_y
        return

    def convert_to_spe11c(self):
        """
        Function to convert spe11b 2D geometry to spe11c 3D geometry
        """
        import gmsh
        from math import pi

        SIZE_DOMAIN_Y = 5000.0

        def z_offset_at(y: float) -> float:
            f = (y - 2500.0) / 2500.0
            return 150.0 * (1.0 - f * f) + 10.0 * y / SIZE_DOMAIN_Y

        gmsh.initialize()
        gmsh.model.add("fluidflower")
        gmsh.model.setCurrent("fluidflower")

        # Add points
        for i, point in enumerate(self.points, start=1):
            gmsh.model.occ.addPoint(*point.xyz, tag=i)

        # Add curves
        for curve in self.curves:
            if curve.curve_type == 'line':
                gmsh.model.occ.addLine(curve.points[0], curve.points[1], tag=curve.tag)

        # Add surfaces
        for surface in self.surfaces:
            curve_tags = [curve_tag for curve_tag in surface.curves]
            wire_tag = gmsh.model.occ.addCurveLoop(curve_tags, tag=surface.tag)
            gmsh.model.occ.addPlaneSurface([wire_tag], tag=surface.tag)
            # gmsh.model.occ.addPlaneSurface(surface.curves, tag=surface.tag)

        # Extrude surfaces to create volume
        gmsh.model.occ.synchronize()

        # # Rotate the model to align with z-axis
        # gmsh.model.occ.rotate(
        #     dimTags=gmsh.model.getEntities(2),
        #     x=0.5 * (max[0] - min[0]), y=min[1], z=0.0,
        #     ax=1.0, ay=0.0, az=0.0,
        #     angle=pi / 2.0
        # )
        # gmsh.model.occ.synchronize()

        frontside_surface_tags = gmsh.model.getEntities(dim=2)
        backside_surface_tags = gmsh.model.occ.copy(frontside_surface_tags)
        backside_dz = z_offset_at(SIZE_DOMAIN_Y)
        gmsh.model.occ.translate(backside_surface_tags, dx=0.0, dy=SIZE_DOMAIN_Y, dz=backside_dz)

        gmsh.model.occ.synchronize()
        def find_corresponding_point(front_point_tag: int, front_curve_points: list, back_curve_points: list) -> int:
            # Find the index of the front point in the front curve points
            index = [p[1] for p in front_curve_points].index(front_point_tag)
            # Return the corresponding back point
            return back_curve_points[index][1]

        def align_curves(front_boundary_curves, back_boundary_curves):
            # Align back_boundary_curves to front_boundary_curves if they are offset by one
            if len(front_boundary_curves) != len(back_boundary_curves):
                raise ValueError("Front and back boundary curves do not match in length")

            aligned_back_boundary_curves = []
            for i, front_curve in enumerate(front_boundary_curves):
                aligned_back_boundary_curves.append(back_boundary_curves[(i + 1) % len(back_boundary_curves)])

            return aligned_back_boundary_curves

        def _make_connecting_spline(source_tag: int, target_tag: int) -> tuple[int, list]:
            source = gmsh.model.getValue(0, source_tag, [])
            target = gmsh.model.getValue(0, target_tag, [])

            print(f"Source: {source}")
            print(f"Target: {target}")
            print(f"Backside dz: {backside_dz}")

            assert abs(source[0] - target[0]) < 1e-8 * SIZE_DOMAIN_Y, f"X coordinates differ: {source[0]} vs {target[0]}"
            assert abs(abs(source[2] - target[2]) - backside_dz) < 1e-8 * SIZE_DOMAIN_Y, f"Z coordinates differ: {source[2]} vs {target[2]}"

            extrusion_length = 5000.0
            num_support_points = 10
            dy = extrusion_length / float(num_support_points + 1)  # first/last point are source/target

            support_point_tags = [source_tag]
            for i in range(num_support_points):
                y_reference = source[1] + float(i + 1) * dy
                support_point_tags.append(
                    gmsh.model.occ.addPoint(
                        x=source[0],
                        y=source[1] + y_reference,
                        z=source[2] + z_offset_at(y_reference)
                    )
                )
            return gmsh.model.occ.addBSpline(support_point_tags + [target_tag]), support_point_tags[1:]

        physical_volumes = {}
        front_point_to_connecting_spline_index = {}
        front_curve_to_connecting_surface_index = {}
        gmsh.model.occ.synchronize()
        for front, back in zip(frontside_surface_tags, backside_surface_tags):
            front_boundary_curves = gmsh.model.getBoundary([front], recursive=False)
            back_boundary_curves = gmsh.model.getBoundary([back], recursive=False)
            aligned_back_boundary_curves = align_curves(front_boundary_curves, back_boundary_curves)

            assert len(front_boundary_curves) == len(aligned_back_boundary_curves)

            bounding_surface_tags = [front[1]]
            for front_curve, back_curve in zip(front_boundary_curves, aligned_back_boundary_curves):
                assert front_curve[1] > 0 and back_curve[1] > 0 \
                       or front_curve[1] < 0 and back_curve[1] < 0

                abs_front_curve = abs(front_curve[1])
                if abs_front_curve in front_curve_to_connecting_surface_index:
                    bounding_surface_tags.append(front_curve_to_connecting_surface_index.get(abs_front_curve))
                else:
                    front_curve_points = gmsh.model.getBoundary([front_curve], recursive=False)
                    back_curve_points = gmsh.model.getBoundary([back_curve], recursive=False)
                    assert len(front_curve_points) == len(back_curve_points)
                    assert len(front_curve_points) == 2

                    pfront_0, pfront_1 = front_curve_points[0][1], front_curve_points[1][1]
                    pback_0 = find_corresponding_point(pfront_0, front_curve_points, back_curve_points)
                    pback_1 = find_corresponding_point(pfront_1, front_curve_points, back_curve_points)
                    # pback_0, pback_1 = back_curve_points[0][1], back_curve_points[1][1]
                    spline1, points1 = front_point_to_connecting_spline_index.get(pfront_0), []
                    spline2, points2 = front_point_to_connecting_spline_index.get(pfront_1), []

                    if spline1 is None:
                        spline1, points1 = _make_connecting_spline(pfront_0, pback_0)
                        front_point_to_connecting_spline_index[pfront_0] = spline1
                    if spline2 is None:
                        spline2, points2 = _make_connecting_spline(pfront_1, pback_1)
                        front_point_to_connecting_spline_index[pfront_1] = spline2

                    curve_loop = [spline1, spline2]
                    if front_curve[1] < 0:
                        curve_loop = list(reversed(curve_loop))
                    wire_tag = gmsh.model.occ.addWire([front_curve[1], *curve_loop, -back_curve[1]])
                    bounding_surface_tags.append(gmsh.model.occ.addSurfaceFilling(wire_tag, tag=wire_tag))
                    front_curve_to_connecting_surface_index[abs_front_curve] = bounding_surface_tags[-1]

                    gmsh.model.occ.remove(dimTags=[(0, t) for t in points1], recursive=False)
                    gmsh.model.occ.remove(dimTags=[(0, t) for t in points2], recursive=False)

            bounding_surface_tags.append(back[1])
            surf_loop = gmsh.model.occ.addSurfaceLoop(bounding_surface_tags)
            volume_tag = gmsh.model.occ.addVolume([surf_loop])
            physical_index = 1  # Example physical group index
            if physical_index not in physical_volumes:
                physical_volumes[physical_index] = {"name": "Volume", "volumes": []}
            physical_volumes[physical_index]["volumes"].append(volume_tag)

        gmsh.model.occ.synchronize()

        gmsh.write("spe11c.brep")

        with open("spe11c.geo", "w") as geo_file:
            geo_file.write('Merge "spe11c.brep";\n')
            for physical_index, properties in physical_volumes.items():
                volume_tags_str = ",".join(str(t) for t in properties["volumes"])
                geo_file.write(f"Physical Volume({physical_index}) = {{{volume_tags_str}}};\n")
            geo_file.write(f"Characteristic Length{{:}} = {self.lc};\n")

        gmsh.finalize()
        return

    def plot_shape_2D(self):
        """
        Function to plot FluidFlower with extra
        """
        import matplotlib.pyplot as plt
        super().plot_shape_2D()

        colours = ['blue', 'mediumseagreen', 'lightskyblue', 'orchid', 'dodgerblue', 'darkcyan']

        # Plot wells, sensors and boxes
        wells = [[2700., 0.0, 300.],
                 [5100., 0.0, 700.]]
        sensors = [[4500., 0., 500.],
                   [5100., 0., 1100.]]
        boxes = [[[3300., 8300.], [0., 1.], [0., 600.]],
                 [[100., 3300.], [0., 1.], [600., 1200.]],
                 [[3300., 7800.], [0., 1.], [100., 400.]]]

        ii = 0
        for i, well in enumerate(wells):
            plt.scatter(well[0], well[2], c=colours[ii + i], label='I-' + str(i + 1))

        ii = 2
        for i, sensor in enumerate(sensors):
            plt.scatter(sensor[0], sensor[2], c=colours[ii + i], label='S-' + str(i + 1))

        box_label = ["Box A", "Box B", "Box C"]
        ii = 4
        for i, box in enumerate(boxes):
            x = [box[0][0], box[0][0], box[0][1], box[0][1], box[0][0]]
            z = [box[2][0], box[2][1], box[2][1], box[2][0], box[2][0]]
            plt.plot(x, z, c=colours[i], linestyle='--', alpha=0.7, label=box_label[i])

        plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")

        plt.tight_layout()
        plt.savefig('fluidflower_base.pdf', format='pdf', dpi=1000)
        plt.close()

        return
