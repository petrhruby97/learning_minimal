#include <Eigen/SVD>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <iostream>

void order_points(Eigen::Vector3d * P, int * perm5, int ix)
{
	double angle1 = std::atan2(P[ix](1), P[ix](0));
	if(angle1 < 0)
		angle1 = angle1 + 2*acos(-1.0);
	
	//obtain the relative angles
	double angles[5];	
	for(int i=0;i<5;++i)
	{
		if(i==ix)
		{
			angles[ix] = 7;
			continue;
		}
		//if negative, add 2*pi to obtain a positive number
		double cur_ang = std::atan2(P[i](1), P[i](0));
		if(cur_ang < 0)
			cur_ang = cur_ang + 2*acos(-1.0);
		//subtract the angle of the longest point from other angles (if negative, add 2*pi)
		double ang = cur_ang - angle1;
		if(ang < 0)
			ang = ang + 2*acos(-1.0);
		angles[i] = ang;
	}

	perm5[0] = ix;
	
	for(int i=1;i<5;++i)
	{
		double min = 7;
		int next = -1;
		for(int j=0;j<5;++j)
		{
			if(angles[j] < min)
			{
				next = j;
				min = angles[j];
			}
		}
		perm5[i] = next;
		angles[next] = 7;
	}
}

bool normalize(Eigen::Vector3d * P, Eigen::Vector3d * Q, std::vector<Eigen::Vector2d> &P1, std::vector<Eigen::Vector2d> &Q1, Eigen::Matrix3d &CP1,Eigen::Matrix3d &CQ1, int * perm5)
{
	//project the points to a sphere and obtain the centroids
	Eigen::Vector3d centroidP = Eigen::Vector3d::Zero();
	Eigen::Vector3d centroidQ = Eigen::Vector3d::Zero();
	for(int i=0;i<5;++i)
	{
		P[i] = P[i]/P[i].norm();
		Q[i] = Q[i]/Q[i].norm();
		centroidP = centroidP + P[i];
		centroidQ = centroidQ + Q[i];
	}
	centroidP = 0.2*centroidP;
	centroidQ = 0.2*centroidQ;
	
	//identify the first point and view
	int ix;
	int view;
	double best = 1000;
	for(int i=0;i<5;++i)
	{
		double ang = P[i].transpose() * centroidP;
		if(ang < best)
		{
			best = ang;
			ix = i;
			view = 0;
		}
		
		ang = Q[i].transpose() * centroidQ;
		if(ang < best)
		{
			best = ang;
			ix = i;
			view = 1;
		}
	}
	int perm2[2];
	perm2[view] = 0;
	perm2[1-view] = 1;
	
	//rotate the centroid to zero and the given point to y axis
	Eigen::Vector3d p0 = centroidP/centroidP.norm();
	Eigen::Vector3d p1 = p0.cross(P[ix]);
	p1 = p1/p1.norm();
	Eigen::Vector3d p2 = p0.cross(p1);
	p2 = p2/p2.norm();
	Eigen::Matrix3d Zp;
	Zp << p0, p1, p2;
	
	Eigen::Vector3d q0 = centroidQ/centroidQ.norm();
	Eigen::Vector3d q1 = q0.cross(Q[ix]);
	q1 = q1/q1.norm();
	Eigen::Vector3d q2 = q0.cross(q1);
	q2 = q2/q2.norm();
	Eigen::Matrix3d Zq;
	Zq << q0, q1, q2;
	
	Eigen::Matrix3d ZZ;
	ZZ << 0,-1,0,0,0,-1,1,0,0;
	Eigen::Matrix3d CP = ZZ * Zp.transpose();
	Eigen::Matrix3d CQ = ZZ * Zq.transpose();	
	
	//rotate the points and project them back to the plane
	for(int i=0;i<5;++i)
	{
		P[i] = CP * P[i];
		P[i] = P[i]/P[i](2);
		
		Q[i] = CQ * Q[i];
		Q[i] = Q[i]/Q[i](2);
	}
	
	//order the points
	if(view)
		order_points(P, perm5, ix);
	else
		order_points(Q, perm5, ix);
	
	//permute the views
	if(!view)
	{
		for(int i=0;i<5;++i)
		{
			Eigen::Vector2d p;
			p(0) = P[perm5[i]](0);
			p(1) = P[perm5[i]](1);
			P1[i] = p;
			
			Eigen::Vector2d q;
			q(0) = Q[perm5[i]](0);
			q(1) = Q[perm5[i]](1);
			Q1[i] = q;
			
		}
		CP1 = CP;
		CQ1 = CQ;
	}
	else
	{
		for(int i=0;i<5;++i)
		{
			Eigen::Vector2d p;
			p(0) = P[perm5[i]](0);
			p(1) = P[perm5[i]](1);
			Q1[i] = p;
			
			Eigen::Vector2d q;
			q(0) = Q[perm5[i]](0);
			q(1) = Q[perm5[i]](1);
			P1[i] = q;
			
		}
		CP1 = CQ;
		CQ1 = CP;
	}
	return view;
}

void extract_pose(double params[40], double solution[9], Eigen::Matrix3d &R, Eigen::Vector3d &t)
{
	const Eigen::Vector3d x1(params[20], params[25], 1);
	const Eigen::Vector3d x2(solution[0] * params[21], solution[0] * params[26], solution[0]);
	const Eigen::Vector3d x3(solution[1] * params[22], solution[1] * params[27], solution[1]);
	const Eigen::Vector3d y1(solution[4] * params[30], solution[4] * params[35], solution[4]);
	const Eigen::Vector3d y2(solution[5] * params[31], solution[5] * params[36], solution[5]);
	const Eigen::Vector3d y3(solution[6] * params[32], solution[6] * params[37], solution[6]);
	const Eigen::Vector3d z2 = x2-x1;
	const Eigen::Vector3d z3 = x3-x1;
	const Eigen::Vector3d z1 = z2.cross(z3);
	const Eigen::Vector3d zz2 = y2-y1;
	const Eigen::Vector3d zz3 = y3-y1;
	const Eigen::Vector3d zz1 = zz2.cross(zz3);
	Eigen::Matrix3d Z;
	Z << z1, z2, z3;
	Eigen::Matrix3d ZZ;
	ZZ << zz1, zz2, zz3;
	
	R = ZZ * Z.inverse();
	t = (y1 - R*x1);
}

struct track_settings
{
	track_settings():
	init_dt_(0.05),   // m2 tStep, t_step, raw interface code initDt
	min_dt_(1e-4),        // m2 tStepMin, raw interface code minDt
	end_zone_factor_(0.05),
	epsilon_(.00001), // m2 CorrectorTolerance
	epsilon2_(epsilon_ * epsilon_), 
	dt_increase_factor_(3.),  // m2 stepIncreaseFactor
	dt_decrease_factor_(1./dt_increase_factor_),  // m2 stepDecreaseFactor not existent in DEFAULT, using what is in track.m2:77 
	infinity_threshold_(1e7), // m2 InfinityThreshold
	infinity_threshold2_(infinity_threshold_ * infinity_threshold_),
	max_corr_steps_(3),  // m2 maxCorrSteps (track.m2 param of rawSetParametersPT corresp to max_corr_steps in NAG.cpp)
	num_successes_before_increase_(4), // m2 numberSuccessesBeforeIncrease
	corr_thresh_(0.0001),
	anch_num_(26)
	{ }

	double init_dt_;   // m2 tStep, t_step, raw interface code initDt
	double min_dt_;        // m2 tStepMin, raw interface code minDt
	double end_zone_factor_;
	double epsilon_; // m2 CorrectorTolerance (chicago.m2, track.m2), raw interface code epsilon (interface2.d, NAG.cpp:rawSwetParametersPT)
	double epsilon2_; 
	double dt_increase_factor_;  // m2 stepIncreaseFactor
	double dt_decrease_factor_;  // m2 stepDecreaseFactor not existent in DEFAULT, using what is in track.m2:77 
	double infinity_threshold_; // m2 InfinityThreshold
	double infinity_threshold2_;
	unsigned max_corr_steps_;  // m2 maxCorrSteps (track.m2 param of rawSetParametersPT corresp to max_corr_steps in NAG.cpp)
	unsigned num_successes_before_increase_; // m2 numberSuccessesBeforeIncrease
	double corr_thresh_;
	unsigned anch_num_;
};

void print_settings(struct track_settings &settings)
{
  std::cerr << " track settings -----------------------------------------------\n";
  const char *names[12] = {
    "init_dt_",
    "min_dt_",
    "end_zone_factor_",
    "epsilon_",
    "epsilon2_",
    "dt_increase_factor_",
    "dt_decrease_factor_",
    "infinity_threshold_",
    "infinity_threshold2_",
    "max_corr_steps_",
    "num_successes_before_increase_",
    "corr_thresh_",
  };
  double *ptr = (double *) &settings;
  for (int i=0; i < 9; ++i)
    std::cerr << names[i] << " = " << *ptr++ << std::endl;
  std::cerr << names[9] << " = " << settings.max_corr_steps_ << std::endl;
  std::cerr << names[10] << " = " << settings.num_successes_before_increase_ << std::endl;
  std::cerr << names[11] << " = " << settings.corr_thresh_ << std::endl;
  std::cerr << "---------------------------------------------------------------\n";
}

bool load_settings(std::string set_file, struct track_settings &settings)
{
	std::ifstream f;
	f.open(set_file);

	if(!f.good())
	{
		f.close();
		std::cout << "Settings file not available\n";

		return 0;
	}

	std::string t;

	//init dt
	double init_dt_;
	f >> init_dt_;
	getline(f, t);

	//min dt
	double min_dt_;
	f >> min_dt_;
	getline(f, t);

	// end zone factor
	double end_zone;
	f >> end_zone;
	getline(f, t);
	
	// corrector tolerance epsilon
	double epsilon;
	f >> epsilon;
	getline(f, t);
	
	// step increase factor
	double increase_factor;
	f >> increase_factor;
	getline(f, t);
	
	// infinity threshold
	double inf_thr;
	f >> inf_thr;
	getline(f, t);
	
	// max corr steps
	unsigned max_corr;
	f >> max_corr;
	getline(f, t);
	
	// num successes before increase
	unsigned succ_bef_inc;
	f >> succ_bef_inc;
	getline(f, t);

	// threshold for the difference between the obtained and the GT solution
	double corr_thresh_;
	f >> corr_thresh_;
	getline(f, t);

	//number of anchors
	int anchors;
	f >> anchors;
	getline(f, t);

	f.close();
	settings.init_dt_ = init_dt_;
	settings.min_dt_ = min_dt_;
	settings.end_zone_factor_ = end_zone;
	settings.epsilon_ = epsilon;
	settings.epsilon2_ = epsilon * epsilon;
	settings.dt_increase_factor_ = increase_factor;
	settings.dt_decrease_factor_ = 1./increase_factor;
	settings.infinity_threshold_ = inf_thr;
	settings.infinity_threshold2_ = inf_thr * inf_thr;
	settings.max_corr_steps_ = max_corr;
	settings.num_successes_before_increase_ = succ_bef_inc;
	settings.corr_thresh_ = corr_thresh_;
	settings.anch_num_ = anchors;

	return 1;
}


template <unsigned N, typename F>
struct minus_array
{ // Speed critical -----------------------------------------
	static inline void 
	multiply_scalar_to_self(F *__restrict__ a, F b)
	{
		for (unsigned i = 0; i < N; ++i, ++a) *a = *a * b;
	}

	static inline void
	negate_self(F * __restrict__ a)
	{
		for (unsigned i = 0; i < N; ++i, ++a) *a = -*a;
	}

	static inline void 
	multiply_self(F * __restrict__ a, const F * __restrict__ b)
	{
		for (unsigned int i=0; i < N; ++i,++a,++b) *a *= *b;
	}

	static inline void 
	add_to_self(F * __restrict__ a, const F * __restrict__ b)
	{
		for (unsigned int i=0; i < N; ++i,++a,++b) *a += *b;
	}

	static inline void 
	add_scalar_to_self(F * __restrict__ a, F b)
	{
		for (unsigned int i=0; i < N; ++i,++a) *a += b;
	}

	static inline void 
	copy(const F * __restrict__ a, F * __restrict__ b)
	{
		memcpy(b, a, N*sizeof(double));
	}

	static inline F
	norm2(const F *__restrict__ a)
	{
		F val = 0;
		F const* __restrict__ end = a+N;
		while (a != end) val += std::norm(*a++);
		return val;
	}
};

//Straight line program for evaluation of the Jacobian of the homotopy function, generated in Macaulay2
inline void evaluate_Hxt(const double * x, const double * params, double * y)
{
	const	double	&X0	=	x[0];		
	const	double	&X1	=	x[1];		
	const	double	&X2	=	x[2];		
	const	double	&X3	=	x[3];		
	const	double	&X4	=	x[4];		
	const	double	&X5	=	x[5];		
	const	double	&X6	=	x[6];		
	const	double	&X7	=	x[7];		
	const	double	&X8	=	x[8];		
	const	double	&X9	=	x[9];		
	const	double	&X10	=	params[0];		
	const	double	&X11	=	params[1];		
	const	double	&X12	=	params[2];		
	const	double	&X13	=	params[3];		
	const	double	&X14	=	params[4];		
	const	double	&X15	=	params[5];		
	const	double	&X16	=	params[6];		
	const	double	&X17	=	params[7];		
	const	double	&X18	=	params[8];		
	const	double	&X19	=	params[9];		
	const	double	&X20	=	params[10];		
	const	double	&X21	=	params[11];		
	const	double	&X22	=	params[12];		
	const	double	&X23	=	params[13];		
	const	double	&X24	=	params[14];		
	const	double	&X25	=	params[15];		
	const	double	&X26	=	params[16];		
	const	double	&X27	=	params[17];		
	const	double	&X28	=	params[18];		
	const	double	&X29	=	params[19];		
	const	double	&X30	=	params[20];		
	const	double	&X31	=	params[21];		
	const	double	&X32	=	params[22];		
	const	double	&X33	=	params[23];		
	const	double	&X34	=	params[24];		
	const	double	&X35	=	params[25];		
	const	double	&X36	=	params[26];		
	const	double	&X37	=	params[27];		
	const	double	&X38	=	params[28];		
	const	double	&X39	=	params[29];		
	const	double	&X40	=	params[30];		
	const	double	&X41	=	params[31];		
	const	double	&X42	=	params[32];		
	const	double	&X43	=	params[33];		
	const	double	&X44	=	params[34];		
	const	double	&X45	=	params[35];		
	const	double	&X46	=	params[36];		
	const	double	&X47	=	params[37];		
	const	double	&X48	=	params[38];		
	const	double	&X49	=	params[39];		
	static	constexpr	double	C0	=	1;	
	static	constexpr	double	C1	=	-1;	
	const	double	G0	=	C1	*	X9;
	const	double	G1	=	C0	+	G0;
	const	double	G2	=	G1	*	X10;
	const	double	G3	=	X9	*	X30;
	const	double	G4	=	G2	+	G3;
	const	double	G5	=	G1	*	X11;
	const	double	G6	=	X9	*	X31;
	const	double	G7	=	G5	+	G6;
	const	double	G8	=	X0	*	G7;
	const	double	G9	=	C1	*	G8;
	const	double	G10	=	G4	+	G9;
	const	double	G11	=	C1	*	G7;
	const	double	G12	=	G10	*	G11;
	const	double	G13	=	G12	+	G12;
	const	double	G14	=	G1	*	X15;
	const	double	G15	=	X9	*	X35;
	const	double	G16	=	G14	+	G15;
	const	double	G17	=	G1	*	X16;
	const	double	G18	=	X9	*	X36;
	const	double	G19	=	G17	+	G18;
	const	double	G20	=	X0	*	G19;
	const	double	G21	=	C1	*	G20;
	const	double	G22	=	G16	+	G21;
	const	double	G23	=	C1	*	G19;
	const	double	G24	=	G22	*	G23;
	const	double	G25	=	G24	+	G24;
	const	double	G26	=	G13	+	G25;
	static	constexpr	double	C2	=	1;	
	const	double	G27	=	C1	*	X0;
	const	double	G28	=	C2	+	G27;
	const	double	G29	=	C1	*	G28;
	const	double	G30	=	G29	+	G29;
	const	double	G31	=	G26	+	G30;
	static	constexpr	double	C3	=	0;	
	const	double	G32	=	G1	*	X12;
	const	double	G33	=	X9	*	X32;
	const	double	G34	=	G32	+	G33;
	const	double	G35	=	X1	*	G34;
	const	double	G36	=	C1	*	G35;
	const	double	G37	=	G8	+	G36;
	const	double	G38	=	G37	*	G7;
	const	double	G39	=	G38	+	G38;
	const	double	G40	=	G1	*	X17;
	const	double	G41	=	X9	*	X37;
	const	double	G42	=	G40	+	G41;
	const	double	G43	=	X1	*	G42;
	const	double	G44	=	C1	*	G43;
	const	double	G45	=	G20	+	G44;
	const	double	G46	=	G45	*	G19;
	const	double	G47	=	G46	+	G46;
	const	double	G48	=	G39	+	G47;
	const	double	G49	=	C1	*	X1;
	const	double	G50	=	X0	+	G49;
	const	double	G51	=	G50	+	G50;
	const	double	G52	=	G48	+	G51;
	const	double	G53	=	G1	*	X13;
	const	double	G54	=	X9	*	X33;
	const	double	G55	=	G53	+	G54;
	const	double	G56	=	X2	*	G55;
	const	double	G57	=	C1	*	G56;
	const	double	G58	=	G8	+	G57;
	const	double	G59	=	G58	*	G7;
	const	double	G60	=	G59	+	G59;
	const	double	G61	=	G1	*	X18;
	const	double	G62	=	X9	*	X38;
	const	double	G63	=	G61	+	G62;
	const	double	G64	=	X2	*	G63;
	const	double	G65	=	C1	*	G64;
	const	double	G66	=	G20	+	G65;
	const	double	G67	=	G66	*	G19;
	const	double	G68	=	G67	+	G67;
	const	double	G69	=	G60	+	G68;
	const	double	G70	=	C1	*	X2;
	const	double	G71	=	X0	+	G70;
	const	double	G72	=	G71	+	G71;
	const	double	G73	=	G69	+	G72;
	const	double	G74	=	G1	*	X14;
	const	double	G75	=	X9	*	X34;
	const	double	G76	=	G74	+	G75;
	const	double	G77	=	X3	*	G76;
	const	double	G78	=	C1	*	G77;
	const	double	G79	=	G8	+	G78;
	const	double	G80	=	G79	*	G7;
	const	double	G81	=	G80	+	G80;
	const	double	G82	=	G1	*	X19;
	const	double	G83	=	X9	*	X39;
	const	double	G84	=	G82	+	G83;
	const	double	G85	=	X3	*	G84;
	const	double	G86	=	C1	*	G85;
	const	double	G87	=	G20	+	G86;
	const	double	G88	=	G87	*	G19;
	const	double	G89	=	G88	+	G88;
	const	double	G90	=	G81	+	G89;
	const	double	G91	=	C1	*	X3;
	const	double	G92	=	X0	+	G91;
	const	double	G93	=	G92	+	G92;
	const	double	G94	=	G90	+	G93;
	const	double	G95	=	G4	+	G36;
	const	double	G96	=	C1	*	G34;
	const	double	G97	=	G95	*	G96;
	const	double	G98	=	G97	+	G97;
	const	double	G99	=	G16	+	G44;
	const	double	G100	=	C1	*	G42;
	const	double	G101	=	G99	*	G100;
	const	double	G102	=	G101	+	G101;
	const	double	G103	=	G98	+	G102;
	const	double	G104	=	C2	+	G49;
	const	double	G105	=	C1	*	G104;
	const	double	G106	=	G105	+	G105;
	const	double	G107	=	G103	+	G106;
	const	double	G108	=	G37	*	G96;
	const	double	G109	=	G108	+	G108;
	const	double	G110	=	G45	*	G100;
	const	double	G111	=	G110	+	G110;
	const	double	G112	=	G109	+	G111;
	const	double	G113	=	C1	*	G50;
	const	double	G114	=	G113	+	G113;
	const	double	G115	=	G112	+	G114;
	const	double	G116	=	G35	+	G57;
	const	double	G117	=	G116	*	G34;
	const	double	G118	=	G117	+	G117;
	const	double	G119	=	G43	+	G65;
	const	double	G120	=	G119	*	G42;
	const	double	G121	=	G120	+	G120;
	const	double	G122	=	G118	+	G121;
	const	double	G123	=	X1	+	G70;
	const	double	G124	=	G123	+	G123;
	const	double	G125	=	G122	+	G124;
	const	double	G126	=	G35	+	G78;
	const	double	G127	=	G126	*	G34;
	const	double	G128	=	G127	+	G127;
	const	double	G129	=	G43	+	G86;
	const	double	G130	=	G129	*	G42;
	const	double	G131	=	G130	+	G130;
	const	double	G132	=	G128	+	G131;
	const	double	G133	=	X1	+	G91;
	const	double	G134	=	G133	+	G133;
	const	double	G135	=	G132	+	G134;
	const	double	G136	=	G4	+	G57;
	const	double	G137	=	C1	*	G55;
	const	double	G138	=	G136	*	G137;
	const	double	G139	=	G138	+	G138;
	const	double	G140	=	G16	+	G65;
	const	double	G141	=	C1	*	G63;
	const	double	G142	=	G140	*	G141;
	const	double	G143	=	G142	+	G142;
	const	double	G144	=	G139	+	G143;
	const	double	G145	=	C2	+	G70;
	const	double	G146	=	C1	*	G145;
	const	double	G147	=	G146	+	G146;
	const	double	G148	=	G144	+	G147;
	const	double	G149	=	G58	*	G137;
	const	double	G150	=	G149	+	G149;
	const	double	G151	=	G66	*	G141;
	const	double	G152	=	G151	+	G151;
	const	double	G153	=	G150	+	G152;
	const	double	G154	=	C1	*	G71;
	const	double	G155	=	G154	+	G154;
	const	double	G156	=	G153	+	G155;
	const	double	G157	=	G116	*	G137;
	const	double	G158	=	G157	+	G157;
	const	double	G159	=	G119	*	G141;
	const	double	G160	=	G159	+	G159;
	const	double	G161	=	G158	+	G160;
	const	double	G162	=	C1	*	G123;
	const	double	G163	=	G162	+	G162;
	const	double	G164	=	G161	+	G163;
	const	double	G165	=	G4	+	G78;
	const	double	G166	=	C1	*	G76;
	const	double	G167	=	G165	*	G166;
	const	double	G168	=	G167	+	G167;
	const	double	G169	=	G16	+	G86;
	const	double	G170	=	C1	*	G84;
	const	double	G171	=	G169	*	G170;
	const	double	G172	=	G171	+	G171;
	const	double	G173	=	G168	+	G172;
	const	double	G174	=	C2	+	G91;
	const	double	G175	=	C1	*	G174;
	const	double	G176	=	G175	+	G175;
	const	double	G177	=	G173	+	G176;
	const	double	G178	=	G79	*	G166;
	const	double	G179	=	G178	+	G178;
	const	double	G180	=	G87	*	G170;
	const	double	G181	=	G180	+	G180;
	const	double	G182	=	G179	+	G181;
	const	double	G183	=	C1	*	G92;
	const	double	G184	=	G183	+	G183;
	const	double	G185	=	G182	+	G184;
	const	double	G186	=	G126	*	G166;
	const	double	G187	=	G186	+	G186;
	const	double	G188	=	G129	*	G170;
	const	double	G189	=	G188	+	G188;
	const	double	G190	=	G187	+	G189;
	const	double	G191	=	C1	*	G133;
	const	double	G192	=	G191	+	G191;
	const	double	G193	=	G190	+	G192;
	const	double	G194	=	G1	*	X20;
	const	double	G195	=	X9	*	X40;
	const	double	G196	=	G194	+	G195;
	const	double	G197	=	X4	*	G196;
	const	double	G198	=	G1	*	X21;
	const	double	G199	=	X9	*	X41;
	const	double	G200	=	G198	+	G199;
	const	double	G201	=	X5	*	G200;
	const	double	G202	=	C1	*	G201;
	const	double	G203	=	G197	+	G202;
	const	double	G204	=	G203	*	G196;
	const	double	G205	=	G204	+	G204;
	const	double	G206	=	G1	*	X25;
	const	double	G207	=	X9	*	X45;
	const	double	G208	=	G206	+	G207;
	const	double	G209	=	X4	*	G208;
	const	double	G210	=	G1	*	X26;
	const	double	G211	=	X9	*	X46;
	const	double	G212	=	G210	+	G211;
	const	double	G213	=	X5	*	G212;
	const	double	G214	=	C1	*	G213;
	const	double	G215	=	G209	+	G214;
	const	double	G216	=	G215	*	G208;
	const	double	G217	=	G216	+	G216;
	const	double	G218	=	G205	+	G217;
	const	double	G219	=	C1	*	X5;
	const	double	G220	=	X4	+	G219;
	const	double	G221	=	G220	+	G220;
	const	double	G222	=	G218	+	G221;
	const	double	G223	=	C1	*	G222;
	const	double	G224	=	G1	*	X22;
	const	double	G225	=	X9	*	X42;
	const	double	G226	=	G224	+	G225;
	const	double	G227	=	X6	*	G226;
	const	double	G228	=	C1	*	G227;
	const	double	G229	=	G197	+	G228;
	const	double	G230	=	G229	*	G196;
	const	double	G231	=	G230	+	G230;
	const	double	G232	=	G1	*	X27;
	const	double	G233	=	X9	*	X47;
	const	double	G234	=	G232	+	G233;
	const	double	G235	=	X6	*	G234;
	const	double	G236	=	C1	*	G235;
	const	double	G237	=	G209	+	G236;
	const	double	G238	=	G237	*	G208;
	const	double	G239	=	G238	+	G238;
	const	double	G240	=	G231	+	G239;
	const	double	G241	=	C1	*	X6;
	const	double	G242	=	X4	+	G241;
	const	double	G243	=	G242	+	G242;
	const	double	G244	=	G240	+	G243;
	const	double	G245	=	C1	*	G244;
	const	double	G246	=	G1	*	X23;
	const	double	G247	=	X9	*	X43;
	const	double	G248	=	G246	+	G247;
	const	double	G249	=	X7	*	G248;
	const	double	G250	=	C1	*	G249;
	const	double	G251	=	G197	+	G250;
	const	double	G252	=	G251	*	G196;
	const	double	G253	=	G252	+	G252;
	const	double	G254	=	G1	*	X28;
	const	double	G255	=	X9	*	X48;
	const	double	G256	=	G254	+	G255;
	const	double	G257	=	X7	*	G256;
	const	double	G258	=	C1	*	G257;
	const	double	G259	=	G209	+	G258;
	const	double	G260	=	G259	*	G208;
	const	double	G261	=	G260	+	G260;
	const	double	G262	=	G253	+	G261;
	const	double	G263	=	C1	*	X7;
	const	double	G264	=	X4	+	G263;
	const	double	G265	=	G264	+	G264;
	const	double	G266	=	G262	+	G265;
	const	double	G267	=	C1	*	G266;
	const	double	G268	=	G1	*	X24;
	const	double	G269	=	X9	*	X44;
	const	double	G270	=	G268	+	G269;
	const	double	G271	=	X8	*	G270;
	const	double	G272	=	C1	*	G271;
	const	double	G273	=	G197	+	G272;
	const	double	G274	=	G273	*	G196;
	const	double	G275	=	G274	+	G274;
	const	double	G276	=	G1	*	X29;
	const	double	G277	=	X9	*	X49;
	const	double	G278	=	G276	+	G277;
	const	double	G279	=	X8	*	G278;
	const	double	G280	=	C1	*	G279;
	const	double	G281	=	G209	+	G280;
	const	double	G282	=	G281	*	G208;
	const	double	G283	=	G282	+	G282;
	const	double	G284	=	G275	+	G283;
	const	double	G285	=	C1	*	X8;
	const	double	G286	=	X4	+	G285;
	const	double	G287	=	G286	+	G286;
	const	double	G288	=	G284	+	G287;
	const	double	G289	=	C1	*	G288;
	const	double	G290	=	C1	*	G200;
	const	double	G291	=	G203	*	G290;
	const	double	G292	=	G291	+	G291;
	const	double	G293	=	C1	*	G212;
	const	double	G294	=	G215	*	G293;
	const	double	G295	=	G294	+	G294;
	const	double	G296	=	G292	+	G295;
	const	double	G297	=	C1	*	G220;
	const	double	G298	=	G297	+	G297;
	const	double	G299	=	G296	+	G298;
	const	double	G300	=	C1	*	G299;
	const	double	G301	=	G201	+	G228;
	const	double	G302	=	G301	*	G200;
	const	double	G303	=	G302	+	G302;
	const	double	G304	=	G213	+	G236;
	const	double	G305	=	G304	*	G212;
	const	double	G306	=	G305	+	G305;
	const	double	G307	=	G303	+	G306;
	const	double	G308	=	X5	+	G241;
	const	double	G309	=	G308	+	G308;
	const	double	G310	=	G307	+	G309;
	const	double	G311	=	C1	*	G310;
	const	double	G312	=	G201	+	G250;
	const	double	G313	=	G312	*	G200;
	const	double	G314	=	G313	+	G313;
	const	double	G315	=	G213	+	G258;
	const	double	G316	=	G315	*	G212;
	const	double	G317	=	G316	+	G316;
	const	double	G318	=	G314	+	G317;
	const	double	G319	=	X5	+	G263;
	const	double	G320	=	G319	+	G319;
	const	double	G321	=	G318	+	G320;
	const	double	G322	=	C1	*	G321;
	const	double	G323	=	G201	+	G272;
	const	double	G324	=	G323	*	G200;
	const	double	G325	=	G324	+	G324;
	const	double	G326	=	G213	+	G280;
	const	double	G327	=	G326	*	G212;
	const	double	G328	=	G327	+	G327;
	const	double	G329	=	G325	+	G328;
	const	double	G330	=	X5	+	G285;
	const	double	G331	=	G330	+	G330;
	const	double	G332	=	G329	+	G331;
	const	double	G333	=	C1	*	G332;
	const	double	G334	=	C1	*	G226;
	const	double	G335	=	G229	*	G334;
	const	double	G336	=	G335	+	G335;
	const	double	G337	=	C1	*	G234;
	const	double	G338	=	G237	*	G337;
	const	double	G339	=	G338	+	G338;
	const	double	G340	=	G336	+	G339;
	const	double	G341	=	C1	*	G242;
	const	double	G342	=	G341	+	G341;
	const	double	G343	=	G340	+	G342;
	const	double	G344	=	C1	*	G343;
	const	double	G345	=	G301	*	G334;
	const	double	G346	=	G345	+	G345;
	const	double	G347	=	G304	*	G337;
	const	double	G348	=	G347	+	G347;
	const	double	G349	=	G346	+	G348;
	const	double	G350	=	C1	*	G308;
	const	double	G351	=	G350	+	G350;
	const	double	G352	=	G349	+	G351;


	const	double	G353	=	C1	*	G352;
	const	double	G354	=	G227	+	G250;
	const	double	G355	=	G354	*	G226;
	const	double	G356	=	G355	+	G355;
	const	double	G357	=	G235	+	G258;
	const	double	G358	=	G357	*	G234;
	const	double	G359	=	G358	+	G358;
	const	double	G360	=	G356	+	G359;
	const	double	G361	=	X6	+	G263;
	const	double	G362	=	G361	+	G361;
	const	double	G363	=	G360	+	G362;
	const	double	G364	=	C1	*	G363;
	const	double	G365	=	G227	+	G272;
	const	double	G366	=	G365	*	G226;
	const	double	G367	=	G366	+	G366;
	const	double	G368	=	G235	+	G280;
	const	double	G369	=	G368	*	G234;
	const	double	G370	=	G369	+	G369;
	const	double	G371	=	G367	+	G370;
	const	double	G372	=	X6	+	G285;
	const	double	G373	=	G372	+	G372;
	const	double	G374	=	G371	+	G373;
	const	double	G375	=	C1	*	G374;
	const	double	G376	=	C1	*	G248;
	const	double	G377	=	G251	*	G376;
	const	double	G378	=	G377	+	G377;
	const	double	G379	=	C1	*	G256;
	const	double	G380	=	G259	*	G379;
	const	double	G381	=	G380	+	G380;
	const	double	G382	=	G378	+	G381;
	const	double	G383	=	C1	*	G264;
	const	double	G384	=	G383	+	G383;
	const	double	G385	=	G382	+	G384;
	const	double	G386	=	C1	*	G385;
	const	double	G387	=	G312	*	G376;
	const	double	G388	=	G387	+	G387;
	const	double	G389	=	G315	*	G379;
	const	double	G390	=	G389	+	G389;
	const	double	G391	=	G388	+	G390;
	const	double	G392	=	C1	*	G319;
	const	double	G393	=	G392	+	G392;
	const	double	G394	=	G391	+	G393;
	const	double	G395	=	C1	*	G394;
	const	double	G396	=	G354	*	G376;
	const	double	G397	=	G396	+	G396;
	const	double	G398	=	G357	*	G379;
	const	double	G399	=	G398	+	G398;
	const	double	G400	=	G397	+	G399;
	const	double	G401	=	C1	*	G361;
	const	double	G402	=	G401	+	G401;
	const	double	G403	=	G400	+	G402;
	const	double	G404	=	C1	*	G403;
	const	double	G405	=	C1	*	G270;
	const	double	G406	=	G273	*	G405;
	const	double	G407	=	G406	+	G406;
	const	double	G408	=	C1	*	G278;
	const	double	G409	=	G281	*	G408;
	const	double	G410	=	G409	+	G409;
	const	double	G411	=	G407	+	G410;
	const	double	G412	=	C1	*	G286;
	const	double	G413	=	G412	+	G412;
	const	double	G414	=	G411	+	G413;
	const	double	G415	=	C1	*	G414;
	const	double	G416	=	G323	*	G405;
	const	double	G417	=	G416	+	G416;
	const	double	G418	=	G326	*	G408;
	const	double	G419	=	G418	+	G418;
	const	double	G420	=	G417	+	G419;
	const	double	G421	=	C1	*	G330;
	const	double	G422	=	G421	+	G421;
	const	double	G423	=	G420	+	G422;
	const	double	G424	=	C1	*	G423;
	const	double	G425	=	G365	*	G405;
	const	double	G426	=	G425	+	G425;
	const	double	G427	=	G368	*	G408;
	const	double	G428	=	G427	+	G427;
	const	double	G429	=	G426	+	G428;
	const	double	G430	=	C1	*	G372;
	const	double	G431	=	G430	+	G430;
	const	double	G432	=	G429	+	G431;
	const	double	G433	=	C1	*	G432;
	const	double	G434	=	C1	*	X10;
	const	double	G435	=	G434	+	X30;
	const	double	G436	=	C1	*	X11;
	const	double	G437	=	G436	+	X31;
	const	double	G438	=	X0	*	G437;
	const	double	G439	=	C1	*	G438;
	const	double	G440	=	G435	+	G439;
	const	double	G441	=	G10	*	G440;
	const	double	G442	=	G441	+	G441;
	const	double	G443	=	C1	*	X15;
	const	double	G444	=	G443	+	X35;
	const	double	G445	=	C1	*	X16;
	const	double	G446	=	G445	+	X36;
	const	double	G447	=	X0	*	G446;
	const	double	G448	=	C1	*	G447;
	const	double	G449	=	G444	+	G448;
	const	double	G450	=	G22	*	G449;
	const	double	G451	=	G450	+	G450;
	const	double	G452	=	G442	+	G451;
	const	double	G453	=	C1	*	X20;
	const	double	G454	=	G453	+	X40;
	const	double	G455	=	X4	*	G454;
	const	double	G456	=	C1	*	X21;
	const	double	G457	=	G456	+	X41;
	const	double	G458	=	X5	*	G457;
	const	double	G459	=	C1	*	G458;
	const	double	G460	=	G455	+	G459;
	const	double	G461	=	G203	*	G460;
	const	double	G462	=	G461	+	G461;
	const	double	G463	=	C1	*	X25;
	const	double	G464	=	G463	+	X45;
	const	double	G465	=	X4	*	G464;
	const	double	G466	=	C1	*	X26;
	const	double	G467	=	G466	+	X46;
	const	double	G468	=	X5	*	G467;
	const	double	G469	=	C1	*	G468;
	const	double	G470	=	G465	+	G469;
	const	double	G471	=	G215	*	G470;
	const	double	G472	=	G471	+	G471;
	const	double	G473	=	G462	+	G472;
	const	double	G474	=	C1	*	G473;
	const	double	G475	=	G452	+	G474;
	const	double	G476	=	C1	*	X12;
	const	double	G477	=	G476	+	X32;
	const	double	G478	=	X1	*	G477;
	const	double	G479	=	C1	*	G478;
	const	double	G480	=	G435	+	G479;
	const	double	G481	=	G95	*	G480;
	const	double	G482	=	G481	+	G481;
	const	double	G483	=	C1	*	X17;
	const	double	G484	=	G483	+	X37;
	const	double	G485	=	X1	*	G484;
	const	double	G486	=	C1	*	G485;
	const	double	G487	=	G444	+	G486;
	const	double	G488	=	G99	*	G487;
	const	double	G489	=	G488	+	G488;
	const	double	G490	=	G482	+	G489;
	const	double	G491	=	C1	*	X22;
	const	double	G492	=	G491	+	X42;
	const	double	G493	=	X6	*	G492;
	const	double	G494	=	C1	*	G493;
	const	double	G495	=	G455	+	G494;
	const	double	G496	=	G229	*	G495;
	const	double	G497	=	G496	+	G496;
	const	double	G498	=	C1	*	X27;
	const	double	G499	=	G498	+	X47;
	const	double	G500	=	X6	*	G499;
	const	double	G501	=	C1	*	G500;
	const	double	G502	=	G465	+	G501;
	const	double	G503	=	G237	*	G502;
	const	double	G504	=	G503	+	G503;
	const	double	G505	=	G497	+	G504;
	const	double	G506	=	C1	*	G505;
	const	double	G507	=	G490	+	G506;
	const	double	G508	=	G438	+	G479;
	const	double	G509	=	G37	*	G508;
	const	double	G510	=	G509	+	G509;
	const	double	G511	=	G447	+	G486;
	const	double	G512	=	G45	*	G511;
	const	double	G513	=	G512	+	G512;
	const	double	G514	=	G510	+	G513;
	const	double	G515	=	G458	+	G494;
	const	double	G516	=	G301	*	G515;
	const	double	G517	=	G516	+	G516;
	const	double	G518	=	G468	+	G501;
	const	double	G519	=	G304	*	G518;
	const	double	G520	=	G519	+	G519;
	const	double	G521	=	G517	+	G520;
	const	double	G522	=	C1	*	G521;
	const	double	G523	=	G514	+	G522;
	const	double	G524	=	C1	*	X13;
	const	double	G525	=	G524	+	X33;
	const	double	G526	=	X2	*	G525;
	const	double	G527	=	C1	*	G526;
	const	double	G528	=	G435	+	G527;
	const	double	G529	=	G136	*	G528;
	const	double	G530	=	G529	+	G529;
	const	double	G531	=	C1	*	X18;
	const	double	G532	=	G531	+	X38;
	const	double	G533	=	X2	*	G532;
	const	double	G534	=	C1	*	G533;
	const	double	G535	=	G444	+	G534;
	const	double	G536	=	G140	*	G535;
	const	double	G537	=	G536	+	G536;
	const	double	G538	=	G530	+	G537;
	const	double	G539	=	C1	*	X23;
	const	double	G540	=	G539	+	X43;
	const	double	G541	=	X7	*	G540;
	const	double	G542	=	C1	*	G541;
	const	double	G543	=	G455	+	G542;
	const	double	G544	=	G251	*	G543;
	const	double	G545	=	G544	+	G544;
	const	double	G546	=	C1	*	X28;
	const	double	G547	=	G546	+	X48;
	const	double	G548	=	X7	*	G547;
	const	double	G549	=	C1	*	G548;
	const	double	G550	=	G465	+	G549;
	const	double	G551	=	G259	*	G550;
	const	double	G552	=	G551	+	G551;
	const	double	G553	=	G545	+	G552;
	const	double	G554	=	C1	*	G553;
	const	double	G555	=	G538	+	G554;
	const	double	G556	=	G438	+	G527;
	const	double	G557	=	G58	*	G556;
	const	double	G558	=	G557	+	G557;
	const	double	G559	=	G447	+	G534;
	const	double	G560	=	G66	*	G559;
	const	double	G561	=	G560	+	G560;
	const	double	G562	=	G558	+	G561;
	const	double	G563	=	G458	+	G542;
	const	double	G564	=	G312	*	G563;
	const	double	G565	=	G564	+	G564;
	const	double	G566	=	G468	+	G549;
	const	double	G567	=	G315	*	G566;
	const	double	G568	=	G567	+	G567;
	const	double	G569	=	G565	+	G568;
	const	double	G570	=	C1	*	G569;
	const	double	G571	=	G562	+	G570;
	const	double	G572	=	G478	+	G527;
	const	double	G573	=	G116	*	G572;
	const	double	G574	=	G573	+	G573;
	const	double	G575	=	G485	+	G534;
	const	double	G576	=	G119	*	G575;
	const	double	G577	=	G576	+	G576;
	const	double	G578	=	G574	+	G577;
	const	double	G579	=	G493	+	G542;
	const	double	G580	=	G354	*	G579;
	const	double	G581	=	G580	+	G580;
	const	double	G582	=	G500	+	G549;
	const	double	G583	=	G357	*	G582;
	const	double	G584	=	G583	+	G583;
	const	double	G585	=	G581	+	G584;
	const	double	G586	=	C1	*	G585;
	const	double	G587	=	G578	+	G586;
	const	double	G588	=	C1	*	X14;
	const	double	G589	=	G588	+	X34;
	const	double	G590	=	X3	*	G589;
	const	double	G591	=	C1	*	G590;
	const	double	G592	=	G435	+	G591;
	const	double	G593	=	G165	*	G592;
	const	double	G594	=	G593	+	G593;
	const	double	G595	=	C1	*	X19;
	const	double	G596	=	G595	+	X39;
	const	double	G597	=	X3	*	G596;
	const	double	G598	=	C1	*	G597;
	const	double	G599	=	G444	+	G598;
	const	double	G600	=	G169	*	G599;
	const	double	G601	=	G600	+	G600;
	const	double	G602	=	G594	+	G601;
	const	double	G603	=	C1	*	X24;
	const	double	G604	=	G603	+	X44;
	const	double	G605	=	X8	*	G604;
	const	double	G606	=	C1	*	G605;
	const	double	G607	=	G455	+	G606;
	const	double	G608	=	G273	*	G607;
	const	double	G609	=	G608	+	G608;
	const	double	G610	=	C1	*	X29;
	const	double	G611	=	G610	+	X49;
	const	double	G612	=	X8	*	G611;
	const	double	G613	=	C1	*	G612;
	const	double	G614	=	G465	+	G613;
	const	double	G615	=	G281	*	G614;
	const	double	G616	=	G615	+	G615;
	const	double	G617	=	G609	+	G616;
	const	double	G618	=	C1	*	G617;
	const	double	G619	=	G602	+	G618;
	const	double	G620	=	G438	+	G591;
	const	double	G621	=	G79	*	G620;
	const	double	G622	=	G621	+	G621;
	const	double	G623	=	G447	+	G598;
	const	double	G624	=	G87	*	G623;
	const	double	G625	=	G624	+	G624;
	const	double	G626	=	G622	+	G625;
	const	double	G627	=	G458	+	G606;
	const	double	G628	=	G323	*	G627;
	const	double	G629	=	G628	+	G628;
	const	double	G630	=	G468	+	G613;
	const	double	G631	=	G326	*	G630;
	const	double	G632	=	G631	+	G631;
	const	double	G633	=	G629	+	G632;
	const	double	G634	=	C1	*	G633;
	const	double	G635	=	G626	+	G634;
	const	double	G636	=	G478	+	G591;
	const	double	G637	=	G126	*	G636;
	const	double	G638	=	G637	+	G637;
	const	double	G639	=	G485	+	G598;
	const	double	G640	=	G129	*	G639;
	const	double	G641	=	G640	+	G640;
	const	double	G642	=	G638	+	G641;
	const	double	G643	=	G493	+	G606;
	const	double	G644	=	G365	*	G643;
	const	double	G645	=	G644	+	G644;
	const	double	G646	=	G500	+	G613;
	const	double	G647	=	G368	*	G646;
	const	double	G648	=	G647	+	G647;
	const	double	G649	=	G645	+	G648;
	const	double	G650	=	C1	*	G649;
	const	double	G651	=	G642	+	G650;

	//std::cout << "SLP1\n";
	y[0] = G31;
	y[1] = C3;
	y[2] = G52;
	y[3] = C3;
	y[4] = G73;
	y[5] = C3;
	y[6] = C3;
	y[7] = G94;
	y[8] = C3;
	y[9] = C3;
	y[10] = G107;
	y[11] = G115;
	y[12] = C3;
	y[13] = C3;
	y[14] = G125;
	y[15] = C3;
	y[16] = C3;
	y[17] = G135;
	y[18] = C3;
	y[19] = C3;
	y[20] = C3;
	y[21] = G148;
	y[22] = G156;
	y[23] = G164;
	y[24] = C3;
	y[25] = C3;
	y[26] = C3;
	y[27] = C3;
	y[28] = C3;
	y[29] = C3;
	y[30] = C3;
	y[31] = C3;
	y[32] = C3;
	y[33] = G177;
	y[34] = G185;
	y[35] = G193;
	y[36] = G223;
	y[37] = G245;
	y[38] = C3;
	y[39] = G267;
	y[40] = C3;
	y[41] = C3;
	y[42] = G289;
	y[43] = C3;
	y[44] = C3;
	y[45] = G300;
	y[46] = C3;
	y[47] = G311;
	y[48] = C3;
	y[49] = G322;
	y[50] = C3;
	y[51] = C3;
	y[52] = G333;
	y[53] = C3;
	y[54] = C3;
	y[55] = G344;
	y[56] = G353;
	y[57] = C3;
	y[58] = C3;
	y[59] = G364;
	y[60] = C3;
	y[61] = C3;
	y[62] = G375;
	y[63] = C3;
	y[64] = C3;
	y[65] = C3;
	y[66] = G386;
	y[67] = G395;
	y[68] = G404;
	y[69] = C3;
	y[70] = C3;
	y[71] = C3;
	y[72] = C3;
	y[73] = C3;
	y[74] = C3;
	y[75] = C3;
	y[76] = C3;
	y[77] = C3;
	y[78] = G415;
	y[79] = G424;
	y[80] = G433;

	y[81] = -G475;
	y[82] = -G507;
	y[83] = -G523;
	y[84] = -G555;
	y[85] = -G571;
	y[86] = -G587;
	y[87] = -G619;
	y[88] = -G635;
	y[89] = -G651;
}

//Straight line program for evaluation of the Jacobian of the homotopy function, generated in Macaulay2
inline void evaluate_HxH(const double * x, const double * params, double * y)
{
	const	double	&X0	=	x[0];		
	const	double	&X1	=	x[1];		
	const	double	&X2	=	x[2];		
	const	double	&X3	=	x[3];		
	const	double	&X4	=	x[4];		
	const	double	&X5	=	x[5];		
	const	double	&X6	=	x[6];		
	const	double	&X7	=	x[7];		
	const	double	&X8	=	x[8];		
	const	double	&X9	=	x[9];		
	const	double	&X10	=	params[0];		
	const	double	&X11	=	params[1];		
	const	double	&X12	=	params[2];		
	const	double	&X13	=	params[3];		
	const	double	&X14	=	params[4];		
	const	double	&X15	=	params[5];		
	const	double	&X16	=	params[6];		
	const	double	&X17	=	params[7];		
	const	double	&X18	=	params[8];		
	const	double	&X19	=	params[9];		
	const	double	&X20	=	params[10];		
	const	double	&X21	=	params[11];		
	const	double	&X22	=	params[12];		
	const	double	&X23	=	params[13];		
	const	double	&X24	=	params[14];		
	const	double	&X25	=	params[15];		
	const	double	&X26	=	params[16];		
	const	double	&X27	=	params[17];		
	const	double	&X28	=	params[18];		
	const	double	&X29	=	params[19];		
	const	double	&X30	=	params[20];		
	const	double	&X31	=	params[21];		
	const	double	&X32	=	params[22];		
	const	double	&X33	=	params[23];		
	const	double	&X34	=	params[24];		
	const	double	&X35	=	params[25];		
	const	double	&X36	=	params[26];		
	const	double	&X37	=	params[27];		
	const	double	&X38	=	params[28];		
	const	double	&X39	=	params[29];		
	const	double	&X40	=	params[30];		
	const	double	&X41	=	params[31];		
	const	double	&X42	=	params[32];		
	const	double	&X43	=	params[33];		
	const	double	&X44	=	params[34];		
	const	double	&X45	=	params[35];		
	const	double	&X46	=	params[36];		
	const	double	&X47	=	params[37];		
	const	double	&X48	=	params[38];		
	const	double	&X49	=	params[39];		
	static	constexpr	double	C0	=	1;	
	static	constexpr	double	C1	=	-1;	
	const	double	G0	=	C1	*	X9;
	const	double	G1	=	C0	+	G0;
	const	double	G2	=	G1	*	X10;
	const	double	G3	=	X9	*	X30;
	const	double	G4	=	G2	+	G3;
	const	double	G5	=	G1	*	X11;
	const	double	G6	=	X9	*	X31;
	const	double	G7	=	G5	+	G6;
	const	double	G8	=	X0	*	G7;
	const	double	G9	=	C1	*	G8;
	const	double	G10	=	G4	+	G9;
	const	double	G11	=	C1	*	G7;
	const	double	G12	=	G10	*	G11;
	const	double	G13	=	G12	+	G12;
	const	double	G14	=	G1	*	X15;
	const	double	G15	=	X9	*	X35;
	const	double	G16	=	G14	+	G15;
	const	double	G17	=	G1	*	X16;
	const	double	G18	=	X9	*	X36;
	const	double	G19	=	G17	+	G18;
	const	double	G20	=	X0	*	G19;
	const	double	G21	=	C1	*	G20;
	const	double	G22	=	G16	+	G21;
	const	double	G23	=	C1	*	G19;
	const	double	G24	=	G22	*	G23;
	const	double	G25	=	G24	+	G24;
	const	double	G26	=	G13	+	G25;
	const	double	C2	=	1	;	
	const	double	G27	=	C1	*	X0;
	const	double	G28	=	C2	+	G27;
	const	double	G29	=	C1	*	G28;
	const	double	G30	=	G29	+	G29;
	const	double	G31	=	G26	+	G30;
	const	double	C3	=	0	;	
	const	double	G32	=	G1	*	X12;
	const	double	G33	=	X9	*	X32;
	const	double	G34	=	G32	+	G33;
	const	double	G35	=	X1	*	G34;
	const	double	G36	=	C1	*	G35;
	const	double	G37	=	G8	+	G36;
	const	double	G38	=	G37	*	G7;
	const	double	G39	=	G38	+	G38;
	const	double	G40	=	G1	*	X17;
	const	double	G41	=	X9	*	X37;
	const	double	G42	=	G40	+	G41;
	const	double	G43	=	X1	*	G42;
	const	double	G44	=	C1	*	G43;
	const	double	G45	=	G20	+	G44;
	const	double	G46	=	G45	*	G19;
	const	double	G47	=	G46	+	G46;
	const	double	G48	=	G39	+	G47;
	const	double	G49	=	C1	*	X1;
	const	double	G50	=	X0	+	G49;
	const	double	G51	=	G50	+	G50;
	const	double	G52	=	G48	+	G51;
	const	double	G53	=	G1	*	X13;
	const	double	G54	=	X9	*	X33;
	const	double	G55	=	G53	+	G54;
	const	double	G56	=	X2	*	G55;
	const	double	G57	=	C1	*	G56;
	const	double	G58	=	G8	+	G57;
	const	double	G59	=	G58	*	G7;
	const	double	G60	=	G59	+	G59;
	const	double	G61	=	G1	*	X18;
	const	double	G62	=	X9	*	X38;
	const	double	G63	=	G61	+	G62;
	const	double	G64	=	X2	*	G63;
	const	double	G65	=	C1	*	G64;
	const	double	G66	=	G20	+	G65;
	const	double	G67	=	G66	*	G19;
	const	double	G68	=	G67	+	G67;
	const	double	G69	=	G60	+	G68;
	const	double	G70	=	C1	*	X2;
	const	double	G71	=	X0	+	G70;
	const	double	G72	=	G71	+	G71;
	const	double	G73	=	G69	+	G72;
	const	double	G74	=	G1	*	X14;
	const	double	G75	=	X9	*	X34;
	const	double	G76	=	G74	+	G75;
	const	double	G77	=	X3	*	G76;
	const	double	G78	=	C1	*	G77;
	const	double	G79	=	G8	+	G78;
	const	double	G80	=	G79	*	G7;
	const	double	G81	=	G80	+	G80;
	const	double	G82	=	G1	*	X19;
	const	double	G83	=	X9	*	X39;
	const	double	G84	=	G82	+	G83;
	const	double	G85	=	X3	*	G84;
	const	double	G86	=	C1	*	G85;
	const	double	G87	=	G20	+	G86;
	const	double	G88	=	G87	*	G19;
	const	double	G89	=	G88	+	G88;
	const	double	G90	=	G81	+	G89;
	const	double	G91	=	C1	*	X3;
	const	double	G92	=	X0	+	G91;
	const	double	G93	=	G92	+	G92;
	const	double	G94	=	G90	+	G93;
	const	double	G95	=	G4	+	G36;
	const	double	G96	=	C1	*	G34;
	const	double	G97	=	G95	*	G96;
	const	double	G98	=	G97	+	G97;
	const	double	G99	=	G16	+	G44;
	const	double	G100	=	C1	*	G42;
	const	double	G101	=	G99	*	G100;
	const	double	G102	=	G101	+	G101;
	const	double	G103	=	G98	+	G102;
	const	double	G104	=	C2	+	G49;
	const	double	G105	=	C1	*	G104;
	const	double	G106	=	G105	+	G105;
	const	double	G107	=	G103	+	G106;
	const	double	G108	=	G37	*	G96;
	const	double	G109	=	G108	+	G108;
	const	double	G110	=	G45	*	G100;
	const	double	G111	=	G110	+	G110;
	const	double	G112	=	G109	+	G111;
	const	double	G113	=	C1	*	G50;
	const	double	G114	=	G113	+	G113;
	const	double	G115	=	G112	+	G114;
	const	double	G116	=	G35	+	G57;
	const	double	G117	=	G116	*	G34;
	const	double	G118	=	G117	+	G117;
	const	double	G119	=	G43	+	G65;
	const	double	G120	=	G119	*	G42;
	const	double	G121	=	G120	+	G120;
	const	double	G122	=	G118	+	G121;
	const	double	G123	=	X1	+	G70;
	const	double	G124	=	G123	+	G123;
	const	double	G125	=	G122	+	G124;
	const	double	G126	=	G35	+	G78;
	const	double	G127	=	G126	*	G34;
	const	double	G128	=	G127	+	G127;
	const	double	G129	=	G43	+	G86;
	const	double	G130	=	G129	*	G42;
	const	double	G131	=	G130	+	G130;
	const	double	G132	=	G128	+	G131;
	const	double	G133	=	X1	+	G91;
	const	double	G134	=	G133	+	G133;
	const	double	G135	=	G132	+	G134;
	const	double	G136	=	G4	+	G57;
	const	double	G137	=	C1	*	G55;
	const	double	G138	=	G136	*	G137;
	const	double	G139	=	G138	+	G138;
	const	double	G140	=	G16	+	G65;
	const	double	G141	=	C1	*	G63;
	const	double	G142	=	G140	*	G141;
	const	double	G143	=	G142	+	G142;
	const	double	G144	=	G139	+	G143;
	const	double	G145	=	C2	+	G70;
	const	double	G146	=	C1	*	G145;
	const	double	G147	=	G146	+	G146;
	const	double	G148	=	G144	+	G147;
	const	double	G149	=	G58	*	G137;
	const	double	G150	=	G149	+	G149;
	const	double	G151	=	G66	*	G141;
	const	double	G152	=	G151	+	G151;
	const	double	G153	=	G150	+	G152;
	const	double	G154	=	C1	*	G71;
	const	double	G155	=	G154	+	G154;
	const	double	G156	=	G153	+	G155;
	const	double	G157	=	G116	*	G137;
	const	double	G158	=	G157	+	G157;
	const	double	G159	=	G119	*	G141;
	const	double	G160	=	G159	+	G159;
	const	double	G161	=	G158	+	G160;
	const	double	G162	=	C1	*	G123;
	const	double	G163	=	G162	+	G162;
	const	double	G164	=	G161	+	G163;
	const	double	G165	=	G4	+	G78;
	const	double	G166	=	C1	*	G76;
	const	double	G167	=	G165	*	G166;
	const	double	G168	=	G167	+	G167;
	const	double	G169	=	G16	+	G86;
	const	double	G170	=	C1	*	G84;
	const	double	G171	=	G169	*	G170;
	const	double	G172	=	G171	+	G171;
	const	double	G173	=	G168	+	G172;
	const	double	G174	=	C2	+	G91;
	const	double	G175	=	C1	*	G174;
	const	double	G176	=	G175	+	G175;
	const	double	G177	=	G173	+	G176;
	const	double	G178	=	G79	*	G166;
	const	double	G179	=	G178	+	G178;
	const	double	G180	=	G87	*	G170;
	const	double	G181	=	G180	+	G180;
	const	double	G182	=	G179	+	G181;
	const	double	G183	=	C1	*	G92;

	const	double	G184	=	G183	+	G183;
	const	double	G185	=	G182	+	G184;
	const	double	G186	=	G126	*	G166;
	const	double	G187	=	G186	+	G186;
	const	double	G188	=	G129	*	G170;
	const	double	G189	=	G188	+	G188;
	const	double	G190	=	G187	+	G189;
	const	double	G191	=	C1	*	G133;
	const	double	G192	=	G191	+	G191;
	const	double	G193	=	G190	+	G192;
	const	double	G194	=	G1	*	X20;
	const	double	G195	=	X9	*	X40;
	const	double	G196	=	G194	+	G195;
	const	double	G197	=	X4	*	G196;
	const	double	G198	=	G1	*	X21;
	const	double	G199	=	X9	*	X41;
	const	double	G200	=	G198	+	G199;
	const	double	G201	=	X5	*	G200;
	const	double	G202	=	C1	*	G201;
	const	double	G203	=	G197	+	G202;
	const	double	G204	=	G203	*	G196;
	const	double	G205	=	G204	+	G204;
	const	double	G206	=	G1	*	X25;
	const	double	G207	=	X9	*	X45;
	const	double	G208	=	G206	+	G207;
	const	double	G209	=	X4	*	G208;
	const	double	G210	=	G1	*	X26;
	const	double	G211	=	X9	*	X46;
	const	double	G212	=	G210	+	G211;
	const	double	G213	=	X5	*	G212;
	const	double	G214	=	C1	*	G213;
	const	double	G215	=	G209	+	G214;
	const	double	G216	=	G215	*	G208;
	const	double	G217	=	G216	+	G216;
	const	double	G218	=	G205	+	G217;
	const	double	G219	=	C1	*	X5;
	const	double	G220	=	X4	+	G219;
	const	double	G221	=	G220	+	G220;
	const	double	G222	=	G218	+	G221;
	const	double	G223	=	C1	*	G222;
	const	double	G224	=	G1	*	X22;
	const	double	G225	=	X9	*	X42;
	const	double	G226	=	G224	+	G225;
	const	double	G227	=	X6	*	G226;
	const	double	G228	=	C1	*	G227;
	const	double	G229	=	G197	+	G228;
	const	double	G230	=	G229	*	G196;
	const	double	G231	=	G230	+	G230;
	const	double	G232	=	G1	*	X27;
	const	double	G233	=	X9	*	X47;
	const	double	G234	=	G232	+	G233;
	const	double	G235	=	X6	*	G234;
	const	double	G236	=	C1	*	G235;
	const	double	G237	=	G209	+	G236;
	const	double	G238	=	G237	*	G208;
	const	double	G239	=	G238	+	G238;
	const	double	G240	=	G231	+	G239;
	const	double	G241	=	C1	*	X6;
	const	double	G242	=	X4	+	G241;
	const	double	G243	=	G242	+	G242;
	const	double	G244	=	G240	+	G243;
	const	double	G245	=	C1	*	G244;
	const	double	G246	=	G1	*	X23;
	const	double	G247	=	X9	*	X43;
	const	double	G248	=	G246	+	G247;
	const	double	G249	=	X7	*	G248;
	const	double	G250	=	C1	*	G249;
	const	double	G251	=	G197	+	G250;
	const	double	G252	=	G251	*	G196;
	const	double	G253	=	G252	+	G252;
	const	double	G254	=	G1	*	X28;
	const	double	G255	=	X9	*	X48;
	const	double	G256	=	G254	+	G255;
	const	double	G257	=	X7	*	G256;
	const	double	G258	=	C1	*	G257;
	const	double	G259	=	G209	+	G258;
	const	double	G260	=	G259	*	G208;
	const	double	G261	=	G260	+	G260;
	const	double	G262	=	G253	+	G261;
	const	double	G263	=	C1	*	X7;
	const	double	G264	=	X4	+	G263;
	const	double	G265	=	G264	+	G264;
	const	double	G266	=	G262	+	G265;
	const	double	G267	=	C1	*	G266;
	const	double	G268	=	G1	*	X24;
	const	double	G269	=	X9	*	X44;
	const	double	G270	=	G268	+	G269;
	const	double	G271	=	X8	*	G270;
	const	double	G272	=	C1	*	G271;
	const	double	G273	=	G197	+	G272;
	const	double	G274	=	G273	*	G196;
	const	double	G275	=	G274	+	G274;
	const	double	G276	=	G1	*	X29;
	const	double	G277	=	X9	*	X49;
	const	double	G278	=	G276	+	G277;
	const	double	G279	=	X8	*	G278;
	const	double	G280	=	C1	*	G279;
	const	double	G281	=	G209	+	G280;
	const	double	G282	=	G281	*	G208;
	const	double	G283	=	G282	+	G282;
	const	double	G284	=	G275	+	G283;
	const	double	G285	=	C1	*	X8;
	const	double	G286	=	X4	+	G285;
	const	double	G287	=	G286	+	G286;
	const	double	G288	=	G284	+	G287;
	const	double	G289	=	C1	*	G288;
	const	double	G290	=	C1	*	G200;
	const	double	G291	=	G203	*	G290;
	const	double	G292	=	G291	+	G291;
	const	double	G293	=	C1	*	G212;
	const	double	G294	=	G215	*	G293;
	const	double	G295	=	G294	+	G294;
	const	double	G296	=	G292	+	G295;
	const	double	G297	=	C1	*	G220;
	const	double	G298	=	G297	+	G297;
	const	double	G299	=	G296	+	G298;
	const	double	G300	=	C1	*	G299;
	const	double	G301	=	G201	+	G228;
	const	double	G302	=	G301	*	G200;
	const	double	G303	=	G302	+	G302;
	const	double	G304	=	G213	+	G236;
	const	double	G305	=	G304	*	G212;
	const	double	G306	=	G305	+	G305;
	const	double	G307	=	G303	+	G306;
	const	double	G308	=	X5	+	G241;
	const	double	G309	=	G308	+	G308;
	const	double	G310	=	G307	+	G309;
	const	double	G311	=	C1	*	G310;
	const	double	G312	=	G201	+	G250;
	const	double	G313	=	G312	*	G200;
	const	double	G314	=	G313	+	G313;
	const	double	G315	=	G213	+	G258;
	const	double	G316	=	G315	*	G212;
	const	double	G317	=	G316	+	G316;
	const	double	G318	=	G314	+	G317;
	const	double	G319	=	X5	+	G263;
	const	double	G320	=	G319	+	G319;
	const	double	G321	=	G318	+	G320;
	const	double	G322	=	C1	*	G321;
	const	double	G323	=	G201	+	G272;
	const	double	G324	=	G323	*	G200;
	const	double	G325	=	G324	+	G324;
	const	double	G326	=	G213	+	G280;
	const	double	G327	=	G326	*	G212;
	const	double	G328	=	G327	+	G327;
	const	double	G329	=	G325	+	G328;
	const	double	G330	=	X5	+	G285;
	const	double	G331	=	G330	+	G330;
	const	double	G332	=	G329	+	G331;
	const	double	G333	=	C1	*	G332;
	const	double	G334	=	C1	*	G226;
	const	double	G335	=	G229	*	G334;
	const	double	G336	=	G335	+	G335;
	const	double	G337	=	C1	*	G234;
	const	double	G338	=	G237	*	G337;
	const	double	G339	=	G338	+	G338;
	const	double	G340	=	G336	+	G339;
	const	double	G341	=	C1	*	G242;
	const	double	G342	=	G341	+	G341;
	const	double	G343	=	G340	+	G342;
	const	double	G344	=	C1	*	G343;
	const	double	G345	=	G301	*	G334;
	const	double	G346	=	G345	+	G345;
	const	double	G347	=	G304	*	G337;
	const	double	G348	=	G347	+	G347;
	const	double	G349	=	G346	+	G348;
	const	double	G350	=	C1	*	G308;
	const	double	G351	=	G350	+	G350;
	const	double	G352	=	G349	+	G351;
	const	double	G353	=	C1	*	G352;
	const	double	G354	=	G227	+	G250;
	const	double	G355	=	G354	*	G226;
	const	double	G356	=	G355	+	G355;
	const	double	G357	=	G235	+	G258;
	const	double	G358	=	G357	*	G234;
	const	double	G359	=	G358	+	G358;
	const	double	G360	=	G356	+	G359;
	const	double	G361	=	X6	+	G263;
	const	double	G362	=	G361	+	G361;
	const	double	G363	=	G360	+	G362;
	const	double	G364	=	C1	*	G363;
	const	double	G365	=	G227	+	G272;
	const	double	G366	=	G365	*	G226;
	const	double	G367	=	G366	+	G366;
	const	double	G368	=	G235	+	G280;
	const	double	G369	=	G368	*	G234;
	const	double	G370	=	G369	+	G369;
	const	double	G371	=	G367	+	G370;
	const	double	G372	=	X6	+	G285;
	const	double	G373	=	G372	+	G372;
	const	double	G374	=	G371	+	G373;
	const	double	G375	=	C1	*	G374;
	const	double	G376	=	C1	*	G248;
	const	double	G377	=	G251	*	G376;
	const	double	G378	=	G377	+	G377;
	const	double	G379	=	C1	*	G256;
	const	double	G380	=	G259	*	G379;
	const	double	G381	=	G380	+	G380;
	const	double	G382	=	G378	+	G381;
	const	double	G383	=	C1	*	G264;
	const	double	G384	=	G383	+	G383;
	const	double	G385	=	G382	+	G384;
	const	double	G386	=	C1	*	G385;
	const	double	G387	=	G312	*	G376;
	const	double	G388	=	G387	+	G387;
	const	double	G389	=	G315	*	G379;
	const	double	G390	=	G389	+	G389;
	const	double	G391	=	G388	+	G390;
	const	double	G392	=	C1	*	G319;
	const	double	G393	=	G392	+	G392;
	const	double	G394	=	G391	+	G393;
	const	double	G395	=	C1	*	G394;
	const	double	G396	=	G354	*	G376;
	const	double	G397	=	G396	+	G396;
	const	double	G398	=	G357	*	G379;
	const	double	G399	=	G398	+	G398;
	const	double	G400	=	G397	+	G399;
	const	double	G401	=	C1	*	G361;
	const	double	G402	=	G401	+	G401;
	const	double	G403	=	G400	+	G402;
	const	double	G404	=	C1	*	G403;
	const	double	G405	=	C1	*	G270;
	const	double	G406	=	G273	*	G405;
	const	double	G407	=	G406	+	G406;
	const	double	G408	=	C1	*	G278;
	const	double	G409	=	G281	*	G408;
	const	double	G410	=	G409	+	G409;
	const	double	G411	=	G407	+	G410;
	const	double	G412	=	C1	*	G286;
	const	double	G413	=	G412	+	G412;
	const	double	G414	=	G411	+	G413;
	const	double	G415	=	C1	*	G414;
	const	double	G416	=	G323	*	G405;
	const	double	G417	=	G416	+	G416;
	const	double	G418	=	G326	*	G408;
	const	double	G419	=	G418	+	G418;
	const	double	G420	=	G417	+	G419;
	const	double	G421	=	C1	*	G330;
	const	double	G422	=	G421	+	G421;
	const	double	G423	=	G420	+	G422;
	const	double	G424	=	C1	*	G423;
	const	double	G425	=	G365	*	G405;
	const	double	G426	=	G425	+	G425;
	const	double	G427	=	G368	*	G408;
	const	double	G428	=	G427	+	G427;
	const	double	G429	=	G426	+	G428;
	const	double	G430	=	C1	*	G372;
	const	double	G431	=	G430	+	G430;
	const	double	G432	=	G429	+	G431;
	const	double	G433	=	C1	*	G432;
	const	double	G434	=	G10	*	G10;
	const	double	G435	=	G22	*	G22;
	const	double	G436	=	G434	+	G435;
	const	double	G437	=	G28	*	G28;
	const	double	G438	=	G436	+	G437;
	const	double	G439	=	G203	*	G203;
	const	double	G440	=	G215	*	G215;
	const	double	G441	=	G439	+	G440;
	const	double	G442	=	G220	*	G220;
	const	double	G443	=	G441	+	G442;
	const	double	G444	=	C1	*	G443;
	const	double	G445	=	G438	+	G444;
	const	double	G446	=	G95	*	G95;
	const	double	G447	=	G99	*	G99;
	const	double	G448	=	G446	+	G447;
	const	double	G449	=	G104	*	G104;
	const	double	G450	=	G448	+	G449;
	const	double	G451	=	G229	*	G229;
	const	double	G452	=	G237	*	G237;
	const	double	G453	=	G451	+	G452;
	const	double	G454	=	G242	*	G242;
	const	double	G455	=	G453	+	G454;
	const	double	G456	=	C1	*	G455;
	const	double	G457	=	G450	+	G456;
	const	double	G458	=	G37	*	G37;
	const	double	G459	=	G45	*	G45;
	const	double	G460	=	G458	+	G459;
	const	double	G461	=	G50	*	G50;
	const	double	G462	=	G460	+	G461;
	const	double	G463	=	G301	*	G301;
	const	double	G464	=	G304	*	G304;
	const	double	G465	=	G463	+	G464;
	const	double	G466	=	G308	*	G308;
	const	double	G467	=	G465	+	G466;
	const	double	G468	=	C1	*	G467;
	const	double	G469	=	G462	+	G468;
	const	double	G470	=	G136	*	G136;
	const	double	G471	=	G140	*	G140;
	const	double	G472	=	G470	+	G471;
	const	double	G473	=	G145	*	G145;
	const	double	G474	=	G472	+	G473;
	const	double	G475	=	G251	*	G251;
	const	double	G476	=	G259	*	G259;
	const	double	G477	=	G475	+	G476;
	const	double	G478	=	G264	*	G264;
	const	double	G479	=	G477	+	G478;
	const	double	G480	=	C1	*	G479;
	const	double	G481	=	G474	+	G480;
	const	double	G482	=	G58	*	G58;
	const	double	G483	=	G66	*	G66;
	const	double	G484	=	G482	+	G483;
	const	double	G485	=	G71	*	G71;
	const	double	G486	=	G484	+	G485;
	const	double	G487	=	G312	*	G312;
	const	double	G488	=	G315	*	G315;
	const	double	G489	=	G487	+	G488;
	const	double	G490	=	G319	*	G319;
	const	double	G491	=	G489	+	G490;
	const	double	G492	=	C1	*	G491;
	const	double	G493	=	G486	+	G492;
	const	double	G494	=	G116	*	G116;
	const	double	G495	=	G119	*	G119;
	const	double	G496	=	G494	+	G495;
	const	double	G497	=	G123	*	G123;
	const	double	G498	=	G496	+	G497;
	const	double	G499	=	G354	*	G354;
	const	double	G500	=	G357	*	G357;
	const	double	G501	=	G499	+	G500;
	const	double	G502	=	G361	*	G361;
	const	double	G503	=	G501	+	G502;
	const	double	G504	=	C1	*	G503;
	const	double	G505	=	G498	+	G504;
	const	double	G506	=	G165	*	G165;
	const	double	G507	=	G169	*	G169;
	const	double	G508	=	G506	+	G507;
	const	double	G509	=	G174	*	G174;
	const	double	G510	=	G508	+	G509;
	const	double	G511	=	G273	*	G273;
	const	double	G512	=	G281	*	G281;
	const	double	G513	=	G511	+	G512;
	const	double	G514	=	G286	*	G286;
	const	double	G515	=	G513	+	G514;
	const	double	G516	=	C1	*	G515;
	const	double	G517	=	G510	+	G516;
	const	double	G518	=	G79	*	G79;
	const	double	G519	=	G87	*	G87;
	const	double	G520	=	G518	+	G519;
	const	double	G521	=	G92	*	G92;
	const	double	G522	=	G520	+	G521;
	const	double	G523	=	G323	*	G323;
	const	double	G524	=	G326	*	G326;
	const	double	G525	=	G523	+	G524;
	const	double	G526	=	G330	*	G330;
	const	double	G527	=	G525	+	G526;
	const	double	G528	=	C1	*	G527;
	const	double	G529	=	G522	+	G528;
	const	double	G530	=	G126	*	G126;
	const	double	G531	=	G129	*	G129;
	const	double	G532	=	G530	+	G531;
	const	double	G533	=	G133	*	G133;
	const	double	G534	=	G532	+	G533;
	const	double	G535	=	G365	*	G365;
	const	double	G536	=	G368	*	G368;
	const	double	G537	=	G535	+	G536;
	const	double	G538	=	G372	*	G372;
	const	double	G539	=	G537	+	G538;
	const	double	G540	=	C1	*	G539;
	const	double	G541	=	G534	+	G540;
	y[0]	=	G31;				
	y[1]	=	C3;				
	y[2]	=	G52;				
	y[3]	=	C3;				
	y[4]	=	G73;				
	y[5]	=	C3;				
	y[6]	=	C3;				
	y[7]	=	G94;				
	y[8]	=	C3;				
	y[9]	=	C3;				
	y[10]	=	G107;				
	y[11]	=	G115;				
	y[12]	=	C3;				
	y[13]	=	C3;				
	y[14]	=	G125;				
	y[15]	=	C3;				
	y[16]	=	C3;				
	y[17]	=	G135;				
	y[18]	=	C3;				
	y[19]	=	C3;				
	y[20]	=	C3;				
	y[21]	=	G148;				
	y[22]	=	G156;				
	y[23]	=	G164;				
	y[24]	=	C3;				
	y[25]	=	C3;				
	y[26]	=	C3;				
	y[27]	=	C3;				
	y[28]	=	C3;				
	y[29]	=	C3;				
	y[30]	=	C3;				
	y[31]	=	C3;				
	y[32]	=	C3;				
	y[33]	=	G177;				
	y[34]	=	G185;				
	y[35]	=	G193;				
	y[36]	=	G223;				
	y[37]	=	G245;				
	y[38]	=	C3;				
	y[39]	=	G267;				
	y[40]	=	C3;				
	y[41]	=	C3;				
	y[42]	=	G289;				
	y[43]	=	C3;				
	y[44]	=	C3;				
	y[45]	=	G300;				
	y[46]	=	C3;				
	y[47]	=	G311;				
	y[48]	=	C3;				
	y[49]	=	G322;				
	y[50]	=	C3;				
	y[51]	=	C3;				
	y[52]	=	G333;				
	y[53]	=	C3;				
	y[54]	=	C3;				
	y[55]	=	G344;				
	y[56]	=	G353;				
	y[57]	=	C3;				
	y[58]	=	C3;				
	y[59]	=	G364;				
	y[60]	=	C3;				
	y[61]	=	C3;				
	y[62]	=	G375;				
	y[63]	=	C3;				
	y[64]	=	C3;				
	y[65]	=	C3;				
	y[66]	=	G386;				
	y[67]	=	G395;				
	y[68]	=	G404;				
	y[69]	=	C3;				
	y[70]	=	C3;				
	y[71]	=	C3;				
	y[72]	=	C3;				
	y[73]	=	C3;				
	y[74]	=	C3;				
	y[75]	=	C3;				
	y[76]	=	C3;				
	y[77]	=	C3;				
	y[78]	=	G415;				
	y[79]	=	G424;				
	y[80]	=	G433;				
					
	y[81]	=	-G445;				
	y[82]	=	-G457;				
	y[83]	=	-G469;				
	y[84]	=	-G481;				
	y[85]	=	-G493;				
	y[86]	=	-G505;				
	y[87]	=	-G517;				
	y[88]	=	-G529;				
	y[89]	=	-G541;				

}

//THE FUNCTION RESPONSIBLE FOR HOMOTOPY CONTINUATION TRACKING
int track(const struct track_settings s, const double s_sols[9], const double params[40], double solution[9], int * num_st)
{
	const unsigned int nve = 9;
	const unsigned int NVEPLUS1 = nve+1;
	const unsigned int NVE2 = nve*nve;

	//initialize variables
	double Hxt[NVEPLUS1 * nve] __attribute__((aligned(16))); 
	double x0t0xtblock[2*NVEPLUS1] __attribute__((aligned(16)));
	double dxdt[NVEPLUS1] __attribute__((aligned(16)));
	double dxB[nve] __attribute__((aligned(16)));
	double *x0t0 = x0t0xtblock;  // t = real running in [0,1]
	double *x0 = x0t0;
	double *t0 = (double *)(x0t0 + nve);
	double *xt = x0t0xtblock + NVEPLUS1; 
	double *x1t1 = xt;      // reusing xt's space to represent x1t1
	double *const HxH=Hxt;  // HxH is reusing Hxt
	double *const dx = dxdt;
	const double *const RHS = Hxt + NVE2;  // Hx or Ht, same storage //// UNUSED:  C<F> *const LHS = Hxt;
	double *const dxA = dx;   // reuse dx for dx4
	double *const dt = (double *)(dxdt + nve);
	const double &t_step = s.init_dt_;  // initial step
	Eigen::Map<const Eigen::Matrix<double, nve, nve>,Eigen::Aligned> AA((double *)Hxt,nve,nve);  // accessors for the data
	static constexpr double the_smallest_number = 1e-13;
	typedef minus_array<nve,double> v;
	typedef minus_array<NVEPLUS1,double> vp;
	
	//We have 3 ARRAYS: x0t0xtblock, dxi, dxdt
	
	//x0t0xtblock: divided in 2 parts
		//part1: x0
		//part2: xt = x1t1
		// dt: 1 element between the parts, contains the current value of parameter t
		
	//dxB
	
	//dxdt = dx = dx4
		//dt 1 element at the end, contains the parameter t
		
	//x0 stores the current solution
	//xt stores the predicted solution
	//dxA, dxB store the intermediate results for the Runge-Kutta predictor
	//dx stores the resulting step of the predictor/corrector

	//initialize the track
	const double* __restrict__ s_s = s_sols;
	int status = 1;
	bool end_zone = false;
	v::copy(s_s, x0);
	*t0 = 0; *dt = t_step;
	unsigned predictor_successes = 0;
	unsigned num_steps = 0;

	// track H(x,t) for t in [0,1]
	while(status==1 && 1 - *t0 > the_smallest_number)
	{
		++num_steps;
		
		//shorten the step if in end zone
		if(!end_zone && 1 - *t0 <= s.end_zone_factor_ + the_smallest_number)
			end_zone = true; // TODO: see if this path coincides with any other path on entry to the end zone
		if(end_zone)
		{
			if (*dt > 1 - *t0)
				*dt = 1 - *t0;
		}
		else if (*dt > 1 - s.end_zone_factor_ - *t0)
			*dt = 1 - s.end_zone_factor_ - *t0;

		/// PREDICTOR /// in: x0t0,dt out: dx
		/*  top-level code for Runge-Kutta-4
		  dx1 := solveHxTimesDXequalsminusHt(x0,t0);
		  dx2 := solveHxTimesDXequalsminusHt(x0+(1/2)*dx1*dt,t0+(1/2)*dt);
		  dx3 := solveHxTimesDXequalsminusHt(x0+(1/2)*dx2*dt,t0+(1/2)*dt);
		  dx4 := solveHxTimesDXequalsminusHt(x0+dx3*dt,t0+dt);
		  (1/6)*dt*(dx1+2*dx2+2*dx3+dx4) */
		vp::copy(x0t0, xt);

		// dx1
		evaluate_Hxt(xt, params, Hxt); // Outputs Hxt
		{
			//solve solveHxTimesDXequalsminusHt(x0,t0);
			const double coeff01 = AA(2,5)/AA(0,5);
			const double c00 = AA(2,0)-coeff01*AA(0,0);
			const double coeff02 = AA(2,1)/AA(1,1);
			const double c04 = coeff02*AA(1,4)+coeff01*AA(0,4);
			const double c06 = AA(2,6)-coeff02*AA(1,6);
			const double d0 = RHS[2]-coeff02*RHS[1]-coeff01*RHS[0];

			const double coeff11 = AA(4,5)/AA(0,5);
			const double c10 = AA(4,0)-coeff11*AA(0,0);
			const double coeff12 = AA(4,2)/AA(3,2);
			const double c14 = coeff12*AA(3,4)+coeff11*AA(0,4);
			const double c17 = AA(4,7)-coeff12*AA(3,7);
			const double d1 = RHS[4]-coeff12*RHS[3]-coeff11*RHS[0];

			const double coeff21 = -AA(5,1)/AA(1,1);
			const double coeff22 = -AA(5,2)/AA(3,2);
			const double c24 = coeff21*AA(1,4)+coeff22*AA(3,4);
			const double c26 = AA(5,6)+coeff21*AA(1,6);
			const double c27 = AA(5,7)+coeff22*AA(3,7);
			const double d2 = RHS[5]+coeff21*RHS[1]+coeff22*RHS[3];

			const double coeff31 = AA(7,5)/AA(0,5);
			const double c30 = AA(7,0)-coeff31*AA(0,0);
			const double coeff32 = AA(7,3)/AA(6,3);
			const double c34 = coeff32*AA(6,4)+coeff31*AA(0,4);
			const double c38 = AA(7,8)-coeff32*AA(6,8);
			const double d3 = RHS[7]-coeff32*RHS[6]-coeff31*RHS[0];

			const double coeff41 = -AA(8,1)/AA(1,1);
			const double coeff42 = -AA(8,3)/AA(6,3);
			const double c44 = coeff41*AA(1,4)+coeff42*AA(6,4);
			const double c46 = AA(8,6)+coeff41*AA(1,6);
			const double c48 = AA(8,8)+coeff42*AA(6,8);
			const double d4 = RHS[8]+coeff41*RHS[1]+coeff42*RHS[6];

			const double co01 = c26/c06;
			const double co02 = c27/c17;
			const double e00 = co01*c00+co02*c10;
			const double e04 = c24+co01*c04+co02*c14;
			const double f0 = d2-co01*d0-co02*d1;

			const double co11 = c46/c06;
			const double co12 = c48/c38;
			const double e10 = co11*c00+co12*c30;
			const double e14 = c44+co11*c04+co12*c34;
			const double f1 = d4-co11*d0-co12*d3;

			const double co2 = e14/e04;
			dxA[0] = (f1-co2*f0)/(co2*e00-e10);
			dxA[4] = (f0 + e00*dxA[0])/e04;
			dxA[6] = (d0-c00*dxA[0]+c04*dxA[4])/c06;
			dxA[7] = (d1-c10*dxA[0]+c14*dxA[4])/c17;
			dxA[8] = (d3-c30*dxA[0]+c34*dxA[4])/c38;
			dxA[1] = (RHS[1]-AA(1,4)*dxA[4]-AA(1,6)*dxA[6])/AA(1,1);
			dxA[2] = (RHS[3]-AA(3,4)*dxA[4]-AA(3,7)*dxA[7])/AA(3,2);
			dxA[3] = (RHS[6]-AA(6,4)*dxA[4]-AA(6,8)*dxA[8])/AA(6,3);
			dxA[5] = (RHS[0]-AA(0,0)*dxA[0]-AA(0,4)*dxA[4])/AA(0,5);
			//dx1 is in dxA
		}

		//dx2
		const double one_half_dt = *dt*0.5;
		v::multiply_scalar_to_self(dxA, one_half_dt);		//0.5*dt*dx1 in dxA
		v::add_to_self(xt, dxA);							//x0 + 0.5*dx*dx1 in xt
		v::multiply_scalar_to_self(dxA, 2.);				//dt*dx1 in dxA
		xt[nve] += one_half_dt;								// t0+.5dt (affects only time parameter, not solution)
		evaluate_Hxt(xt, params, Hxt);
		{
			//solveHxTimesDXequalsminusHt(x0+(1/2)*dx1*dt,t0+(1/2)*dt);
			const double coeff01 = AA(2,5)/AA(0,5);
			const double c00 = AA(2,0)-coeff01*AA(0,0);
			const double coeff02 = AA(2,1)/AA(1,1);
			const double c04 = coeff02*AA(1,4)+coeff01*AA(0,4);
			const double c06 = AA(2,6)-coeff02*AA(1,6);
			const double d0 = RHS[2]-coeff02*RHS[1]-coeff01*RHS[0];

			const double coeff11 = AA(4,5)/AA(0,5);
			const double c10 = AA(4,0)-coeff11*AA(0,0);
			const double coeff12 = AA(4,2)/AA(3,2);
			const double c14 = coeff12*AA(3,4)+coeff11*AA(0,4);
			const double c17 = AA(4,7)-coeff12*AA(3,7);
			const double d1 = RHS[4]-coeff12*RHS[3]-coeff11*RHS[0];

			const double coeff21 = -AA(5,1)/AA(1,1);
			const double coeff22 = -AA(5,2)/AA(3,2);
			const double c24 = coeff21*AA(1,4)+coeff22*AA(3,4);
			const double c26 = AA(5,6)+coeff21*AA(1,6);
			const double c27 = AA(5,7)+coeff22*AA(3,7);
			const double d2 = RHS[5]+coeff21*RHS[1]+coeff22*RHS[3];

			const double coeff31 = AA(7,5)/AA(0,5);
			const double c30 = AA(7,0)-coeff31*AA(0,0);
			const double coeff32 = AA(7,3)/AA(6,3);
			const double c34 = coeff32*AA(6,4)+coeff31*AA(0,4);
			const double c38 = AA(7,8)-coeff32*AA(6,8);
			const double d3 = RHS[7]-coeff32*RHS[6]-coeff31*RHS[0];

			const double coeff41 = -AA(8,1)/AA(1,1);
			const double coeff42 = -AA(8,3)/AA(6,3);
			const double c44 = coeff41*AA(1,4)+coeff42*AA(6,4);
			const double c46 = AA(8,6)+coeff41*AA(1,6);
			const double c48 = AA(8,8)+coeff42*AA(6,8);
			const double d4 = RHS[8]+coeff41*RHS[1]+coeff42*RHS[6];

			const double co01 = c26/c06;
			const double co02 = c27/c17;
			const double e00 = co01*c00+co02*c10;
			const double e04 = c24+co01*c04+co02*c14;
			const double f0 = d2-co01*d0-co02*d1;

			const double co11 = c46/c06;
			const double co12 = c48/c38;
			const double e10 = co11*c00+co12*c30;
			const double e14 = c44+co11*c04+co12*c34;
			const double f1 = d4-co11*d0-co12*d3;

			const double co2 = e14/e04;
			dxB[0] = (f1-co2*f0)/(co2*e00-e10);
			dxB[4] = (f0 + e00*dxB[0])/e04;
			dxB[6] = (d0-c00*dxB[0]+c04*dxB[4])/c06;
			dxB[7] = (d1-c10*dxB[0]+c14*dxB[4])/c17;
			dxB[8] = (d3-c30*dxB[0]+c34*dxB[4])/c38;
			dxB[1] = (RHS[1]-AA(1,4)*dxB[4]-AA(1,6)*dxB[6])/AA(1,1);
			dxB[2] = (RHS[3]-AA(3,4)*dxB[4]-AA(3,7)*dxB[7])/AA(3,2);
			dxB[3] = (RHS[6]-AA(6,4)*dxB[4]-AA(6,8)*dxB[8])/AA(6,3);
			dxB[5] = (RHS[0]-AA(0,0)*dxB[0]-AA(0,4)*dxB[4])/AA(0,5);
			//dx2 in dxB, dt*dx1 in dxA
		}

		// dx3
		v::multiply_scalar_to_self(dxB, one_half_dt);		//0.5*dt*dx2 in dxB
		v::copy(x0t0, xt);									//x0 in xt
		v::add_to_self(xt, dxB);							//x0 + 0.5*dt*dx2 in xt
		v::multiply_scalar_to_self(dxB, 4);					//2*dt*dx2 in dxB
		v::add_to_self(dxA, dxB);							//dt*dx1 + 2*dt*dx2 in dxA
		evaluate_Hxt(xt, params, Hxt);
		{
			//solveHxTimesDXequalsminusHt(x0+(1/2)*dx2*dt,t0+(1/2)*dt);
		  	const double coeff01 = AA(2,5)/AA(0,5);
			const double c00 = AA(2,0)-coeff01*AA(0,0);
			const double coeff02 = AA(2,1)/AA(1,1);
			const double c04 = coeff02*AA(1,4)+coeff01*AA(0,4);
			const double c06 = AA(2,6)-coeff02*AA(1,6);
			const double d0 = RHS[2]-coeff02*RHS[1]-coeff01*RHS[0];

			const double coeff11 = AA(4,5)/AA(0,5);
			const double c10 = AA(4,0)-coeff11*AA(0,0);
			const double coeff12 = AA(4,2)/AA(3,2);
			const double c14 = coeff12*AA(3,4)+coeff11*AA(0,4);
			const double c17 = AA(4,7)-coeff12*AA(3,7);
			const double d1 = RHS[4]-coeff12*RHS[3]-coeff11*RHS[0];

			const double coeff21 = -AA(5,1)/AA(1,1);
			const double coeff22 = -AA(5,2)/AA(3,2);
			const double c24 = coeff21*AA(1,4)+coeff22*AA(3,4);
			const double c26 = AA(5,6)+coeff21*AA(1,6);
			const double c27 = AA(5,7)+coeff22*AA(3,7);
			const double d2 = RHS[5]+coeff21*RHS[1]+coeff22*RHS[3];

			const double coeff31 = AA(7,5)/AA(0,5);
			const double c30 = AA(7,0)-coeff31*AA(0,0);
			const double coeff32 = AA(7,3)/AA(6,3);
			const double c34 = coeff32*AA(6,4)+coeff31*AA(0,4);
			const double c38 = AA(7,8)-coeff32*AA(6,8);
			const double d3 = RHS[7]-coeff32*RHS[6]-coeff31*RHS[0];

			const double coeff41 = -AA(8,1)/AA(1,1);
			const double coeff42 = -AA(8,3)/AA(6,3);
			const double c44 = coeff41*AA(1,4)+coeff42*AA(6,4);
			const double c46 = AA(8,6)+coeff41*AA(1,6);
			const double c48 = AA(8,8)+coeff42*AA(6,8);
			const double d4 = RHS[8]+coeff41*RHS[1]+coeff42*RHS[6];

			const double co01 = c26/c06;
			const double co02 = c27/c17;
			const double e00 = co01*c00+co02*c10;
			const double e04 = c24+co01*c04+co02*c14;
			const double f0 = d2-co01*d0-co02*d1;

			const double co11 = c46/c06;
			const double co12 = c48/c38;
			const double e10 = co11*c00+co12*c30;
			const double e14 = c44+co11*c04+co12*c34;
			const double f1 = d4-co11*d0-co12*d3;

			const double co2 = e14/e04;
			dxB[0] = (f1-co2*f0)/(co2*e00-e10);
			dxB[4] = (f0 + e00*dxB[0])/e04;
			dxB[6] = (d0-c00*dxB[0]+c04*dxB[4])/c06;
			dxB[7] = (d1-c10*dxB[0]+c14*dxB[4])/c17;
			dxB[8] = (d3-c30*dxB[0]+c34*dxB[4])/c38;
			dxB[1] = (RHS[1]-AA(1,4)*dxB[4]-AA(1,6)*dxB[6])/AA(1,1);
			dxB[2] = (RHS[3]-AA(3,4)*dxB[4]-AA(3,7)*dxB[7])/AA(3,2);
			dxB[3] = (RHS[6]-AA(6,4)*dxB[4]-AA(6,8)*dxB[8])/AA(6,3);
			dxB[5] = (RHS[0]-AA(0,0)*dxB[0]-AA(0,4)*dxB[4])/AA(0,5);
			//dx3 in dxB, dt*dx1 + 2*dt*dx2 in dxA
		}

		// dx4
		v::multiply_scalar_to_self(dxB, *dt);	//dt*dx3 in dxB
		vp::copy(x0t0, xt);						//x0 in xt
		v::add_to_self(xt, dxB);				//x0 + dt*dx3 in xt
		v::multiply_scalar_to_self(dxB, 2);		//2*dt*dx3 in dxB
		v::add_to_self(dxA, dxB);				//dt*dx1 + 2*dt*dx2 + 2*dt*dx3 in dxA
		xt[nve] = *t0 + *dt;					//t0+dt (update of t parameter)
		evaluate_Hxt(xt, params, Hxt);

		{
			//solveHxTimesDXequalsminusHt(x0+dx3*dt,t0+dt);
		  	const double coeff01 = AA(2,5)/AA(0,5);
			const double c00 = AA(2,0)-coeff01*AA(0,0);
			const double coeff02 = AA(2,1)/AA(1,1);
			const double c04 = coeff02*AA(1,4)+coeff01*AA(0,4);
			const double c06 = AA(2,6)-coeff02*AA(1,6);
			const double d0 = RHS[2]-coeff02*RHS[1]-coeff01*RHS[0];

			const double coeff11 = AA(4,5)/AA(0,5);
			const double c10 = AA(4,0)-coeff11*AA(0,0);
			const double coeff12 = AA(4,2)/AA(3,2);
			const double c14 = coeff12*AA(3,4)+coeff11*AA(0,4);
			const double c17 = AA(4,7)-coeff12*AA(3,7);
			const double d1 = RHS[4]-coeff12*RHS[3]-coeff11*RHS[0];

			const double coeff21 = -AA(5,1)/AA(1,1);
			const double coeff22 = -AA(5,2)/AA(3,2);
			const double c24 = coeff21*AA(1,4)+coeff22*AA(3,4);
			const double c26 = AA(5,6)+coeff21*AA(1,6);
			const double c27 = AA(5,7)+coeff22*AA(3,7);
			const double d2 = RHS[5]+coeff21*RHS[1]+coeff22*RHS[3];

			const double coeff31 = AA(7,5)/AA(0,5);
			const double c30 = AA(7,0)-coeff31*AA(0,0);
			const double coeff32 = AA(7,3)/AA(6,3);
			const double c34 = coeff32*AA(6,4)+coeff31*AA(0,4);
			const double c38 = AA(7,8)-coeff32*AA(6,8);
			const double d3 = RHS[7]-coeff32*RHS[6]-coeff31*RHS[0];

			const double coeff41 = -AA(8,1)/AA(1,1);
			const double coeff42 = -AA(8,3)/AA(6,3);
			const double c44 = coeff41*AA(1,4)+coeff42*AA(6,4);
			const double c46 = AA(8,6)+coeff41*AA(1,6);
			const double c48 = AA(8,8)+coeff42*AA(6,8);
			const double d4 = RHS[8]+coeff41*RHS[1]+coeff42*RHS[6];

			const double co01 = c26/c06;
			const double co02 = c27/c17;
			const double e00 = co01*c00+co02*c10;
			const double e04 = c24+co01*c04+co02*c14;
			const double f0 = d2-co01*d0-co02*d1;

			const double co11 = c46/c06;
			const double co12 = c48/c38;
			const double e10 = co11*c00+co12*c30;
			const double e14 = c44+co11*c04+co12*c34;
			const double f1 = d4-co11*d0-co12*d3;

			const double co2 = e14/e04;
			dxB[0] = (f1-co2*f0)/(co2*e00-e10);
			dxB[4] = (f0 + e00*dxB[0])/e04;
			dxB[6] = (d0-c00*dxB[0]+c04*dxB[4])/c06;
			dxB[7] = (d1-c10*dxB[0]+c14*dxB[4])/c17;
			dxB[8] = (d3-c30*dxB[0]+c34*dxB[4])/c38;
			dxB[1] = (RHS[1]-AA(1,4)*dxB[4]-AA(1,6)*dxB[6])/AA(1,1);
			dxB[2] = (RHS[3]-AA(3,4)*dxB[4]-AA(3,7)*dxB[7])/AA(3,2);
			dxB[3] = (RHS[6]-AA(6,4)*dxB[4]-AA(6,8)*dxB[8])/AA(6,3);
			dxB[5] = (RHS[0]-AA(0,0)*dxB[0]-AA(0,4)*dxB[4])/AA(0,5);
			//dx4 in dxB
		}
		v::multiply_scalar_to_self(dxB, *dt);	//dt*dx4 in dxB
		v::add_to_self(dxA, dxB);				//dt*dx1 + 2*dt*dx2 + 2*dt*dx3 + dt*dx4 in dxB
		v::multiply_scalar_to_self(dxA, 1./6.);	//(1/6) * (dt*dx1 + 2*dt*dx2 + 2*dt*dx3 + dt*dx4 in dxB) in dxA = dx (the RK prediction)

		// make prediction
		vp::copy(x0t0, x1t1);
		vp::add_to_self(x1t1, dxdt);

		/// CORRECTOR ///
		unsigned n_corr_steps = 0;
		bool is_successful;
		do
		{
			++n_corr_steps;
			evaluate_HxH(x1t1, params, HxH);

			const double coeff01 = AA(2,5)/AA(0,5);
			const double c00 = AA(2,0)-coeff01*AA(0,0);
			const double coeff02 = AA(2,1)/AA(1,1);
			const double c04 = coeff02*AA(1,4)+coeff01*AA(0,4);
			const double c06 = AA(2,6)-coeff02*AA(1,6);
			const double d0 = RHS[2]-coeff02*RHS[1]-coeff01*RHS[0];

			const double coeff11 = AA(4,5)/AA(0,5);
			const double c10 = AA(4,0)-coeff11*AA(0,0);
			const double coeff12 = AA(4,2)/AA(3,2);
			const double c14 = coeff12*AA(3,4)+coeff11*AA(0,4);
			const double c17 = AA(4,7)-coeff12*AA(3,7);
			const double d1 = RHS[4]-coeff12*RHS[3]-coeff11*RHS[0];

			const double coeff21 = -AA(5,1)/AA(1,1);
			const double coeff22 = -AA(5,2)/AA(3,2);
			const double c24 = coeff21*AA(1,4)+coeff22*AA(3,4);
			const double c26 = AA(5,6)+coeff21*AA(1,6);
			const double c27 = AA(5,7)+coeff22*AA(3,7);
			const double d2 = RHS[5]+coeff21*RHS[1]+coeff22*RHS[3];

			const double coeff31 = AA(7,5)/AA(0,5);
			const double c30 = AA(7,0)-coeff31*AA(0,0);
			const double coeff32 = AA(7,3)/AA(6,3);
			const double c34 = coeff32*AA(6,4)+coeff31*AA(0,4);
			const double c38 = AA(7,8)-coeff32*AA(6,8);
			const double d3 = RHS[7]-coeff32*RHS[6]-coeff31*RHS[0];

			const double coeff41 = -AA(8,1)/AA(1,1);
			const double coeff42 = -AA(8,3)/AA(6,3);
			const double c44 = coeff41*AA(1,4)+coeff42*AA(6,4);
			const double c46 = AA(8,6)+coeff41*AA(1,6);
			const double c48 = AA(8,8)+coeff42*AA(6,8);
			const double d4 = RHS[8]+coeff41*RHS[1]+coeff42*RHS[6];

			const double co01 = c26/c06;
			const double co02 = c27/c17;
			const double e00 = co01*c00+co02*c10;
			const double e04 = c24+co01*c04+co02*c14;
			const double f0 = d2-co01*d0-co02*d1;

			const double co11 = c46/c06;
			const double co12 = c48/c38;
			const double e10 = co11*c00+co12*c30;
			const double e14 = c44+co11*c04+co12*c34;
			const double f1 = d4-co11*d0-co12*d3;

			const double co2 = e14/e04;
			dx[0] = (f1-co2*f0)/(co2*e00-e10);
			dx[4] = (f0 + e00*dx[0])/e04;
			dx[6] = (d0-c00*dx[0]+c04*dx[4])/c06;
			dx[7] = (d1-c10*dx[0]+c14*dx[4])/c17;
			dx[8] = (d3-c30*dx[0]+c34*dx[4])/c38;
			dx[1] = (RHS[1]-AA(1,4)*dx[4]-AA(1,6)*dx[6])/AA(1,1);
			dx[2] = (RHS[3]-AA(3,4)*dx[4]-AA(3,7)*dx[7])/AA(3,2);

			dx[3] = (RHS[6]-AA(6,4)*dx[4]-AA(6,8)*dx[8])/AA(6,3);
			dx[5] = (RHS[0]-AA(0,0)*dx[0]-AA(0,4)*dx[4])/AA(0,5);


			v::add_to_self(x1t1, dx);
			is_successful = v::norm2(dx) < s.epsilon2_ * v::norm2(x1t1);
		} while (!is_successful && n_corr_steps < s.max_corr_steps_);

		if (!is_successful)
		{ // predictor failure
			predictor_successes = 0;
			*dt *= s.dt_decrease_factor_;
			if (*dt < s.min_dt_)
				status = 5; // slight difference to SLP-imp.hpp:612
		}
		else
		{ // predictor success
			++predictor_successes;
			std::swap(x1t1,x0t0);
			x0 = x0t0;
			t0 = (double *) (x0t0 + nve);
			xt = x1t1;
			if (predictor_successes >= s.num_successes_before_increase_)
			{
				predictor_successes = 0;
				*dt *= s.dt_increase_factor_;
			}
		}
		if (v::norm2(x0) > s.infinity_threshold2_)
			status = 4;
	}

	v::copy(x0, solution); // record the solution
	*num_st = num_steps;
	//t_s->num_steps = num_steps;
	//t_s->t = *t0; // TODO try to include this in the previous memcpy
	if(status == 1) status = 2;

	return status;
}
