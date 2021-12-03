#include <Eigen/SVD>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <iostream>


struct track_settings
{
	track_settings():
	init_dt_(0.05),   // m2 tStep, t_step, raw interface code initDt
	min_dt_(1e-4),        // m2 tStepMin, raw interface code minDt
	end_zone_factor_(0.05),
	epsilon_(4e-2), // m2 CorrectorTolerance
	epsilon2_(epsilon_ * epsilon_), 
	dt_increase_factor_(3.),  // m2 stepIncreaseFactor
	dt_decrease_factor_(1./dt_increase_factor_),  // m2 stepDecreaseFactor not existent in DEFAULT, using what is in track.m2:77 
	infinity_threshold_(1e7), // m2 InfinityThreshold
	infinity_threshold2_(infinity_threshold_ * infinity_threshold_),
	max_corr_steps_(9),  // m2 maxCorrSteps (track.m2 param of rawSetParametersPT corresp to max_corr_steps in NAG.cpp)
	num_successes_before_increase_(4), // m2 numberSuccessesBeforeIncrease
	corr_thresh_(0.00001),
	anch_num_(134)
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
	const double &X0 = x[0];
	const double &X1 = x[1];
	const double &X2 = x[2];
	const double &X3 = x[3];
	const double &X4 = x[4];
	const double &X5 = x[5];
	const double &X6 = x[6];
	const double &X7 = x[7];
	const double &X8 = x[8];
	const double &X9 = x[9];
	const double &X10 = x[10];
	const double &X11 = x[11];
	const double &X12 = x[12];
	const double &X13 = params[0];
	const double &X14 = params[1];
	const double &X15 = params[2];
	const double &X16 = params[3];
	const double &X17 = params[4];
	const double &X18 = params[5];
	const double &X19 = params[6];
	const double &X20 = params[7];
	const double &X21 = params[8];
	const double &X22 = params[9];
	const double &X23 = params[10];
	const double &X24 = params[11];
	const double &X25 = params[12];
	const double &X26 = params[13];
	const double &X27 = params[14];
	const double &X28 = params[15];
	const double &X29 = params[16];
	const double &X30 = params[17];
	const double &X31 = params[18];
	const double &X32 = params[19];
	const double &X33 = params[20];
	const double &X34 = params[21];
	const double &X35 = params[22];
	const double &X36 = params[23];
	const double &X37 = params[24];
	const double &X38 = params[25];
	const double &X39 = params[26];
	const double &X40 = params[27];
	const double &X41 = params[28];
	const double &X42 = params[29];
	const double &X43 = params[30];
	const double &X44 = params[31];
	const double &X45 = params[32];
	const double &X46 = params[33];
	const double &X47 = params[34];
	const double &X48 = params[35];
	const double &X49 = params[36];
	const double &X50 = params[37];
	const double &X51 = params[38];
	const double &X52 = params[39];
	const double &X53 = params[40];
	const double &X54 = params[41];
	const double &X55 = params[42];
	const double &X56 = params[43];
	const double &X57 = params[44];
	const double &X58 = params[45];
	const double &X59 = params[46];
	const double &X60 = params[47];

	static constexpr double C0 = 1;
	static constexpr double C1 = -1;
	const double G0 = C1 * X12;
	const double G1 = C0 + G0;
	const double G2 = G1 * X13;
	const double G3 = X12 * X37;
	const double G4 = G2 + G3;
	const double G5 = G1 * X14;
	const double G6 = X12 * X38;
	const double G7 = G5 + G6;
	const double G8 = X0 * G7;
	const double G9 = C1 * G8;
	const double G10 = G4 + G9;
	const double G11 = C1 * G7;
	const double G12 = G10 * G11;
	const double G13 = G12 + G12;
	const double G14 = G1 * X17;
	const double G15 = X12 * X41;
	const double G16 = G14 + G15;
	const double G17 = G1 * X18;
	const double G18 = X12 * X42;
	const double G19 = G17 + G18;
	const double G20 = X0 * G19;
	const double G21 = C1 * G20;
	const double G22 = G16 + G21;
	const double G23 = C1 * G19;
	const double G24 = G22 * G23;
	const double G25 = G24 + G24;
	const double G26 = G13 + G25;
	static constexpr double C2 = 1;;
	const double G27 = C1 * X0;
	const double G28 = C2 + G27;
	const double G29 = C1 * G28;
	const double G30 = G29 + G29;
	const double G31 = G26 + G30;
	static constexpr double C3 = 0;;
	const double G32 = G1 * X15;
	const double G33 = X12 * X39;
	const double G34 = G32 + G33;
	const double G35 = X1 * G34;
	const double G36 = C1 * G35;
	const double G37 = G8 + G36;
	const double G38 = G37 * G7;
	const double G39 = G38 + G38;
	const double G40 = G1 * X19;
	const double G41 = X12 * X43;
	const double G42 = G40 + G41;
	const double G43 = X1 * G42;
	const double G44 = C1 * G43;
	const double G45 = G20 + G44;
	const double G46 = G45 * G19;
	const double G47 = G46 + G46;
	const double G48 = G39 + G47;
	const double G49 = C1 * X1;
	const double G50 = X0 + G49;
	const double G51 = G50 + G50;
	const double G52 = G48 + G51;
	const double G53 = G1 * X16;
	const double G54 = X12 * X40;
	const double G55 = G53 + G54;
	const double G56 = X2 * G55;
	const double G57 = C1 * G56;
	const double G58 = G8 + G57;
	const double G59 = G58 * G7;
	const double G60 = G59 + G59;
	const double G61 = G1 * X20;
	const double G62 = X12 * X44;
	const double G63 = G61 + G62;
	const double G64 = X2 * G63;
	const double G65 = C1 * G64;
	const double G66 = G20 + G65;
	const double G67 = G66 * G19;
	const double G68 = G67 + G67;
	const double G69 = G60 + G68;
	const double G70 = C1 * X2;
	const double G71 = X0 + G70;
	const double G72 = G71 + G71;
	const double G73 = G69 + G72;
	const double G74 = G4 + G36;
	const double G75 = C1 * G34;
	const double G76 = G74 * G75;
	const double G77 = G76 + G76;
	const double G78 = G16 + G44;
	const double G79 = C1 * G42;
	const double G80 = G78 * G79;
	const double G81 = G80 + G80;
	const double G82 = G77 + G81;
	const double G83 = C2 + G49;
	const double G84 = C1 * G83;
	const double G85 = G84 + G84;
	const double G86 = G82 + G85;
	const double G87 = G37 * G75;
	const double G88 = G87 + G87;
	const double G89 = G45 * G79;
	const double G90 = G89 + G89;
	const double G91 = G88 + G90;
	const double G92 = C1 * G50;
	const double G93 = G92 + G92;
	const double G94 = G91 + G93;
	const double G95 = G35 + G57;
	const double G96 = G95 * G34;
	const double G97 = G96 + G96;
	const double G98 = G43 + G65;
	const double G99 = G98 * G42;
	const double G100 = G99 + G99;
	const double G101 = G97 + G100;
	const double G102 = X1 + G70;
	const double G103 = G102 + G102;
	const double G104 = G101 + G103;
	const double G105 = G4 + G57;
	const double G106 = C1 * G55;
	const double G107 = G105 * G106;
	const double G108 = G107 + G107;
	const double G109 = G16 + G65;
	const double G110 = C1 * G63;
	const double G111 = G109 * G110;
	const double G112 = G111 + G111;
	const double G113 = G108 + G112;
	const double G114 = C2 + G70;
	const double G115 = C1 * G114;
	const double G116 = G115 + G115;
	const double G117 = G113 + G116;
	const double G118 = G58 * G106;
	const double G119 = G118 + G118;
	const double G120 = G66 * G110;
	const double G121 = G120 + G120;
	const double G122 = G119 + G121;
	const double G123 = C1 * G71;
	const double G124 = G123 + G123;
	const double G125 = G122 + G124;
	const double G126 = G95 * G106;
	const double G127 = G126 + G126;
	const double G128 = G98 * G110;
	const double G129 = G128 + G128;
	const double G130 = G127 + G129;
	const double G131 = C1 * G102;
	const double G132 = G131 + G131;
	const double G133 = G130 + G132;
	const double G134 = G1 * X21;
	const double G135 = X12 * X45;
	const double G136 = G134 + G135;
	const double G137 = X3 * G136;
	const double G138 = G1 * X22;
	const double G139 = X12 * X46;
	const double G140 = G138 + G139;
	const double G141 = X4 * G140;
	const double G142 = C1 * G141;
	const double G143 = G137 + G142;
	const double G144 = G143 * G136;
	const double G145 = G144 + G144;
	const double G146 = G1 * X25;
	const double G147 = X12 * X49;
	const double G148 = G146 + G147;
	const double G149 = X3 * G148;
	const double G150 = G1 * X26;
	const double G151 = X12 * X50;
	const double G152 = G150 + G151;
	const double G153 = X4 * G152;
	const double G154 = C1 * G153;
	const double G155 = G149 + G154;
	const double G156 = G155 * G148;
	const double G157 = G156 + G156;
	const double G158 = G145 + G157;
	const double G159 = C1 * X4;
	const double G160 = X3 + G159;
	const double G161 = G160 + G160;
	const double G162 = G158 + G161;
	const double G163 = C1 * G162;
	const double G164 = G1 * X23;
	const double G165 = X12 * X47;
	const double G166 = G164 + G165;
	const double G167 = X5 * G166;
	const double G168 = C1 * G167;
	const double G169 = G137 + G168;
	const double G170 = G169 * G136;
	const double G171 = G170 + G170;
	const double G172 = G1 * X27;
	const double G173 = X12 * X51;
	const double G174 = G172 + G173;
	const double G175 = X5 * G174;
	const double G176 = C1 * G175;
	const double G177 = G149 + G176;
	const double G178 = G177 * G148;
	const double G179 = G178 + G178;
	const double G180 = G171 + G179;
	const double G181 = C1 * X5;
	const double G182 = X3 + G181;
	const double G183 = G182 + G182;
	const double G184 = G180 + G183;
	const double G185 = C1 * G184;
	const double G186 = G1 * X24;
	const double G187 = X12 * X48;
	const double G188 = G186 + G187;
	const double G189 = X6 * G188;
	const double G190 = C1 * G189;
	const double G191 = G137 + G190;
	const double G192 = G191 * G136;
	const double G193 = G192 + G192;
	const double G194 = G1 * X28;
	const double G195 = X12 * X52;
	const double G196 = G194 + G195;
	const double G197 = X6 * G196;
	const double G198 = C1 * G197;
	const double G199 = G149 + G198;
	const double G200 = G199 * G148;
	const double G201 = G200 + G200;
	const double G202 = G193 + G201;
	const double G203 = C1 * X6;
	const double G204 = X3 + G203;
	const double G205 = G204 + G204;
	const double G206 = G202 + G205;
	const double G207 = C1 * G206;
	const double G208 = C1 * G140;
	const double G209 = G143 * G208;
	const double G210 = G209 + G209;
	const double G211 = C1 * G152;
	const double G212 = G155 * G211;
	const double G213 = G212 + G212;
	const double G214 = G210 + G213;
	const double G215 = C1 * G160;
	const double G216 = G215 + G215;
	const double G217 = G214 + G216;
	const double G218 = C1 * G217;
	const double G219 = G141 + G168;
	const double G220 = G219 * G140;
	const double G221 = G220 + G220;
	const double G222 = G153 + G176;
	const double G223 = G222 * G152;
	const double G224 = G223 + G223;
	const double G225 = G221 + G224;
	const double G226 = X4 + G181;
	const double G227 = G226 + G226;
	const double G228 = G225 + G227;
	const double G229 = C1 * G228;
	const double G230 = G141 + G190;
	const double G231 = G230 * G140;
	const double G232 = G231 + G231;
	const double G233 = G153 + G198;
	const double G234 = G233 * G152;
	const double G235 = G234 + G234;
	const double G236 = G232 + G235;
	const double G237 = X4 + G203;
	const double G238 = G237 + G237;
	const double G239 = G236 + G238;
	const double G240 = C1 * G239;
	const double G241 = C1 * G166;
	const double G242 = G169 * G241;
	const double G243 = G242 + G242;
	const double G244 = C1 * G174;
	const double G245 = G177 * G244;
	const double G246 = G245 + G245;
	const double G247 = G243 + G246;
	const double G248 = C1 * G182;
	const double G249 = G248 + G248;
	const double G250 = G247 + G249;
	const double G251 = C1 * G250;
	const double G252 = G219 * G241;
	const double G253 = G252 + G252;
	const double G254 = G222 * G244;
	const double G255 = G254 + G254;
	const double G256 = G253 + G255;
	const double G257 = C1 * G226;
	const double G258 = G257 + G257;
	const double G259 = G256 + G258;
	const double G260 = C1 * G259;
	const double G261 = G167 + G190;
	const double G262 = G261 * G166;
	const double G263 = G262 + G262;
	const double G264 = G175 + G198;
	const double G265 = G264 * G174;
	const double G266 = G265 + G265;
	const double G267 = G263 + G266;
	const double G268 = X5 + G203;
	const double G269 = G268 + G268;
	const double G270 = G267 + G269;
	const double G271 = C1 * G270;
	const double G272 = C1 * G188;
	const double G273 = G191 * G272;
	const double G274 = G273 + G273;
	const double G275 = C1 * G196;
	const double G276 = G199 * G275;
	const double G277 = G276 + G276;
	const double G278 = G274 + G277;
	const double G279 = C1 * G204;
	const double G280 = G279 + G279;
	const double G281 = G278 + G280;
	const double G282 = C1 * G281;
	const double G283 = G230 * G272;
	const double G284 = G283 + G283;
	const double G285 = G233 * G275;
	const double G286 = G285 + G285;
	const double G287 = G284 + G286;
	const double G288 = C1 * G237;
	const double G289 = G288 + G288;
	const double G290 = G287 + G289;
	const double G291 = C1 * G290;
	const double G292 = G261 * G272;
	const double G293 = G292 + G292;
	const double G294 = G264 * G275;
	const double G295 = G294 + G294;
	const double G296 = G293 + G295;
	const double G297 = C1 * G268;
	const double G298 = G297 + G297;
	const double G299 = G296 + G298;
	const double G300 = C1 * G299;
	const double G301 = G1 * X29;
	const double G302 = X12 * X53;
	const double G303 = G301 + G302;
	const double G304 = X7 * G303;
	const double G305 = G1 * X30;
	const double G306 = X12 * X54;
	const double G307 = G305 + G306;
	const double G308 = X8 * G307;
	const double G309 = C1 * G308;
	const double G310 = G304 + G309;
	const double G311 = G310 * G303;
	const double G312 = G311 + G311;
	const double G313 = G1 * X33;
	const double G314 = X12 * X57;
	const double G315 = G313 + G314;
	const double G316 = X7 * G315;
	const double G317 = G1 * X34;
	const double G318 = X12 * X58;
	const double G319 = G317 + G318;
	const double G320 = X8 * G319;
	const double G321 = C1 * G320;
	const double G322 = G316 + G321;
	const double G323 = G322 * G315;
	const double G324 = G323 + G323;
	const double G325 = G312 + G324;
	const double G326 = C1 * X8;
	const double G327 = X7 + G326;
	const double G328 = G327 + G327;
	const double G329 = G325 + G328;
	const double G330 = C1 * G329;
	const double G331 = G1 * X31;
	const double G332 = X12 * X55;
	const double G333 = G331 + G332;
	const double G334 = X9 * G333;
	const double G335 = C1 * G334;
	const double G336 = G304 + G335;
	const double G337 = G336 * G303;
	const double G338 = G337 + G337;
	const double G339 = G1 * X35;
	const double G340 = X12 * X59;
	const double G341 = G339 + G340;
	const double G342 = X9 * G341;
	const double G343 = C1 * G342;
	const double G344 = G316 + G343;
	const double G345 = G344 * G315;
	const double G346 = G345 + G345;
	const double G347 = G338 + G346;
	const double G348 = C1 * X9;
	const double G349 = X7 + G348;
	const double G350 = G349 + G349;
	const double G351 = G347 + G350;
	const double G352 = C1 * G351;
	const double G353 = G1 * X32;
	const double G354 = X12 * X56;
	const double G355 = G353 + G354;
	const double G356 = X10 * G355;
	const double G357 = C1 * G356;
	const double G358 = G304 + G357;
	const double G359 = G358 * G303;
	const double G360 = G359 + G359;
	const double G361 = G1 * X36;
	const double G362 = X12 * X60;
	const double G363 = G361 + G362;
	const double G364 = G363 + X11;
	const double G365 = X10 * G364;
	const double G366 = C1 * G365;
	const double G367 = G316 + G366;
	const double G368 = G367 * G315;
	const double G369 = G368 + G368;
	const double G370 = G360 + G369;
	const double G371 = C1 * X10;
	const double G372 = X7 + G371;
	const double G373 = G372 + G372;
	const double G374 = G370 + G373;
	const double G375 = C1 * G374;
	const double G376 = C1 * G307;
	const double G377 = G310 * G376;
	const double G378 = G377 + G377;
	const double G379 = C1 * G319;
	const double G380 = G322 * G379;
	const double G381 = G380 + G380;
	const double G382 = G378 + G381;
	const double G383 = C1 * G327;
	const double G384 = G383 + G383;
	const double G385 = G382 + G384;
	const double G386 = C1 * G385;
	const double G387 = G308 + G335;
	const double G388 = G387 * G307;
	const double G389 = G388 + G388;
	const double G390 = G320 + G343;
	const double G391 = G390 * G319;
	const double G392 = G391 + G391;
	const double G393 = G389 + G392;
	const double G394 = X8 + G348;
	const double G395 = G394 + G394;
	const double G396 = G393 + G395;
	const double G397 = C1 * G396;
	const double G398 = G308 + G357;
	const double G399 = G398 * G307;
	const double G400 = G399 + G399;
	const double G401 = G320 + G366;
	const double G402 = G401 * G319;
	const double G403 = G402 + G402;
	const double G404 = G400 + G403;
	const double G405 = X8 + G371;
	const double G406 = G405 + G405;
	const double G407 = G404 + G406;
	const double G408 = C1 * G407;
	const double G409 = C1 * G333;
	const double G410 = G336 * G409;
	const double G411 = G410 + G410;
	const double G412 = C1 * G341;
	const double G413 = G344 * G412;
	const double G414 = G413 + G413;
	const double G415 = G411 + G414;
	const double G416 = C1 * G349;
	const double G417 = G416 + G416;
	const double G418 = G415 + G417;
	const double G419 = C1 * G418;
	const double G420 = G387 * G409;
	const double G421 = G420 + G420;
	const double G422 = G390 * G412;
	const double G423 = G422 + G422;
	const double G424 = G421 + G423;
	const double G425 = C1 * G394;
	const double G426 = G425 + G425;
	const double G427 = G424 + G426;
	const double G428 = C1 * G427;
	const double G429 = G334 + G357;
	const double G430 = G429 * G333;
	const double G431 = G430 + G430;
	const double G432 = G342 + G366;
	const double G433 = G432 * G341;
	const double G434 = G433 + G433;
	const double G435 = G431 + G434;
	const double G436 = X9 + G371;
	const double G437 = G436 + G436;
	const double G438 = G435 + G437;
	const double G439 = C1 * G438;
	const double G440 = C1 * G355;
	const double G441 = G358 * G440;
	const double G442 = G441 + G441;
	const double G443 = C1 * G364;
	const double G444 = G367 * G443;
	const double G445 = G444 + G444;
	const double G446 = G442 + G445;
	const double G447 = C1 * G372;
	const double G448 = G447 + G447;
	const double G449 = G446 + G448;
	const double G450 = C1 * G449;
	const double G451 = G398 * G440;
	const double G452 = G451 + G451;
	const double G453 = G401 * G443;
	const double G454 = G453 + G453;
	const double G455 = G452 + G454;
	const double G456 = C1 * G405;
	const double G457 = G456 + G456;
	const double G458 = G455 + G457;
	const double G459 = C1 * G458;
	const double G460 = G429 * G440;
	const double G461 = G460 + G460;
	const double G462 = G432 * G443;
	const double G463 = G462 + G462;
	const double G464 = G461 + G463;
	const double G465 = C1 * G436;
	const double G466 = G465 + G465;
	const double G467 = G464 + G466;
	const double G468 = C1 * G467;
	const double G469 = G367 * G371;
	const double G470 = G469 + G469;
	const double G471 = C1 * G470;
	const double G472 = G401 * G371;
	const double G473 = G472 + G472;
	const double G474 = C1 * G473;
	const double G475 = G432 * G371;
	const double G476 = G475 + G475;
	const double G477 = C1 * G476;
	const double G478 = C1 * X13;
	const double G479 = G478 + X37;
	const double G480 = C1 * X14;
	const double G481 = G480 + X38;
	const double G482 = X0 * G481;
	const double G483 = C1 * G482;
	const double G484 = G479 + G483;
	const double G485 = G10 * G484;
	const double G486 = G485 + G485;
	const double G487 = C1 * X17;
	const double G488 = G487 + X41;
	const double G489 = C1 * X18;
	const double G490 = G489 + X42;
	const double G491 = X0 * G490;
	const double G492 = C1 * G491;
	const double G493 = G488 + G492;
	const double G494 = G22 * G493;
	const double G495 = G494 + G494;
	const double G496 = G486 + G495;
	const double G497 = C1 * X21;
	const double G498 = G497 + X45;
	const double G499 = X3 * G498;
	const double G500 = C1 * X22;
	const double G501 = G500 + X46;
	const double G502 = X4 * G501;
	const double G503 = C1 * G502;
	const double G504 = G499 + G503;
	const double G505 = G143 * G504;
	const double G506 = G505 + G505;
	const double G507 = C1 * X25;
	const double G508 = G507 + X49;
	const double G509 = X3 * G508;
	const double G510 = C1 * X26;
	const double G511 = G510 + X50;
	const double G512 = X4 * G511;
	const double G513 = C1 * G512;
	const double G514 = G509 + G513;
	const double G515 = G155 * G514;
	const double G516 = G515 + G515;
	const double G517 = G506 + G516;
	const double G518 = C1 * G517;
	const double G519 = G496 + G518;
	const double G520 = C1 * X15;
	const double G521 = G520 + X39;
	const double G522 = X1 * G521;
	const double G523 = C1 * G522;
	const double G524 = G479 + G523;
	const double G525 = G74 * G524;
	const double G526 = G525 + G525;
	const double G527 = C1 * X19;
	const double G528 = G527 + X43;
	const double G529 = X1 * G528;
	const double G530 = C1 * G529;
	const double G531 = G488 + G530;
	const double G532 = G78 * G531;
	const double G533 = G532 + G532;
	const double G534 = G526 + G533;
	const double G535 = C1 * X23;
	const double G536 = G535 + X47;
	const double G537 = X5 * G536;
	const double G538 = C1 * G537;
	const double G539 = G499 + G538;
	const double G540 = G169 * G539;
	const double G541 = G540 + G540;
	const double G542 = C1 * X27;
	const double G543 = G542 + X51;
	const double G544 = X5 * G543;
	const double G545 = C1 * G544;
	const double G546 = G509 + G545;
	const double G547 = G177 * G546;
	const double G548 = G547 + G547;
	const double G549 = G541 + G548;
	const double G550 = C1 * G549;
	const double G551 = G534 + G550;
	const double G552 = G482 + G523;
	const double G553 = G37 * G552;
	const double G554 = G553 + G553;
	const double G555 = G491 + G530;
	const double G556 = G45 * G555;
	const double G557 = G556 + G556;
	const double G558 = G554 + G557;
	const double G559 = G502 + G538;
	const double G560 = G219 * G559;
	const double G561 = G560 + G560;
	const double G562 = G512 + G545;
	const double G563 = G222 * G562;
	const double G564 = G563 + G563;
	const double G565 = G561 + G564;
	const double G566 = C1 * G565;
	const double G567 = G558 + G566;
	const double G568 = C1 * X16;
	const double G569 = G568 + X40;
	const double G570 = X2 * G569;
	const double G571 = C1 * G570;
	const double G572 = G479 + G571;
	const double G573 = G105 * G572;
	const double G574 = G573 + G573;
	const double G575 = C1 * X20;
	const double G576 = G575 + X44;
	const double G577 = X2 * G576;
	const double G578 = C1 * G577;
	const double G579 = G488 + G578;
	const double G580 = G109 * G579;
	const double G581 = G580 + G580;
	const double G582 = G574 + G581;
	const double G583 = C1 * X24;
	const double G584 = G583 + X48;
	const double G585 = X6 * G584;
	const double G586 = C1 * G585;
	const double G587 = G499 + G586;
	const double G588 = G191 * G587;
	const double G589 = G588 + G588;
	const double G590 = C1 * X28;
	const double G591 = G590 + X52;
	const double G592 = X6 * G591;
	const double G593 = C1 * G592;
	const double G594 = G509 + G593;
	const double G595 = G199 * G594;
	const double G596 = G595 + G595;
	const double G597 = G589 + G596;
	const double G598 = C1 * G597;
	const double G599 = G582 + G598;
	const double G600 = G482 + G571;
	const double G601 = G58 * G600;
	const double G602 = G601 + G601;
	const double G603 = G491 + G578;
	const double G604 = G66 * G603;
	const double G605 = G604 + G604;
	const double G606 = G602 + G605;
	const double G607 = G502 + G586;
	const double G608 = G230 * G607;
	const double G609 = G608 + G608;
	const double G610 = G512 + G593;
	const double G611 = G233 * G610;
	const double G612 = G611 + G611;
	const double G613 = G609 + G612;
	const double G614 = C1 * G613;
	const double G615 = G606 + G614;
	const double G616 = G522 + G571;
	const double G617 = G95 * G616;
	const double G618 = G617 + G617;
	const double G619 = G529 + G578;
	const double G620 = G98 * G619;
	const double G621 = G620 + G620;
	const double G622 = G618 + G621;
	const double G623 = G537 + G586;
	const double G624 = G261 * G623;
	const double G625 = G624 + G624;
	const double G626 = G544 + G593;
	const double G627 = G264 * G626;
	const double G628 = G627 + G627;
	const double G629 = G625 + G628;
	const double G630 = C1 * G629;
	const double G631 = G622 + G630;
	const double G632 = C1 * X29;
	const double G633 = G632 + X53;
	const double G634 = X7 * G633;
	const double G635 = C1 * X30;
	const double G636 = G635 + X54;
	const double G637 = X8 * G636;
	const double G638 = C1 * G637;
	const double G639 = G634 + G638;
	const double G640 = G310 * G639;
	const double G641 = G640 + G640;
	const double G642 = C1 * X33;
	const double G643 = G642 + X57;
	const double G644 = X7 * G643;
	const double G645 = C1 * X34;
	const double G646 = G645 + X58;
	const double G647 = X8 * G646;
	const double G648 = C1 * G647;
	const double G649 = G644 + G648;
	const double G650 = G322 * G649;
	const double G651 = G650 + G650;
	const double G652 = G641 + G651;
	const double G653 = C1 * G652;
	const double G654 = G496 + G653;
	const double G655 = C1 * X31;
	const double G656 = G655 + X55;
	const double G657 = X9 * G656;
	const double G658 = C1 * G657;
	const double G659 = G634 + G658;
	const double G660 = G336 * G659;
	const double G661 = G660 + G660;
	const double G662 = C1 * X35;
	const double G663 = G662 + X59;
	const double G664 = X9 * G663;
	const double G665 = C1 * G664;
	const double G666 = G644 + G665;
	const double G667 = G344 * G666;
	const double G668 = G667 + G667;
	const double G669 = G661 + G668;
	const double G670 = C1 * G669;
	const double G671 = G534 + G670;
	const double G672 = G637 + G658;
	const double G673 = G387 * G672;
	const double G674 = G673 + G673;
	const double G675 = G647 + G665;
	const double G676 = G390 * G675;
	const double G677 = G676 + G676;
	const double G678 = G674 + G677;
	const double G679 = C1 * G678;
	const double G680 = G558 + G679;
	const double G681 = C1 * X32;
	const double G682 = G681 + X56;
	const double G683 = X10 * G682;
	const double G684 = C1 * G683;
	const double G685 = G634 + G684;
	const double G686 = G358 * G685;
	const double G687 = G686 + G686;
	const double G688 = C1 * X36;
	const double G689 = G688 + X60;
	const double G690 = X10 * G689;
	const double G691 = C1 * G690;
	const double G692 = G644 + G691;
	const double G693 = G367 * G692;
	const double G694 = G693 + G693;
	const double G695 = G687 + G694;
	const double G696 = C1 * G695;
	const double G697 = G582 + G696;
	const double G698 = G637 + G684;
	const double G699 = G398 * G698;
	const double G700 = G699 + G699;
	const double G701 = G647 + G691;
	const double G702 = G401 * G701;
	const double G703 = G702 + G702;
	const double G704 = G700 + G703;
	const double G705 = C1 * G704;
	const double G706 = G606 + G705;
	const double G707 = G657 + G684;
	const double G708 = G429 * G707;
	const double G709 = G708 + G708;
	const double G710 = G664 + G691;
	const double G711 = G432 * G710;
	const double G712 = G711 + G711;
	const double G713 = G709 + G712;
	const double G714 = C1 * G713;
	const double G715 = G622 + G714;
	y[0] = G31;
	y[1] = C3;
	y[2] = G52;
	y[3] = C3;
	y[4] = G73;
	y[5] = C3;
	y[6] = G31;
	y[7] = C3;
	y[8] = G52;
	y[9] = C3;
	y[10] = G73;
	y[11] = C3;
	y[12] = C3;
	y[13] = G86;
	y[14] = G94;
	y[15] = C3;
	y[16] = C3;
	y[17] = G104;
	y[18] = C3;
	y[19] = G86;
	y[20] = G94;
	y[21] = C3;
	y[22] = C3;
	y[23] = G104;
	y[24] = C3;
	y[25] = C3;
	y[26] = C3;
	y[27] = G117;
	y[28] = G125;
	y[29] = G133;
	y[30] = C3;
	y[31] = C3;
	y[32] = C3;
	y[33] = G117;
	y[34] = G125;
	y[35] = G133;
	y[36] = G163;
	y[37] = G185;
	y[38] = C3;
	y[39] = G207;
	y[40] = C3;
	y[41] = C3;
	y[42] = C3;
	y[43] = C3;
	y[44] = C3;
	y[45] = C3;
	y[46] = C3;
	y[47] = C3;
	y[48] = G218;
	y[49] = C3;
	y[50] = G229;
	y[51] = C3;
	y[52] = G240;
	y[53] = C3;
	y[54] = C3;
	y[55] = C3;
	y[56] = C3;
	y[57] = C3;
	y[58] = C3;
	y[59] = C3;
	y[60] = C3;
	y[61] = G251;
	y[62] = G260;
	y[63] = C3;
	y[64] = C3;
	y[65] = G271;
	y[66] = C3;
	y[67] = C3;
	y[68] = C3;
	y[69] = C3;
	y[70] = C3;
	y[71] = C3;
	y[72] = C3;
	y[73] = C3;
	y[74] = C3;
	y[75] = G282;
	y[76] = G291;
	y[77] = G300;
	y[78] = C3;
	y[79] = C3;
	y[80] = C3;
	y[81] = C3;
	y[82] = C3;
	y[83] = C3;
	y[84] = C3;
	y[85] = C3;
	y[86] = C3;
	y[87] = C3;
	y[88] = C3;
	y[89] = C3;
	y[90] = G330;
	y[91] = G352;
	y[92] = C3;
	y[93] = G375;
	y[94] = C3;
	y[95] = C3;
	y[96] = C3;
	y[97] = C3;
	y[98] = C3;
	y[99] = C3;
	y[100] = C3;
	y[101] = C3;
	y[102] = G386;
	y[103] = C3;
	y[104] = G397;
	y[105] = C3;
	y[106] = G408;
	y[107] = C3;
	y[108] = C3;
	y[109] = C3;
	y[110] = C3;
	y[111] = C3;
	y[112] = C3;
	y[113] = C3;
	y[114] = C3;
	y[115] = G419;
	y[116] = G428;
	y[117] = C3;
	y[118] = C3;
	y[119] = G439;
	y[120] = C3;
	y[121] = C3;
	y[122] = C3;
	y[123] = C3;
	y[124] = C3;
	y[125] = C3;
	y[126] = C3;
	y[127] = C3;
	y[128] = C3;
	y[129] = G450;
	y[130] = G459;
	y[131] = G468;
	y[132] = C3;
	y[133] = C3;
	y[134] = C3;
	y[135] = C3;
	y[136] = C3;
	y[137] = C3;
	y[138] = C3;
	y[139] = C3;
	y[140] = C3;
	y[141] = G471;
	y[142] = G474;
	y[143] = G477;

	y[144] = -G519;
	y[145] = -G551;
	y[146] = -G567;
	y[147] = -G599;
	y[148] = -G615;
	y[149] = -G631;
	y[150] = -G654;
	y[151] = -G671;
	y[152] = -G680;
	y[153] = -G697;
	y[154] = -G706;
	y[155] = -G715;

}

//Straight line program for evaluation of the Jacobian of the homotopy function, generated in Macaulay2
inline void evaluate_HxH(const double * x, const double * params, double * y)
{
	const double &X0 = x[0];
	const double &X1 = x[1];
	const double &X2 = x[2];
	const double &X3 = x[3];
	const double &X4 = x[4];
	const double &X5 = x[5];
	const double &X6 = x[6];
	const double &X7 = x[7];
	const double &X8 = x[8];
	const double &X9 = x[9];
	const double &X10 = x[10];
	const double &X11 = x[11];
	const double &X12 = x[12];
	const double &X13 = params[0];
	const double &X14 = params[1];
	const double &X15 = params[2];
	const double &X16 = params[3];
	const double &X17 = params[4];
	const double &X18 = params[5];
	const double &X19 = params[6];
	const double &X20 = params[7];
	const double &X21 = params[8];
	const double &X22 = params[9];
	const double &X23 = params[10];
	const double &X24 = params[11];
	const double &X25 = params[12];
	const double &X26 = params[13];
	const double &X27 = params[14];
	const double &X28 = params[15];
	const double &X29 = params[16];
	const double &X30 = params[17];
	const double &X31 = params[18];
	const double &X32 = params[19];
	const double &X33 = params[20];
	const double &X34 = params[21];
	const double &X35 = params[22];
	const double &X36 = params[23];
	const double &X37 = params[24];
	const double &X38 = params[25];
	const double &X39 = params[26];
	const double &X40 = params[27];
	const double &X41 = params[28];
	const double &X42 = params[29];
	const double &X43 = params[30];
	const double &X44 = params[31];
	const double &X45 = params[32];
	const double &X46 = params[33];
	const double &X47 = params[34];
	const double &X48 = params[35];
	const double &X49 = params[36];
	const double &X50 = params[37];
	const double &X51 = params[38];
	const double &X52 = params[39];
	const double &X53 = params[40];
	const double &X54 = params[41];
	const double &X55 = params[42];
	const double &X56 = params[43];
	const double &X57 = params[44];
	const double &X58 = params[45];
	const double &X59 = params[46];
	const double &X60 = params[47];

	static constexpr double C0 = 1;
	static constexpr double C1 = -1;
	const double G0 = C1 * X12;
	const double G1 = C0 + G0;
	const double G2 = G1 * X13;
	const double G3 = X12 * X37;
	const double G4 = G2 + G3;
	const double G5 = G1 * X14;
	const double G6 = X12 * X38;
	const double G7 = G5 + G6;
	const double G8 = X0 * G7;
	const double G9 = C1 * G8;
	const double G10 = G4 + G9;
	const double G11 = C1 * G7;
	const double G12 = G10 * G11;
	const double G13 = G12 + G12;
	const double G14 = G1 * X17;
	const double G15 = X12 * X41;
	const double G16 = G14 + G15;
	const double G17 = G1 * X18;
	const double G18 = X12 * X42;
	const double G19 = G17 + G18;
	const double G20 = X0 * G19;
	const double G21 = C1 * G20;
	const double G22 = G16 + G21;
	const double G23 = C1 * G19;
	const double G24 = G22 * G23;
	const double G25 = G24 + G24;
	const double G26 = G13 + G25;
	static constexpr double C2 = 1;
	const double G27 = C1 * X0;
	const double G28 = C2 + G27;
	const double G29 = C1 * G28;
	const double G30 = G29 + G29;
	const double G31 = G26 + G30;
	static constexpr double C3 = 0;
	const double G32 = G1 * X15;
	const double G33 = X12 * X39;
	const double G34 = G32 + G33;
	const double G35 = X1 * G34;
	const double G36 = C1 * G35;
	const double G37 = G8 + G36;
	const double G38 = G37 * G7;
	const double G39 = G38 + G38;
	const double G40 = G1 * X19;
	const double G41 = X12 * X43;
	const double G42 = G40 + G41;
	const double G43 = X1 * G42;
	const double G44 = C1 * G43;
	const double G45 = G20 + G44;
	const double G46 = G45 * G19;
	const double G47 = G46 + G46;
	const double G48 = G39 + G47;
	const double G49 = C1 * X1;
	const double G50 = X0 + G49;
	const double G51 = G50 + G50;
	const double G52 = G48 + G51;
	const double G53 = G1 * X16;
	const double G54 = X12 * X40;
	const double G55 = G53 + G54;
	const double G56 = X2 * G55;
	const double G57 = C1 * G56;
	const double G58 = G8 + G57;
	const double G59 = G58 * G7;
	const double G60 = G59 + G59;
	const double G61 = G1 * X20;
	const double G62 = X12 * X44;
	const double G63 = G61 + G62;
	const double G64 = X2 * G63;
	const double G65 = C1 * G64;
	const double G66 = G20 + G65;
	const double G67 = G66 * G19;
	const double G68 = G67 + G67;
	const double G69 = G60 + G68;
	const double G70 = C1 * X2;
	const double G71 = X0 + G70;
	const double G72 = G71 + G71;
	const double G73 = G69 + G72;
	const double G74 = G4 + G36;
	const double G75 = C1 * G34;
	const double G76 = G74 * G75;
	const double G77 = G76 + G76;
	const double G78 = G16 + G44;
	const double G79 = C1 * G42;
	const double G80 = G78 * G79;
	const double G81 = G80 + G80;
	const double G82 = G77 + G81;
	const double G83 = C2 + G49;
	const double G84 = C1 * G83;
	const double G85 = G84 + G84;
	const double G86 = G82 + G85;
	const double G87 = G37 * G75;
	const double G88 = G87 + G87;
	const double G89 = G45 * G79;
	const double G90 = G89 + G89;
	const double G91 = G88 + G90;
	const double G92 = C1 * G50;
	const double G93 = G92 + G92;
	const double G94 = G91 + G93;
	const double G95 = G35 + G57;
	const double G96 = G95 * G34;
	const double G97 = G96 + G96;
	const double G98 = G43 + G65;
	const double G99 = G98 * G42;
	const double G100 = G99 + G99;
	const double G101 = G97 + G100;
	const double G102 = X1 + G70;
	const double G103 = G102 + G102;
	const double G104 = G101 + G103;
	const double G105 = G4 + G57;
	const double G106 = C1 * G55;
	const double G107 = G105 * G106;
	const double G108 = G107 + G107;
	const double G109 = G16 + G65;
	const double G110 = C1 * G63;
	const double G111 = G109 * G110;
	const double G112 = G111 + G111;
	const double G113 = G108 + G112;
	const double G114 = C2 + G70;
	const double G115 = C1 * G114;
	const double G116 = G115 + G115;
	const double G117 = G113 + G116;
	const double G118 = G58 * G106;
	const double G119 = G118 + G118;
	const double G120 = G66 * G110;
	const double G121 = G120 + G120;
	const double G122 = G119 + G121;
	const double G123 = C1 * G71;
	const double G124 = G123 + G123;
	const double G125 = G122 + G124;
	const double G126 = G95 * G106;
	const double G127 = G126 + G126;
	const double G128 = G98 * G110;
	const double G129 = G128 + G128;
	const double G130 = G127 + G129;
	const double G131 = C1 * G102;
	const double G132 = G131 + G131;
	const double G133 = G130 + G132;
	const double G134 = G1 * X21;
	const double G135 = X12 * X45;
	const double G136 = G134 + G135;
	const double G137 = X3 * G136;
	const double G138 = G1 * X22;
	const double G139 = X12 * X46;
	const double G140 = G138 + G139;
	const double G141 = X4 * G140;
	const double G142 = C1 * G141;
	const double G143 = G137 + G142;
	const double G144 = G143 * G136;
	const double G145 = G144 + G144;
	const double G146 = G1 * X25;
	const double G147 = X12 * X49;
	const double G148 = G146 + G147;
	const double G149 = X3 * G148;
	const double G150 = G1 * X26;
	const double G151 = X12 * X50;
	const double G152 = G150 + G151;
	const double G153 = X4 * G152;
	const double G154 = C1 * G153;
	const double G155 = G149 + G154;
	const double G156 = G155 * G148;
	const double G157 = G156 + G156;
	const double G158 = G145 + G157;
	const double G159 = C1 * X4;
	const double G160 = X3 + G159;
	const double G161 = G160 + G160;
	const double G162 = G158 + G161;
	const double G163 = C1 * G162;
	const double G164 = G1 * X23;
	const double G165 = X12 * X47;
	const double G166 = G164 + G165;
	const double G167 = X5 * G166;
	const double G168 = C1 * G167;
	const double G169 = G137 + G168;
	const double G170 = G169 * G136;
	const double G171 = G170 + G170;
	const double G172 = G1 * X27;
	const double G173 = X12 * X51;
	const double G174 = G172 + G173;
	const double G175 = X5 * G174;
	const double G176 = C1 * G175;
	const double G177 = G149 + G176;
	const double G178 = G177 * G148;
	const double G179 = G178 + G178;
	const double G180 = G171 + G179;
	const double G181 = C1 * X5;
	const double G182 = X3 + G181;
	const double G183 = G182 + G182;
	const double G184 = G180 + G183;
	const double G185 = C1 * G184;
	const double G186 = G1 * X24;
	const double G187 = X12 * X48;
	const double G188 = G186 + G187;
	const double G189 = X6 * G188;
	const double G190 = C1 * G189;
	const double G191 = G137 + G190;
	const double G192 = G191 * G136;
	const double G193 = G192 + G192;
	const double G194 = G1 * X28;
	const double G195 = X12 * X52;
	const double G196 = G194 + G195;
	const double G197 = X6 * G196;
	const double G198 = C1 * G197;
	const double G199 = G149 + G198;
	const double G200 = G199 * G148;
	const double G201 = G200 + G200;
	const double G202 = G193 + G201;
	const double G203 = C1 * X6;
	const double G204 = X3 + G203;
	const double G205 = G204 + G204;
	const double G206 = G202 + G205;
	const double G207 = C1 * G206;
	const double G208 = C1 * G140;
	const double G209 = G143 * G208;
	const double G210 = G209 + G209;
	const double G211 = C1 * G152;
	const double G212 = G155 * G211;
	const double G213 = G212 + G212;
	const double G214 = G210 + G213;
	const double G215 = C1 * G160;
	const double G216 = G215 + G215;
	const double G217 = G214 + G216;
	const double G218 = C1 * G217;
	const double G219 = G141 + G168;
	const double G220 = G219 * G140;
	const double G221 = G220 + G220;
	const double G222 = G153 + G176;
	const double G223 = G222 * G152;
	const double G224 = G223 + G223;
	const double G225 = G221 + G224;
	const double G226 = X4 + G181;
	const double G227 = G226 + G226;
	const double G228 = G225 + G227;
	const double G229 = C1 * G228;
	const double G230 = G141 + G190;
	const double G231 = G230 * G140;
	const double G232 = G231 + G231;
	const double G233 = G153 + G198;
	const double G234 = G233 * G152;
	const double G235 = G234 + G234;
	const double G236 = G232 + G235;
	const double G237 = X4 + G203;
	const double G238 = G237 + G237;
	const double G239 = G236 + G238;
	const double G240 = C1 * G239;
	const double G241 = C1 * G166;
	const double G242 = G169 * G241;
	const double G243 = G242 + G242;
	const double G244 = C1 * G174;
	const double G245 = G177 * G244;
	const double G246 = G245 + G245;
	const double G247 = G243 + G246;
	const double G248 = C1 * G182;
	const double G249 = G248 + G248;
	const double G250 = G247 + G249;
	const double G251 = C1 * G250;
	const double G252 = G219 * G241;
	const double G253 = G252 + G252;
	const double G254 = G222 * G244;
	const double G255 = G254 + G254;
	const double G256 = G253 + G255;
	const double G257 = C1 * G226;
	const double G258 = G257 + G257;
	const double G259 = G256 + G258;
	const double G260 = C1 * G259;
	const double G261 = G167 + G190;
	const double G262 = G261 * G166;
	const double G263 = G262 + G262;
	const double G264 = G175 + G198;
	const double G265 = G264 * G174;
	const double G266 = G265 + G265;
	const double G267 = G263 + G266;
	const double G268 = X5 + G203;
	const double G269 = G268 + G268;
	const double G270 = G267 + G269;
	const double G271 = C1 * G270;
	const double G272 = C1 * G188;
	const double G273 = G191 * G272;
	const double G274 = G273 + G273;
	const double G275 = C1 * G196;
	const double G276 = G199 * G275;
	const double G277 = G276 + G276;
	const double G278 = G274 + G277;
	const double G279 = C1 * G204;
	const double G280 = G279 + G279;
	const double G281 = G278 + G280;
	const double G282 = C1 * G281;
	const double G283 = G230 * G272;
	const double G284 = G283 + G283;
	const double G285 = G233 * G275;
	const double G286 = G285 + G285;
	const double G287 = G284 + G286;
	const double G288 = C1 * G237;
	const double G289 = G288 + G288;
	const double G290 = G287 + G289;
	const double G291 = C1 * G290;
	const double G292 = G261 * G272;
	const double G293 = G292 + G292;
	const double G294 = G264 * G275;
	const double G295 = G294 + G294;
	const double G296 = G293 + G295;
	const double G297 = C1 * G268;
	const double G298 = G297 + G297;
	const double G299 = G296 + G298;
	const double G300 = C1 * G299;
	const double G301 = G1 * X29;
	const double G302 = X12 * X53;
	const double G303 = G301 + G302;
	const double G304 = X7 * G303;
	const double G305 = G1 * X30;
	const double G306 = X12 * X54;
	const double G307 = G305 + G306;
	const double G308 = X8 * G307;
	const double G309 = C1 * G308;
	const double G310 = G304 + G309;
	const double G311 = G310 * G303;
	const double G312 = G311 + G311;
	const double G313 = G1 * X33;
	const double G314 = X12 * X57;
	const double G315 = G313 + G314;
	const double G316 = X7 * G315;
	const double G317 = G1 * X34;
	const double G318 = X12 * X58;
	const double G319 = G317 + G318;
	const double G320 = X8 * G319;
	const double G321 = C1 * G320;
	const double G322 = G316 + G321;
	const double G323 = G322 * G315;
	const double G324 = G323 + G323;
	const double G325 = G312 + G324;
	const double G326 = C1 * X8;
	const double G327 = X7 + G326;
	const double G328 = G327 + G327;
	const double G329 = G325 + G328;
	const double G330 = C1 * G329;
	const double G331 = G1 * X31;
	const double G332 = X12 * X55;
	const double G333 = G331 + G332;
	const double G334 = X9 * G333;
	const double G335 = C1 * G334;
	const double G336 = G304 + G335;
	const double G337 = G336 * G303;
	const double G338 = G337 + G337;
	const double G339 = G1 * X35;
	const double G340 = X12 * X59;
	const double G341 = G339 + G340;
	const double G342 = X9 * G341;
	const double G343 = C1 * G342;
	const double G344 = G316 + G343;
	const double G345 = G344 * G315;
	const double G346 = G345 + G345;
	const double G347 = G338 + G346;
	const double G348 = C1 * X9;
	const double G349 = X7 + G348;
	const double G350 = G349 + G349;
	const double G351 = G347 + G350;
	const double G352 = C1 * G351;
	const double G353 = G1 * X32;
	const double G354 = X12 * X56;
	const double G355 = G353 + G354;
	const double G356 = X10 * G355;
	const double G357 = C1 * G356;
	const double G358 = G304 + G357;
	const double G359 = G358 * G303;
	const double G360 = G359 + G359;
	const double G361 = G1 * X36;
	const double G362 = X12 * X60;
	const double G363 = G361 + G362;
	const double G364 = G363 + X11;
	const double G365 = X10 * G364;
	const double G366 = C1 * G365;
	const double G367 = G316 + G366;
	const double G368 = G367 * G315;
	const double G369 = G368 + G368;
	const double G370 = G360 + G369;
	const double G371 = C1 * X10;
	const double G372 = X7 + G371;
	const double G373 = G372 + G372;
	const double G374 = G370 + G373;
	const double G375 = C1 * G374;
	const double G376 = C1 * G307;
	const double G377 = G310 * G376;
	const double G378 = G377 + G377;
	const double G379 = C1 * G319;
	const double G380 = G322 * G379;
	const double G381 = G380 + G380;
	const double G382 = G378 + G381;
	const double G383 = C1 * G327;
	const double G384 = G383 + G383;
	const double G385 = G382 + G384;
	const double G386 = C1 * G385;
	const double G387 = G308 + G335;
	const double G388 = G387 * G307;
	const double G389 = G388 + G388;
	const double G390 = G320 + G343;
	const double G391 = G390 * G319;
	const double G392 = G391 + G391;
	const double G393 = G389 + G392;
	const double G394 = X8 + G348;
	const double G395 = G394 + G394;
	const double G396 = G393 + G395;
	const double G397 = C1 * G396;
	const double G398 = G308 + G357;
	const double G399 = G398 * G307;
	const double G400 = G399 + G399;
	const double G401 = G320 + G366;
	const double G402 = G401 * G319;
	const double G403 = G402 + G402;
	const double G404 = G400 + G403;
	const double G405 = X8 + G371;
	const double G406 = G405 + G405;
	const double G407 = G404 + G406;
	const double G408 = C1 * G407;
	const double G409 = C1 * G333;
	const double G410 = G336 * G409;
	const double G411 = G410 + G410;
	const double G412 = C1 * G341;
	const double G413 = G344 * G412;
	const double G414 = G413 + G413;
	const double G415 = G411 + G414;
	const double G416 = C1 * G349;
	const double G417 = G416 + G416;
	const double G418 = G415 + G417;
	const double G419 = C1 * G418;
	const double G420 = G387 * G409;
	const double G421 = G420 + G420;
	const double G422 = G390 * G412;
	const double G423 = G422 + G422;
	const double G424 = G421 + G423;
	const double G425 = C1 * G394;
	const double G426 = G425 + G425;
	const double G427 = G424 + G426;
	const double G428 = C1 * G427;
	const double G429 = G334 + G357;
	const double G430 = G429 * G333;
	const double G431 = G430 + G430;
	const double G432 = G342 + G366;
	const double G433 = G432 * G341;
	const double G434 = G433 + G433;
	const double G435 = G431 + G434;
	const double G436 = X9 + G371;
	const double G437 = G436 + G436;
	const double G438 = G435 + G437;
	const double G439 = C1 * G438;
	const double G440 = C1 * G355;
	const double G441 = G358 * G440;
	const double G442 = G441 + G441;
	const double G443 = C1 * G364;
	const double G444 = G367 * G443;
	const double G445 = G444 + G444;
	const double G446 = G442 + G445;
	const double G447 = C1 * G372;
	const double G448 = G447 + G447;
	const double G449 = G446 + G448;
	const double G450 = C1 * G449;
	const double G451 = G398 * G440;
	const double G452 = G451 + G451;
	const double G453 = G401 * G443;
	const double G454 = G453 + G453;
	const double G455 = G452 + G454;
	const double G456 = C1 * G405;
	const double G457 = G456 + G456;
	const double G458 = G455 + G457;
	const double G459 = C1 * G458;
	const double G460 = G429 * G440;
	const double G461 = G460 + G460;
	const double G462 = G432 * G443;
	const double G463 = G462 + G462;
	const double G464 = G461 + G463;
	const double G465 = C1 * G436;
	const double G466 = G465 + G465;
	const double G467 = G464 + G466;
	const double G468 = C1 * G467;
	const double G469 = G367 * G371;
	const double G470 = G469 + G469;
	const double G471 = C1 * G470;
	const double G472 = G401 * G371;
	const double G473 = G472 + G472;
	const double G474 = C1 * G473;
	const double G475 = G432 * G371;
	const double G476 = G475 + G475;
	const double G477 = C1 * G476;
	const double G478 = G10 * G10;
	const double G479 = G22 * G22;
	const double G480 = G478 + G479;
	const double G481 = G28 * G28;
	const double G482 = G480 + G481;
	const double G483 = G143 * G143;
	const double G484 = G155 * G155;
	const double G485 = G483 + G484;
	const double G486 = G160 * G160;
	const double G487 = G485 + G486;
	const double G488 = C1 * G487;
	const double G489 = G482 + G488;
	const double G490 = G74 * G74;
	const double G491 = G78 * G78;
	const double G492 = G490 + G491;
	const double G493 = G83 * G83;
	const double G494 = G492 + G493;
	const double G495 = G169 * G169;
	const double G496 = G177 * G177;
	const double G497 = G495 + G496;
	const double G498 = G182 * G182;
	const double G499 = G497 + G498;
	const double G500 = C1 * G499;
	const double G501 = G494 + G500;
	const double G502 = G37 * G37;
	const double G503 = G45 * G45;
	const double G504 = G502 + G503;
	const double G505 = G50 * G50;
	const double G506 = G504 + G505;
	const double G507 = G219 * G219;
	const double G508 = G222 * G222;
	const double G509 = G507 + G508;
	const double G510 = G226 * G226;
	const double G511 = G509 + G510;
	const double G512 = C1 * G511;
	const double G513 = G506 + G512;
	const double G514 = G105 * G105;
	const double G515 = G109 * G109;
	const double G516 = G514 + G515;
	const double G517 = G114 * G114;
	const double G518 = G516 + G517;
	const double G519 = G191 * G191;
	const double G520 = G199 * G199;
	const double G521 = G519 + G520;
	const double G522 = G204 * G204;
	const double G523 = G521 + G522;
	const double G524 = C1 * G523;
	const double G525 = G518 + G524;
	const double G526 = G58 * G58;
	const double G527 = G66 * G66;
	const double G528 = G526 + G527;
	const double G529 = G71 * G71;
	const double G530 = G528 + G529;
	const double G531 = G230 * G230;
	const double G532 = G233 * G233;
	const double G533 = G531 + G532;
	const double G534 = G237 * G237;
	const double G535 = G533 + G534;
	const double G536 = C1 * G535;
	const double G537 = G530 + G536;
	const double G538 = G95 * G95;
	const double G539 = G98 * G98;
	const double G540 = G538 + G539;
	const double G541 = G102 * G102;
	const double G542 = G540 + G541;
	const double G543 = G261 * G261;
	const double G544 = G264 * G264;
	const double G545 = G543 + G544;
	const double G546 = G268 * G268;
	const double G547 = G545 + G546;
	const double G548 = C1 * G547;
	const double G549 = G542 + G548;
	const double G550 = G310 * G310;
	const double G551 = G322 * G322;
	const double G552 = G550 + G551;
	const double G553 = G327 * G327;
	const double G554 = G552 + G553;
	const double G555 = C1 * G554;
	const double G556 = G482 + G555;
	const double G557 = G336 * G336;
	const double G558 = G344 * G344;
	const double G559 = G557 + G558;
	const double G560 = G349 * G349;
	const double G561 = G559 + G560;
	const double G562 = C1 * G561;
	const double G563 = G494 + G562;
	const double G564 = G387 * G387;
	const double G565 = G390 * G390;
	const double G566 = G564 + G565;
	const double G567 = G394 * G394;
	const double G568 = G566 + G567;
	const double G569 = C1 * G568;
	const double G570 = G506 + G569;
	const double G571 = G358 * G358;
	const double G572 = G367 * G367;
	const double G573 = G571 + G572;
	const double G574 = G372 * G372;
	const double G575 = G573 + G574;
	const double G576 = C1 * G575;
	const double G577 = G518 + G576;
	const double G578 = G398 * G398;
	const double G579 = G401 * G401;
	const double G580 = G578 + G579;
	const double G581 = G405 * G405;
	const double G582 = G580 + G581;
	const double G583 = C1 * G582;
	const double G584 = G530 + G583;
	const double G585 = G429 * G429;
	const double G586 = G432 * G432;
	const double G587 = G585 + G586;
	const double G588 = G436 * G436;
	const double G589 = G587 + G588;
	const double G590 = C1 * G589;
	const double G591 = G542 + G590;
	y[0] = G31;
	y[1] = C3;
	y[2] = G52;
	y[3] = C3;
	y[4] = G73;
	y[5] = C3;
	y[6] = G31;
	y[7] = C3;
	y[8] = G52;
	y[9] = C3;
	y[10] = G73;
	y[11] = C3;
	y[12] = C3;
	y[13] = G86;
	y[14] = G94;
	y[15] = C3;
	y[16] = C3;
	y[17] = G104;
	y[18] = C3;
	y[19] = G86;
	y[20] = G94;
	y[21] = C3;
	y[22] = C3;
	y[23] = G104;
	y[24] = C3;
	y[25] = C3;
	y[26] = C3;
	y[27] = G117;
	y[28] = G125;
	y[29] = G133;
	y[30] = C3;
	y[31] = C3;
	y[32] = C3;
	y[33] = G117;
	y[34] = G125;
	y[35] = G133;
	y[36] = G163;
	y[37] = G185;
	y[38] = C3;
	y[39] = G207;
	y[40] = C3;
	y[41] = C3;
	y[42] = C3;
	y[43] = C3;
	y[44] = C3;
	y[45] = C3;
	y[46] = C3;
	y[47] = C3;
	y[48] = G218;
	y[49] = C3;
	y[50] = G229;
	y[51] = C3;
	y[52] = G240;
	y[53] = C3;
	y[54] = C3;
	y[55] = C3;
	y[56] = C3;
	y[57] = C3;
	y[58] = C3;
	y[59] = C3;
	y[60] = C3;
	y[61] = G251;
	y[62] = G260;
	y[63] = C3;
	y[64] = C3;
	y[65] = G271;
	y[66] = C3;
	y[67] = C3;
	y[68] = C3;
	y[69] = C3;
	y[70] = C3;
	y[71] = C3;
	y[72] = C3;
	y[73] = C3;
	y[74] = C3;
	y[75] = G282;
	y[76] = G291;
	y[77] = G300;
	y[78] = C3;
	y[79] = C3;
	y[80] = C3;
	y[81] = C3;
	y[82] = C3;
	y[83] = C3;
	y[84] = C3;
	y[85] = C3;
	y[86] = C3;
	y[87] = C3;
	y[88] = C3;
	y[89] = C3;
	y[90] = G330;
	y[91] = G352;
	y[92] = C3;
	y[93] = G375;
	y[94] = C3;
	y[95] = C3;
	y[96] = C3;
	y[97] = C3;
	y[98] = C3;
	y[99] = C3;
	y[100] = C3;
	y[101] = C3;
	y[102] = G386;
	y[103] = C3;
	y[104] = G397;
	y[105] = C3;
	y[106] = G408;
	y[107] = C3;
	y[108] = C3;
	y[109] = C3;
	y[110] = C3;
	y[111] = C3;
	y[112] = C3;
	y[113] = C3;
	y[114] = C3;
	y[115] = G419;
	y[116] = G428;
	y[117] = C3;
	y[118] = C3;
	y[119] = G439;
	y[120] = C3;
	y[121] = C3;
	y[122] = C3;
	y[123] = C3;
	y[124] = C3;
	y[125] = C3;
	y[126] = C3;
	y[127] = C3;
	y[128] = C3;
	y[129] = G450;
	y[130] = G459;
	y[131] = G468;
	y[132] = C3;
	y[133] = C3;
	y[134] = C3;
	y[135] = C3;
	y[136] = C3;
	y[137] = C3;
	y[138] = C3;
	y[139] = C3;
	y[140] = C3;
	y[141] = G471;
	y[142] = G474;
	y[143] = G477;

	y[144] = -G489;
	y[145] = -G501;
	y[146] = -G513;
	y[147] = -G525;
	y[148] = -G537;
	y[149] = -G549;
	y[150] = -G556;
	y[151] = -G563;
	y[152] = -G570;
	y[153] = -G577;
	y[154] = -G584;
	y[155] = -G591;		

}

//THE FUNCTION RESPONSIBLE FOR HOMOTOPY CONTINUATION TRACKING
int track(const struct track_settings s, const double s_sols[9], const double params[40], double solution[9], int * num_st)
{
	const unsigned int nve = 12;
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
	Eigen::Map<const Eigen::Matrix<double, nve, 1>, Eigen::Aligned > bb(RHS);
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
			const double c00 = AA(2,0)-AA(2,4)*AA(0,0)/AA(0,4);
			const double c01 = AA(2,1)-AA(2,5)*AA(1,1)/AA(1,5);
			const double c03 = -AA(2,4)*AA(0,3)/AA(0,4)-AA(2,5)*AA(1,3)/AA(1,5);
			const double d0 = bb(2) - (AA(2,4)/AA(0,4))*bb(0) - (AA(2,5)/AA(1,5))*bb(1); 

			const double c10 = AA(4,0)-AA(4,4)*AA(0,0)/AA(0,4);
			const double c12 = AA(4,2)-AA(4,6)*AA(3,2)/AA(3,6);
			const double c13 = -AA(4,4)*AA(0,3)/AA(0,4)-AA(4,6)*AA(3,3)/AA(3,6);
			const double d1 = bb(4) - (AA(4,4)/AA(0,4))*bb(0) - (AA(4,6)/AA(3,6))*bb(3);

			const double c21 = AA(5,1)-AA(5,5)*AA(1,1)/AA(1,5);
			const double c22 = AA(5,2)-AA(5,6)*AA(3,2)/AA(3,6);
			const double c23 = -AA(5,5)*AA(1,3)/AA(1,5)-AA(5,6)*AA(3,3)/AA(3,6);
			const double d2 = bb(5) - (AA(5,5)/AA(1,5))*bb(1) - (AA(5,6)/AA(3,6))*bb(3);

			const double e33 = -c21*c03/c01 - c22*c13/c12 + c23;
			const double e30 = c21*c00/c01 + c22*c10/c12;
			const double f3 = d2 - (c21/c01)*d0 - (c22/c12)*d1;

			const double e1 = -c00/c01 - (c03*e30)/(c01*e33);
			const double f1 = d0/c01 - (c03*f3)/(c01*e33);
			const double e2 = -c10/c12 - (c13*e30)/(c12*e33);
			const double f2 = d1/c12 - (c13*f3)/(c12*e33);

			const double c47 = -AA(8,8)*AA(6,7)/AA(6,8) - AA(8,9)*AA(7,7)/AA(7,9);
			const double c40 = -AA(8,0) - AA(8,1)*e1 + AA(8,8)*AA(6,0)/AA(6,8) + AA(8,9)*AA(7,1)*e1/AA(7,9);
			const double d4 = bb(8) - AA(8,1)*f1 - (AA(8,8)/AA(6,8))*bb(6) - (AA(8,9)/AA(7,9))*(bb(7)-AA(7,1)*f1);

			const double c9 = AA(9,2)*e2 + AA(9,7)*c40/c47;
			const double d9 = bb(9) - AA(9,2)*f2 - AA(9,7)*d4/c47;

			const double c010 = AA(10,0) + AA(10,2)*e2 - AA(10,8)*AA(6,0)/AA(6,8) - AA(10,8)*AA(6,7)*c40/(AA(6,8)*c47);
			const double d010 = bb(10) - AA(10,2)*f2 - (AA(10,8)/AA(6,8))*bb(6) + (AA(10,8)*AA(6,7)*d4)/(AA(6,8)*c47);

			const double c011 = AA(11,1)*e1 + AA(11,2)*e2 - AA(11,9)*AA(7,1)*e1/AA(7,9) - AA(11,9)*AA(7,7)*c40/(AA(7,9)*c47);
			const double d011 = bb(11) - AA(11,1)*f1 - AA(11,2)*f2 - AA(11,9)*bb(7)/AA(7,9) + AA(11,9)*AA(7,1)*f1/AA(7,9) + (AA(11,9)*AA(7,7)*d4)/(AA(7,9)*c47);

			const double e010 = c010 - AA(10,11)*c9/AA(9,11);
			const double e1010 = AA(10,10)-AA(10,11)*AA(9,10)/AA(9,11);
			const double f10 = d010-AA(10,11)*d9/AA(9,11);

			const double g0 = c011 - AA(11,10)*e010/e1010 - AA(11,11)*c9/AA(9,11) + AA(11,11)*AA(9,10)*e010/(AA(9,11)*e1010);
			const double h0 = d011 - AA(11,10)*f10/e1010 - AA(11,11)*d9/AA(9,11) + AA(11,11)*AA(9,10)*f10/(AA(9,11)*e1010);

		  	dxA[0] = h0/g0;
		  	
		  	dxA[10] = (f10-e010*dxA[0])/e1010;
		  	dxA[11] = (d9-c9*dxA[0]-AA(9,10)*dxA[10])/AA(9,11);

		  	dxA[7] = (c40*dxA[0]+d4)/c47;
		  	dxA[2] = e2*dxA[0]+f2;
		  	dxA[1] = e1*dxA[0]+f1;
		  	dxA[3] = (e30*dxA[0]+f3)/e33;

		  	dxA[4] = (bb(0)-AA(0,0)*dxA[0]-AA(0,3)*dxA[3])/AA(0,4);
		  	dxA[5] = (bb(1)-AA(1,1)*dxA[1]-AA(1,3)*dxA[3])/AA(1,5);
		  	dxA[6] = (bb(3)-AA(3,2)*dxA[2]-AA(3,3)*dxA[3])/AA(3,6);
		  	dxA[8] = (bb(6)-AA(6,0)*dxA[0]-AA(6,7)*dxA[7])/AA(6,8);
		  	dxA[9] = (bb(7)-AA(7,1)*dxA[1]-AA(7,7)*dxA[7])/AA(7,9);
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
			const double c00 = AA(2,0)-AA(2,4)*AA(0,0)/AA(0,4);
			const double c01 = AA(2,1)-AA(2,5)*AA(1,1)/AA(1,5);
			const double c03 = -AA(2,4)*AA(0,3)/AA(0,4)-AA(2,5)*AA(1,3)/AA(1,5);
			const double d0 = bb(2) - (AA(2,4)/AA(0,4))*bb(0) - (AA(2,5)/AA(1,5))*bb(1); 

			const double c10 = AA(4,0)-AA(4,4)*AA(0,0)/AA(0,4);
			const double c12 = AA(4,2)-AA(4,6)*AA(3,2)/AA(3,6);
			const double c13 = -AA(4,4)*AA(0,3)/AA(0,4)-AA(4,6)*AA(3,3)/AA(3,6);
			const double d1 = bb(4) - (AA(4,4)/AA(0,4))*bb(0) - (AA(4,6)/AA(3,6))*bb(3);

			const double c21 = AA(5,1)-AA(5,5)*AA(1,1)/AA(1,5);
			const double c22 = AA(5,2)-AA(5,6)*AA(3,2)/AA(3,6);
			const double c23 = -AA(5,5)*AA(1,3)/AA(1,5)-AA(5,6)*AA(3,3)/AA(3,6);
			const double d2 = bb(5) - (AA(5,5)/AA(1,5))*bb(1) - (AA(5,6)/AA(3,6))*bb(3);

			const double e33 = -c21*c03/c01 - c22*c13/c12 + c23;
			const double e30 = c21*c00/c01 + c22*c10/c12;
			const double f3 = d2 - (c21/c01)*d0 - (c22/c12)*d1;

			const double e1 = -c00/c01 - (c03*e30)/(c01*e33);
			const double f1 = d0/c01 - (c03*f3)/(c01*e33);
			const double e2 = -c10/c12 - (c13*e30)/(c12*e33);
			const double f2 = d1/c12 - (c13*f3)/(c12*e33);

			const double c47 = -AA(8,8)*AA(6,7)/AA(6,8) - AA(8,9)*AA(7,7)/AA(7,9);
			const double c40 = -AA(8,0) - AA(8,1)*e1 + AA(8,8)*AA(6,0)/AA(6,8) + AA(8,9)*AA(7,1)*e1/AA(7,9);
			const double d4 = bb(8) - AA(8,1)*f1 - (AA(8,8)/AA(6,8))*bb(6) - (AA(8,9)/AA(7,9))*(bb(7)-AA(7,1)*f1);

			const double c9 = AA(9,2)*e2 + AA(9,7)*c40/c47;
			const double d9 = bb(9) - AA(9,2)*f2 - AA(9,7)*d4/c47;

			const double c010 = AA(10,0) + AA(10,2)*e2 - AA(10,8)*AA(6,0)/AA(6,8) - AA(10,8)*AA(6,7)*c40/(AA(6,8)*c47);
			const double d010 = bb(10) - AA(10,2)*f2 - (AA(10,8)/AA(6,8))*bb(6) + (AA(10,8)*AA(6,7)*d4)/(AA(6,8)*c47);

			const double c011 = AA(11,1)*e1 + AA(11,2)*e2 - AA(11,9)*AA(7,1)*e1/AA(7,9) - AA(11,9)*AA(7,7)*c40/(AA(7,9)*c47);
			const double d011 = bb(11) - AA(11,1)*f1 - AA(11,2)*f2 - AA(11,9)*bb(7)/AA(7,9) + AA(11,9)*AA(7,1)*f1/AA(7,9) + (AA(11,9)*AA(7,7)*d4)/(AA(7,9)*c47);

			const double e010 = c010 - AA(10,11)*c9/AA(9,11);
			const double e1010 = AA(10,10)-AA(10,11)*AA(9,10)/AA(9,11);
			const double f10 = d010-AA(10,11)*d9/AA(9,11);

			const double g0 = c011 - AA(11,10)*e010/e1010 - AA(11,11)*c9/AA(9,11) + AA(11,11)*AA(9,10)*e010/(AA(9,11)*e1010);
			const double h0 = d011 - AA(11,10)*f10/e1010 - AA(11,11)*d9/AA(9,11) + AA(11,11)*AA(9,10)*f10/(AA(9,11)*e1010);

			dxB[0] = h0/g0;
      	
			dxB[10] = (f10-e010*dxB[0])/e1010;
			dxB[11] = (d9-c9*dxB[0]-AA(9,10)*dxB[10])/AA(9,11);

			dxB[7] = (c40*dxB[0]+d4)/c47;
			dxB[2] = e2*dxB[0]+f2;
			dxB[1] = e1*dxB[0]+f1;
			dxB[3] = (e30*dxB[0]+f3)/e33;

			dxB[4] = (bb(0)-AA(0,0)*dxB[0]-AA(0,3)*dxB[3])/AA(0,4);
			dxB[5] = (bb(1)-AA(1,1)*dxB[1]-AA(1,3)*dxB[3])/AA(1,5);
			dxB[6] = (bb(3)-AA(3,2)*dxB[2]-AA(3,3)*dxB[3])/AA(3,6);
			dxB[8] = (bb(6)-AA(6,0)*dxB[0]-AA(6,7)*dxB[7])/AA(6,8);
			dxB[9] = (bb(7)-AA(7,1)*dxB[1]-AA(7,7)*dxB[7])/AA(7,9);
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
		  	const double c00 = AA(2,0)-AA(2,4)*AA(0,0)/AA(0,4);
			const double c01 = AA(2,1)-AA(2,5)*AA(1,1)/AA(1,5);
			const double c03 = -AA(2,4)*AA(0,3)/AA(0,4)-AA(2,5)*AA(1,3)/AA(1,5);
			const double d0 = bb(2) - (AA(2,4)/AA(0,4))*bb(0) - (AA(2,5)/AA(1,5))*bb(1); 

			const double c10 = AA(4,0)-AA(4,4)*AA(0,0)/AA(0,4);
			const double c12 = AA(4,2)-AA(4,6)*AA(3,2)/AA(3,6);
			const double c13 = -AA(4,4)*AA(0,3)/AA(0,4)-AA(4,6)*AA(3,3)/AA(3,6);
			const double d1 = bb(4) - (AA(4,4)/AA(0,4))*bb(0) - (AA(4,6)/AA(3,6))*bb(3);

			const double c21 = AA(5,1)-AA(5,5)*AA(1,1)/AA(1,5);
			const double c22 = AA(5,2)-AA(5,6)*AA(3,2)/AA(3,6);
			const double c23 = -AA(5,5)*AA(1,3)/AA(1,5)-AA(5,6)*AA(3,3)/AA(3,6);
			const double d2 = bb(5) - (AA(5,5)/AA(1,5))*bb(1) - (AA(5,6)/AA(3,6))*bb(3);

			const double e33 = -c21*c03/c01 - c22*c13/c12 + c23;
			const double e30 = c21*c00/c01 + c22*c10/c12;
			const double f3 = d2 - (c21/c01)*d0 - (c22/c12)*d1;

			const double e1 = -c00/c01 - (c03*e30)/(c01*e33);
			const double f1 = d0/c01 - (c03*f3)/(c01*e33);
			const double e2 = -c10/c12 - (c13*e30)/(c12*e33);
			const double f2 = d1/c12 - (c13*f3)/(c12*e33);

			const double c47 = -AA(8,8)*AA(6,7)/AA(6,8) - AA(8,9)*AA(7,7)/AA(7,9);
			const double c40 = -AA(8,0) - AA(8,1)*e1 + AA(8,8)*AA(6,0)/AA(6,8) + AA(8,9)*AA(7,1)*e1/AA(7,9);
			const double d4 = bb(8) - AA(8,1)*f1 - (AA(8,8)/AA(6,8))*bb(6) - (AA(8,9)/AA(7,9))*(bb(7)-AA(7,1)*f1);

			const double c9 = AA(9,2)*e2 + AA(9,7)*c40/c47;
			const double d9 = bb(9) - AA(9,2)*f2 - AA(9,7)*d4/c47;

			const double c010 = AA(10,0) + AA(10,2)*e2 - AA(10,8)*AA(6,0)/AA(6,8) - AA(10,8)*AA(6,7)*c40/(AA(6,8)*c47);
			const double d010 = bb(10) - AA(10,2)*f2 - (AA(10,8)/AA(6,8))*bb(6) + (AA(10,8)*AA(6,7)*d4)/(AA(6,8)*c47);

			const double c011 = AA(11,1)*e1 + AA(11,2)*e2 - AA(11,9)*AA(7,1)*e1/AA(7,9) - AA(11,9)*AA(7,7)*c40/(AA(7,9)*c47);
			const double d011 = bb(11) - AA(11,1)*f1 - AA(11,2)*f2 - AA(11,9)*bb(7)/AA(7,9) + AA(11,9)*AA(7,1)*f1/AA(7,9) + (AA(11,9)*AA(7,7)*d4)/(AA(7,9)*c47);

			const double e010 = c010 - AA(10,11)*c9/AA(9,11);
			const double e1010 = AA(10,10)-AA(10,11)*AA(9,10)/AA(9,11);
			const double f10 = d010-AA(10,11)*d9/AA(9,11);

			const double g0 = c011 - AA(11,10)*e010/e1010 - AA(11,11)*c9/AA(9,11) + AA(11,11)*AA(9,10)*e010/(AA(9,11)*e1010);
			const double h0 = d011 - AA(11,10)*f10/e1010 - AA(11,11)*d9/AA(9,11) + AA(11,11)*AA(9,10)*f10/(AA(9,11)*e1010);

			dxB[0] = h0/g0;

			dxB[10] = (f10-e010*dxB[0])/e1010;
			dxB[11] = (d9-c9*dxB[0]-AA(9,10)*dxB[10])/AA(9,11);

			dxB[7] = (c40*dxB[0]+d4)/c47;
			dxB[2] = e2*dxB[0]+f2;
			dxB[1] = e1*dxB[0]+f1;
			dxB[3] = (e30*dxB[0]+f3)/e33;

			dxB[4] = (bb(0)-AA(0,0)*dxB[0]-AA(0,3)*dxB[3])/AA(0,4);
			dxB[5] = (bb(1)-AA(1,1)*dxB[1]-AA(1,3)*dxB[3])/AA(1,5);
			dxB[6] = (bb(3)-AA(3,2)*dxB[2]-AA(3,3)*dxB[3])/AA(3,6);
			dxB[8] = (bb(6)-AA(6,0)*dxB[0]-AA(6,7)*dxB[7])/AA(6,8);
			dxB[9] = (bb(7)-AA(7,1)*dxB[1]-AA(7,7)*dxB[7])/AA(7,9);
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
		  	const double c00 = AA(2,0)-AA(2,4)*AA(0,0)/AA(0,4);
			const double c01 = AA(2,1)-AA(2,5)*AA(1,1)/AA(1,5);
			const double c03 = -AA(2,4)*AA(0,3)/AA(0,4)-AA(2,5)*AA(1,3)/AA(1,5);
			const double d0 = bb(2) - (AA(2,4)/AA(0,4))*bb(0) - (AA(2,5)/AA(1,5))*bb(1); 

			const double c10 = AA(4,0)-AA(4,4)*AA(0,0)/AA(0,4);
			const double c12 = AA(4,2)-AA(4,6)*AA(3,2)/AA(3,6);
			const double c13 = -AA(4,4)*AA(0,3)/AA(0,4)-AA(4,6)*AA(3,3)/AA(3,6);
			const double d1 = bb(4) - (AA(4,4)/AA(0,4))*bb(0) - (AA(4,6)/AA(3,6))*bb(3);

			const double c21 = AA(5,1)-AA(5,5)*AA(1,1)/AA(1,5);
			const double c22 = AA(5,2)-AA(5,6)*AA(3,2)/AA(3,6);
			const double c23 = -AA(5,5)*AA(1,3)/AA(1,5)-AA(5,6)*AA(3,3)/AA(3,6);
			const double d2 = bb(5) - (AA(5,5)/AA(1,5))*bb(1) - (AA(5,6)/AA(3,6))*bb(3);

			const double e33 = -c21*c03/c01 - c22*c13/c12 + c23;
			const double e30 = c21*c00/c01 + c22*c10/c12;
			const double f3 = d2 - (c21/c01)*d0 - (c22/c12)*d1;

			const double e1 = -c00/c01 - (c03*e30)/(c01*e33);
			const double f1 = d0/c01 - (c03*f3)/(c01*e33);
			const double e2 = -c10/c12 - (c13*e30)/(c12*e33);
			const double f2 = d1/c12 - (c13*f3)/(c12*e33);

			const double c47 = -AA(8,8)*AA(6,7)/AA(6,8) - AA(8,9)*AA(7,7)/AA(7,9);
			const double c40 = -AA(8,0) - AA(8,1)*e1 + AA(8,8)*AA(6,0)/AA(6,8) + AA(8,9)*AA(7,1)*e1/AA(7,9);
			const double d4 = bb(8) - AA(8,1)*f1 - (AA(8,8)/AA(6,8))*bb(6) - (AA(8,9)/AA(7,9))*(bb(7)-AA(7,1)*f1);

			const double c9 = AA(9,2)*e2 + AA(9,7)*c40/c47;
			const double d9 = bb(9) - AA(9,2)*f2 - AA(9,7)*d4/c47;

			const double c010 = AA(10,0) + AA(10,2)*e2 - AA(10,8)*AA(6,0)/AA(6,8) - AA(10,8)*AA(6,7)*c40/(AA(6,8)*c47);
			const double d010 = bb(10) - AA(10,2)*f2 - (AA(10,8)/AA(6,8))*bb(6) + (AA(10,8)*AA(6,7)*d4)/(AA(6,8)*c47);

			const double c011 = AA(11,1)*e1 + AA(11,2)*e2 - AA(11,9)*AA(7,1)*e1/AA(7,9) - AA(11,9)*AA(7,7)*c40/(AA(7,9)*c47);
			const double d011 = bb(11) - AA(11,1)*f1 - AA(11,2)*f2 - AA(11,9)*bb(7)/AA(7,9) + AA(11,9)*AA(7,1)*f1/AA(7,9) + (AA(11,9)*AA(7,7)*d4)/(AA(7,9)*c47);

			const double e010 = c010 - AA(10,11)*c9/AA(9,11);
			const double e1010 = AA(10,10)-AA(10,11)*AA(9,10)/AA(9,11);
			const double f10 = d010-AA(10,11)*d9/AA(9,11);

			const double g0 = c011 - AA(11,10)*e010/e1010 - AA(11,11)*c9/AA(9,11) + AA(11,11)*AA(9,10)*e010/(AA(9,11)*e1010);
			const double h0 = d011 - AA(11,10)*f10/e1010 - AA(11,11)*d9/AA(9,11) + AA(11,11)*AA(9,10)*f10/(AA(9,11)*e1010);

			dxB[0] = h0/g0;

			dxB[10] = (f10-e010*dxB[0])/e1010;
			dxB[11] = (d9-c9*dxB[0]-AA(9,10)*dxB[10])/AA(9,11);

			dxB[7] = (c40*dxB[0]+d4)/c47;
			dxB[2] = e2*dxB[0]+f2;
			dxB[1] = e1*dxB[0]+f1;
			dxB[3] = (e30*dxB[0]+f3)/e33;

			dxB[4] = (bb(0)-AA(0,0)*dxB[0]-AA(0,3)*dxB[3])/AA(0,4);
			dxB[5] = (bb(1)-AA(1,1)*dxB[1]-AA(1,3)*dxB[3])/AA(1,5);
			dxB[6] = (bb(3)-AA(3,2)*dxB[2]-AA(3,3)*dxB[3])/AA(3,6);
			dxB[8] = (bb(6)-AA(6,0)*dxB[0]-AA(6,7)*dxB[7])/AA(6,8);
			dxB[9] = (bb(7)-AA(7,1)*dxB[1]-AA(7,7)*dxB[7])/AA(7,9);
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

			const double c00 = AA(2,0)-AA(2,4)*AA(0,0)/AA(0,4);
			const double c01 = AA(2,1)-AA(2,5)*AA(1,1)/AA(1,5);
			const double c03 = -AA(2,4)*AA(0,3)/AA(0,4)-AA(2,5)*AA(1,3)/AA(1,5);
			const double d0 = bb(2) - (AA(2,4)/AA(0,4))*bb(0) - (AA(2,5)/AA(1,5))*bb(1); 

			const double c10 = AA(4,0)-AA(4,4)*AA(0,0)/AA(0,4);
			const double c12 = AA(4,2)-AA(4,6)*AA(3,2)/AA(3,6);
			const double c13 = -AA(4,4)*AA(0,3)/AA(0,4)-AA(4,6)*AA(3,3)/AA(3,6);
			const double d1 = bb(4) - (AA(4,4)/AA(0,4))*bb(0) - (AA(4,6)/AA(3,6))*bb(3);

			const double c21 = AA(5,1)-AA(5,5)*AA(1,1)/AA(1,5);
			const double c22 = AA(5,2)-AA(5,6)*AA(3,2)/AA(3,6);
			const double c23 = -AA(5,5)*AA(1,3)/AA(1,5)-AA(5,6)*AA(3,3)/AA(3,6);
			const double d2 = bb(5) - (AA(5,5)/AA(1,5))*bb(1) - (AA(5,6)/AA(3,6))*bb(3);

			const double e33 = -c21*c03/c01 - c22*c13/c12 + c23;
			const double e30 = c21*c00/c01 + c22*c10/c12;
			const double f3 = d2 - (c21/c01)*d0 - (c22/c12)*d1;

			const double e1 = -c00/c01 - (c03*e30)/(c01*e33);
			const double f1 = d0/c01 - (c03*f3)/(c01*e33);
			const double e2 = -c10/c12 - (c13*e30)/(c12*e33);
			const double f2 = d1/c12 - (c13*f3)/(c12*e33);

			const double c47 = -AA(8,8)*AA(6,7)/AA(6,8) - AA(8,9)*AA(7,7)/AA(7,9);
			const double c40 = -AA(8,0) - AA(8,1)*e1 + AA(8,8)*AA(6,0)/AA(6,8) + AA(8,9)*AA(7,1)*e1/AA(7,9);
			const double d4 = bb(8) - AA(8,1)*f1 - (AA(8,8)/AA(6,8))*bb(6) - (AA(8,9)/AA(7,9))*(bb(7)-AA(7,1)*f1);

			const double c9 = AA(9,2)*e2 + AA(9,7)*c40/c47;
			const double d9 = bb(9) - AA(9,2)*f2 - AA(9,7)*d4/c47;

			const double c010 = AA(10,0) + AA(10,2)*e2 - AA(10,8)*AA(6,0)/AA(6,8) - AA(10,8)*AA(6,7)*c40/(AA(6,8)*c47);
			const double d010 = bb(10) - AA(10,2)*f2 - (AA(10,8)/AA(6,8))*bb(6) + (AA(10,8)*AA(6,7)*d4)/(AA(6,8)*c47);

			const double c011 = AA(11,1)*e1 + AA(11,2)*e2 - AA(11,9)*AA(7,1)*e1/AA(7,9) - AA(11,9)*AA(7,7)*c40/(AA(7,9)*c47);
			const double d011 = bb(11) - AA(11,1)*f1 - AA(11,2)*f2 - AA(11,9)*bb(7)/AA(7,9) + AA(11,9)*AA(7,1)*f1/AA(7,9) + (AA(11,9)*AA(7,7)*d4)/(AA(7,9)*c47);

			const double e010 = c010 - AA(10,11)*c9/AA(9,11);
			const double e1010 = AA(10,10)-AA(10,11)*AA(9,10)/AA(9,11);
			const double f10 = d010-AA(10,11)*d9/AA(9,11);

			const double g0 = c011 - AA(11,10)*e010/e1010 - AA(11,11)*c9/AA(9,11) + AA(11,11)*AA(9,10)*e010/(AA(9,11)*e1010);
			const double h0 = d011 - AA(11,10)*f10/e1010 - AA(11,11)*d9/AA(9,11) + AA(11,11)*AA(9,10)*f10/(AA(9,11)*e1010);

			dx[0] = h0/g0;

			dx[10] = (f10-e010*dx[0])/e1010;
			dx[11] = (d9-c9*dx[0]-AA(9,10)*dx[10])/AA(9,11);

			dx[7] = (c40*dx[0]+d4)/c47;
			dx[2] = e2*dx[0]+f2;
			dx[1] = e1*dx[0]+f1;
			dx[3] = (e30*dx[0]+f3)/e33;

			dx[4] = (bb(0)-AA(0,0)*dx[0]-AA(0,3)*dx[3])/AA(0,4);
			dx[5] = (bb(1)-AA(1,1)*dx[1]-AA(1,3)*dx[3])/AA(1,5);
			dx[6] = (bb(3)-AA(3,2)*dx[2]-AA(3,3)*dx[3])/AA(3,6);
			dx[8] = (bb(6)-AA(6,0)*dx[0]-AA(6,7)*dx[7])/AA(6,8);
			dx[9] = (bb(7)-AA(7,1)*dx[1]-AA(7,7)*dx[7])/AA(7,9);

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
