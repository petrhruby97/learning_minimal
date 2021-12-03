#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <Eigen/Core>
#include <numeric>
#include <math.h>
#include <chrono>

//#include <homotopy.hxx>
//#include <homotopy_ransac.hxx>
#include <homotopy_nn_ransac.hxx>
//#include <homotopy_nn_multi_ransac.hxx>
//#include <nister_ransac.hxx>

void load_anchors(std::string model_dir, std::vector< std::vector<double> > &anchors, std::vector< std::vector<double> > &starts)
{
	//open the file
	std::ifstream fa;
	fa.open(model_dir + "/anchors.txt");
	int m;
	fa >> m;

	anchors = std::vector< std::vector<double> >(m);
	starts = std::vector< std::vector<double> >(m);
	
	//load the anchors
	for(int i=0;i<m;++i)
	{
		//load the points
		std::vector<double> c_anchor(20);
		for(int j=0;j<20;j++)
		{
			double u;
			fa >> u;
			c_anchor[j] = u;
		}
		anchors[i] = c_anchor;

		//load the depths and convert them to the solution
		double first_depth;
		fa >> first_depth;
		std::vector<double> c_start(9);
		for(int j=0;j<9;j++)
		{
			double u;
			fa >> u;
			c_start[j] = u/first_depth;
		}
		starts[i] = c_start;
	}
}

void load_pair(std::ifstream &match_f, std::ifstream &conf_f, unsigned int &size, std::vector<Eigen::Vector2d> &points1, std::vector<Eigen::Vector2d> &points2)
{
	match_f.read(reinterpret_cast<char*>(&size), sizeof(unsigned int));

	//load the confidence
	std::vector<bool> conf = std::vector<bool>(size);
	for(int i=0;i<size;++i)
	{
		double num;
		conf_f.read(reinterpret_cast<char*>(&num), sizeof(double));
		if(num <= 0.85)
			conf[i] = 1;
		else
			conf[i] = 0;
	}
	int sum = std::accumulate(conf.begin(), conf.end(), 0);

	points1 = std::vector<Eigen::Vector2d>(sum);
	points2 = std::vector<Eigen::Vector2d>(sum);
	int pos = 0;

	for(int i=0;i<size;++i)
	{    		
		double num;
		//read the first point
		Eigen::Vector2d p1;
		match_f.read(reinterpret_cast<char*>(&num), sizeof(double));
		p1(0) = num;
		match_f.read(reinterpret_cast<char*>(&num), sizeof(double));
		p1(1) = num;

		//read the second point
		Eigen::Vector2d p2;
		match_f.read(reinterpret_cast<char*>(&num), sizeof(double));
		p2(0) = num;
		match_f.read(reinterpret_cast<char*>(&num), sizeof(double));
		p2(1) = num;

		if(conf[i])
		{
			points1[pos] = p1;
			points2[pos] = p2;
			++pos;
		}
	}
}

void load_K(std::ifstream &K_f, Eigen::Matrix3d &K1, Eigen::Matrix3d &K2)
{
	double k1;
	double k2;
	double k3;
	double k4;
	double k5;
	double k6;
	double k7;
	double k8;
	double k9;

	K_f.read(reinterpret_cast<char*>(&k1), sizeof(double));
	K_f.read(reinterpret_cast<char*>(&k2), sizeof(double));
	K_f.read(reinterpret_cast<char*>(&k3), sizeof(double));
	K_f.read(reinterpret_cast<char*>(&k4), sizeof(double));
	K_f.read(reinterpret_cast<char*>(&k5), sizeof(double));
	K_f.read(reinterpret_cast<char*>(&k6), sizeof(double));
	K_f.read(reinterpret_cast<char*>(&k7), sizeof(double));
	K_f.read(reinterpret_cast<char*>(&k8), sizeof(double));
	K_f.read(reinterpret_cast<char*>(&k9), sizeof(double));

	K1 << k1, k4, k7, k2, k5, k8, k3, k6, k9;

	//K1 = (Mat_<double>(3,3) << k1, k4, k7, k2, k5, k8, k3, k6, k9);
	

	K_f.read(reinterpret_cast<char*>(&k1), sizeof(double));
	K_f.read(reinterpret_cast<char*>(&k2), sizeof(double));
	K_f.read(reinterpret_cast<char*>(&k3), sizeof(double));
	K_f.read(reinterpret_cast<char*>(&k4), sizeof(double));
	K_f.read(reinterpret_cast<char*>(&k5), sizeof(double));
	K_f.read(reinterpret_cast<char*>(&k6), sizeof(double));
	K_f.read(reinterpret_cast<char*>(&k7), sizeof(double));
	K_f.read(reinterpret_cast<char*>(&k8), sizeof(double));
	K_f.read(reinterpret_cast<char*>(&k9), sizeof(double));

	K2 << k1, k4, k7, k2, k5, k8, k3, k6, k9;

	//K2 = (Mat_<double>(3,3) << k1, k4, k7, k2, k5, k8, k3, k6, k9);
}

void load_Rt(std::ifstream &R_f, std::ifstream &t_f, Eigen::Matrix3d &R, Eigen::Vector3d &t)
{
	double k1;
	double k2;
	double k3;
	double k4;
	double k5;
	double k6;
	double k7;
	double k8;
	double k9;

	R_f.read(reinterpret_cast<char*>(&k1), sizeof(double));
	R_f.read(reinterpret_cast<char*>(&k2), sizeof(double));
	R_f.read(reinterpret_cast<char*>(&k3), sizeof(double));
	R_f.read(reinterpret_cast<char*>(&k4), sizeof(double));
	R_f.read(reinterpret_cast<char*>(&k5), sizeof(double));
	R_f.read(reinterpret_cast<char*>(&k6), sizeof(double));
	R_f.read(reinterpret_cast<char*>(&k7), sizeof(double));
	R_f.read(reinterpret_cast<char*>(&k8), sizeof(double));
	R_f.read(reinterpret_cast<char*>(&k9), sizeof(double));

	//R = (Mat_<double>(3,3) << k1, k4, k7, k2, k5, k8, k3, k6, k9);
	R << k1, k4, k7, k2, k5, k8, k3, k6, k9;

	t_f.read(reinterpret_cast<char*>(&k1), sizeof(double));
	t_f.read(reinterpret_cast<char*>(&k2), sizeof(double));
	t_f.read(reinterpret_cast<char*>(&k3), sizeof(double));

	//t = (Mat_<double>(3,1) << k1, k2, k3);
	t << k1, k2, k3;
}

void load_NN(std::string model_dir/*, float * tr*/, std::vector<std::vector<float>> &ws, std::vector<std::vector<float>> &bs, std::vector<std::vector<float>> &ps, std::vector<int> &a_, std::vector<int> &b_)
{
	/*std::ifstream ft;
	ft.open(model_dir+"/transform.txt");
	for(int i=0;i<14*20;++i)
	{
		float u;
		ft >> u;
		tr[i] = u;
	}	
	ft.close();*/

	std::ifstream fnn;
	fnn.open(model_dir+"/nn.txt");
	int layers;
	fnn >> layers;
	ws = std::vector<std::vector<float>>(layers);
	bs = std::vector<std::vector<float>>(layers);
	ps = std::vector<std::vector<float>>(layers-1);
	a_ = std::vector<int>(layers);
	b_ = std::vector<int>(layers);
	for(int i=0;i<layers;++i)
	{
		int a;
		int b;
		fnn >> a;
		fnn >> b;
		a_[i] = a;
		b_[i] = b;

		std::vector<float> __attribute__((aligned(16))) cw(a*b);
		for(int j=0;j<a*b;++j)
		{
			float u;
			fnn >> u;
			cw[j] = u;
		}
		ws[i] = cw;
		
		fnn >> a;
		fnn >> b;
		std::vector<float> __attribute__((aligned(16))) cb(a);
		for(int j=0;j<a;++j)
		{
			float u;
			fnn >> u;
			cb[j] = u;
		}
		bs[i] = cb;

		if(i==layers-1)
			break;

		fnn >> a;
		fnn >> b;
		std::vector<float> __attribute__((aligned(16))) cp(a);
		for(int j=0;j<a;++j)
		{
			float u;
			fnn >> u;
			cp[j] = u;
		}
		ps[i] = cp;
	}
	fnn.close();
	
}

void calibrate_pts(Eigen::Matrix3d K, std::vector<Eigen::Vector2d> points, std::vector<Eigen::Vector2d> &c_points)
{	
	c_points = std::vector<Eigen::Vector2d>(points.size());
	for(int i=0;i<points.size();++i)
	{
		Eigen::Vector2d cal_p;
		cal_p(0) = (points[i](0)-K(0,2))/K(0,0);
		cal_p(1) = (points[i](1)-K(1,2))/K(1,1);

		c_points[i] = cal_p;
	}
}

Eigen::Vector4d R2q(Eigen::Matrix3d R)
{
	Eigen::Vector4d q;
	
	//construct the quaternion
	q(0) = R.trace()+1;
	q(1) = R(2,1) - R(1,2);
	q(2) = R(0,2) - R(2,0);
	q(3) = R(1,0) - R(0,1);	

	//normalize the quaternion
	q.normalize();

	return q;
}

double evaluate_R(Eigen::Matrix3d R, Eigen::Matrix3d gtR)
{
	//obtain the quaternion from both rotation matrices
	Eigen::Vector4d q = R2q(R);
	Eigen::Vector4d gtq = R2q(gtR);

	//compute the distance between the quaternions
	double eps = 1e-15;
	double dot = (q.transpose() * gtq);
	double loss_q = 1 - dot*dot;
	double err_q = std::acos(1-2*loss_q);

	return err_q;
}

double evaluate_t(Eigen::Vector3d t, Eigen::Vector3d gtT)
{
	double n1 = t.norm();
	double n2 = gtT.norm();
	double cos = (gtT.transpose() * t);
	cos = cos/(n1*n2);
	cos = std::abs(cos);
	double err = std::acos(cos);
	if(cos > 1)
		err = 0;
	return err;
}

int main(int argc, char* argv[])
{
	if(argc < 5)
	{
		std::cout << "Run as:\ntesting test_data_folder model_folder trainParam testParam\n";
		return 0;
	}

	std::string data_folder(argv[1]);
	std::string model_folder(argv[2]);
	std::string set_file(argv[3]);
	std::string test_set_file(argv[4]);

	//load the settings and store them to the track_settings structure (update to contain more settings)
	track_settings settings;
	bool succ_load = load_settings(set_file, settings);
	if(!succ_load) return 0;

	//load the settings and store them to the track_settings structure (update to contain more settings)
	test_settings ts;
	succ_load = load_test_settings(test_set_file, ts);
	if(!succ_load) return 0;
	
	//set the number of RANSAC samples per one problem
	if(argc >= 6)
	{
		unsigned samples = std::stoi(argv[5]);
		ts.num_samples_ = samples;
	}

	//open the test data files
	std::ifstream match_f;
	match_f.open(data_folder+"/matches.bin", std::ios::binary);
	std::ifstream conf_f;
	conf_f.open(data_folder+"/confidence.bin", std::ios::binary);
	std::ifstream K_f;
	K_f.open(data_folder+"/K.bin", std::ios::binary);

	//open the ground truth files
	std::ifstream R_f;
	R_f.open(data_folder+"/R.bin", std::ios::binary);
	std::ifstream t_f;
	t_f.open(data_folder+"/t.bin", std::ios::binary);

	unsigned int seqs; //number of camera pairs in the data set
	match_f.read(reinterpret_cast<char*>(&seqs), sizeof(unsigned int));

	//load the MLP
	std::vector<std::vector<float>> ws;
	std::vector<std::vector<float>> bs;
	std::vector<std::vector<float>> ps;
	std::vector<int> a_;
	std::vector<int> b_;
	load_NN(model_folder, ws, bs, ps, a_, b_);
	
	//load the anchors
	std::vector< std::vector<double> > anchors;
	std::vector< std::vector<double> > starts;
	load_anchors(model_folder, anchors, starts);

	//initialize the solver and the RANSAC (where the solver is one of the member variables and its function run is called in every step of RANSAC, another one is the max number of samples)
	HomotopyRansac r = HomotopyRansac(anchors, starts, settings);
	HomotopyNNRansac rn = HomotopyNNRansac(anchors, starts, ws, bs, ps, a_, b_, settings, ts.use_trash_);

	//initialize the statistics
	std::vector<long> times(seqs);
	std::vector<bool> mask(50000);
	double succ_tracks;
	int good = 0;
	double rot_avg = 0;
	double tran_avg = 0;
	double time_avg = 0;

	//TESTING LOOP
	for(int i=0;i<seqs;++i)
	{
		//load the matches
		unsigned int size;
		std::vector<Eigen::Vector2d> points1;
		std::vector<Eigen::Vector2d> points2;
		load_pair(match_f, conf_f, size, points1, points2);

		//load the calibration cameras
		Eigen::Matrix3d K1;
		Eigen::Matrix3d K2;
		load_K(K_f, K1, K2);

		//load the ground truth rotation and translation
		Eigen::Matrix3d Rgt;
		Eigen::Vector3d tgt;
		load_Rt(R_f, t_f, Rgt, tgt);

		//calibrate the points
		std::vector<Eigen::Vector2d> u1;
		std::vector<Eigen::Vector2d> u2;
		calibrate_pts(K1,points1,u1);
		calibrate_pts(K2,points2,u2);

		//intialize variables that are to be found in the Ransac loop
		Eigen::Matrix3d E;
		Eigen::Matrix3d R;
		Eigen::Vector3d t;

		//run RANSAC to obtain the essential matrix E, pose (R, t) and inlier mask m
		std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
		if(ts.use_NN_)
			succ_tracks = rn.run(u1, u2, points1, points2, K1, K2, E, R, t, mask, ts);
		else
			succ_tracks = r.run(u1, u2, points1, points2, K1, K2, E, R, t, mask, ts);
		std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();

		//count the inliers
		int inl_count = 0;
		for(int j=0;j<u1.size();++j)
		{
			inl_count += mask[j];
		}

		//evaluate the rotation and translation
		double err_r = evaluate_R(R, Rgt);
		double err_t = evaluate_t(t, tgt);
		if(!succ_tracks)
		{
			err_r = 3.141592654;
			err_t = 3.141592654/2;
		}
		
		//update the statistics
		if(err_r <= 0.175 && err_t <= 0.175)
		{
			++good;
			rot_avg += err_r;
			tran_avg += err_t;
		}
		time_avg += duration;
	}
	
	std::cout << (double)(100.0*good) / (double)(seqs) << "% correctly estimated poses\n";
	std::cout << "Average rotation error: " << rot_avg/(double)good << "\n";
	std::cout << "Average translation error: " << tran_avg/(double)good << "\n";
	std::cout << time_avg/(double)seqs << " microseconds per camera pair\n";

	//std::cout << (double)good / (double)(seqs) << " " << rot_avg / (double)good << " " << tran_avg / (double)good << "\n";

	return 0;
}
