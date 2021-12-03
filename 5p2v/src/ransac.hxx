#include <iostream>
#include <vector>
#include <stdlib.h>
#include <time.h>
#include <Eigen/Core>
#include <chrono>
#include <string>

typedef struct
{
	long solver_time;
	long nn_time;
} meta_info;

struct test_settings
{
	bool use_NN_;
	unsigned num_samples_;
	bool only_succ_;
	bool use_trash_;
	double err_threshold_;
};

bool load_test_settings(std::string set_file, struct test_settings &settings)
{
	std::ifstream f;
	f.open(set_file);

	if(!f.good())
	{
		f.close();
		std::cout << "Test settings file not available\n";

		return 0;
	}

	std::string t;

	//use NN
	bool use_NN_;
	f >> use_NN_;
	getline(f, t);

	//num samples
	unsigned num_samples_;
	f >> num_samples_;
	getline(f, t);

	//count only successful tracks
	bool only_succ_;
	f >> only_succ_;
	getline(f, t);

	//use a trash bin
	bool use_trash_;
	f >> use_trash_;
	getline(f, t);

	//threshold for the Sampson error
	unsigned err_threshold_;
	f >> err_threshold_;
	getline(f, t);

	//update the structure
	settings.use_NN_ = use_NN_;
	settings.num_samples_ = num_samples_;
	settings.only_succ_ = only_succ_;
	settings.use_trash_ = use_trash_;
	settings.err_threshold_ = err_threshold_;

	return 1;
}

class Ransac
{
public:
	inline void sample(int n, int * s)
	{
		for(int i=0;i<5;++i)
		{
			//obtain a random value from 0 to n-i-1
			int next = rand()%(n-i);

			//insert the point to the structure
			for(int j=0;j<i;++j)
			{
				//heuristic, does not guarantee no repetitions but it may reduce the probability of it
				if(next == s[i])
					++next;
			}
			s[i] = next;
		}
	}

	virtual int minimal_solver(const Eigen::Vector2d points1[5], const Eigen::Vector2d points2[5], Eigen::Matrix3d * E, Eigen::Matrix3d * R, Eigen::Vector3d * t) = 0;

	
	int run(std::vector<Eigen::Vector2d> u1, std::vector<Eigen::Vector2d> u2, std::vector<Eigen::Vector2d> p1, std::vector<Eigen::Vector2d> p2, Eigen::Matrix3d K1, Eigen::Matrix3d K2, Eigen::Matrix3d &E, Eigen::Matrix3d &R, Eigen::Vector3d &t, std::vector<bool> &mask, struct test_settings ts_)
	{
		int succ_tracks = 0;
		ts = ts_;
		
		//initialize the variables
		Eigen::Vector2d *  smpl1 = new Eigen::Vector2d[5];
		Eigen::Vector2d * smpl2 = new Eigen::Vector2d[5];
		int * smpl = new int[5];
		Eigen::Matrix3d curE[10];
		Eigen::Matrix3d curR[10];
		Eigen::Vector3d curT[10];
		Eigen::Matrix3d K1m = K1.inverse();
		Eigen::Matrix3d K2m = K2.inverse();
		Eigen::Matrix3d F;
		std::vector<bool> cur_mask(mask.size());

		int best_val = 0;

		//initialize the random sampler
		srand(time(NULL));
		//srand(20211117);

		//MAIN RANSAC LOOP
		for(unsigned int i=0;i<ts.num_samples_;++i)
		{
			//sample a five tuple from the data
			sample(u1.size(), smpl);
			for(int j=0;j<5;++j)
			{
				smpl1[j] = u1[smpl[j]];
				smpl2[j] = u2[smpl[j]];
			}

			//call the minimal solver to get the model and the relative pose
			int sols = minimal_solver(smpl1, smpl2, curE, curR, curT);

			//update the number of succ tracks
			if(sols)
			{
				++succ_tracks;
			}

			//find the inliers for all relative poses
			for(unsigned int a=0;a<sols;++a)
			{
				//find the inliers to the model and evaluate the function
				F = K2m.transpose() * curE[a] * K1m;
				int val = inliers(F, ts.err_threshold_, cur_mask, p1, p2);
			
				//if the found solution is so-far the best one, update the model
				if(val > best_val)
				{
					best_val = val;
					E = curE[a];
					R = curR[a];
					t = curT[a];
					std::swap(cur_mask, mask);
				}
			}
			
		}

		return succ_tracks;
	}	

protected:
	struct test_settings ts;
	
	int inliers(const Eigen::Matrix3d F, const double threshold, std::vector<bool> &mask, const std::vector<Eigen::Vector2d> &p1, const std::vector<Eigen::Vector2d> &p2)
	{
		int ret = 0;
		const double E_00 = F(0, 0);
		const double E_01 = F(0, 1);
		const double E_02 = F(0, 2);
		const double E_10 = F(1, 0);
		const double E_11 = F(1, 1);
		const double E_12 = F(1, 2);
		const double E_20 = F(2, 0);
		const double E_21 = F(2, 1);
		const double E_22 = F(2, 2);
		for(int i=0;i<p1.size();++i)
		{
			//evaluate the Sampson error of the sample
			const double x1_0 = p1[i](0);
			const double x1_1 = p1[i](1);
			const double x2_0 = p2[i](0);
			const double x2_1 = p2[i](1);

			const double Ex1_0 = E_00 * x1_0 + E_01 * x1_1 + E_02;
			const double Ex1_1 = E_10 * x1_0 + E_11 * x1_1 + E_12;
			const double Ex1_2 = E_20 * x1_0 + E_21 * x1_1 + E_22;

			const double Etx2_0 = E_00 * x2_0 + E_10 * x2_1 + E_20;
			const double Etx2_1 = E_01 * x2_0 + E_11 * x2_1 + E_21;

			const double x2tEx1 = x2_0 * Ex1_0 + x2_1 * Ex1_1 + Ex1_2;

			double err = x2tEx1 * x2tEx1 / (Ex1_0 * Ex1_0 + Ex1_1 * Ex1_1 + Etx2_0 * Etx2_0 + Etx2_1 * Etx2_1);


			//if the error is <= threshold, count it as an inlier
			if(err <= threshold)
			{
				++ret;
				mask[i] = 1;
			}
			else
			{
				mask[i] = 0;
			}
			
		}
		return ret;
	}
};
