#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <math.h>
#include <string>

#include <Eigen/Core>
#include <Eigen/QR>
#include <Eigen/Geometry>

#include <homotopy.hxx>

#define Float double
typedef unsigned char ind;
static constexpr Float tol = 1e-3;
typedef std::complex<Float> complex;
using namespace std::chrono;
//using namespace std;

bool load_anchors(std::string data_file,
				std::vector<std::vector<Float>> &problems,
				std::vector<std::vector<Float>> &start,
				std::vector<std::vector<Float>> &depths)
{
	std::ifstream f;
	f.open(data_file);

	if(!f.good())
	{
		f.close();
		std::cout << "Anchor file not available\n";
		return 0;
	}

	int n;
	f >> n;
	std::cout << n << " anchors\n";

	problems = std::vector<std::vector<Float>>(n);
	start = std::vector<std::vector<Float>>(n);
	depths = std::vector<std::vector<Float>>(n);

	//load the problems
	for(int i=0;i<n;i++)
	{
		std::vector<Float> problem(24);
		std::vector<Float> cst(12);
		std::vector<Float> depth(13);
	
		//load the points
		for(int j=0;j<24;j++)
		{
			Float u;
			f >> u;		

			problem[j] = u;
		}
		problems[i] = problem;
		
		//load the depths and convert them to the solution
		Float first_depth;
		f >> first_depth;
		depth[0] = first_depth;
		for(int j=0;j<11;j++)
		{
			Float u;
			f >> u;		

			cst[j] = u/first_depth;
			depth[j+1] = u;
		}
		Float l;
		f >> l;
		cst[11] = l;
		depth[12] = l;
		
		start[i] = cst;
		depths[i] = depth;
	}
	f.close();
	return 1;
}

void load_NN(std::string model_dir, std::vector<std::vector<float>> &ws, std::vector<std::vector<float>> &bs, std::vector<std::vector<float>> &ps, std::vector<int> &a_, std::vector<int> &b_)
{
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
		
		std::cout << a << " " << b << "\n";

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

int main(int argc, char **argv)
{
	if(argc < 4)
	{
		std::cout << "Run as:\n labels test_data model_folder trainParam\n where test_data is the file with problems on which the model will be tested, model_folder is the trained model of the solver, and trainParam is a file with settings\n";
		return 0;
	}
	std::string data_file(argv[1]);
	std::cout << "Extracting data from file " << data_file << ".\n";
	std::string model_folder(argv[2]);
	std::cout << "Model folder " << model_folder << "\n";
	std::string set_file(argv[3]);
	std::cout << "Extracting settings from file " << set_file << ".\n";

	track_settings settings;
	//load the settings and store them to the track_settings structure (update to contain more settings)
	bool succ_load = load_settings(set_file, settings);
	if(!succ_load) return 0;

	//load the anchors
	std::vector<std::vector<Float>> anchors;
	std::vector<std::vector<Float>> start_a;
	std::vector<std::vector<Float>> depths_a;
	succ_load = load_anchors(model_folder+"/anchors.txt",anchors,start_a,depths_a);
	if(!succ_load) return 0;
	int m = anchors.size();

	//load the NN
	std::vector<std::vector<float>> ws;
	std::vector<std::vector<float>> bs;
	std::vector<std::vector<float>> ps;
	std::vector<int> a_;
	std::vector<int> b_;
	load_NN(model_folder, ws, bs, ps, a_, b_);
	int layers = b_.size();

	//initialize the variables for the tracking
	Float params[48];
	static double solution[12];
	int num_steps;
	char tr;

	//initialize the statistics
	long total_track = 0;
	long total = 0;
	int succ = 0;
	int all = 0;

	//open the data file
	std::ifstream f;
	f.open(data_file);
	if(!f.good())
	{
		f.close();
		std::cout << "Training data file not available\n";
		return 0;
	}
	int n;
	f >> n;
	std::cout << n << " problems\n";

	//load every problem, select the starting problem with the classifier, track HC to the loaded problem, and evaluate the result
	for(int i=0;i<n;++i)
	{
		if(!(i%10000)) std::cout << i << "\n";
	
		//load the problem
		Float problem[24];
		Float depths[12];
		Float gt_sol[12];

		//load the points
		for(int j=0;j<24;j++)
		{
			Float u;
			f >> u;

			problem[j] = u;
		}

		//load the depths
		Float first_depth;
		f >> first_depth;
		depths[0] = first_depth;
		for(int j=0;j<11;j++)
		{
			Float u;
			f >> u;		

			gt_sol[j] = u/first_depth;
			depths[j+1] = u;
		}
		Float l;
		f >> l;
		gt_sol[11] = l;
		depths[12] = l;

		//SELECT THE INITIAL P-S PAIR
		high_resolution_clock::time_point ta1 = high_resolution_clock::now();

		float orig[24];
		for(int a=0;a<24;++a)
			orig[a] = (float)problem[a];

		//evaluate the MLP
		Eigen::Map<Eigen::VectorXf> input_n2(orig,24);
		Eigen::VectorXf input_ = input_n2;
		Eigen::VectorXf output_;
		for(int i=0;i<layers;++i)
		{
			float * ws_ = &ws[i][0];
			float * bs_ = &bs[i][0];
			float * ps_ = &ps[i][0];
			const Eigen::Map< const Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>, Eigen::Aligned > weights = Eigen::Map< const Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>, Eigen::Aligned >(ws_,a_[i],b_[i]);
			const Eigen::Map<const Eigen::VectorXf> bias = Eigen::Map<const Eigen::VectorXf>(bs_,a_[i]);

			output_ = weights*input_+bias;

			if(i==layers-1) break;

			const Eigen::Map<const Eigen::VectorXf> prelu = Eigen::Map<const Eigen::VectorXf>(ps_,a_[i]);
			input_ = output_.cwiseMax(output_.cwiseProduct(prelu));
		}
		
		//find the output with the highest score
		double best = -1000;
		int p = 0;
		
		for(int j=1;j<a_[layers-1];++j)
		{
			if(output_(j) > best)
			{
				best = output_(j);
				p = j;
			}
		}
		p = p-1;
		if(p==-1)
			continue;

		//update the time for the classification
		high_resolution_clock::time_point ta2 = high_resolution_clock::now();
		auto duration_a = duration_cast<microseconds>(ta2 - ta1).count();
		total = total + duration_a;

		//copy the start problem
		for(int a=0;a<24;a++)
		{
			params[a] = anchors[p][a];
			params[a+24] = problem[a];
		}
		int k=0;

		//copy the start solution
		Float start[12];
		for(int a=0;a<12;++a)
			start[a] = start_a[p][a];

		//track the problem
		high_resolution_clock::time_point t1 = high_resolution_clock::now();
		
		int status = track(settings, start, params, solution, &num_steps);
		
		high_resolution_clock::time_point t2 = high_resolution_clock::now();
		auto duration = duration_cast<microseconds>(t2 - t1).count();
		total_track = total_track + duration;
		total = total + duration;

		//evaluate the solution
		if(status == 2)
		{
			//compute the difference between the obtained and the expected solutions
			Float diff = 0;
			for(int a=0;a<11;a++)
			{
				Float cdiff = solution[a] - gt_sol[a];
				diff = diff + cdiff * cdiff;
			}

			if(diff <= settings.corr_thresh_)
			{
				++succ;
			}
			
		}

	}
	
	
	std::cout << "\n";
	std::cout << "Time of tracking " << total_track << "µs, " << (double)total_track/(double)(all) << "µs per track" << " \n";
	std::cout << succ << " successful problems out of " << n << ", " << (100.0*succ)/(double)n << "% \n";
	std::cout << "Time " << (double)total/(double)n << "\n";
	std::cout << "\n";

	return 0;
}
