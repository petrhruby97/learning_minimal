#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <math.h>
#include <string>

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

int main(int argc, char **argv)
{

	//load the arguments
	if(argc < 6)
	{
		std::cout << "Run as:\n labels train_data model_folder output_folder dataset_name trainParam\n where train_data is the file with problems used for training the model, model_folder is the folder from which the data (i.e. the anchors) are loaded, output_folder is the folder to which the output is stored, dataset_name is a string with a name of the dataset (multiple datasets like training and testing dataset will be prepared, they will have different names) and trainParam is a file with settings\n";
		return 0;
	}
	std::string data_file(argv[1]);
	std::cout << "Extracting data from file " << data_file << ".\n";
	std::string model_folder(argv[2]);
	std::cout << "Model folder " << model_folder << "\n";
	std::string output_folder(argv[3]);
	std::cout << "Output folder " << output_folder << "\n";
	std::string dataset_name(argv[4]);
	std::cout << "Name of the dataset under which the labels will be saved:  " << dataset_name << "\n";
	std::string set_file(argv[5]);
	std::cout << "Extracting settings from file " << set_file << ".\n";

	//load the settings and store them to the track_settings structure (update to contain more settings)
	track_settings settings;
	bool succ_load = load_settings(set_file, settings);
	if(!succ_load) return 0;

	//load the anchors
	std::vector<std::vector<Float>> anchors;
	std::vector<std::vector<Float>> start_a;
	std::vector<std::vector<Float>> depths_a;
	succ_load = load_anchors(model_folder+"/anchors.txt",anchors,start_a,depths_a);
	if(!succ_load) return 0;
	int m = anchors.size();

	//initialize the variables for the tracking
	Float params[48];
	static double solution[12];
	int num_steps;
	int succ = 0;
	long total_track = 0;
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

	//problems are stored to file X_<NAME>.txt, labels to file Y_<NAME>.txt
	std::ofstream xf;
	xf.open(output_folder+"/X_"+dataset_name+".txt");
	std::ofstream lf;
	lf.open(output_folder+"/Y_"+dataset_name+".txt");	

	//track from each anchor to each training/testing problem, if successful, store the problem and the anchor ID
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

		//try all anchors and save the labels of the successful labels
		bool covered = 0;
		for(int j=0;j<m;++j)
		{
			//copy the start problem and the target problem
			for(int a=0;a<24;a++)
			{
				params[a] = anchors[j][a];
				params[24+a] = problem[a];
			}
			int k = 0;

			Float start[9];
			for(int a=0;a<9;++a)
				start[a] = start_a[j][a];

			//track the problem
			high_resolution_clock::time_point t1 = high_resolution_clock::now();
			int status = track(settings, start, params, solution, &num_steps);
			high_resolution_clock::time_point t2 = high_resolution_clock::now();
			auto duration = duration_cast<microseconds>(t2 - t1).count();
			total_track = total_track + duration;

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

				//if the difference is small, store the problem with the label of the current anchor
				if(diff <= settings.corr_thresh_)
				{
					//update the statistics
					succ += 1;
					covered = 1;

					//store the label (and the transformed problem)
					lf << (j+1) << "\n";
					for(int a=0;a<24;++a)
						xf << problem[a] << " ";
					xf << "\n";
				}
				
			}
			
		}

		//if the problem is not covered, store it with zero (TRASH) label
		if(!covered)
		{
			lf << 0 << "\n";
			for(int a=0;a<24;++a)
				xf << problem[a] << " ";
			xf << "\n";
		}
	}

	lf.close();
	xf.close();
	
	
	std::cout << "Time of tracking " << total_track << "µs, " << (double)total_track/(double)(all) << "µs per track" << " \n";
	std::cout << succ << " successful tracks out of " << all << "\n";

	return 0;
}
