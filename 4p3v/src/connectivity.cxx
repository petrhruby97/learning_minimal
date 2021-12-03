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

bool load_data(std::string data_file,
				std::vector<std::vector<Float>> &problems,
				std::vector<std::vector<Float>> &start,
				std::vector<std::vector<Float>> &depths)
{
	std::ifstream f;
	f.open(data_file);

	if(!f.good())
	{
		f.close();
		std::cout << "Data file not available\n";
		return 0;
	}

	int n;
	f >> n;
	std::cout << n << " problems\n";

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

typedef struct
{
	std::vector<int> nbs;
	std::vector<int> dists;
} node;

void store_graph(std::vector<node> G, std::string output_folder)
{
	std::cout << "STORING GRAPH\n";
	std::ofstream nb;
	nb.open(output_folder+"/connectivity.txt");
	nb << G.size() << "\n";
	for(int i=0;i<G.size();i++)
	{
		nb << G[i].nbs.size() << " ";
		for(int j=0;j<G[i].nbs.size();j++)
		{
			nb << G[i].nbs[j] << " " << G[i].dists[j] << " ";
		}
		nb << "\n";
	}
	nb.close();
}

int main(int argc, char **argv)
{
	if(argc < 4)
	{
		std::cout << "Run as:\n connectivity anchor_data model_folder trainParam\n where anchor_data is the file with problems from which the anchors are extracted, model_folder is the folder to which the output is stored and trainParam is a file with settings\n";
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
	//return 0;

	//load the data
	std::vector<std::vector<Float>> problems;
	std::vector<std::vector<Float>> start;
	std::vector<std::vector<Float>> depths;
	succ_load = load_data(data_file,problems,start,depths);
	if(!succ_load) return 0;
	int n = problems.size();

	//initialize the variables for the tracking
	Float params[48];
	static double solution[12];
	std::vector<node> G(n);
	int num_steps;

	//initialize the statistics
	long total_track = 0;
	int succ = 0;
	int all = 0;
	
	//perform the tracking
	for(int i=0;i<n;i++)
	{
	
		//add the current node as the neighbor of itself
		G[i].nbs.push_back(i);
		G[i].dists.push_back(0);
	
		//copy the start problem
		for(int a=0;a<24;a++)
		{
			params[a] = problems[i][a];
		}

		for(int j=0;j<n;j++)
		{
			if(i==j) continue;
			
			//copy the target problem
			for(int a=0;a<24;a++)
			{
				params[24+a] = problems[j][a];
			}
			
			Float cur_start[12];
			for(int a=0;a<12;++a)
				cur_start[a] = start[i][a];		

			//track the problem
			high_resolution_clock::time_point t1 = high_resolution_clock::now();
			int status = track(settings, cur_start, params, solution, &num_steps);
			high_resolution_clock::time_point t2 = high_resolution_clock::now();
			auto duration = duration_cast<microseconds>(t2 - t1).count();
			total_track = total_track + duration;

			//evaluate the solution
			if(status == 2)
			{

				//find the distance between the fabricated and the obtained solution
				Float diff = 0;
				for(int a=0;a<11;a++)
				{
					Float cdiff = solution[a] - start[j][a];
					diff = diff + cdiff * cdiff;
				}

				//if the difference is small, update the graph and the statistics
				if(diff <= settings.corr_thresh_)
				{
					//SUCCESSFUL TRACK
					std::cout << "Anchors: " << i << ", " << j << "\n";
					std::cout << "Diff: " << diff << "\n";
					std::cout << "\n";

					//update the connectivity graph
					G[i].nbs.push_back(j);
					G[i].dists.push_back(num_steps);
					
					succ += 1;
				}
			}

			all+=1;
		}
	}

	std::cout << "Time of tracking " << total_track << "µs, " << (double)total_track/(double)(all) << "µs per track" << " \n";
	std::cout << succ << " successful tracks out of " << all << "\n";

	store_graph(G, model_folder);
	return 0;
}
