#include <iostream>
#include <vector>
#include <fstream>
#include <iomanip>
#include <string>

typedef struct
{
	std::vector<int> nbs;
	std::vector<int> dists;
} node;

//load the expected number of anchors
bool load_settings(std::string set_file, int &anch_num)
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

	anch_num = anchors;

	return 1;
}

int main(int argc, char **argv)
{
	//parse the input
	if(argc < 4)
	{
		std::cout << "Run as:\n anchors anchor_data model_folder trainParam\n where anchor_data is the file with problems from which the anchors are extracted, model_folder is the folder from which the model is loaded and trainParam is a file with settings\n";
		return 0;
	}
	std::string data_file(argv[1]);
	std::cout << "Extracting anchors from file " << data_file << ".\n";
	std::string model_folder(argv[2]);
	std::cout << "Model folder " << model_folder << "\n";
	std::string set_file(argv[3]);
	std::cout << "Extracting settings from file " << set_file << ".\n";

	//load the settings
	int anch_num;
	bool succ_load = load_settings(set_file, anch_num);
	if(!succ_load) return 0;

	//load the connectivity graph G
	std::ifstream f;
	f.open(model_folder+"/connectivity.txt");
	if(!f.good())
	{
		std::cout << "Connectivity matrix connectivity.txt not available in the selected input folder\n";
		return 0;
	}
	int n;
	f >> n;
	std::vector<node> G(n);
	for(int i=0;i<n;i++)
	{
		node cur;
		int count;
		f >> count;
		for(int j=0;j<count;j++)
		{
			int nb;
			f >> nb;
			cur.nbs.push_back(nb);
			int dist;
			f >> dist;
			cur.dists.push_back(dist);
		}
		G[i] = cur;
	}
	f.close();

	std::ifstream fa;
	fa.open(data_file);
	if(!fa.good())
	{
		std::cout << "File with anchor candidates not available\n";
	}
	
	//set up the array telling which nodes are active (=not yet covered)
	bool active[n];
	for(int i=0;i<n;i++)
		active[i] = 1;

	//find the dominating set of a given number of anchors
	std::vector<int> dom;
	int dominated = 0;
	while(1)
	{
		//select the node with the highest number of active neighbors
		int best = -1;
		int best_deg = 0;
		for(int i=0;i<n;i++)
		{
		
			//find the number of active neighbors
			int cur_deg = 0;
			for(unsigned int j=0;j<G[i].nbs.size();j++)
			{
				const int cur = G[i].nbs[j];
				if(active[cur])
				{
					++cur_deg;
				}
			}
			
			//update
			if(cur_deg > best_deg)
			{
				best_deg = cur_deg;
				best = i;
			}
		}

		//if no good node has been found, terminate
		if(best == -1)
			break;

		//make the selected node and all its neighbors inactive
		active[best] = 0;
		for(int j=0;j<G[best].nbs.size();j++)
		{
			active[G[best].nbs[j]] = 0;
		}

		//update the number of dominated problems
		dom.push_back(best);
		dominated = dominated + best_deg;

		if(dom.size() >= anch_num)
			break;

	}

	//print the statistics
	std::cout << "Size of dominating set: " << dom.size() << "\n";
	std::cout << dominated << " / " << n << " problems dominated (" << 100*(double)dominated/(double)n << "%)\n";

	//load the anchor candidates and store those which have been selected
	int na;
	fa >> na;
	std::vector<std::vector<double>> anchors(na);
	for(int i=0;i<na;++i)
	{
		std::vector<double> anchor(37);
		
		//load the anchor candidates
		for(int j=0;j<37;j++)
		{
			double u;
			fa >> u;

			anchor[j] = u;
		}
		anchors[i] = anchor;
	}
	fa.close();

	//store the selected anchors
	std::ofstream fff;
	fff.open(model_folder+"/anchors.txt");
	fff << dom.size() << "\n";
	for(int i=0;i<dom.size();++i)
	{
		std::vector<double> anchor = anchors[dom[i]];
		for(int j=0;j<37;++j)
		{
			fff << std::setprecision(15) << anchor[j] << " ";
		}
		fff << "\n";
	}
	fff.close();

	return 0;
}
