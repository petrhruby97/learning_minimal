#include <iostream>
#include <vector>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <Eigen/Core>
#include <Eigen/QR>
#include <Eigen/Geometry>

#include <homotopy_ransac.hxx>

class HomotopyNNRansac : public HomotopyRansac
{
public:
	HomotopyNNRansac(std::vector<std::vector<double>> anchors_, std::vector<std::vector<double>> starts_, std::vector<std::vector<float>> &ws_, std::vector<std::vector<float>> &bs_, std::vector<std::vector<float>> &ps_, std::vector<int> &a_, std::vector<int> &b_, struct track_settings s_, bool trash_) : HomotopyRansac(anchors_, starts_, s_)
	{
		trash = trash_;

		layers = ws_.size();
		a = &a_[0];
		b = &b_[0];

		ws = new float*[ws_.size()];
		bs = new float*[ws_.size()];
		ps = new float*[ws_.size()-1];
		for(int i=0;i<ws_.size();++i)
		{
			ws[i] = &ws_[i][0];
			bs[i] = &bs_[i][0];
			if(i < layers-1)
				ps[i] = &ps_[i][0];
		}

		ws0 = &ws_[0][0];
		ws1 = &ws_[1][0];
		ws2 = &ws_[2][0];
		ws3 = &ws_[3][0];
		ws4 = &ws_[4][0];
		ws5 = &ws_[5][0];
		ws6 = &ws_[6][0];

		bs0 = &bs_[0][0];
		bs1 = &bs_[1][0];
		bs2 = &bs_[2][0];
		bs3 = &bs_[3][0];
		bs4 = &bs_[4][0];
		bs5 = &bs_[5][0];
		bs6 = &bs_[6][0];

		ps0 = &ps_[0][0];
		ps1 = &ps_[1][0];
		ps2 = &ps_[2][0];
		ps3 = &ps_[3][0];
		ps4 = &ps_[4][0];
		ps5 = &ps_[5][0];
	}

	
protected:
	bool trash;

	int layers;
	int * a;
	int * b;
	
	float ** ws;
	float ** bs;
	float ** ps;

	
	float * ws0;
	float * ws1;
	float * ws2;
	float * ws3;
	float * ws4;
	float * ws5;
	float * ws6;

	float * bs0;
	float * bs1;
	float * bs2;
	float * bs3;
	float * bs4;
	float * bs5;
	float * bs6;

	float * ps0;
	float * ps1;
	float * ps2;
	float * ps3;
	float * ps4;
	float * ps5;

	int select_anchor(double * problem) override
	{
		//prepare the input for the network
		float orig[20];
		for(int i=0;i<20;++i)
			orig[i] = problem[i];
		Eigen::Map<Eigen::VectorXf> input_n2(orig,20);

		//evaluate the MLP
		Eigen::VectorXf input_ = input_n2;
		Eigen::VectorXf output_;
		for(int i=0;i<layers;++i)
		{
			const Eigen::Map< const Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>, Eigen::Aligned > weights = Eigen::Map< const Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>, Eigen::Aligned >(ws[i],a[i],b[i]);
			const Eigen::Map<const Eigen::VectorXf> bias = Eigen::Map<const Eigen::VectorXf>(bs[i],a[i]);

			output_ = weights*input_+bias;

			if(i==layers-1) break;

			const Eigen::Map<const Eigen::VectorXf> prelu = Eigen::Map<const Eigen::VectorXf>(ps[i],a[i]);
			input_ = output_.cwiseMax(output_.cwiseProduct(prelu));
		}
		
		//select the anchor with the highest score
		double best = -1000;
		int p = 0;
		int start = 1;
		if(trash)
			start = 0;
		for(int j=start;j<a[layers-1];++j)
		{
			if(output_(j) > best)
			{
				best = output_(j);
				p = j;
			}
		}
		p = p-1;
		
		return p;
	}

};
