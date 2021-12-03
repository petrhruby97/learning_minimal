#include <iostream>
#include <vector>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <Eigen/Core>
#include <Eigen/QR>
#include <Eigen/Geometry>
#include <chrono>

#include <ransac.hxx>
#include <homotopy.hxx>

class HomotopyRansac : public Ransac
{
public:
	HomotopyRansac(std::vector<std::vector<double>> anchors_, std::vector<std::vector<double>> starts_, struct track_settings s_)
	{
		anchors = anchors_;
		starts = starts_;
		s = s_;
	}

	int minimal_solver(const Eigen::Vector2d points1[5], const Eigen::Vector2d points2[5], Eigen::Matrix3d * E, Eigen::Matrix3d * R, Eigen::Vector3d * t)
	{
		//obtain homogeneous points
		Eigen::Vector3d P[5];
		Eigen::Vector3d Q[5];
		for(int i=0;i<5;++i)
		{
			P[i] = points1[i].homogeneous();
			Q[i] = points2[i].homogeneous();
		}
		
		//normalize the problem
		std::vector<Eigen::Vector2d> P1(5);
		std::vector<Eigen::Vector2d> Q1(5);
		Eigen::Matrix3d CP = Eigen::Matrix3d::Identity();
		Eigen::Matrix3d CQ = Eigen::Matrix3d::Identity();
		int perm5[5];
		bool swap = normalize(P,Q,P1,Q1,CP,CQ,perm5);
		
		//copy the normalized problem
		double problem[20];
		for(int k=0;k<5;k++)
		{
			problem[k] = P1[k](0);
			problem[k+5] = P1[k](1);
			problem[k+10] = Q1[k](0);
			problem[k+15] = Q1[k](1);
		}

		//select the anchor
		int p = select_anchor(problem);

		//skip the sample if the TRASH has been selected
		if(p==-1) return 0;

		//copy the parameters of the anchor and the problem
		double params[40];
		for(int i=0;i<20;++i)
		{
			params[i] = anchors[p][i];
			params[i+20] = problem[i];
		}
		
		//perform the homotopy continuation to find the solution
		double solution[9];
		double start[9];
		int num_steps;
		int status = track(s, &starts[p][0], params, solution, &num_steps);
		
		//if a solution has been found, extract the pose and the essential matrix
		if(status == 2)
		{
			//extract the relative pose
			Eigen::Matrix3d R0;
			Eigen::Vector3d t0;
			extract_pose(params, solution, R0, t0);

			//obtain the essential matrix
			Eigen::Matrix3d tx;
			tx << 0, -t0(2), t0(1), t0(2), 0, -t0(0), -t0(1), t0(0), 0;
			Eigen::Matrix3d E0 = tx * R0;

			//adjust the pose with the transformation matrices
			if(swap)
			{
				E[0] = CP.transpose() * E0.transpose() * CQ;
				R[0] = CP.transpose() * R0.transpose() * CQ;
				t[0] = -CP.transpose() * R0.transpose() * t0;
			}
			else
			{
				E[0] = CQ.transpose() * E0 * CP;
				R[0] = CQ.transpose() * R0 * CP;
				t[0] = CQ.transpose() * t0;
			}
			
			return 1;
		}

		return 0;
	}

protected:
	std::vector< std::vector<double> > anchors;
	std::vector<std::vector<double>> starts;
	struct track_settings s;

	virtual int select_anchor(double * problem)
	{
		return 0;
	}

	

};
