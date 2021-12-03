#include <iostream>
#include <vector>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <Eigen/Core>
#include <Eigen/QR>
#include <Eigen/Geometry>
#include <Eigen/Eigenvalues>
#include <chrono>

#include <ransac.hxx>

class NisterRansac : public Ransac
{
public:

	int minimal_solver(const Eigen::Vector2d points1[5], const Eigen::Vector2d points2[5], Eigen::Matrix3d * Es, Eigen::Matrix3d * Rs, Eigen::Vector3d * ts)
	{
		
	
		// Step 1: Extraction of the nullspace x, y, z, w.

		Eigen::Matrix<double, Eigen::Dynamic, 9> Q(5, 9);
		for (size_t i = 0; i < 5; ++i)
		{
			const double x1_0 = points1[i](0);
			const double x1_1 = points1[i](1);
			const double x2_0 = points2[i](0);
			const double x2_1 = points2[i](1);
			Q(i, 0) = x1_0 * x2_0;
			Q(i, 1) = x1_1 * x2_0;
			Q(i, 2) = x2_0;
			Q(i, 3) = x1_0 * x2_1;
			Q(i, 4) = x1_1 * x2_1;
			Q(i, 5) = x2_1;
			Q(i, 6) = x1_0;
			Q(i, 7) = x1_1;
			Q(i, 8) = 1;
		}

		// Extract the 4 Eigen vectors corresponding to the smallest singular values.
		const Eigen::JacobiSVD<Eigen::Matrix<double, Eigen::Dynamic, 9>> svd(Q, Eigen::ComputeFullV);
		const Eigen::Matrix<double, 9, 4> E = svd.matrixV().block<9, 4>(0, 5);

		// Step 3: Gauss-Jordan elimination with partial pivoting on A.

		Eigen::Matrix<double, 10, 20> A;
		#include "essential_matrix_poly.h"
		Eigen::Matrix<double, 10, 10> AA =
		  A.block<10, 10>(0, 0).partialPivLu().solve(A.block<10, 10>(0, 10));

		// Step 4: Expansion of the determinant polynomial of the 3x3 polynomial
		//         matrix B to obtain the tenth degree polynomial.

		Eigen::Matrix<double, 13, 3> B;
		for (size_t i = 0; i < 3; ++i)
		{
			B(0, i) = 0;
			B(4, i) = 0;
			B(8, i) = 0;
			B.block<3, 1>(1, i) = AA.block<1, 3>(i * 2 + 4, 0);
			B.block<3, 1>(5, i) = AA.block<1, 3>(i * 2 + 4, 3);
			B.block<4, 1>(9, i) = AA.block<1, 4>(i * 2 + 4, 6);
			B.block<3, 1>(0, i) -= AA.block<1, 3>(i * 2 + 5, 0);
			B.block<3, 1>(4, i) -= AA.block<1, 3>(i * 2 + 5, 3);
			B.block<4, 1>(8, i) -= AA.block<1, 4>(i * 2 + 5, 6);
		}

		// Step 5: Extraction of roots from the degree 10 polynomial.
		Eigen::Matrix<double, 11, 1> coeffs;
		#include "essential_matrix_coeffs.h"

		Eigen::VectorXd roots_real;
		Eigen::VectorXd roots_imag;
		if (!FindPolynomialRootsCompanionMatrix(coeffs, &roots_real, &roots_imag))
		{
			//return {};
			return 0;
		}

		std::vector<Eigen::Matrix3d> models;
		models.reserve(roots_real.size());

		int pos = 0;
		for (Eigen::VectorXd::Index i = 0; i < roots_imag.size(); ++i)
		{
			const double kMaxRootImag = 1e-10;
			if (std::abs(roots_imag(i)) > kMaxRootImag)
			{
			  continue;
			}

			const double z1 = roots_real(i);
			const double z2 = z1 * z1;
			const double z3 = z2 * z1;
			const double z4 = z3 * z1;

			Eigen::Matrix3d Bz;
			for (size_t j = 0; j < 3; ++j) 
			{
				Bz(j, 0) = B(0, j) * z3 + B(1, j) * z2 + B(2, j) * z1 + B(3, j);
				Bz(j, 1) = B(4, j) * z3 + B(5, j) * z2 + B(6, j) * z1 + B(7, j);
				Bz(j, 2) = B(8, j) * z4 + B(9, j) * z3 + B(10, j) * z2 + B(11, j) * z1 + B(12, j);
			}

			const Eigen::JacobiSVD<Eigen::Matrix3d> svd(Bz, Eigen::ComputeFullV);
			const Eigen::Vector3d X = svd.matrixV().block<3, 1>(0, 2);

			const double kMaxX3 = 1e-10;
			if (std::abs(X(2)) < kMaxX3)
			{
				continue;
			}

			Eigen::MatrixXd essential_vec = E.col(0) * (X(0) / X(2)) +
								            E.col(1) * (X(1) / X(2)) + E.col(2) * z1 +
								            E.col(3);
			essential_vec /= essential_vec.norm();

			const Eigen::Matrix3d essential_matrix = Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(essential_vec.data());
			Es[pos] = essential_matrix;
			++pos;
		}

		return pos;
	}

protected:

	// Remove leading zero coefficients.
	Eigen::VectorXd RemoveLeadingZeros(const Eigen::VectorXd& coeffs) const
	{
		Eigen::VectorXd::Index num_zeros = 0;
		for (; num_zeros < coeffs.size(); ++num_zeros)
		{
			if (coeffs(num_zeros) != 0)
			{
				break;
			}
		}
		return coeffs.tail(coeffs.size() - num_zeros);
	}

	// Remove trailing zero coefficients.
	Eigen::VectorXd RemoveTrailingZeros(const Eigen::VectorXd& coeffs) const
	{
		Eigen::VectorXd::Index num_zeros = 0;
		for (; num_zeros < coeffs.size(); ++num_zeros)
		{
			if (coeffs(coeffs.size() - 1 - num_zeros) != 0)
			{
				break;
			}
		}
		return coeffs.head(coeffs.size() - num_zeros);
	}

	bool FindLinearPolynomialRoots(const Eigen::VectorXd& coeffs, Eigen::VectorXd* real, Eigen::VectorXd* imag) const
	{
		//CHECK_EQ(coeffs.size(), 2);

		if (coeffs(0) == 0)
		{
			return false;
		}

		if (real != nullptr)
		{
			real->resize(1);
			(*real)(0) = -coeffs(1) / coeffs(0);
		}

		if (imag != nullptr)
		{
			imag->resize(1);
			(*imag)(0) = 0;
		}

		return true;
	}

	bool FindQuadraticPolynomialRoots(const Eigen::VectorXd& coeffs, Eigen::VectorXd* real, Eigen::VectorXd* imag) const
	{
	  //CHECK_EQ(coeffs.size(), 3);

	  const double a = coeffs(0);
	  if (a == 0) {
		return FindLinearPolynomialRoots(coeffs.tail(2), real, imag);
	  }

	  const double b = coeffs(1);
	  const double c = coeffs(2);
	  if (b == 0 && c == 0) {
		if (real != nullptr) {
		  real->resize(1);
		  (*real)(0) = 0;
		}
		if (imag != nullptr) {
		  imag->resize(1);
		  (*imag)(0) = 0;
		}
		return true;
	  }

	  const double d = b * b - 4 * a * c;

	  if (d >= 0) {
		const double sqrt_d = std::sqrt(d);
		if (real != nullptr) {
		  real->resize(2);
		  if (b >= 0) {
		    (*real)(0) = (-b - sqrt_d) / (2 * a);
		    (*real)(1) = (2 * c) / (-b - sqrt_d);
		  } else {
		    (*real)(0) = (2 * c) / (-b + sqrt_d);
		    (*real)(1) = (-b + sqrt_d) / (2 * a);
		  }
		}
		if (imag != nullptr) {
		  imag->resize(2);
		  imag->setZero();
		}
	  } else {
		if (real != nullptr) {
		  real->resize(2);
		  real->setConstant(-b / (2 * a));
		}
		if (imag != nullptr) {
		  imag->resize(2);
		  (*imag)(0) = std::sqrt(-d) / (2 * a);
		  (*imag)(1) = -(*imag)(0);
		}
	  }

	  return true;
	}

	bool FindPolynomialRootsCompanionMatrix(const Eigen::VectorXd& coeffs_all,
                                        Eigen::VectorXd* real,
                                        Eigen::VectorXd* imag) const
	{

	Eigen::VectorXd coeffs = RemoveLeadingZeros(coeffs_all);

	const int degree = coeffs.size() - 1;

	if (degree <= 0)
	{
		return false;
	}
	else if (degree == 1)
	{
		return FindLinearPolynomialRoots(coeffs, real, imag);
	}
	else if (degree == 2)
	{
		return FindQuadraticPolynomialRoots(coeffs, real, imag);
	}

	// Remove the coefficients where zero is a solution.
	coeffs = RemoveTrailingZeros(coeffs);

	// Check if only zero is a solution.
	if (coeffs.size() == 1)
	{
		if (real != nullptr)
		{
			real->resize(1);
	  		(*real)(0) = 0;
		}
		if (imag != nullptr)
		{
			imag->resize(1);
			(*imag)(0) = 0;
		}
		return true;
	}

	// Fill the companion matrix.
	Eigen::MatrixXd C(coeffs.size() - 1, coeffs.size() - 1);
	C.setZero();
	for (Eigen::MatrixXd::Index i = 1; i < C.rows(); ++i)
	{
		C(i, i - 1) = 1;
	}
	C.row(0) = -coeffs.tail(coeffs.size() - 1) / coeffs(0);

	// Solve for the roots of the polynomial.
	Eigen::EigenSolver<Eigen::MatrixXd> solver(C, false);
	if (solver.info() != Eigen::Success)
	{
		return false;
	}

	// If there are trailing zeros, we must add zero as a solution.
	const int effective_degree = coeffs.size() - 1 < degree ? coeffs.size() : coeffs.size() - 1;

	if (real != nullptr)
	{
		real->resize(effective_degree);
		real->head(coeffs.size() - 1) = solver.eigenvalues().real();
		if (effective_degree > coeffs.size() - 1)
		{
			(*real)(real->size() - 1) = 0;
		}
	}
	if (imag != nullptr)
	{
		imag->resize(effective_degree);
		imag->head(coeffs.size() - 1) = solver.eigenvalues().imag();
		if (effective_degree > coeffs.size() - 1)
		{
			(*imag)(imag->size() - 1) = 0;
		}
	}

	return true;
}

};
