#ifndef DDX_WLOP_HEADER
#define DDX_WLOP_HEADER
#pragma once 

#include <vector>
#include <memory>
#include <Eigen/Dense>
#include "KDTreeFlann.h"
#include "KDTreeSearchParam.h"
#include <random>
#include <omp.h>
#include <iostream>
#include <fstream>

namespace DDX {

class WLOP
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
private:

	double mRadius;
	double mMu;
	int mSelect_percentage;
	bool require_uniform_sampling;
	
	std::vector<Eigen::Vector3d>& original_point;
	std::vector<Eigen::Vector3d> sample_points ;



public:
	WLOP(std::vector<Eigen::Vector3d>& point , double radius ,int percentage  ,double mu = 0.45 ,bool density = false)
		:
		original_point(point),
		mRadius (radius) ,
		mMu(mu) ,
		mSelect_percentage(percentage) ,
		require_uniform_sampling (density)
	{};
	
	void Run (int iter_number) 
	{
		size_t number_of_original = original_point.size() ;

		size_t number_of_sample = static_cast<size_t> (float(number_of_original) * 
																(mSelect_percentage / float(100.0))) ;

		std::vector<Eigen::Vector3d> update_sample_points(number_of_sample);

		// 1 按照百分比生成下采样的点
		sample_points.reserve (number_of_sample);
		generate_sample_points(number_of_original ,number_of_sample);
		std::cout << "number_of_sample " << number_of_sample << std::endl;
		std::cout << "sample_points size " << sample_points.size() << std::endl;
		// 2 初始化一个原始的kd tree 
		const auto original_kd_tree = create_Kdtree (original_point);
		
		std::vector<double> original_density (number_of_original);
		std::vector<double> samples_density (number_of_sample);
		if (require_uniform_sampling) 
		{
			for (int i = 0 ; i < (int)original_point.size() ;++i)
			{
				original_density[i] = compute_density_weight_for_original_point (original_kd_tree ,original_point[i] ,mRadius);
				
			}
		}
		
		// 3 迭代更新
		for (int iter_n = 0; iter_n < iter_number; ++iter_n) 
		{
			std::cout << "iter " << iter_n << std::endl;

			// 创建采样点的kd tree 
			const auto sample_kd_tree = create_Kdtree (sample_points);
			
			
			for (int i = 0 ; i < (int)sample_points.size() ; ++i)
			{
				samples_density[i] = compute_density_weight_for_sample_point (sample_kd_tree ,sample_points[i] ,mRadius);
			}
			
			for (int i = 0 ; i < (int)sample_points.size() ; ++i) {
				update_sample_points[i] = compute_update_sample_point (
											i ,
											original_kd_tree ,
											sample_kd_tree ,
											original_density ,
											samples_density ,
											mRadius);
			}

			for (int i = 0 ; i < (int)update_sample_points.size() ; ++i) 
			{
				sample_points[i] = update_sample_points[i];
			}
		}
	} ;
	
	void write_obj (const std::string& path) 
	{
		std::ofstream delaunay3d (path.c_str());
		for (int i = 0 ; i < sample_points.size() ; ++i) {
			delaunay3d << "v " 
					<< sample_points[i][0] << " "
					<< sample_points[i][1] << " "
					<< sample_points[i][2] << std::endl ;
		}
		delaunay3d.close ();
	}
private:	
	std::unique_ptr<KDTreeFlann> create_Kdtree (std::vector<Eigen::Vector3d>& point) 
	{
		Eigen::MatrixXd dataset0 = Eigen::Map<Eigen::MatrixXd>(reinterpret_cast<double*>(point.data()),3,point.size());
		return std::make_unique<KDTreeFlann> (dataset0) ;
	};

	void generate_sample_points (size_t number_of_original, size_t number_of_sample ) 
	{
		size_t diff_number = number_of_original - number_of_sample;
		std::random_device rd ; 
		std::mt19937 gen(rd()) ; 
		constexpr int int_max = std::numeric_limits<int>::max () ;
		std::uniform_int_distribution<int> dis(int(0),int_max) ;
		std::vector<int> vist (number_of_original ,-1);
		int count_indx = 0;
		while (count_indx < diff_number )
		{
			int randint = dis (gen) ;
			const auto i13x = randint % (int)number_of_original ;
			if (vist[i13x] < 0) {
				vist[i13x] = 1;
				count_indx++;
			}
		}

		for (size_t i = 0 ; i < number_of_original ; ++i) 
		{
			if (vist[i] < 0)
				sample_points.push_back (original_point[i]);
		}
	}
	

	double compute_density_weight_for_original_point (const std::unique_ptr<KDTreeFlann>& original_kd_tree ,
													  Eigen::Vector3d const&p ,
													  double radius )
	{
		
		double radius2 = radius * radius;
		double density_weight = (double)1.0;
		double iradius16 = -(double)4.0 / radius2;

		std::vector<int> index ;
		std::vector<double> distance ;
		const auto k = original_kd_tree->SearchRadius (p ,radius ,index ,distance) ;

		for (int i = 0 ; i < k ; ++i) 
		{
			if (distance[i] < 1e-8) continue;
			density_weight += std::exp(distance[i] * iradius16);
		}

		return double(1.0) / density_weight;
	};

	double compute_density_weight_for_sample_point (const std::unique_ptr<KDTreeFlann>& sample_kd_tree  ,
													Eigen::Vector3d const&p ,
													double radius)
	{
		//Compute density weight
		double radius2 = radius * radius;
		double density_weight = (double)1.0;
		double iradius16 = -(double)4.0 / radius2;

		std::vector<int> index ;
		std::vector<double> distance ;
		const auto k = sample_kd_tree->SearchRadius (p ,radius ,index ,distance) ;

		for (int i = 0 ; i < k ; ++i) 
		{
			density_weight += std::exp(distance[i] * iradius16);
		}

		return density_weight;

	};

	Eigen::Vector3d compute_update_sample_point (int indx ,
									  const std::unique_ptr<KDTreeFlann>& original_kd_tree,
									  const std::unique_ptr<KDTreeFlann>& sample_kd_tree ,
									  const std::vector<double>& sample_densities ,
									  const std::vector<double> original_densities ,
									  double radius) 
	{
		bool is_original_densities_empty = original_densities.empty();
		bool is_sample_densities_empty = sample_densities.empty();

		double radius2 = radius * radius;
		double iradius16 = -(double)4.0 / radius2;
		

		// Compute Average Term
		Eigen::Vector3d average (0 , 0 ,0);
		if (true) 
		{	
			
			double average_weight_sum = (double)0.0;

			std::vector<int> index ;
			std::vector<double> distance ; 
			const auto k = original_kd_tree->SearchRadius (sample_points[indx] , radius ,index ,distance);

			for (int j = 0 ; j < k ; j++)
			{
				double dist2 = distance[j];
				if (dist2 < 1e-10) continue;

				double weight = (double)0.0;

				weight = std::exp(dist2 * iradius16);

				if (!is_original_densities_empty)
				{
					weight *= original_densities[index[j]];
				}
				average_weight_sum += weight;
    			average += original_point[index[j]] * weight;
			}

			if (k == 0 || average_weight_sum < 1e-20)
			{
				average = sample_points[indx] ;
			}
			else
			{
				average = average / average_weight_sum;
			}
		}

		//Compute repulsion term
		Eigen::Vector3d repulsion (0.0 , 0.0 ,0.0);
		if (true) 
		{
			double weight = (double)1.0;
			double repulsion_weight_sum = (double)0.0;
			std::vector<int> index ;
			std::vector<double> distance ; 
			const auto k = sample_kd_tree->SearchRadius (sample_points[indx] , radius ,index ,distance);

			for (int j = 0 ; j < k ; j++)
			{
				double dist2 = distance[j];
				if (dist2 < 1e-10) continue;
				double dist = std::sqrt(dist2);
				weight = std::exp(dist2 * iradius16) * std::pow(double(1.0) / dist ,2); // L1
				if (!is_sample_densities_empty)
				{
					weight *= sample_densities[index[j]];
				}

				Eigen::Vector3d diff = sample_points[indx] - sample_points[index[j]];
				repulsion_weight_sum += weight;
    			repulsion = repulsion + diff * weight;
			}
			
			if (k < 3 || repulsion_weight_sum < double(1e-20)) 
			{
				repulsion = Eigen::Vector3d(0.0 ,0.0 ,0.0);
			}

			else 
			{
				repulsion = repulsion / repulsion_weight_sum;
			}
			
		}

		return average + mMu * repulsion ;	
	};

};

}

#endif	/* DDX_WLOP_HEADER */