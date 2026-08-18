// Computer Vision Plugin.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
#if _MSC_VER // this is defined when compiling with Visual Studio
#define EXPORT_API __declspec(dllexport) // Visual Studio needs annotating exported functions with this
#else
#define EXPORT_API // XCode does not need annotating exported functions, so define is empty
#endif

#include <fstream>
#include <opencv2/opencv.hpp>
#include "opencv2/core.hpp"
#include <opencv2/calib3d/calib3d.hpp>
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/imgproc.hpp"

// ------------------------------------------------------------------------
// Plugin itself
// Link following functions C-style (required for plugins)
extern "C"
{
	struct Vector2
	{
		float x;
		float y;
	};

	struct Vector3
	{
		float x;
		float y;
		float z;
	};

	struct Vector4
	{
		float x;
		float y;
		float z;
		float w;
	};

	struct Matrix
	{
		float fx;
		float fy;
		float px;
		float py;
		float width;
		float height;
	};

	// The functions we will call from Unity.
	//
	void transform(const cv::Point3d& pt, const cv::Mat& rvec, const cv::Mat& tvec, cv::Point3d& ptTrans)
	{
		cv::Mat R;
		cv::Rodrigues(rvec, R);

		cv::Mat matPt(3, 1, CV_64F);
		matPt.at<double>(0, 0) = pt.x;
		matPt.at<double>(1, 0) = pt.y;
		matPt.at<double>(2, 0) = pt.z;

		cv::Mat matPtTrans = R * matPt + tvec;
		ptTrans.x = matPtTrans.at<double>(0, 0);
		ptTrans.y = matPtTrans.at<double>(1, 0);
		ptTrans.z = matPtTrans.at<double>(2, 0);
	}

	void recoverPoseFromPnP(const std::vector<cv::Point3d>& objectPoints1, const cv::Mat& rvec1, 
							const cv::Mat& tvec1, const std::vector<cv::Point2d>& imagePoints2,
							const cv::Mat& cameraMatrix, cv::Mat& rvec1to2, cv::Mat& tvec1to2)
	{
		cv::Mat R1;
		cv::Rodrigues(rvec1, R1);

		/* transform object points in camera frame */
		std::vector<cv::Point3d> objectPoints1InCam;
		for (size_t i = 0; i < objectPoints1.size(); i++) {
			cv::Point3d ptTrans;
			transform(objectPoints1[i], rvec1, tvec1, ptTrans);
			objectPoints1InCam.push_back(ptTrans);
		}

		cv::solvePnPRansac(objectPoints1InCam, imagePoints2, cameraMatrix, cv::noArray(), rvec1to2, tvec1to2, false, cv::SOLVEPNP_ITERATIVE);
	}

	EXPORT_API void PNP(Vector3 position1, Vector4 rotation1, Matrix cameraIntrinsics, Vector3 points3DUnity[8], Vector2 points2DUnity[8], 
						Vector3 position2[1], Vector4 rotation2[1])
	{
		/* Quaternion to rotation matrix */
		/* TODO 2.1.1 Create quaternion from input data */
		cv::Vec4f quaternion1{ 0.0f, 0.0f, 0.0f, 1.0f };

		/* TODO 2.1.2 Convert quaternion to rotation matrix */
		cv::Mat rotationMatrix1(cv::Size(3, 3), CV_64FC1);

		/* TODO 2.1.3 Convert rotation matrix to rotation vector */
		cv::Mat rotationVector1;

		/* TODO 2.2 Create translation vector from input data */
		cv::Mat translationVector1 = cv::Mat::zeros(3, 1, CV_64F);

		/* TODO 2.3 Create camera intrinsic matrix from input data */
		cv::Mat cameraMatrix = cv::Mat::zeros(3, 3, CV_64F);

		/* Convert Unity feature points and their projections to OpenCV format */
		std::vector<cv::Point2d> points2DOpenCV;
		std::vector<cv::Point3d> points3DOpenCV;

		for (int i = 0; i < 8; i++)
		{
			points2DOpenCV.push_back(cv::Point2d(points2DUnity[i].x, cameraIntrinsics.height - points2DUnity[i].y));
			points3DOpenCV.push_back(cv::Point3d(points3DUnity[i].x, -points3DUnity[i].y, points3DUnity[i].z));
		}

		/* PNP */
		cv::Mat R1;
		Rodrigues(rotationVector1, R1);
		R1 = R1.t();
		translationVector1 = -R1 * translationVector1;
		Rodrigues(R1, rotationVector1);

		cv::Mat rotationVector2, translationVector2;
		recoverPoseFromPnP(points3DOpenCV, rotationVector1, translationVector1, points2DOpenCV, cameraMatrix, rotationVector2, translationVector2);

		/* TODO 2.4 Inverse results */

		/* TODO 2.5 Save position */
		position2[0].x = 0;
		position2[0].y = 0;
		position2[0].z = 0;

		/* Save orientation */
		/* - TODO 2.6.1 Convert rotation vector to rotation matrix */
		/* - TODO 2.6.2 Convert rotation matrix to quaternion */
		/* - TODO 2.6.3 Normalize the quaternion */
	}
} // end of export C block

/* Detect feature points with SIFT. Match feature points with FLANN */
void DetectorFLANNSift(cv::Mat image1, cv::Mat image2, std::ofstream& measure)
{
	cv::Ptr<cv::Feature2D> detectorSift;
	cv::Ptr<cv::DescriptorMatcher> matcherFlann;

	std::vector<cv::KeyPoint> keypoints1, keypoints2;
	cv::Mat descriptors1, descriptors2;

	std::vector<std::vector<cv::DMatch> > knn_matches;
	std::vector<cv::DMatch> good_matches;

	const float ratio_thresh = 0.75f;
	/* SIFT detector */
	/* TODO 1.2.1 Intialize SIFT detector */
	/* TODO 1.2.2 Detect keypoints and descriptors in both images */
	/* TODO 1.6.1 Compute execution time of SIFT */
	auto durationSift = 0;

	/* Save execution time and number of feature points */
	measure << "SIFT detectors execution time: " << durationSift << std::endl;
	measure << "SIFT detected " << keypoints1.size() << " and " << keypoints2.size() << " keypoints" << std::endl;

	/* Save output image */
	cv::Mat output;
	cv::drawKeypoints(image1, keypoints1, output);
	cv::imwrite("SIFT.jpg", output);

	/* FLANNMatcher */
	/* TODO 1.3.1 Initialize FLANN Matcher */
	/* TODO 1.3.2 Use knnMatch to match feature points in the two images */
	/* TODO 1.3.3 Filter the matches using Lowe's ratio test. Save the result in good_matches */
	
	/* TODO 1.6.3 Compute execution time of FLANN */
	auto durationFlann = 0;

	/* Save execution time and number of matched feature points */
	measure << "FLANN matcher execution time: " << durationFlann << std::endl;
	measure << "FLANN matched " << good_matches.size() << " keypoints" << std::endl;
	measure << std::endl;

	/* Save results */
	cv::Mat image_matches;
	drawMatches(image1, keypoints1, image2, keypoints2, good_matches, image_matches, cv::Scalar::all(-1),
		cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	cv::imwrite("SIFT_Flann.jpg", image_matches);
}

/* Detect feature points with SIFT. Match feature points with BF */
void DetectorBFSift(cv::Mat image1, cv::Mat image2, std::ofstream& measure)
{
	cv::Ptr<cv::Feature2D> detectorSift;
	cv::Ptr<cv::DescriptorMatcher> matcherBF;

	std::vector<cv::KeyPoint> keypoints1, keypoints2;
	cv::Mat descriptors1, descriptors2;

	std::vector<cv::DMatch> matches;
	std::vector<cv::DMatch> good_matches;

	int maximumSize = 20;

	/* SIFT detector */
	/* TODO 1.2.1 Intialize SIFT detector */
	/* TODO 1.2.2 Detect keypoints and descriptors in both images */
	/* TODO 1.6.1 Compute execution time of SIFT */
	auto durationSift = 0;

	/* Save execution time and number of feature points */
	measure << "SIFT detectors execution time: " << durationSift << std::endl;
	measure << "SIFT detected " << keypoints1.size() << " and " << keypoints2.size() << " keypoints" << std::endl;

	/* BFMatcher */
	/* TODO 1.4.1 Initialize BFMatcher */
	/* TODO 1.4.2 Match the feature points in the two images */
	/* Save in good_matches the best 20 results */
	sort(begin(matches), end(matches), [](cv::DMatch a, cv::DMatch b) { return a.distance < b.distance; });

	if (maximumSize > matches.size())
	{
		maximumSize = matches.size();
	}

	for (size_t i = 0; i < maximumSize; i++)
	{
		good_matches.push_back(matches[i]);
	}

	/* TODO 1.6.4 Compute execution time of BF */
	auto durationBF = 0;

	/* Save execution time and number of matched feature points */
	measure << "BF matcher execution time: " << durationBF << std::endl;
	measure << "BF matched " << good_matches.size() << " keypoints" << std::endl;
	measure << std::endl;

	/* Save results */
	cv::Mat image_matches;
	drawMatches(image1, keypoints1, image2, keypoints2, good_matches, image_matches, cv::Scalar::all(-1),
		cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	cv::imwrite("SIFT_BF.jpg", image_matches);
}

/* Detect feature points with ORB. Match feature points with BF */
void DetectorBFOrb(cv::Mat image1, cv::Mat image2, std::ofstream& measure)
{
	cv::Ptr<cv::ORB> detectorOrb;
	cv::Ptr<cv::DescriptorMatcher> matcherBF;

	std::vector<cv::KeyPoint> keypoints1, keypoints2;
	cv::Mat descriptors1, descriptors2;

	std::vector<cv::DMatch> matches;
	std::vector<cv::DMatch> good_matches;

	int maximumSize = 20;

	/* ORB detector */
	/* TODO 1.5.1 Intialize ORB detector */
	/* TODO 1.5.2 Detect keypoints and descriptors in both images */
	/* TODO 1.6.2 Compute execution time of ORB */
	auto durationOrb = 0;

	/* Save execution time and number of feature points */
	measure << "ORB detectors execution time: " << durationOrb << std::endl;
	measure << "ORB detected " << keypoints1.size() << " and " << keypoints2.size() << " keypoints" << std::endl;

	/* Save output image */
	cv::Mat output;
	cv::drawKeypoints(image1, keypoints1, output);
	cv::imwrite("ORB.jpg", output);

	/* BFMatcher */
	/* TODO 1.4.1 Initialize BFMatcher */
	/* TODO 1.4.2 Match the feature points in the two images */
	/* Save in good_matches the best 20 results */
	sort(begin(matches), end(matches), [](cv::DMatch a, cv::DMatch b) { return a.distance < b.distance; });

	if (maximumSize > matches.size())
	{
		maximumSize = matches.size();
	}

	for (size_t i = 0; i < maximumSize; i++)
	{
		good_matches.push_back(matches[i]);
	}

	/* TODO 1.6.4 Compute execution time of BF */
	auto durationBF = 0;

	/* Save execution time and number of matched feature points */
	measure << "BF matcher execution time: " << durationBF << std::endl;
	measure << "BF matched " << good_matches.size() << " keypoints" << std::endl;
	measure << std::endl;

	/* Save results */
	cv::Mat image_matches;
	drawMatches(image1, keypoints1, image2, keypoints2, good_matches, image_matches, cv::Scalar::all(-1),
		cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	cv::imwrite("ORB_BF.jpg", image_matches);
}

void FeatureDetection()
{
	/* TODO 1.1 Read images created by Unity */
	cv::Mat image1 = cv::imread("path\\to\\image0.jpg", cv::IMREAD_COLOR);
	cv::Mat image2 = cv::imread("path\\to\\image1.jpg", cv::IMREAD_COLOR);

	/* Output file for measurements */
	std::ofstream measure;
	measure.open("measurements.txt");

	/* Detect feature points with SIFT. Match feature points with FLANN */
	DetectorFLANNSift(image1, image2, measure);

	/* Detect feature points with SIFT. Match feature points with BF */
	DetectorBFSift(image1, image2, measure);

	/* Detect feature points with ORB. Match feature points with BF.
	 * What would happen if we tried to match ORB detected feature points using FLANN?
	 */
	DetectorBFOrb(image1, image2, measure);

	/* TODO 1.7 Implement the following keypoints detectors and descriptors extractors: 
	 * SURF, FAST, STAR, BRISK, GFTT, KAZE, AKAZE
	 */
	
	measure.close();
}

int main()
{
	FeatureDetection();
	return 0;
}