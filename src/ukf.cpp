#include "ukf.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  n_x_ = 5;
  lambda_ = 3 - n_x_;
  n_aug_ = 7;
  // initial state vector
  x_ = VectorXd(n_x_); // px, py, v, 

  // initial covariance matrix
  P_ = MatrixXd(n_x_, n_x_);

  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);
  // Process noise standard deviation longitudinal acceleration in m/s^2
  //std_a_ = 30;
  std_a_ = 2;

  // Process noise standard deviation yaw acceleration in rad/s^2
//  std_yawdd_ = 30;
  std_yawdd_ = 0.6;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  /**
  TODO:

  Complete the initialization. See ukf.h for other member properties.

  Hint: one or more values initialized above might be wildly off...
  */
  P_ << 1, 0, 0, 0, 0,
	  0, 1, 0, 0, 0,
	  0, 0, 1, 0, 0,
	  0, 0, 0, 1, 0,
	  0, 0, 0, 0, 1;
#define Uses_TUNE_P
#ifdef Uses_TUNE_P
  P_(0, 0) = std_laspx_ * std_laspx_;
  P_(1, 1) = std_laspy_ * std_laspy_;
#endif
  //set weights
  weights_ = VectorXd(2 * n_aug_ + 1);
  weights_(0) = lambda_ / (n_aug_ + lambda_);
  for (int i = 1; i <(2 * n_aug_ + 1); i++) {
	  weights_(i) = 0.5 / (lambda_ + n_aug_);
  }
  is_initialized_ = false;
#ifdef   Uses_LIDAR_LINEAR_UPDATE
  H_laser_ = MatrixXd(2, 5); // 2, 5 = n_x_
  H_laser_ << 1, 0, 0, 0, 0,
	          0, 1, 0, 0, 0;
  R_laser_ = MatrixXd(2, 2);
  R_laser_ << std_laspx_ * std_laspx_, 0,
	  0, std_laspy_ * std_laspy_; 

#endif

}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
	/**
	TODO:

	Complete this function! Make sure you switch between lidar and radar
	measurements.
	*/

	if (!is_initialized_) {
		// first measurement
		cout << "UKF: " << endl;
		
		if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
			/**
			Convert radar from polar to cartesian coordinates and initialize state.
			*/
			double rho, phi, px, py;
			rho = meas_package.raw_measurements_[0];
			phi = meas_package.raw_measurements_[1];
			py = rho * sin(phi);
			px = rho * cos(phi);
			x_ << px, py, INIT_BIKE_VELOCITY, INIT_BIKE_YAW, INIT_BIKE_YAW_D;
		}
		else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
			/**measurement_pack
			Initialize state.
			*/
			x_ << meas_package.raw_measurements_[0], meas_package.raw_measurements_[1], INIT_BIKE_VELOCITY, INIT_BIKE_YAW, INIT_BIKE_YAW_D;
		}

		// done initializing, no need to predict or update
		time_us_ = meas_package.timestamp_;

		is_initialized_ = true;
		return;
	}

	double dt = (meas_package.timestamp_ - time_us_) / 1000000.0;	//dt - expressed in seconds
	time_us_ = meas_package.timestamp_;
	Prediction(dt);
	if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
		UpdateRadar(meas_package);
	}
	else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
		UpdateLidar(meas_package);
	}
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  /**
  TODO:

  Complete this function! Estimate the object's location. Modify the state
  vector, x_. Predict sigma points, the state, and the state covariance matrix.
  */
	MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);
	AugmentedSigmaPoints(&Xsig_aug);
	
	SigmaPointPrediction(Xsig_aug, delta_t, &Xsig_pred_);

	VectorXd x_pred = VectorXd(n_x_);
	MatrixXd P_pred = MatrixXd(n_x_, n_x_);
	PredictMeanAndCovariance(&x_pred, &P_pred);
	x_ = x_pred;
	P_ = P_pred;

}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use lidar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the lidar NIS.
  */
#ifdef Uses_LIDAR_SIGMA_UPDATE
	int n_z = 2;
	VectorXd z_pred = VectorXd(n_z);
	MatrixXd S = MatrixXd(n_z, n_z);
	MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);
	PredictLidarMeasurement(&z_pred, &S, &Zsig);

	VectorXd z = VectorXd(2);
	z << meas_package.raw_measurements_[0],
		meas_package.raw_measurements_[1];

	MatrixXd Tc = MatrixXd(n_x_, n_z);

	//calculate cross correlation matrix
	Tc.fill(0.0);
	for (int i = 0; i < (2 * n_aug_ + 1); i++) {

		VectorXd z_diff = Zsig.col(i) - z_pred;

		// state difference
		VectorXd x_diff = Xsig_pred_.col(i) - x_;
		
		Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
	}
	//calculate Kalman gain K;
	//Kalman gain K;
	MatrixXd K = Tc * S.inverse();

	//update state mean and covariance matrix
	//residual
	VectorXd z_diff = z - z_pred;

	x_ = x_ + K * z_diff;
	P_ = P_ - K*S*K.transpose();
	NIS_laser_ = z_diff.transpose() * S.inverse() * z_diff;
#endif
#ifdef Uses_LIDAR_LINEAR_UPDATE
	VectorXd z = VectorXd(2);
	z << meas_package.raw_measurements_[0],
		meas_package.raw_measurements_[1];

	VectorXd z_pred = H_laser_ * x_;
	VectorXd y = z - z_pred;
	MatrixXd Ht = H_laser_.transpose();
	MatrixXd S = H_laser_ * P_ * Ht + R_laser_;
	MatrixXd Si = S.inverse();
	MatrixXd PHt = P_ * Ht;
	MatrixXd K = PHt * Si;

	//new estimate
	x_ = x_ + (K * y);
	long x_size = x_.size();
	MatrixXd I = MatrixXd::Identity(x_size, x_size);
	P_ = (I - K * H_laser_) * P_;
	NIS_laser_ = y.transpose() * S.inverse() * y;
#endif
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use radar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the radar NIS.
  */
	int n_z = 3;
	VectorXd z_pred = VectorXd(n_z);
	MatrixXd S = MatrixXd(n_z, n_z);
	MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);
	PredictRadarMeasurement( &z_pred, &S, &Zsig);

	VectorXd z = VectorXd(3);
	z << meas_package.raw_measurements_[0], 
		meas_package.raw_measurements_[1], 
		meas_package.raw_measurements_[2];
	MatrixXd Tc = MatrixXd(n_x_, n_z);

	//calculate cross correlation matrix
	Tc.fill(0.0);
	for (int i = 0; i < (2 * n_aug_ + 1); i++) {

		VectorXd z_diff = Zsig.col(i) - z_pred;
		//angle normalization
		while (z_diff(1)> M_PI) z_diff(1) -= 2.*M_PI;
		while (z_diff(1)<-M_PI) z_diff(1) += 2.*M_PI;

		// state difference
		VectorXd x_diff = Xsig_pred_.col(i) - x_;
		//angle normalization
		while (x_diff(3)> M_PI) x_diff(3) -= 2.*M_PI;
		while (x_diff(3)<-M_PI) x_diff(3) += 2.*M_PI;
		Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
	}
	//calculate Kalman gain K;
	//Kalman gain K;
	MatrixXd K = Tc * S.inverse();

	//update state mean and covariance matrix
	//residual
	VectorXd z_diff = z - z_pred;

	//angle normalization
	while (z_diff(1)> M_PI) z_diff(1) -= 2.*M_PI;
	while (z_diff(1)<-M_PI) z_diff(1) += 2.*M_PI;
	x_ = x_ + K * z_diff;
	P_ = P_ - K*S*K.transpose();
	// calculate radar NIS
	NIS_radar_ = z_diff.transpose() * S.inverse() * z_diff;
}
void UKF::AugmentedSigmaPoints(MatrixXd* Xsig_out) {
	//create augmented mean vector
	VectorXd x_aug = VectorXd(n_aug_);
	x_aug.fill(0.0);
	x_aug.head(n_x_) = x_;
	//create augmented state covariance
	MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);
	P_aug.fill(0.0);
	P_aug.topLeftCorner(n_x_, n_x_) = P_;
	P_aug(n_x_, n_x_) = std_a_*std_a_;
	P_aug(n_x_ + 1, n_x_ + 1) = std_yawdd_*std_yawdd_;
	//create sigma point matrix
	MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);
	//create square root matrix
	MatrixXd S = P_aug.llt().matrixL();
	Xsig_aug.col(0) = x_aug;
	for (int i = 0; i < n_aug_; i++) {
		Xsig_aug.col(1 + i) = x_aug + sqrt(lambda_ + n_aug_) * S.col(i);
		Xsig_aug.col(1 + n_aug_ + i) = x_aug - sqrt(lambda_ + n_aug_) * S.col(i);
	}
	*Xsig_out = Xsig_aug;
}
void UKF::SigmaPointPrediction(MatrixXd& Xsig_aug, double delta_t, MatrixXd* Xsig_out ) {
	const float epsilon = 0.0001;
	MatrixXd Xsig_pred = MatrixXd(n_x_, 2 * n_aug_ + 1);
	for (int i = 0; i < (2 * n_aug_ + 1); i++) {
		float px = Xsig_aug(0, i);
		float py = Xsig_aug(1, i);
		float v = Xsig_aug(2, i);
		float yaw = Xsig_aug(3, i);
		float yawd = Xsig_aug(4, i);
		double nu_a = Xsig_aug(5, i);
		double nu_yawdd = Xsig_aug(6, i);
		if (fabs(yawd) > epsilon) {
			Xsig_pred(0, i) = px + v / yawd * (sin(yaw + yawd *delta_t) - sin(yaw));
			Xsig_pred(1, i) = py + v / yawd * (-cos(yaw + yawd *delta_t) + cos(yaw));
			Xsig_pred(3, i) = yaw + yawd * delta_t;
		}
		else {
			Xsig_pred(0, i) = px + v * cos(yaw * delta_t);
			Xsig_pred(1, i) = py + v * sin(yaw * delta_t);
			Xsig_pred(3, i) = yaw;
		}
		Xsig_pred(2, i) = v;
		Xsig_pred(4, i) = yawd;
		// add noise
		Xsig_pred(0, i) += 0.5 * delta_t * delta_t * cos(yaw) * nu_a;
		Xsig_pred(1, i) += 0.5 * delta_t * delta_t * sin(yaw) * nu_a;
		Xsig_pred(2, i) += delta_t * nu_a;
		Xsig_pred(3, i) += 0.5 * delta_t * delta_t * nu_yawdd;
		Xsig_pred(4, i) += delta_t * nu_yawdd;
	}
	*Xsig_out = Xsig_pred;
}

void UKF::PredictMeanAndCovariance(VectorXd* x_out, MatrixXd* P_out) {

	//create vector for predicted state
	VectorXd x = VectorXd(n_x_);

	//create covariance matrix for prediction
	MatrixXd P = MatrixXd(n_x_, n_x_);

	//predict state mean
	x.fill(0.0);
	for (int i = 0; i <(2 * n_aug_ + 1); i++) {
		x += weights_(i) * Xsig_pred_.col(i);
	}
	//predict state covariance matrix
	P.fill(0.0);
	for (int i = 0; i <(2 * n_aug_ + 1); i++) {
		// state difference
		VectorXd x_diff = Xsig_pred_.col(i) - x;
		//angle normalization
		while (x_diff(3)> M_PI) x_diff(3) -= 2.*M_PI;
		while (x_diff(3)<-M_PI) x_diff(3) += 2.*M_PI;

		P = P + weights_(i) * x_diff * x_diff.transpose();
	}
	*x_out = x;
	*P_out = P;
}
void UKF::PredictRadarMeasurement(VectorXd* z_out, MatrixXd* S_out, MatrixXd *Zsig_out) {
	//create matrix for sigma points in measurement space
	const float epsilon = 0.0001;
	int n_z = 3;
	MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);

	//mean predicted measurement
	VectorXd z_pred = VectorXd(n_z);

	//measurement covariance matrix S
	MatrixXd S = MatrixXd(n_z, n_z);

	//transform sigma points into measurement space
	Zsig.fill(0.0);
	for (int i = 0; i < (2 * n_aug_ + 1); i++) {
		double px = Xsig_pred_(0, i);
		double py = Xsig_pred_(1, i);
		double v = Xsig_pred_(2, i);
		double yaw = Xsig_pred_(3, i);
		double yawd = Xsig_pred_(4, i);
		double rho = sqrt(px*px + py*py);
		double phi = atan2(py, px);
		double rho1 = rho;
		if (rho1 < epsilon)
			rho1 = epsilon;
		double rhod = (px * cos(yaw)*v + py * sin(yaw) * v) / rho1;
		Zsig(0, i) = rho;
		Zsig(1, i) = phi;
		Zsig(2, i) = rhod;
	}
	//calculate mean predicted measurement
	z_pred.fill(0.0);
	for (int i = 0; i < (2 * n_aug_ + 1); i++) {
		z_pred += Zsig.col(i) * weights_(i);
	}
	//calculate measurement covariance matrix S
	S.fill(0.0);
	for (int i = 0; i < (2 * n_aug_ + 1); i++) {
		VectorXd x_diff = Zsig.col(i) - z_pred;
		//angle normalization
		while (x_diff(1)> M_PI) x_diff(1) -= 2.*M_PI;
		while (x_diff(1)<-M_PI) x_diff(1) += 2.*M_PI;

		S += weights_(i) * x_diff * x_diff.transpose();
	}
	// add noises
	S(0, 0) += std_radr_ * std_radr_;
	S(1, 1) += std_radphi_ * std_radphi_;
	S(2, 2) += std_radrd_ * std_radrd_;

	//write result
	*z_out = z_pred;
	*S_out = S;
	*Zsig_out = Zsig;

}
#ifdef Uses_LIDAR_SIGMA_UPDATE
void UKF::PredictLidarMeasurement(VectorXd* z_out, MatrixXd* S_out, MatrixXd *Zsig_out) {

	//create matrix for sigma points in measurement space
	int n_z = 2;
	MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);

	//mean predicted measurement
	VectorXd z_pred = VectorXd(n_z);

	//measurement covariance matrix S
	MatrixXd S = MatrixXd(n_z, n_z);

	//transform sigma points into measurement space
	Zsig.fill(0.0);
	for (int i = 0; i < (2 * n_aug_ + 1); i++) {
		double px = Xsig_pred_(0, i);
		double py = Xsig_pred_(1, i);

		Zsig(0, i) = px;
		Zsig(1, i) = py;
	}
	//calculate mean predicted measurement
	z_pred.fill(0.0);
	for (int i = 0; i < (2 * n_aug_ + 1); i++) {
		z_pred += Zsig.col(i) * weights_(i);
	}
	//calculate measurement covariance matrix S
	S.fill(0.0);
	for (int i = 0; i < (2 * n_aug_ + 1); i++) {
		VectorXd x_diff = Zsig.col(i) - z_pred;

		S += weights_(i) * x_diff * x_diff.transpose();
	}
	// add noises
	S(0, 0) += std_laspx_ * std_laspx_;
	S(1, 1) += std_laspy_ * std_laspy_;

	//write result
	*z_out = z_pred;
	*S_out = S;
	*Zsig_out = Zsig;
}
#endif
