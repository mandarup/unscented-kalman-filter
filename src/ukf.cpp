#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>
#include <fstream>
#include<vector>

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


    // Process noise standard deviation longitudinal acceleration in m/s^2
    std_a_ = .3;

    // Process noise standard deviation yaw acceleration in rad/s^2
    std_yawdd_ = 0.3;

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
    is_initialized_ = false;

    previous_timestamp_ = 0;

    //set state dimension
    n_x_ = 5;

    //set augmented dimension
    n_aug_ = 7;

    ///* Sigma point spreading parameter
    lambda_ = 3 - n_aug_;

    ///* Weights of sigma points
    weights_ = VectorXd(2 * n_aug_ + 1);

    ///* predicted sigma points matrix
    Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);

    // initial state vector
    x_ = VectorXd(n_x_);

    // initial covariance matrix
    // P_ = MatrixXd(n_x_, n_x_);
    P_ = MatrixXd::Identity(n_x_,n_x_);


    // set weights
    double weight_0 = lambda_/(lambda_+n_aug_);
    weights_(0) = weight_0;
    for (int i=1; i<2*n_aug_+1; i++) {  //2n+1 weights
         double weight = 0.5/(n_aug_+lambda_);
         weights_(i) = weight;
    }


}

UKF::~UKF() {}


void UKF::UpdateParamsFromFile(){
    static const std::streamsize max = std::numeric_limits<std::streamsize>::max();

    std::vector<double> attrs;
    double value;
    string param;

    std::ifstream file;
    file.open("../params.txt");

    while(file >> param >> value)
    {
        attrs.push_back(value);
    }

    // Process noise standard deviation longitudinal acceleration in m/s^2
    std_a_ = attrs[0];

    // Process noise standard deviation yaw acceleration in rad/s^2
    std_yawdd_ = attrs[1];

    // Laser measurement noise standard deviation position1 in m
    std_laspx_ = attrs[2];

    // Laser measurement noise standard deviation position2 in m
    std_laspy_ = attrs[3];

    // Radar measurement noise standard deviation radius in m
    std_radr_ = attrs[4];

    // Radar measurement noise standard deviation angle in rad
    std_radphi_ = attrs[5];

    // Radar measurement noise standard deviation radius change in m/s
    std_radrd_ = attrs[6];

    std::cout << "Init from file " << endl;

    return;
}


/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {

    /*****************************************************************************
    *  Initialization
    ****************************************************************************/
    if (!is_initialized_) {
        UpdateParamsFromFile();

        InitializeStateVector(meas_package);
    }

    /*****************************************************************************
    *  Control Structure
    ****************************************************************************/
    float delta_t = (meas_package.timestamp_ - previous_timestamp_) / 1000000.0;	//dt - expressed in seconds


    cout << "==========" << endl;
    cout << "prediction" << endl;
    cout << "==========" << endl;
    Prediction(delta_t);

    cout << "==========" << endl;
    cout << "update" << endl;
    cout << "==========" << endl;

    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
        // Radar updates
        cout << "radar" << endl;

        // UpdateRadar(meas_package);
    } else if(meas_package.sensor_type_ == MeasurementPackage::LASER) {
        cout << "lidar " << endl;
        // Laser updates
        UpdateLidar(meas_package);
    }

    cout << "end update" << endl;
    previous_timestamp_ = meas_package.timestamp_;


    // print the output
    cout << "x_ = " << x_ << endl;
    cout << "P_ = " << P_ << endl;

    // return;

}


void UKF::InitializeStateVector(MeasurementPackage meas_package){
    // first measurement
    cout << "UKF: " << endl;
    float px, py;

    x_.fill(0.0);

    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
        /**
        Convert radar from polar to cartesian coordinates and initialize state.
        */
        cout << "init radar" << endl;
        float rho = meas_package.raw_measurements_[0];
        float theta = meas_package.raw_measurements_[1];
        px = rho * cos(theta);
        py = rho * sin(theta);
        x_(0) = px;
        x_(1) = py;

    }
    else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
        /**
        Initialize state.
        */
        // cout << "init lidar" << endl;
        px = meas_package.raw_measurements_[0]; // px
        py = meas_package.raw_measurements_[1]; // py
        x_(0) = px;
        x_(1) = py;

        // // use smaller variance for lidar
        // P_(1,1) = 0.5;
        // P_(0,0) = 0.5;
    }

    previous_timestamp_ = meas_package.timestamp_;

    // done initializing, no need to predict or update
    is_initialized_ = true;
    // cout << "done initialization" << endl;

    return;
}


MatrixXd UKF::GenerateSigmaPoints(){
    /***********************
    Create augmented states
    ************************/

    //create augmented mean vector
    VectorXd x_aug = VectorXd(7);

    //create augmented state covariance
    MatrixXd P_aug = MatrixXd(7, 7);

    //create sigma point matrix
    MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);


    //create augmented mean state
    x_aug.head(5) = x_;
    x_aug(5) = 0;
    x_aug(6) = 0;

    //create augmented covariance matrix
    P_aug.fill(0.0);
    P_aug.topLeftCorner(5,5) = P_;
    P_aug(5,5) = std_a_*std_a_;
    P_aug(6,6) = std_yawdd_*std_yawdd_;

    //create square root matrix
    MatrixXd L = P_aug.llt().matrixL();

    //create augmented sigma points
    Xsig_aug.col(0)  = x_aug;
    for (int i = 0; i< n_aug_; i++)
    {
     Xsig_aug.col(i+1)       = x_aug + sqrt(lambda_+n_aug_) * L.col(i);
     Xsig_aug.col(i+1+n_aug_) = x_aug - sqrt(lambda_+n_aug_) * L.col(i);
    }

    return Xsig_aug;
}


void UKF::PredictSigmaPoints(double delta_t, MatrixXd Xsig_aug){
    /*****************************************************************************
    *  predict sigma points
    ****************************************************************************/


     //predict sigma points
    for (int i = 0; i< 2*n_aug_+1; i++)
    {
      //extract values for better readability
      double p_x = Xsig_aug(0,i);
      double p_y = Xsig_aug(1,i);
      double v = Xsig_aug(2,i);
      double yaw = Xsig_aug(3,i);
      double yawd = Xsig_aug(4,i);
      double nu_a = Xsig_aug(5,i);
      double nu_yawdd = Xsig_aug(6,i);

      //predicted state values
      double px_p, py_p;

      //avoid division by zero
      if (fabs(yawd) > 0.001) {
         px_p = p_x + v/yawd * ( sin (yaw + yawd*delta_t) - sin(yaw));
         py_p = p_y + v/yawd * ( cos(yaw) - cos(yaw+yawd*delta_t) );
      }
      else {
         px_p = p_x + v*delta_t*cos(yaw);
         py_p = p_y + v*delta_t*sin(yaw);
      }

      double v_p = v;
      double yaw_p = yaw + yawd*delta_t;
      double yawd_p = yawd;

      //add noise
      px_p = px_p + 0.5*nu_a*delta_t*delta_t * cos(yaw);
      py_p = py_p + 0.5*nu_a*delta_t*delta_t * sin(yaw);
      v_p = v_p + nu_a*delta_t;

      yaw_p = yaw_p + 0.5*nu_yawdd*delta_t*delta_t;
      yawd_p = yawd_p + nu_yawdd*delta_t;

      //write predicted sigma point into right column
      Xsig_pred_(0,i) = px_p;
      Xsig_pred_(1,i) = py_p;
      Xsig_pred_(2,i) = v_p;
      Xsig_pred_(3,i) = yaw_p;
      Xsig_pred_(4,i) = yawd_p;
    }

    return;
}




/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
    /**
    Modify the state vector, x_. Predict sigma points, the state,
    and the state covariance matrix.
    */

    MatrixXd Xsig_aug = GenerateSigmaPoints();
    // cout << "create augmented state \n" << Xsig_aug << "\n" ;
    PredictSigmaPoints(delta_t, Xsig_aug);
    // cout << "predicted sigma points" << endl;


    /*****************************************************************************
    *  predict mean and covariance
    ****************************************************************************/


    //predicted state mean
    // x_.fill(0.0);
    // for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //iterate over sigma points
    //     x_ = x_+ weights_(i) * Xsig_pred_.col(i);
    // }
    x_ = Xsig_pred_ * weights_ ;

    //predicted state covariance matrix
    P_.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //iterate over sigma points

        // state difference
        VectorXd x_diff = Xsig_pred_.col(i) - x_;
        //angle normalization
        x_diff(3) = NormalizeAngle(x_diff(3));

        P_ = P_ + weights_(i) * x_diff * x_diff.transpose() ;
        // cout << "P_ = " << P_ << endl;
    }

    return;

}




/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
  Use lidar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  calculate the lidar NIS.
  */

    MatrixXd H_ = MatrixXd(2, 5);
    H_ <<   1, 0, 0, 0, 0,
          0, 1, 0, 0, 0;
    // VectorXd z = meas_package.raw_measurements_;
    VectorXd z = VectorXd(2);
    z <<    meas_package.raw_measurements_(0),
           meas_package.raw_measurements_(1);

    VectorXd z_pred = H_ * x_;
    VectorXd z_diff = z - z_pred;
    MatrixXd Ht = H_.transpose();

    MatrixXd R_ = MatrixXd(2, 2);
    R_ <<    std_laspx_*std_laspx_, 0,
            0, std_laspy_*std_laspy_;

    MatrixXd S = H_ * P_ * Ht + R_;
    MatrixXd Si = S.inverse();
    MatrixXd PHt = P_ * Ht;
    MatrixXd K = PHt * Si;

    NIS_laser_ = z_diff.transpose() * Si * z_diff;

    //new estimate
    x_ = x_ + (K * z_diff);
    long x_size = x_.size();
    MatrixXd I = MatrixXd::Identity(x_size, x_size);
    P_ = (I - K * H_) * P_;

    return;
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

  /*******************************************************************************
 * Predict Radar Sigma Points
 ******************************************************************************/
  VectorXd z = meas_package.raw_measurements_;


  //set measurement dimension, radar can measure r, phi, and r_dot
  int n_z_ = 3;

  //create matrix for sigma points in measurement space
  MatrixXd Zsig_ = MatrixXd(n_z_, 2 * n_aug_ + 1);


  //transform sigma points into measurement space
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points

    // extract values for better readibility
    double p_x = Xsig_pred_(0,i);
    double p_y = Xsig_pred_(1,i);
    double v  = Xsig_pred_(2,i);
    double yaw = Xsig_pred_(3,i);

    double v1 = cos(yaw)*v;
    double v2 = sin(yaw)*v;

    // double r = sqrt(p_x*p_x + p_y*p_y);
    // // check for singular case
    // double eps = 0.0001;
    // double sqrt2 = 1.4142136;
    // if (r < eps*sqrt2){
    //   p_x=eps;
    //   p_y=eps;
    //   r=eps*sqrt2;
    // }
    // // measurement model
    // Zsig_(0,i) = r;                         //r
    // Zsig_(1,i) = atan2(p_y,p_x);            //phi
    // Zsig_(2,i) = (p_x*v1 + p_y*v2)/r;       //r_dot

    // // measurement model
    Zsig_(0,i) = sqrt(p_x*p_x + p_y*p_y);                        //r
    Zsig_(1,i) = atan2(p_y,p_x);                                 //phi
    Zsig_(2,i) = (p_x*v1 + p_y*v2 ) / sqrt(p_x*p_x + p_y*p_y);   //r_dot
  }

  //mean predicted measurement
  VectorXd z_pred = VectorXd(n_z_);
  z_pred.fill(0.0);
  for (int i=0; i < 2*n_aug_+1; i++) {
      z_pred = z_pred + weights_(i) * Zsig_.col(i);
  }

  //measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z_,n_z_);
  S.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points
    //residual
    VectorXd z_diff = Zsig_.col(i) - z_pred;

    //angle normalization
    // while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    // while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;
    // NormalizeAngle(&z_diff(1));
    z_diff(1) = NormalizeAngle(z_diff(1));


    S = S + weights_(i) * z_diff * z_diff.transpose();
  }

  //add measurement noise covariance matrix
  MatrixXd R = MatrixXd(n_z_,n_z_);
  R <<    std_radr_*std_radr_, 0, 0,
          0, std_radphi_*std_radphi_, 0,
          0, 0,std_radrd_*std_radrd_;
  S = S + R;


  /*******************************************************************************
 * Update Radar
 ******************************************************************************/

 //create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z_);

  //calculate cross correlation matrix
  Tc.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points

    //residual
    VectorXd z_diff = Zsig_.col(i) - z_pred;
    //angle normalization
    // while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    // while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;
    // NormalizeAngle(&z_diff(1));
    z_diff(1) = NormalizeAngle(z_diff(1));



    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    //angle normalization
    // while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
    // while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;
    // NormalizeAngle(&x_diff(3));
    x_diff(3) = NormalizeAngle(x_diff(3));


    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  //Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  //residual
  VectorXd z_diff = z - z_pred;

  //angle normalization
  // while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
  // while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;
  // NormalizeAngle(&z_diff(1));
  z_diff(1) = NormalizeAngle(z_diff(1));


  NIS_radar_ = z_diff.transpose() * S.inverse() * z_diff;


  //update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  P_ = P_ - K*S*K.transpose();

  return;
}


// void UKF::NormalizeAngle(double *angle){
//     // while (*angle > M_PI){
//     //     *angle -= 2. * M_PI;
//     // }
//     // while (*angle < -M_PI){
//     //     *angle += 2. * M_PI;
//     // }
//     *angle = fmod(*angle, M_PI);
//     cout << "angle normalization" << *angle<<  endl;
//
// }

double UKF::NormalizeAngle(double angle){
    double angle_out = angle;

    // while (angle_out > M_PI){
    //     angle_out -= 2. * M_PI;
    // }
    // while (angle_out < -M_PI){
    //     angle_out += 2. * M_PI;
    // }
    angle_out = fmod(angle, M_PI);
    // cout << "angle normalization " << angle_out << " should equal " << fmod(angle, M_PI) <<  endl;
    return angle_out;

}

// double UKF::NormalizeAngle(double phi)
// {
//   const double Max = M_PI;
//   const double Min = -M_PI;
//
//   return phi < Min
//     ? Max + std::fmod(phi - Min, Max - Min)
//     : std::fmod(phi - Min, Max - Min) + Min;
// }
