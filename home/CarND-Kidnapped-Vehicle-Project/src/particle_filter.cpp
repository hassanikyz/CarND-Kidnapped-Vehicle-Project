/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>
#include <cassert>
#include "helper_functions.h"

using std::string;
using std::vector;

#include <random> // Need this for sampling from distributions

using std::normal_distribution;
using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  num_particles = 900;  // TODO: Set the number of particles

  std::default_random_engine gen;
  double std_x, std_y, std_theta;  // Standard deviations for x, y, and theta

  std_x = std[0];
  std_y = std[1];
  std_theta = std[2];

  // This line creates a normal (Gaussian) distribution for x
  normal_distribution<double> dist_x(x, std_x);

  // This line create normal distributions for y and theta
  normal_distribution<double> dist_y(y, std_y);
  normal_distribution<double> dist_theta(theta, std_theta);

  weights = std::vector<double>(static_cast<unsigned long>(num_particles), 1.0);
  particles = std::vector<Particle>(static_cast<unsigned long>(num_particles)); 
  for (int i = 0; i < num_particles; ++i) {
    
    //Take samples from these normal distributions like this: 
    particles[i].x = dist_x(gen); //   where "gen" is the random engine initialized earlier.
    particles[i].y = dist_y(gen);
    particles[i].theta = dist_theta(gen);
 	particles[i].weight = weights[i];
    particles[i].id = i; // make particle's identifier a particle's initial position in the particles array
  }

  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  std::default_random_engine gen;
  normal_distribution<double> dist_x(0.0, std_pos[0]);
  normal_distribution<double> dist_y(0.0, std_pos[1]);
  normal_distribution<double> dist_theta(0.0, std_pos[2]);

  // for each particle predict its position and add Gaussian noise
  for (auto &part : particles) {
    // to avoid divid by zero
    if (fabs(yaw_rate) < 0.000001) {
      // predict without Gaussian noise with 0 yaw rate
      part.x += velocity * delta_t * cos(part.theta);
      part.y += velocity * delta_t * sin(part.theta);
    } else {
      // w/o Gaussian noise but with non-zero yaw-rate
      double yr_delta_t = yaw_rate * delta_t; 
      double v_dividedby_yaw_rate = velocity / yaw_rate;
      part.x += v_dividedby_yaw_rate * (sin(part.theta + yr_delta_t) - sin(part.theta));
      part.y += v_dividedby_yaw_rate * (cos(part.theta) - cos(part.theta + yr_delta_t));
      part.theta += yr_delta_t;
    }


    //  add Gaussian noise
    part.x += dist_x(gen);
    part.y += dist_y(gen);
    part.theta += dist_theta(gen);
  }
  

}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
  // let's find the closest predicted particle to each by going through all one by one
  for (auto &obs : observations) {
    // initialize min value to max possible double
    double mindist = numeric_limits<double>::max();

    // initialize particle id to -1 to make sure that a mapping was found for each observation
    obs.id = -1;

    // find the closest match
    for (auto const &pred_obs : predicted) {
      double curdistance = dist(pred_obs.x, pred_obs.y, obs.x, obs.y);

      // update the closest match if found even a closer one
      if (curdistance <= mindist) {
        mindist = curdistance;
        obs.id = pred_obs.id;
      }
    }

    // assert that we found some mapping
    assert(obs.id != -1);
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */


   // steps in sequence for each particle
  for (auto j = 0; j < particles.size(); j++) {
    Particle const &particle = particles[j];
    //
    // Step 1: transform vehicle observations to the map coordinates
    //

    unsigned int num_of_obser = observations.size();
    vector<LandmarkObs> transf_observations(num_of_obser);
    for (auto i = 0; i < num_of_obser; i++) {
      double costheta = cos(particle.theta);
      double sintheta = sin(particle.theta);

      LandmarkObs obs = observations[i];
      transf_observations[i].x = particle.x + costheta * obs.x - sintheta * obs.y;
      transf_observations[i].y = particle.y + sintheta * obs.x + costheta * obs.y;
      transf_observations[i].id = -1;  // at this point we don't know corresponding observation yet
    }


    //
    // Step 2: Associate transformed observation with landmark identifier
    //

    // build an array of landmarks that are within the sensor range
    vector<LandmarkObs> landmarks;
    for (unsigned int k=0; k<map_landmarks.landmark_list.size(); ++k) {
//       single_landmark_s landmark = map_landmarks.landmark_list[k];
      //Get id and x,y coordinates
      float lm_x = map_landmarks.landmark_list[k].x_f;
      float lm_y = map_landmarks.landmark_list[k].y_f;
      int lm_id = map_landmarks.landmark_list[k].id_i;
      double dist_particle = dist(particle.x, particle.y, lm_x, lm_y);
       if ( dist_particle <= sensor_range) {
            LandmarkObs lmark_obs = {
                .id = lm_id,
                .x = static_cast<double>(lm_x),
                .y = static_cast<double>(lm_y),
            };
       		landmarks.push_back(lmark_obs);
       }
    }
      

    // There should be at least one landmark within the sensor range
    assert(!landmarks.empty());

    // Now associate transformed observations with landmarks
    dataAssociation(landmarks, transf_observations);


    //
    // Step 3: update particle's weight
    //

    // Now determine measurement probabilities
    vector<double> obsprob(transf_observations.size());
    particles[j].weight = 1.0;  // set to 1 for multiplication in the end of the loop
    for (auto i = 0; i < observations.size(); i++) {
      LandmarkObs tobs = transf_observations[i];
      LandmarkObs nearest_landmark = {
          .id = -1,  // not important here
          .x = static_cast<double>(map_landmarks.landmark_list[tobs.id - 1].x_f), // landmark index starts at 1
          .y = static_cast<double>(map_landmarks.landmark_list[tobs.id - 1].y_f),
      };

      // helper variables
      double x_diff_2 = pow(tobs.x - nearest_landmark.x, 2.0);
      double y_diff_2 = pow(tobs.y - nearest_landmark.y, 2.0);
      double std_x_2 = pow(std_landmark[0], 2.0);
      double std_y_2 = pow(std_landmark[1], 2.0);

      // formula of multivariate Gaussian probability
      obsprob[i] = (1 / (2 * M_PI * std_landmark[0] * std_landmark[1])) *
                                     exp(-(x_diff_2 / (2 * std_x_2) + y_diff_2 / (2 * std_y_2)));

      // (3.2) combine probabilities (particle's final weight)
      particles[j].weight *= obsprob[i];
    }

    // set calculated particle weight in the weights array
    weights[j] = particles[j].weight;
  }
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  default_random_engine gen;
  discrete_distribution<size_t> dist_index(weights.begin(), weights.end());

  vector<Particle> resampled_particles(particles.size());

  for (unsigned int i = 0; i < particles.size(); i++) {
    resampled_particles[i] = particles[dist_index(gen)];

  }

  particles = resampled_particles;
  

}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}