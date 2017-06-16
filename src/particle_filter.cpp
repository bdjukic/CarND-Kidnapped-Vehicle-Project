/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	default_random_engine gen;
	
	num_particles = 100;
	
	// Create normal (Gaussian) distributions for x, y, and theta
	normal_distribution<double> dist_x(0, std[0]);
	normal_distribution<double> dist_y(0, std[1]);
	normal_distribution<double> dist_theta(0, std[2]);
	
	for (int i = 0; i < num_particles; i++) {
		double initial_weight = 1.0;
		double noise_x = dist_x(gen);
		double noise_y = dist_y(gen);
		double noise_theta = dist_theta(gen);	 
		
		Particle particle;
		
		particle.id = i;
		particle.x = x + noise_x;
		particle.y = y + noise_y;
		particle.theta = theta + noise_theta;
		particle.weight = initial_weight;
		
		particles.push_back(particle);
		weights.push_back(initial_weight);
	}
	
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	default_random_engine gen;
	
	// Create normal (Gaussian) distributions for predicted x, y, and theta
	normal_distribution<double> dist_x(0, std_pos[0]);
	normal_distribution<double> dist_y(0, std_pos[1]);
	normal_distribution<double> dist_theta(0, std_pos[2]);
		
	for (auto& particle : particles) {
		if (fabs(yaw_rate) < 0.00001) {
			// Cover the case when vehcile is going straight
			particle.x += velocity * delta_t * cos(particle.theta);
			particle.y += velocity * delta_t * sin(particle.theta);
		} else {
			particle.x += velocity / yaw_rate * (sin(particle.theta + yaw_rate * delta_t) - sin(particle.theta));
			particle.y += velocity / yaw_rate * (cos(particle.theta) - cos(particle.theta + yaw_rate * delta_t));
			particle.theta += yaw_rate * delta_t;
		}
		
		// Creating noise for x, y, and theta
		double noise_x = dist_x(gen);
		double noise_y = dist_y(gen);
		double noise_theta = dist_theta(gen);
		
		// Update particle with the new predicted states
		particle.x += noise_x;
		particle.y += noise_y;
		particle.theta += noise_theta;	 
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {	
	for (auto& observation : observations) {
		double minimal_distance = std::numeric_limits<double>::max();
		
		for (unsigned int predicted_index = 0; predicted_index < predicted.size(); predicted_index++) {
			LandmarkObs prediction = predicted[predicted_index];
			
			double current_distance = dist(observation.x, observation.y, prediction.x, prediction.y);
			
			// Assigning a corresponding landmark ID to the observation based on nearest neighbour method. 
			if (current_distance < minimal_distance) {
				minimal_distance = current_distance;
				observation.id = prediction.id;
			}
		}
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], std::vector<LandmarkObs> observations, Map map_landmarks) {
	for (auto& particle : particles) {
		// Transform observation coords from vehicle's to map's coordinate system
		vector<LandmarkObs> transformed_observations;
		
		for (unsigned int observation_index = 0; observation_index < observations.size(); observation_index++) {
			LandmarkObs observation = observations[observation_index];
			
			double transformed_x = cos(particle.theta) * observation.x - sin(particle.theta) * observation.y + particle.x;
			double transformed_y = sin(particle.theta) * observation.x + cos(particle.theta) * observation.y + particle.y;
			
			transformed_observations.push_back(LandmarkObs{ observation.id, transformed_x, transformed_y });
		}
		
		// Find landmarks around the current particle which are in the range of the sensor
		vector<LandmarkObs> landmarks_in_sensor_range;
		
		for (unsigned int landmark_index = 0; landmark_index < map_landmarks.landmark_list.size(); landmark_index++) {
			float landmark_x = map_landmarks.landmark_list[landmark_index].x_f;
			float landmark_y = map_landmarks.landmark_list[landmark_index].y_f;
			int landmark_id = map_landmarks.landmark_list[landmark_index].id_i;
			
			if (dist(particle.x, particle.y, landmark_x, landmark_y) < sensor_range) {
				landmarks_in_sensor_range.push_back(LandmarkObs{ landmark_id, landmark_x, landmark_y });
			}
		} 
		
		// Associate observations with the related map landmarks
		dataAssociation(landmarks_in_sensor_range, transformed_observations);
		
		// Resetting particle's weight
		particle.weight = 1.0;
		
		for (unsigned int observation_index = 0; observation_index < transformed_observations.size(); observation_index++) {
			LandmarkObs observation = transformed_observations[observation_index];
			LandmarkObs landmark;
			
			// Find related landmark for the observation
			for (unsigned int landmark_index = 0; landmark_index < landmarks_in_sensor_range.size(); landmark_index++) {
				LandmarkObs current_landmark = landmarks_in_sensor_range[landmark_index];
				
				if (current_landmark.id == observation.id) {
					landmark = current_landmark;
				}
			}
			
			// Calculating the particle's final weight using Multivariate-Gaussian probability
			double first_exponent_argument = pow(landmark.x - observation.x, 2) / (2.0 * pow(std_landmark[0], 2));
			double second_exponent_argument = pow(landmark.y - observation.y, 2) / (2.0 * pow(std_landmark[1], 2));
			
			particle.weight *= (1.0 / (2.0 * M_PI * std_landmark[0] * std_landmark[1])) * exp(-(first_exponent_argument + second_exponent_argument));  
		}
	}
}

void ParticleFilter::resample() {
	default_random_engine gen;
	vector<Particle> resampled_particles;
    
    for (int particle_index = 0; particle_index < num_particles; particle_index++) {
    cout << particles[particle_index].weight << endl;
    
    	weights.push_back(particles[particle_index].weight);
	}

	// Random starting index for the wheel
	uniform_int_distribution<int> uniintdist(0, num_particles - 1);
	int random_starting_index = uniintdist(gen);

	double maximum_weight = *max_element(weights.begin(), weights.end());
	uniform_real_distribution<double> unirealdist(0.0, maximum_weight);

	double beta = 0.0;

	for (int particle_index = 0; particle_index < num_particles; particle_index++) {
		beta += unirealdist(gen) * 2.0;
    
		while (beta > weights[random_starting_index]) {
			beta -= weights[random_starting_index];
			random_starting_index = (random_starting_index + 1) % num_particles;
		}
    
		resampled_particles.push_back(particles[random_starting_index]);
	}
	
	weights.clear();
	particles = resampled_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
