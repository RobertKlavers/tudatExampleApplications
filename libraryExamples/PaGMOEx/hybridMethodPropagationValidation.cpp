/*    Copyright (c) 2010-2018, Delft University of Technology
 *    All rigths reserved
 *
 *    This file is part of the Tudat. Redistribution and use in source and
 *    binary forms, with or without modification, are permitted exclusively
 *    under the terms of the Modified BSD license. You should have received
 *    a copy of the license with this file. If not, please or visit:
 *    http://tudat.tudelft.nl/LICENSE.
 */

#include <Eigen/Dense>
#include <cmath>
#include <Tudat/Astrodynamics/LowThrustTrajectories/hybridMethod.h>
#include "Tudat/Astrodynamics/Ephemerides/approximatePlanetPositions.h"
#include "Tudat/Astrodynamics/BasicAstrodynamics/celestialBodyConstants.h"
#include "Problems/applicationOutput.h"
#include <json.hpp>
#include <boost/format.hpp>

using json = nlohmann::json;

void propagateArc(int numberOfSteps, double archLength) {

}

int main() {
    using namespace tudat;
    using namespace tudat::input_output;
    using namespace tudat::simulation_setup;
    using namespace tudat::low_thrust_trajectories;
    using namespace tudat::propagators;
    using namespace low_thrust_trajectories;
    spice_interface::loadStandardSpiceKernels();

    // 0 Load configuration files
    const std::string cppFilePath(__FILE__);
    const std::string cppFolder = cppFilePath.substr(0, cppFilePath.find_last_of("/\\") + 1);
    const std::string configurationFile = "hybridMethodPropagationValidationConfiguration.json";

    // Read JSON file
    std::ifstream inputstream(cppFolder + configurationFile);
    json input_data;
    inputstream >> input_data;

    // Read Hybrid Optimisation Settings
    inputstream.close();
    inputstream.open(cppFolder + "OptimizerSettings.json");
    json optimizer_settings;
    inputstream >> optimizer_settings;

    // 1 Set up Environment
    const std::string propagationCase = input_data["case"];
    json caseConfiguration = input_data[propagationCase];

    // Vehicle Settings
    double maximumThrust = caseConfiguration["vehicle"]["maximumThrust"];
    double specificImpulse = caseConfiguration["vehicle"]["specificImpulse"];
    double initialMass = caseConfiguration["vehicle"]["initialMass"];

    std::string bodyToPropagate = "Vehicle";
    std::string centralBody = "Earth";

    std::function<double(const double)> specificImpulseFunction = [=](const double currentTime) {
        return specificImpulse;
    };

    // Initial Trajectory Settings
    double initialTime = 0.0 * physical_constants::JULIAN_DAY;
    double timeOfFlight = caseConfiguration["trajectory"]["timeOfFlight"].get<double>() * physical_constants::JULIAN_DAY;

    // Define body settings for simulation.
    std::vector<std::string> bodiesToCreate;
    bodiesToCreate.push_back(centralBody);

    // Create body objects.
    std::map<std::string, std::shared_ptr<simulation_setup::BodySettings> > bodySettings =
            simulation_setup::getDefaultBodySettings(bodiesToCreate,
                                                     initialTime - 300.0,
                                                     initialTime + timeOfFlight + 300.0);
    for (auto & i : bodiesToCreate) {
        bodySettings[i]->ephemerisSettings->resetFrameOrientation("ECLIPJ2000");
        bodySettings[i]->rotationModelSettings->resetOriginalFrame("ECLIPJ2000");
    }

    // Create the BodyMap given inital settings
    simulation_setup::NamedBodyMap bodyMap = createBodies(bodySettings);

    // Create spacecraft object.
    bodyMap[bodyToPropagate] = std::make_shared<simulation_setup::Body>();

    // Finalize body creation.
    setGlobalFrameBodyEphemerides(bodyMap, centralBody, "ECLIPJ2000");

    double centralBodyGravitationalParameter = bodyMap.at(
            centralBody)->getGravityFieldModel()->getGravitationalParameter();

    // TODO: Refactor this out of HybridMethodModel
    Eigen::Vector6d epsilon_upper(optimizer_settings["epsilon_upper"].get<std::vector<double>>().data());
    Eigen::Vector6d constraint_weights(optimizer_settings["weight_constraints"].get<std::vector<double>>().data());
    double weightMass = optimizer_settings["weight_mass"];
    double weightTOF = optimizer_settings["weight_tof"];
    // Set vehicle mass.
    bodyMap[bodyToPropagate]->setConstantBodyMass(initialMass);

    // Validation Cases Setup
    std::vector<int> numberOfStepsVector = {30, 40, 50};
    std::vector<double> arcLenghts = {3.0, 4.0, 5.0};
    std::vector<PropagationType> propagationTypes = {PropagationType::tangential, PropagationType::radial, PropagationType::outofplane};

    // 2 Set up initial Conditions
    // Retrieve initial and final Keplerian Elements
    Eigen::Vector6d initialKeplerianElements(caseConfiguration["trajectory"]["initialKeplerianElements"].get<std::vector<double>>().data());
    Eigen::Vector6d finalKeplerianElements(caseConfiguration["trajectory"]["initialKeplerianElements"].get<std::vector<double>>().data());

    // Initial and final states in cartesian coordinates.
    Eigen::Vector6d stateAtDeparture = orbital_element_conversions::convertKeplerianToCartesianElements(
            initialKeplerianElements, centralBodyGravitationalParameter);
    Eigen::Vector6d stateAtArrival = orbital_element_conversions::convertKeplerianToCartesianElements(
            finalKeplerianElements, centralBodyGravitationalParameter);

    // Used for CI
    std::shared_ptr<numerical_integrators::IntegratorSettings<double> > integratorSettings =
            std::make_shared<numerical_integrators::IntegratorSettings<double> >
                    (numerical_integrators::rungeKutta4, 0.0, 2.0 * mathematical_constants::PI / 30);

    // Cannot make the model without initial and final costates (yet)
    Eigen::VectorXd initialCostates = Eigen::VectorXd::Map(caseConfiguration["trajectory"]["initialCostates"].get<std::vector<double>>().data(), 6);
    Eigen::VectorXd finalCostates = Eigen::VectorXd::Map(caseConfiguration["trajectory"]["finalCostates"].get<std::vector<double>>().data(), 6);

    for (PropagationType propagationType: propagationTypes) {
        std::cout << "== " << propagationType << " ==" << std::endl;
        std::shared_ptr<simulation_setup::HybridOptimisationSettings> hybridOptimisationSettings =
                std::make_shared<simulation_setup::HybridOptimisationSettings>(epsilon_upper, constraint_weights, weightMass, weightTOF, false, propagationType);

        HybridMethodModel hybridMethodModel = HybridMethodModel(
                stateAtDeparture, stateAtArrival, initialCostates, finalCostates, maximumThrust,
                specificImpulse,
                timeOfFlight,
                bodyMap, bodyToPropagate, centralBody, integratorSettings, hybridOptimisationSettings);

        std::map<double, Eigen::VectorXd> benchmark_trajectory = hybridMethodModel.propagateTrajectoryBenchmark(10.0);
        std::pair<std::map< double, Eigen::VectorXd >, std::map< double, Eigen::VectorXd >> ci_trajectory_output = hybridMethodModel.getTrajectoryOutput();


        input_output::writeDataMapToTextFile( ci_trajectory_output.first,
                                              "OA_" + std::to_string(propagationType) + "_CI_Trajectory.dat",
                                              tudat_pagmo_applications::getOutputPath(),
                                              "",
                                              std::numeric_limits< double >::digits10,
                                              std::numeric_limits< double >::digits10,
                                              "," );

        input_output::writeDataMapToTextFile( benchmark_trajectory,
                                              "OA_" + std::to_string(propagationType) + "_Benchmark_Trajectory.dat",
                                              tudat_pagmo_applications::getOutputPath(),
                                              "",
                                              std::numeric_limits< double >::digits10,
                                              std::numeric_limits< double >::digits10,
                                              "," );

        for (int numberOfSteps: numberOfStepsVector) {
            for(double arcLength: arcLenghts) {
                std::cout << "  Running for nk: " << numberOfSteps << ", t_OA: " << arcLength << " days\n";
                std::map<double, Eigen::Vector6d> oa_trajectory = hybridMethodModel.propagateTrajectoryOA(arcLength * physical_constants::JULIAN_DAY, numberOfSteps);
                string fileName = "OA_" +std::to_string(propagationType) + "_" + std::to_string(numberOfSteps) + "_" + std::to_string(int (arcLength)) + "_Trajectory.dat";
                input_output::writeDataMapToTextFile( oa_trajectory,
                                                      fileName,
                                                      tudat_pagmo_applications::getOutputPath(),
                                                      "",
                                                      std::numeric_limits< double >::digits10,
                                                      std::numeric_limits< double >::digits10,
                                                      "," );

            }
        }
    }







        // Set up the single leg model



        // std::pair<std::map< double, Eigen::VectorXd >, std::map< double, Eigen::VectorXd >> optimalTrajectory = hybridMethodModelTest.getTrajectoryOutput();
        // std::pair<Eigen::VectorXd, Eigen::Vector6d> fitnessResults = hybridMethodModelTest.calculateFitness();
        //
        // std::cout << "Final Mass: " << hybridMethodModelTest.getMassAtTimeOfFlight() << std::endl;
        // std::cout << "Fitness: " << fitnessResults.first.sum() << std::endl;
        //
        //
        // // Temporary stuff to directly store the dependent variable history (hopefully containing thrust acceleration profile)
        // std::cout << "Exporting Trajectory History!!" << std::endl;
        // input_output::writeDataMapToTextFile( optimalTrajectory.first,
        //                                       propagationCase + "_Trajectory.dat",
        //                                       tudat_pagmo_applications::getOutputPath(),
        //                                       "",
        //                                       std::numeric_limits< double >::digits10,
        //                                       std::numeric_limits< double >::digits10,
        //                                       "," );
        // input_output::writeDataMapToTextFile( optimalTrajectory.second,
        //                                       propagationCase + "_DependentVariables.dat",
        //                                       tudat_pagmo_applications::getOutputPath(),
        //                                       "",
        //                                       std::numeric_limits< double >::digits10,
        //                                       std::numeric_limits< double >::digits10,
        //                                       "," );


    // TODO 5 Export results

    return EXIT_SUCCESS;
}
