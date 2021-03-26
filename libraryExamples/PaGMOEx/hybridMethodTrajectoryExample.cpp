/*    Copyright (c) 2010-2018, Delft University of Technology
 *    All rigths reserved
 *
 *    This file is part of the Tudat. Redistribution and use in source and
 *    binary forms, with or without modification, are permitted exclusively
 *    under the terms of the Modified BSD license. You should have received
 *    a copy of the license with this file. If not, please or visit:
 *    http://tudat.tudelft.nl/LICENSE.
 */

#include <boost/shared_ptr.hpp>
#include <Eigen/Dense>
#include <cmath>
#include <iostream>
#include <Tudat/Astrodynamics/LowThrustTrajectories/hybridMethod.h>
#include <Tudat/Astrodynamics/LowThrustTrajectories/hybridOptimisationSetup.h>

#include "Tudat/SimulationSetup/tudatSimulationHeader.h"
#include "Tudat/Astrodynamics/Ephemerides/approximatePlanetPositions.h"
#include "pagmo/algorithms/de1220.hpp"
#include "pagmo/algorithms/de.hpp"
#include "Problems/applicationOutput.h"
#include "Tudat/Astrodynamics/BasicAstrodynamics/celestialBodyConstants.h"

// std::map< double, Eigen::Vector6d > propagateBenchmark() {
//     std::cout << "omg " << std::endl;
// }

int main() {
    using namespace tudat;
    using namespace tudat::input_output;
    using namespace tudat::simulation_setup;
    using namespace tudat::low_thrust_trajectories;
    using namespace tudat::propagators;

    using namespace low_thrust_trajectories;

    spice_interface::loadStandardSpiceKernels();

    double maximumThrust = 0.401706;
    double specificImpulse = 3300.0;
    double initialMass = 1200.0;

    std::function<double(const double)> specificImpulseFunction = [=](const double currentTime) {
        return specificImpulse;
    };

    double julianDate = 0.0 * physical_constants::JULIAN_DAY;
    double timeOfFlight = 218 * physical_constants::JULIAN_DAY;

    // Define body settings for simulation.
    std::vector<std::string> bodiesToCreate;
    // bodiesToCreate.push_back( "Sun" );
    bodiesToCreate.push_back("Earth");

    // Create body objects.
    std::map<std::string, std::shared_ptr<simulation_setup::BodySettings> > bodySettings =
            simulation_setup::getDefaultBodySettings(bodiesToCreate, julianDate - 300.0,
                                                     julianDate + timeOfFlight + 300.0);
    for (unsigned int i = 0; i < bodiesToCreate.size(); i++) {
        bodySettings[bodiesToCreate.at(i)]->ephemerisSettings->resetFrameOrientation("ECLIPJ2000");
        bodySettings[bodiesToCreate.at(i)]->rotationModelSettings->resetOriginalFrame("ECLIPJ2000");
    }
    simulation_setup::NamedBodyMap bodyMap = createBodies(bodySettings);


    // Create spacecraft object.
    bodyMap["Vehicle"] = std::make_shared<simulation_setup::Body>();

    // Finalize body creation.
    setGlobalFrameBodyEphemerides(bodyMap, "Earth", "ECLIPJ2000");


    std::string bodyToPropagate = "Vehicle";
    std::string centralBody = "Earth";
    double centralBodyGravitationalParameter = bodyMap.at(
            centralBody)->getGravityFieldModel()->getGravitationalParameter();

    // Set vehicle mass.
    bodyMap[bodyToPropagate]->setConstantBodyMass(initialMass);

    // Initial and final states in keplerian elements.
    // Eigen::Vector6d initialKeplerianElements = (
    //         Eigen::Vector6d( ) << 7000.0e3,
    //         0.001,
    //         0.005,
    //         1.0e-12,
    //         1.0e-12,
    //         1.0e-12).finished( );
    // Eigen::Vector6d finalKeplerianElements = (
    //         Eigen::Vector6d( ) << 9000.0e3,
    //         0.001,
    //         5.0 * mathematical_constants::PI/180.0,
    //         1.0e-12,
    //         1.0e-12,
    //         1.0e-12 ).finished( );

    std::cout << "Setting up initial statestuff" << std::endl;

    // Initial and final states in keplerian elements.
    Eigen::Vector6d initialKeplerianElements = (
            Eigen::Vector6d() << 24050.9e3,
                    0.725,
                    7.0 * mathematical_constants::PI / 180.0,
                    1.0e-12,
                    1.0e-12,
                    1.0e-12).finished();
    Eigen::Vector6d initialTestKeplerianElements = (
            Eigen::Vector6d() << 6927.0e3,
                    1.0e-12,
                    28.5 * mathematical_constants::PI / 180.0,
                    1.0e-12,
                    1.0e-12,
                    1.0e-12).finished();
    Eigen::Vector6d finalKeplerianElements = (
            Eigen::Vector6d() << 42165.0e3,
                    1.0e-12,
                    1.0e-12,
                    1.0e-12,
                    1.0e-12,
                    1.0e-12).finished();
    std::cout << "Setting up costates:" << std::endl;

    Eigen::VectorXd initialTestCostates(6);
    initialTestCostates << -0.021195, -38.677447, 42.482983, 499.322515, -67.364508, 10.0;

    Eigen::VectorXd finalTestCostates(6);
    finalTestCostates << -0.000078, 24.905070, 2.186083, 499.380629, 43.545684, 10.0;

    // std::cout << initialTestCostates.transpose() << " <> " << finalTestCostates.transpose() << std::endl;

    // Initial and final states in cartesian coordinates.
    Eigen::Vector6d stateAtDeparture = orbital_element_conversions::convertKeplerianToCartesianElements(
            initialTestKeplerianElements, bodyMap["Earth"]->getGravityFieldModel()->getGravitationalParameter());
    Eigen::Vector6d stateAtArrival;

    // stateAtArrival = orbital_element_conversions::convertKeplerianToCartesianElements(
    //         finalKeplerianElements, bodyMap["Earth"]->getGravityFieldModel()->getGravitationalParameter());

    // std::cout << initialKeplerianElements << std::endl;
    // std::cout << finalKeplerianElements << std::endl;
    //
    // std::cout << stateAtDeparture << std::endl;
    // std::cout << stateAtArrival << std::endl;

    // Define integrator settings.
    double numberOfSteps = 15000;
    double stepSize = (timeOfFlight) / static_cast< double >( numberOfSteps );
    std::shared_ptr<numerical_integrators::IntegratorSettings<double> > weirdIntegratorSettings =
            std::make_shared<numerical_integrators::IntegratorSettings<double> >
                    (numerical_integrators::rungeKutta4, 0.0, stepSize);

    // Define optimisation algorithm.
    algorithm optimisationAlgorithm{pagmo::de()};
    optimisationAlgorithm.set_verbosity(1);

    std::shared_ptr<simulation_setup::OptimisationSettings> optimisationSettings =
            std::make_shared<simulation_setup::OptimisationSettings>(optimisationAlgorithm, 1, 150, 1.0e-3);

    // Create object with list of dependent variables
    std::vector<std::shared_ptr<SingleDependentVariableSaveSettings> > dependentVariablesList;
    dependentVariablesList.push_back(std::make_shared<SingleAccelerationDependentVariableSaveSettings>(
            basic_astrodynamics::thrust_acceleration, bodyToPropagate, bodyToPropagate, 0));
    std::shared_ptr<DependentVariableSaveSettings> dependentVariablesToSave =
            std::make_shared<DependentVariableSaveSettings>(dependentVariablesList, false);



    std::cout << "Making HybridMethodModelTest" << std::endl;
    HybridMethodModel hybridMethodModelTest = HybridMethodModel(
            stateAtDeparture, stateAtArrival, initialTestCostates, finalTestCostates, maximumThrust, specificImpulse,
            timeOfFlight,
            bodyMap, bodyToPropagate, centralBody, weirdIntegratorSettings);

    std::vector<double> testEpochsToSave;
    for (int i = 0; i <= numberOfSteps; i++) {
        testEpochsToSave.push_back(i * stepSize);
    }
    std::map<double, Eigen::Vector6d> propagatedTestTrajectory;
    std::cout << "Propagating HybridMethod Test" << std::endl;
    hybridMethodModelTest.propagateTrajectory(testEpochsToSave, propagatedTestTrajectory);

    input_output::writeDataMapToTextFile(propagatedTestTrajectory,
                                         "HybridMethodTrajectoryTest.dat",
                                         tudat_pagmo_applications::getOutputPath(),
                                         "",
                                         std::numeric_limits<double>::digits10,
                                         std::numeric_limits<double>::digits10,
                                         ",");
    return EXIT_SUCCESS;
    // Create hybrid method trajectory.
    HybridMethod hybridMethod = HybridMethod(stateAtDeparture, stateAtArrival, centralBodyGravitationalParameter,
                                             initialMass,
                                             maximumThrust, specificImpulse,
                                             timeOfFlight, bodyMap, bodyToPropagate, centralBody,
                                             weirdIntegratorSettings,
                                             optimisationSettings);
    std::shared_ptr<HybridMethodModel> hybridMethodModel = hybridMethod.getOptimalHybridMethodModel();

    // Results for full propagation
    std::map<double, Eigen::Vector6d> propagatedTrajectory;
    // std::vector< double > epochsToSaveResults;
    // for ( int i = 0 ; i <= numberOfSteps ; i++ )
    // {
    //     epochsToSaveResults.push_back( i * stepSize );
    // }
    // // Propagate trajectory using CI
    // hybridMethod.getTrajectory(epochsToSaveResults, propagatedTrajectory);
    //
    // std::string caseName = "out-of-plane";

    // Propagate trajectory using OA
    // std::vector<int> OAArcSteps = {30, 40, 50};             // n_k
    // std::vector<double> OAArcLengths = {3.0, 4.0, 5.0};     // t_OA

    // for (int numberOfOASteps : OAArcSteps) {
    //     for (double OAArcLength : OAArcLengths) {
    //         int numberOfOASteps = 40;
    //         double OAArcLength = 4.0;
    //
    //         std::cout << "Propagating Arc for n_k: " << numberOfOASteps << ", t_OA: " << OAArcLength << " days" << std::endl;
    //         // Length of OA Arc in Seconds
    //         double OAArcTime = OAArcLength * physical_constants::JULIAN_DAY;
    //
    //         // minimum number of arcs needed to reach TOF
    //         int numberOfOAarcs = (int) floor(timeOfFlight / OAArcTime);
    //
    //         // Initialize propagation things
    //         double currentTime = 0.0;
    //         Eigen::Vector6d currentState = stateAtDeparture;
    //
    //         std::map< double, Eigen::Vector6d >  resultStates;
    //         resultStates[0] = stateAtDeparture;
    //
    //         for (int i = 0; i < numberOfOAarcs; i++) {
    //             std::pair< double, Eigen::Vector6d > finalArcResult = hybridMethodModel->computeAverages( currentState, currentTime, numberOfOASteps, OAArcTime );
    //
    //             Eigen::Vector6d stateAfterArc = finalArcResult.second;
    //             currentTime = finalArcResult.first;
    //             // currentTime += OAArcTime;
    //             currentState = stateAfterArc;
    //             resultStates[currentTime] = currentState;
    //         }
    //
    //         std::pair< double, Eigen::Vector6d > finalOAState = hybridMethodModel->computeAverages( currentState, currentTime, numberOfOASteps, timeOfFlight - currentTime );
    //         resultStates[finalOAState.first] = finalOAState.second;
    //
    //         std::stringstream fileNameOA;
    //         // fileNameOA << "OATrajectory_" << caseName << "_" << numberOfOASteps << "_" << OAArcLength << ".dat";
    //         fileNameOA << "OATrajectory_" << "CaseBTest" << "_" << numberOfOASteps << "_" << OAArcLength << ".dat";

    std::cout << " ---- Results ---- " << std::endl;
    std::cout << "m_prop:" << initialMass - hybridMethodModel->getMassAtTimeOfFlight() << std::endl;

    // input_output::writeDataMapToTextFile( resultStates,
    //                                       fileNameOA.str(),
    //                                       tudat_pagmo_applications::getOutputPath( ),
    //                                       "",
    //                                       std::numeric_limits< double >::digits10,
    //                                       std::numeric_limits< double >::digits10,
    //                                       "," );
    //     }
    // }

    // take average progression, add those to the initialState and outputput as cartesian
    std::vector<double> epochsToSaveResults;
    std::map<double, Eigen::VectorXd> thrustAccelerationProfile;
    for (int i = 0; i <= numberOfSteps; i++) {
        epochsToSaveResults.push_back(i * stepSize);
    }
    hybridMethod.getTrajectory(epochsToSaveResults, propagatedTrajectory);

    input_output::writeDataMapToTextFile(propagatedTrajectory,
                                         "HybridMethodTrajectory.dat",
                                         tudat_pagmo_applications::getOutputPath(),
                                         "",
                                         std::numeric_limits<double>::digits10,
                                         std::numeric_limits<double>::digits10,
                                         ",");

    // Final statement.
    // The exit code EXIT_SUCCESS indicates that the program was successfully executed.
    return EXIT_SUCCESS;
}
