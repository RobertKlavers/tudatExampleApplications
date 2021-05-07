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
#include <json.hpp>

using json = nlohmann::json;

int main() {
    using namespace tudat;
    using namespace tudat::input_output;
    using namespace tudat::simulation_setup;
    using namespace tudat::low_thrust_trajectories;
    using namespace tudat::propagators;

    using namespace low_thrust_trajectories;

    spice_interface::loadStandardSpiceKernels();
    // Retrieve the input json file
    const std::string cppFilePath( __FILE__ );
    const std::string cppFolder = cppFilePath.substr( 0 , cppFilePath.find_last_of("/\\")+1 );

    const std::string testCase = "FullOptimization";

    // Read JSON file
    std::ifstream inputstream(cppFolder + "hybridMethod" + testCase + ".json");
    json input_data;
    inputstream >> input_data;

    // Vehicle Settings
    double maximumThrust = input_data["vehicle"]["maximumThrust"];
    double specificImpulse = input_data["vehicle"]["specificImpulse"];
    double initialMass = input_data["vehicle"]["initialMass"];

    std::function<double(const double)> specificImpulseFunction = [=](const double currentTime) {
        return specificImpulse;
    };

    // Trajectory Settings
    double julianDate = 0.0 * physical_constants::JULIAN_DAY;
    double timeOfFlight = input_data["trajectory"]["timeOfFlight"].get<double>() * physical_constants::JULIAN_DAY;

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


    // Define integrator settings.
    // int numberOfSteps = input_data["optimization"]["numberOfSteps"];
    // double stepSize = (timeOfFlight) / static_cast< double >( numberOfSteps );
    double stepSizeOpt = 500.0;
    std::shared_ptr<numerical_integrators::IntegratorSettings<double> > integratorSettings =
            std::make_shared<numerical_integrators::IntegratorSettings<double> >
                    (numerical_integrators::rungeKutta4, 0.0, stepSizeOpt);

    // ---- Case Specific Propagation ----
    if (testCase == "SingleCostatePropagation") {
        // Retrieve initial and final costates for simplified test
        Eigen::VectorXd initialTestCostates = Eigen::VectorXd::Map(input_data["trajectory"]["constantCostates"].get<std::vector<double>>().data(), 6);
        Eigen::VectorXd finalTestCostates = Eigen::VectorXd::Map(input_data["trajectory"]["constantCostates"].get<std::vector<double>>().data(), 6);
        std::cout << initialTestCostates.transpose() << " <> " << finalTestCostates.transpose() << std::endl;

        // Retrieve initial and final Keplerian Elements
        Eigen::Vector6d initialKeplerianElements(input_data["trajectory"]["initialKeplerianElements"].get<std::vector<double>>().data());
        // Initial and final states in cartesian coordinates.
        Eigen::Vector6d stateAtDeparture = orbital_element_conversions::convertKeplerianToCartesianElements(
                initialKeplerianElements, bodyMap["Earth"]->getGravityFieldModel()->getGravitationalParameter());
        Eigen::Vector6d stateAtArrival;

        std::cout << "Making Test Hybrid Method Model" << std::endl;
        HybridMethodModel hybridMethodModelTest = HybridMethodModel(
                stateAtDeparture, stateAtArrival, initialTestCostates, finalTestCostates, maximumThrust, specificImpulse,
                timeOfFlight,
                bodyMap, bodyToPropagate, centralBody, integratorSettings);
    } else if (testCase == "LinearCostatePropagation") {
        // Retrieve initial and final costates for simplified test
        Eigen::VectorXd initialTestCostates = Eigen::VectorXd::Map(input_data["trajectory"]["initialCostates"].get<std::vector<double>>().data(), 6);
        Eigen::VectorXd finalTestCostates = Eigen::VectorXd::Map(input_data["trajectory"]["finalCostates"].get<std::vector<double>>().data(), 6);
        std::cout << initialTestCostates.transpose() << " <> " << finalTestCostates.transpose() << std::endl;

        // Retrieve initial and final Keplerian Elements
        Eigen::Vector6d initialKeplerianElements(input_data["trajectory"]["initialKeplerianElements"].get<std::vector<double>>().data());
        // Initial and final states in cartesian coordinates.
        Eigen::Vector6d stateAtDeparture = orbital_element_conversions::convertKeplerianToCartesianElements(
                initialKeplerianElements, bodyMap["Earth"]->getGravityFieldModel()->getGravitationalParameter());
        Eigen::Vector6d stateAtArrival;

        std::cout << "Making Test Hybrid Method Model" << std::endl;
        HybridMethodModel hybridMethodModelTest = HybridMethodModel(
                stateAtDeparture, stateAtArrival, initialTestCostates, finalTestCostates, maximumThrust, specificImpulse,
                timeOfFlight,
                bodyMap, bodyToPropagate, centralBody, integratorSettings);

        std::pair<std::map< double, Eigen::VectorXd >, std::map< double, Eigen::VectorXd >> optimalTrajectory = hybridMethodModelTest.getTrajectoryOutput();


        // Temporary stuff to directly store the dependent variable history (hopefully containing thrust acceleration profile)
        std::cout << "Exporting Dependent Variable History!!" << std::endl;
        input_output::writeDataMapToTextFile( optimalTrajectory.first,
                                              "HybridMethodFinalTrajectoryHistory.dat",
                                              tudat_pagmo_applications::getOutputPath(),
                                              "",
                                              std::numeric_limits< double >::digits10,
                                              std::numeric_limits< double >::digits10,
                                              "," );
        input_output::writeDataMapToTextFile( optimalTrajectory.second,
                                              "HybridMethodFinalDependentVariableHistory.dat",
                                              tudat_pagmo_applications::getOutputPath(),
                                              "",
                                              std::numeric_limits< double >::digits10,
                                              std::numeric_limits< double >::digits10,
                                              "," );


    } else if (testCase == "FullOptimization") {
        // Retrieve initial and final Keplerian Elements
        Eigen::Vector6d initialKeplerianElements(input_data["trajectory"]["initialKeplerianElements"].get<std::vector<double>>().data());
        Eigen::Vector6d finalKeplerianElements(input_data["trajectory"]["finalKeplerianElements"].get<std::vector<double>>().data());

        Eigen::Vector6d stateAtDeparture = orbital_element_conversions::convertKeplerianToCartesianElements(
                initialKeplerianElements, bodyMap["Earth"]->getGravityFieldModel()->getGravitationalParameter());
        Eigen::Vector6d stateAtArrival = orbital_element_conversions::convertKeplerianToCartesianElements(
                finalKeplerianElements, bodyMap["Earth"]->getGravityFieldModel()->getGravitationalParameter());

        // Define optimisation algorithm.
        // algorithm optimisationAlgorithm{pagmo::de1220(input_data["optimization"]["numberOfGenerations"].get<int>())};
        algorithm optimisationAlgorithm{pagmo::de1220()};
        // algorithm optimisationAlgorithm{pagmo::de(input_data["optimization"]["numberOfGenerations"].get<int>(), 0.6, 0.8, 2, 1e-6, 1e-6)};
        optimisationAlgorithm.set_verbosity(0);

        std::shared_ptr<simulation_setup::OptimisationSettings> optimisationSettings =
                std::make_shared<simulation_setup::OptimisationSettings>(optimisationAlgorithm,
                                                                         input_data["optimization"]["numberOfGenerations"],
                                                                         input_data["optimization"]["numberOfIndividualsPerPopulation"],
                                                                         input_data["optimization"]["relativeToleranceConstraints"]);

        const std::pair< double, double > initialAndFinalMEEcostatesBounds = std::make_pair( - 1.0e4, 1.0e4 );

        HybridMethod hybridMethod = HybridMethod(stateAtDeparture, stateAtArrival, centralBodyGravitationalParameter,
                                                 initialMass,
                                                 maximumThrust, specificImpulse,
                                                 timeOfFlight, bodyMap, bodyToPropagate, centralBody,
                                                 integratorSettings,
                                                 optimisationSettings, initialAndFinalMEEcostatesBounds);

        std::shared_ptr<HybridMethodModel> hybridMethodModel = hybridMethod.getOptimalHybridMethodModel();

        std::pair<std::map< double, Eigen::VectorXd >, std::map< double, Eigen::VectorXd >> optimalTrajectory = hybridMethodModel->getTrajectoryOutput();


        // Temporary stuff to directly store the dependent variable history (hopefully containing thrust acceleration profile)
        std::cout << "Exporting Dependent Variable History!!" << std::endl;
        input_output::writeDataMapToTextFile( optimalTrajectory.first,
                                              "HybridMethodFinalTrajectoryHistory.dat",
                                              tudat_pagmo_applications::getOutputPath(),
                                              "",
                                              std::numeric_limits< double >::digits10,
                                              std::numeric_limits< double >::digits10,
                                              "," );
        input_output::writeDataMapToTextFile( optimalTrajectory.second,
                                              "HybridMethodFinalDependentVariableHistory.dat",
                                              tudat_pagmo_applications::getOutputPath(),
                                              "",
                                              std::numeric_limits< double >::digits10,
                                              std::numeric_limits< double >::digits10,
                                              "," );



        // Results for full propagation
        // std::map<double, Eigen::Vector6d> propagatedTrajectory;
        // std::cout << " ---- Results ---- " << std::endl;
        // std::cout << "m_prop:" << initialMass - hybridMethodModel->getMassAtTimeOfFlight() << std::endl;
        // // take average progression, add those to the initialState and outputput as cartesian
        // std::vector<double> epochsToSaveResults;
        // std::map<double, Eigen::VectorXd> thrustAccelerationProfile;
        // for (int i = 0; i <= numberOfSteps; i++) {
        //     epochsToSaveResults.push_back(i * stepSize);
        // }
        // hybridMethod.getTrajectory(epochsToSaveResults, propagatedTrajectory);
        //
        // input_output::writeDataMapToTextFile(propagatedTrajectory,
        //                                      "HybridMethodTrajectory.dat",
        //                                      tudat_pagmo_applications::getOutputPath(),
        //                                      "",
        //                                      std::numeric_limits<double>::digits10,
        //                                      std::numeric_limits<double>::digits10,
        //                                      ",");

    }

    // // Create object with list of dependent variables
    // // TODO Pass these on to the HybridMethodModel
    // std::vector<std::shared_ptr<SingleDependentVariableSaveSettings> > dependentVariablesList;
    // dependentVariablesList.push_back(std::make_shared<SingleAccelerationDependentVariableSaveSettings>(
    //         basic_astrodynamics::thrust_acceleration, bodyToPropagate, bodyToPropagate, 0));
    // std::shared_ptr<DependentVariableSaveSettings> dependentVariablesToSave =
    //         std::make_shared<DependentVariableSaveSettings>(dependentVariablesList, false);

    return EXIT_SUCCESS;

    // Create hybrid method trajectory.
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



    // input_output::writeDataMapToTextFile( resultStates,
    //                                       fileNameOA.str(),
    //                                       tudat_pagmo_applications::getOutputPath( ),
    //                                       "",
    //                                       std::numeric_limits< double >::digits10,
    //                                       std::numeric_limits< double >::digits10,
    //                                       "," );
    //     }
    // }

    // Final statement.
    // The exit code EXIT_SUCCESS indicates that the program was successfully executed.
    return EXIT_SUCCESS;
}
