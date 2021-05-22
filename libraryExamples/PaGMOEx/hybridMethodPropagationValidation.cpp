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

    // 0 Load configuration files
    const std::string cppFilePath(__FILE__);
    const std::string cppFolder = cppFilePath.substr(0, cppFilePath.find_last_of("/\\") + 1);
    const std::string configurationFile = "hybridMethodPropagationValidationConfiguration.json";

    // Read JSON file
    std::ifstream inputstream(cppFolder + configurationFile);
    json input_data;
    inputstream >> input_data;

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

    // Set vehicle mass.
    bodyMap[bodyToPropagate]->setConstantBodyMass(initialMass);

    // Integrator Settings
    double stepSize = caseConfiguration["optimization"]["stepSize"];
    std::shared_ptr<numerical_integrators::IntegratorSettings<double> > integratorSettings =
            std::make_shared<numerical_integrators::IntegratorSettings<double> >
                    (numerical_integrators::rungeKutta4, 0.0, stepSize);


    if (propagationCase == "SimpleCostates") {
        // Retrieve initial and final costates for simplified test
        Eigen::VectorXd initialTestCostates = Eigen::VectorXd::Map(caseConfiguration["trajectory"]["initialCostates"].get<std::vector<double>>().data(), 6);
        Eigen::VectorXd finalTestCostates = Eigen::VectorXd::Map(caseConfiguration["trajectory"]["finalCostates"].get<std::vector<double>>().data(), 6);

        // TODO: Refactor this out of HybridMethodModel
        Eigen::Vector6d epsilon_upper = Eigen::Vector6d::Zero();
        Eigen::Vector6d constraint_weights = Eigen::Vector6d::Zero();
        double weightMass = 0.0;
        double weightTOF = 0.0;

        std::shared_ptr<simulation_setup::HybridOptimisationSettings> hybridOptimisationSettings =
                std::make_shared<simulation_setup::HybridOptimisationSettings>(epsilon_upper, constraint_weights, weightMass, weightTOF, true);

        // 2 Set up initial Conditions
        // Retrieve initial and final Keplerian Elements
        Eigen::Vector6d initialKeplerianElements(caseConfiguration["trajectory"]["initialKeplerianElements"].get<std::vector<double>>().data());
        // Initial and final states in cartesian coordinates.
        Eigen::Vector6d stateAtDeparture = orbital_element_conversions::convertKeplerianToCartesianElements(
                initialKeplerianElements, centralBodyGravitationalParameter);
        Eigen::Vector6d stateAtArrival;

        // Set up the single leg model
        HybridMethodModel hybridMethodModelTest = HybridMethodModel(
                stateAtDeparture, stateAtArrival, initialTestCostates, finalTestCostates, maximumThrust,
                specificImpulse,
                timeOfFlight,
                bodyMap, bodyToPropagate, centralBody, integratorSettings, hybridOptimisationSettings);

        std::pair<std::map< double, Eigen::VectorXd >, std::map< double, Eigen::VectorXd >> optimalTrajectory = hybridMethodModelTest.getTrajectoryOutput();
    }

    // TODO 5 Export results

    return EXIT_SUCCESS;
}
