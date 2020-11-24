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

#include "Tudat/Astrodynamics/Ephemerides/approximatePlanetPositions.h"
#include "pagmo/algorithms/de1220.hpp"
#include "Problems/applicationOutput.h"
#include "Tudat/Astrodynamics/BasicAstrodynamics/celestialBodyConstants.h"

int main( )
{
    using namespace tudat;
    using namespace tudat::input_output;
    using namespace tudat::simulation_setup;
    using namespace tudat::low_thrust_trajectories;

    using namespace low_thrust_trajectories;

    spice_interface::loadStandardSpiceKernels( );

    double maximumThrust = 0.450;
    double specificImpulse = 3000.0;
    double mass = 1800.0;
    double initialMass = mass;

    std::function< double( const double ) > specificImpulseFunction = [ = ] ( const double currentTime )
    {
        return specificImpulse;
    };

    double julianDate = 1000.0 * physical_constants::JULIAN_DAY;
    double timeOfFlight = 100.0 * physical_constants::JULIAN_DAY;

    // Define body settings for simulation.
    std::vector< std::string > bodiesToCreate;
    bodiesToCreate.push_back( "Sun" );
    bodiesToCreate.push_back( "Earth" );

    // Create body objects.
    std::map< std::string, std::shared_ptr< simulation_setup::BodySettings > > bodySettings =
            simulation_setup::getDefaultBodySettings( bodiesToCreate, julianDate - 300.0, julianDate + timeOfFlight + 300.0 );
    for( unsigned int i = 0; i < bodiesToCreate.size( ); i++ )
    {
        bodySettings[ bodiesToCreate.at( i ) ]->ephemerisSettings->resetFrameOrientation( "J2000" );
        bodySettings[ bodiesToCreate.at( i ) ]->rotationModelSettings->resetOriginalFrame( "J2000" );
    }
    simulation_setup::NamedBodyMap bodyMap = createBodies( bodySettings );


    // Create spacecraft object.
    bodyMap[ "Vehicle" ] = std::make_shared< simulation_setup::Body >( );

    // Finalize body creation.
    setGlobalFrameBodyEphemerides( bodyMap, "SSB", "J2000" );


    std::string bodyToPropagate = "Vehicle";
    std::string centralBody = "Earth";
    double centralBodyGravitationalParameter = bodyMap.at( centralBody )->getGravityFieldModel( )->getGravitationalParameter( );

    // Set vehicle mass.
    bodyMap[ bodyToPropagate ]->setConstantBodyMass( mass );

    // Initial and final states in keplerian elements.
    Eigen::Vector6d initialKeplerianElements = ( Eigen::Vector6d( ) << 24505.9e3, 0.725, 7.0 * mathematical_constants::PI / 180.0,
            0.0, 0.0, 0.0 ).finished( );
    Eigen::Vector6d finalKeplerianElements = ( Eigen::Vector6d( ) << 42164.65e3, 5.53e-4, 7.41e-5 * mathematical_constants::PI / 180.0,
            0.0, 0.0, 0.0 ).finished( );

    // Initial and final states in cartesian coordinates.
    Eigen::Vector6d stateAtDeparture = orbital_element_conversions::convertKeplerianToCartesianElements(
            initialKeplerianElements, bodyMap[ "Earth" ]->getGravityFieldModel()->getGravitationalParameter() );
    Eigen::Vector6d stateAtArrival = orbital_element_conversions::convertKeplerianToCartesianElements(
            finalKeplerianElements, bodyMap[ "Earth" ]->getGravityFieldModel()->getGravitationalParameter() );

    // Define integrator settings.
    double stepSize = ( timeOfFlight ) / static_cast< double >( 40000 );
    std::shared_ptr< numerical_integrators::IntegratorSettings< double > > integratorSettings =
            std::make_shared< numerical_integrators::IntegratorSettings< double > >
                    ( numerical_integrators::rungeKutta4, 0.0, stepSize );

    // Define optimisation algorithm.
    algorithm optimisationAlgorithm{ pagmo::de1220() };

    std::shared_ptr< simulation_setup::OptimisationSettings > optimisationSettings =
            std::make_shared< simulation_setup::OptimisationSettings >( optimisationAlgorithm, 1, 10, 1.0e-3 );

    HybridMethodModel = HybridMethodModel();

    // Create hybrid method trajectory.
    HybridMethod hybridMethod = HybridMethod( stateAtDeparture, stateAtArrival, centralBodyGravitationalParameter, initialMass,
                                              maximumThrust, specificImpulse,
                                              timeOfFlight, bodyMap, bodyToPropagate, centralBody, integratorSettings,
                                              optimisationSettings );
    int numberSteps = 10;
    std::map< double, Eigen::Vector6d > trajectory;
    std::vector< double > epochsVector;

    for ( int i = 1 ; i <= numberSteps ; i++ )
    {
        epochsVector.push_back( timeOfFlight / numberSteps * i );
    }

    hybridMethod.getTrajectory( epochsVector, trajectory );

    input_output::writeDataMapToTextFile( trajectory,
                                          "HybridMethodTrajectory.dat",
                                          tudat_pagmo_applications::getOutputPath( ),
                                          "",
                                          std::numeric_limits< double >::digits10,
                                          std::numeric_limits< double >::digits10,
                                          "," );

    // Final statement.
    // The exit code EXIT_SUCCESS indicates that the program was successfully executed.
    return EXIT_SUCCESS;
}
