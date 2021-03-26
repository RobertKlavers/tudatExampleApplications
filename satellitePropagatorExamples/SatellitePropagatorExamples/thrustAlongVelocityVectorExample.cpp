/*    Copyright (c) 2010-2019, Delft University of Technology
 *    All rigths reserved
 *
 *    This file is part of the Tudat. Redistribution and use in source and
 *    binary forms, with or without modification, are permitted exclusively
 *    under the terms of the Modified BSD license. You should have received
 *    a copy of the license with this file. If not, please or visit:
 *    http://tudat.tudelft.nl/LICENSE.
 */

#include <Tudat/SimulationSetup/tudatSimulationHeader.h>

#include "SatellitePropagatorExamples/applicationOutput.h"

class TVCGuidance
{
public:

    TVCGuidance( ){ }

    ~TVCGuidance( ){ }

    void updateGuidance( const double currentTime )
    {

        // Implement your guidance model here, using your specific algorithm
        double currentThrustAngle = tudat::mathematical_constants::PI / 4;
        bodyFixedThrustDirection_ << 0.0, 0.0, 1.0;

        // Ensure that direction is a unit vector
        bodyFixedThrustDirection_.normalize( );
    }

    Eigen::Vector3d getBodyFixedThrustDirection( )
    {
        return bodyFixedThrustDirection_;
    }

protected:

    Eigen::Vector3d bodyFixedThrustDirection_;

    double initialAngle_;

    double angleRate_;

    double referenceTime_;

};


//! Execute propagation of orbit of Asterix around the Earth.
int main( )
{
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////            USING STATEMENTS              //////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    using namespace tudat;
    using namespace tudat::simulation_setup;
    using namespace tudat::propagators;
    using namespace tudat::numerical_integrators;
    using namespace tudat::basic_mathematics;
    using namespace tudat::basic_astrodynamics;

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////     CREATE ENVIRONMENT AND VEHICLE       //////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    //Load spice kernels.
    spice_interface::loadStandardSpiceKernels( );

    double maximumThrust = 0.350;
    double specificImpulse = 2000.0;
    double vehicleMass = 2000.0;

    double julianDate = 0.0 * physical_constants::JULIAN_DAY;
    double timeOfFlight = 20.0 * physical_constants::JULIAN_DAY;

    // Create Earth object
    // Define body settings for simulation.
    std::vector< std::string > bodiesToCreate;
    bodiesToCreate.push_back( "Earth" );

    std::map< std::string, std::shared_ptr< simulation_setup::BodySettings > > bodySettings =
            simulation_setup::getDefaultBodySettings( bodiesToCreate, julianDate - 300.0, julianDate + timeOfFlight + 300.0 );
    for( unsigned int i = 0; i < bodiesToCreate.size( ); i++ )
    {
        bodySettings[ bodiesToCreate.at( i ) ]->ephemerisSettings->resetFrameOrientation( "J2000" );
        bodySettings[ bodiesToCreate.at( i ) ]->rotationModelSettings->resetOriginalFrame( "J2000" );
    }
    // // Create body objects.
    // std::map< std::string, std::shared_ptr< BodySettings > > bodySettings =
    //         getDefaultBodySettings( bodiesToCreate );

    NamedBodyMap bodyMap = createBodies( bodySettings );

    bodyMap[ "Vehicle" ] = std::make_shared< simulation_setup::Body >( );
    bodyMap[ "Vehicle" ]->setConstantBodyMass( vehicleMass );

    // Finalize body creation.
    setGlobalFrameBodyEphemerides( bodyMap, "SSB", "J2000" );

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////             CREATE ACCELERATIONS            ///////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // Define propagator settings variables.
    SelectedAccelerationMap accelerationMap;
    std::vector< std::string > bodiesToPropagate;
    std::string centralBody = "Earth";

    std::shared_ptr< TVCGuidance > tvcGuidance = std::make_shared<TVCGuidance>();

// Retrieve required functions from TVC object
    std::function< void( const double ) > updateFunction =
            std::bind( &TVCGuidance::updateGuidance, tvcGuidance, std::placeholders::_1 );
    std::function< Eigen::Vector3d( ) > thrustDirectionFunction =
            std::bind( &TVCGuidance::getBodyFixedThrustDirection, tvcGuidance );

// Create thrust magnitude settings, with constant thrust magnitude and specific impulse
//     double thrustMagnitude = 1.0E3;
//     double specificImpulse = 300;
    std::shared_ptr< ThrustMagnitudeSettings > tvcThrustMagnitudeSettings =
            std::make_shared< FromFunctionThrustMagnitudeSettings >(
                    [ = ]( const double ){ return maximumThrust; },
                    [ = ]( const double ){ return specificImpulse; },
                    [ ]( const double ){ return true; },
                    thrustDirectionFunction,
                    updateFunction );

    // Define thrust settings
    std::shared_ptr< ThrustDirectionGuidanceSettings > thrustDirectionGuidanceSettings =
            std::make_shared< ThrustDirectionFromStateGuidanceSettings >( centralBody, true, false );
    std::shared_ptr< ThrustMagnitudeSettings > constantThrustMagnitudeSettings =
            std::make_shared< ConstantThrustMagnitudeSettings >( maximumThrust, specificImpulse );

    // Define acceleration model settings.
    std::map< std::string, std::vector< std::shared_ptr< AccelerationSettings > > > accelerationsOfVehicle;
    accelerationsOfVehicle[ "Vehicle" ].push_back(
                std::make_shared< ThrustAccelerationSettings >( thrustDirectionGuidanceSettings, tvcThrustMagnitudeSettings ) );
    accelerationsOfVehicle[ "Earth" ].push_back( std::make_shared< AccelerationSettings >( central_gravity ) );

    accelerationMap[ "Vehicle" ] = accelerationsOfVehicle;

    bodiesToPropagate.push_back( "Vehicle" );

    std::vector< std::string > centralBodies;
    centralBodies.push_back( centralBody );
    double centralBodyGravitationalParameter = bodyMap.at( centralBody )->getGravityFieldModel( )->getGravitationalParameter( );

    // Create acceleration models and propagation settings.
    basic_astrodynamics::AccelerationMap accelerationModelMap = createAccelerationModelsMap(
                bodyMap, accelerationMap, bodiesToPropagate, centralBodies );

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////             CREATE PROPAGATION SETTINGS            ////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


    Eigen::Vector6d initialKeplerianElements = ( Eigen::Vector6d( ) << 24505.9e3, 0.725, 7.0 * mathematical_constants::PI / 180.0,
            7.0 * mathematical_constants::PI / 180.0, 7.0 * mathematical_constants::PI / 180.0, 1.0e-12 ).finished( );
    Eigen::Vector6d stateAtDeparture = orbital_element_conversions::convertKeplerianToCartesianElements(
            initialKeplerianElements, centralBodyGravitationalParameter );

    // Define propagation termination conditions (stop after 20 days weeks).
    std::shared_ptr< PropagationTimeTerminationSettings > terminationSettings =
            std::make_shared< propagators::PropagationTimeTerminationSettings >( timeOfFlight );

    // Define settings for propagation of translational dynamics.
    std::shared_ptr< TranslationalStatePropagatorSettings< double > > translationalPropagatorSettings =
            std::make_shared< TranslationalStatePropagatorSettings< double > >(
                centralBodies, accelerationModelMap, bodiesToPropagate, stateAtDeparture, terminationSettings );

    // Create mass rate models
    std::shared_ptr< MassRateModelSettings > massRateModelSettings =
            std::make_shared< FromThrustMassModelSettings >( true );
    std::map< std::string, std::shared_ptr< basic_astrodynamics::MassRateModel > > massRateModels;
    massRateModels[ "Vehicle" ] = createMassRateModel(
                "Vehicle", massRateModelSettings, bodyMap, accelerationModelMap );

    // Create settings for propagating the mass of the vehicle.
    std::vector< std::string > bodiesWithMassToPropagate;
    bodiesWithMassToPropagate.push_back( "Vehicle" );

    Eigen::VectorXd initialBodyMasses = Eigen::VectorXd( 1 );
    initialBodyMasses( 0 ) = vehicleMass;

    std::shared_ptr< SingleArcPropagatorSettings< double > > massPropagatorSettings =
            std::make_shared< MassPropagatorSettings< double > >(
                bodiesWithMassToPropagate, massRateModels, initialBodyMasses, terminationSettings );

    // Create list of propagation settings.
    std::vector< std::shared_ptr< SingleArcPropagatorSettings< double > > > propagatorSettingsVector;
    propagatorSettingsVector.push_back( translationalPropagatorSettings );
    propagatorSettingsVector.push_back( massPropagatorSettings );

    // Create propagation settings for mass and translational dynamics concurrently
    std::shared_ptr< PropagatorSettings< double > > propagatorSettings =
            std::make_shared< MultiTypePropagatorSettings< double > >( propagatorSettingsVector, terminationSettings );

    // Define integrator settings
    std::shared_ptr< IntegratorSettings< > > integratorSettings =
            std::make_shared< IntegratorSettings< > >( rungeKutta4, 0.0, 50.0 );

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////             PROPAGATE ORBIT            ////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


    // Create simulation object and propagate dynamics.
    SingleArcDynamicsSimulator< > dynamicsSimulator(
                bodyMap, integratorSettings, propagatorSettings, true, false, false );

    // Retrieve numerical solutions for state and dependent variables
    std::map< double, Eigen::Matrix< double, Eigen::Dynamic, 1 > > numericalSolution =
            dynamicsSimulator.getEquationsOfMotionNumericalSolution( );

    std::string outputSubFolder = "ThrustAlongVelocityExample/";

    // Write satellite propagation history to file.
    input_output::writeDataMapToTextFile( numericalSolution,
                                          "velocityVectorThrustExample.dat",
                                          tudat_applications::getOutputPath( ) + outputSubFolder,
                                          "",
                                          std::numeric_limits< double >::digits10,
                                          std::numeric_limits< double >::digits10,
                                          "," );

}
