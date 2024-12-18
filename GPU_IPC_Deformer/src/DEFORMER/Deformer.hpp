
#pragma once

#include "WrinkleDeformer.cuh"
#include "WrapDeformer.cuh"
#include "SimulationContext.hpp"
#include "Simulator.hpp"



class Deformer {

public:

    Deformer(SimulationContext& context, Simulator& simulator);



public:
    
    void getHostMesh(std::unique_ptr<GeometryManager>& instance);

    void DeformerPipeline();


private:
    SimulationContext& ctx;
    Simulator& sim;

};


