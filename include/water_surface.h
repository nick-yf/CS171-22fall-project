//
// Created by nick on 23-1-2.
//

#ifndef CS171_HW5_WATER_SURFACE_H
#define CS171_HW5_WATER_SURFACE_H

#include "mesh.h"
#include "time_system.h"

class WavePacket {
public:
private:
    friend class WaterSurface;

    static constexpr unsigned simulation_steps_per_fixed_update_time = 20;
    static constexpr Float fixed_delta_time = Time::fixed_delta_time / Float(simulation_steps_per_fixed_update_time);

    int limitation = 0;

    std::vector<float> wave_length;
    std::vector<float> wave_number;
    std::vector<float> amplitude;
    std::vector<std::pair<Vec2, Vec2>> p;

};

class WaterSurface : public Mesh {
public:
    WaterSurface(int limit);

private:
    int limitation = 0;
    WavePacket wave_packet;
    std::vector<Vec3> local_or_world_pos;

};

#endif //CS171_HW5_WATER_SURFACE_H