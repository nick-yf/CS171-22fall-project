//
// Created by nick on 23-1-2.
//

#ifndef CS171_HW5_WATER_SURFACE_H
#define CS171_HW5_WATER_SURFACE_H

#include "mesh.h"
#include "time_system.h"

struct WaveParticle{
    float amplitude;
    float radius;
    float dispersion_angle;
    float surviving_time;
    Vec3 position[3];
    Vec3 propagate[3];
    Vec3 horizontal[3];
};

struct Sphere{
    float radius;
    float radius_on_surface;
    float old_radius_on_surface;
    Vec3 center;
    Vec3 old_center;
    Vec3 acceleration;
    Vec3 velocity;
    Vec3 local_velo;
    std::shared_ptr<Mesh> mesh;
};

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

    /// constructor

    WaterSurface(int limit, UVec2 sizes, float dx_local, std::shared_ptr<Mesh> other_mesh);

    WaterSurface(const WaterSurface &) = default;

    WaterSurface(WaterSurface &&) = default;

    WaterSurface &operator=(const WaterSurface &) = default;

    WaterSurface &operator=(WaterSurface &&) = default;

    virtual ~WaterSurface() override = default;

    /// interfaces
    void FixedUpdate() override;

private:
    static constexpr unsigned simulation_steps_per_fixed_update_time = 20;
    static constexpr Float fixed_delta_time = Time::fixed_delta_time / Float(simulation_steps_per_fixed_update_time);

    Float dx_local;
    int limitation;

    UVec2 vertex_sizes;
    std::vector<Vec3> original_positions;
    std::vector<Vec3> water_vertices;

    // particles attribute
    int timer;
    std::default_random_engine e;
    std::uniform_real_distribution<float> u;
    bool reflect = true;
    float wave_speed = 1.0f;
    std::vector<WaveParticle> particles;
    std::vector<Vec3> directions;
    // other object
    float density;
    float drag_coef;
    float lift_coef;
    Sphere sphere;

    /// simulation pipeline

    void LocalToWorldPositions();

    void IterateWaveParticles();

    void ComputeObjectForces();

    void IterateObjects();

    void GenerateWaveParticles();

    void RenderHeightFields();

    void WorldToLocalPositions();

    void Simulate(unsigned num_steps);

    /// rendering

    void UpdateMeshVertices();

    /// supporting methods

    [[nodiscard]] size_t Get1DIndex(int iw, int ih) const;

    float rectangle_func(float x);

    WaveParticle create_particle(Vec3 position, Vec3 propagate, float dispersion_angle, float amplitude);
};

#endif //CS171_HW5_WATER_SURFACE_H
