//
// Created by nick on 23-1-2.
//

#include "water_surface.h"
#include "object.h"
#include "transform.h"
#include <iostream>

WaterSurface::WaterSurface(int limit, UVec2 sizes, float dx_local) : Mesh(std::vector<MeshVertex>(sizes.x * sizes.y),
                                                                          std::vector<UVec3>(
                                                                                  (sizes.x - 1) * (sizes.y - 1) * 2),
                                                                          GL_STREAM_DRAW, GL_STATIC_DRAW,
                                                                          true),
                                                                     limitation(limit),
                                                                     original_positions(sizes.x * sizes.y),
                                                                     water_vertices(sizes.x * sizes.y),
                                                                     vertex_sizes(sizes),
                                                                     dx_local(dx_local),
                                                                     particles(limit) {
    float local_width = sizes.x * dx_local;
    float local_height = sizes.y * dx_local;
#pragma omp parallel for
    for (int height = 0; height < sizes.y; ++height) {
        for (int width = 0; width < sizes.x; ++width) {
            water_vertices.at(this->Get1DIndex(width, height)) = Vec3(width * dx_local - 0.5f * local_width,
                                                                      0,
                                                                      height * dx_local - 0.5f * local_height);
        }
    }
    original_positions = water_vertices;

    for (int i = 0; i < limit; ++i) {
        Vec3 position_tmp = Vec3(-0.5f * local_width, 0, -0.5f * local_height);
        Vec3 propagate_tmp = glm::normalize(
                Vec3{glm::cos((i + 0.5f) * glm::radians(90.0f) / this->limitation),
                     0.0f,
                     glm::sin((i + 0.5f) * glm::radians(90.0f) / this->limitation)});
        Vec3 horizon_tmp = glm::normalize(glm::cross(propagate_tmp, Vec3{0.0f, 1.0f, 0.0f}));
        this->particles.at(i).radius = 1.0f;
        this->particles.at(i).amplitude = 5.0f;
        this->particles.at(i).dispersion_angle = 90;
        this->particles.at(i).position = position_tmp;
        this->particles.at(i).original_position = position_tmp;
        this->particles.at(i).propagate = propagate_tmp;
        this->particles.at(i).horizontal = horizon_tmp;
    }

    // initialize mesh vertices
    UpdateMeshVertices();

    // initialize mesh indices
#pragma omp parallel for
    for (int ih = 0; ih < sizes.y - 1; ++ih)
        for (int iw = 0; iw < sizes.x - 1; ++iw) {
            size_t i_indices = (size_t(ih) * size_t(sizes.x - 1) + size_t(iw)) << 1;

            auto i = Get1DIndex(iw, ih);
            auto r = Get1DIndex(iw + 1, ih);
            auto u = Get1DIndex(iw, ih + 1);
            auto ru = Get1DIndex(iw + 1, ih + 1);

            indices[i_indices] = UVec3(i, r, u);
            indices[i_indices + 1] = UVec3(r, ru, u);
        }
    glBindVertexArray(vao);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(UVec3) * indices.size(), indices.data(), GL_STATIC_DRAW);
    glBindVertexArray(0);
}

void WaterSurface::UpdateMeshVertices() {
    // set vertex positions
    for (size_t i = 0; i < water_vertices.size(); ++i)
        vertices[i].position = water_vertices[i];

    // reset vertex normals
    auto compute_normal = [&](auto v1, auto v2, auto v3) {
        return glm::normalize(glm::cross(vertices[v2].position - vertices[v1].position,
                                         vertices[v3].position - vertices[v1].position));
    };

#pragma omp parallel for
    for (int ih = 0; ih < this->vertex_sizes.y; ++ih)
        for (int iw = 0; iw < this->vertex_sizes.x; ++iw) {
            constexpr Float w_small = Float(0.125);
            constexpr Float w_large = Float(0.25);

            auto i = Get1DIndex(iw, ih);
            auto l = Get1DIndex(iw - 1, ih);
            auto r = Get1DIndex(iw + 1, ih);
            auto u = Get1DIndex(iw, ih + 1);
            auto d = Get1DIndex(iw, ih - 1);
            auto lu = Get1DIndex(iw - 1, ih + 1);
            auto rd = Get1DIndex(iw + 1, ih - 1);
            auto &normal = vertices[i].normal;

            normal = {0, 0, 0};

            if (iw > 0 && ih < this->vertex_sizes.y - 1) {
                normal += compute_normal(l, i, lu) * w_small;
                normal += compute_normal(i, u, lu) * w_small;
            }
            if (iw < this->vertex_sizes.x - 1 && ih < this->vertex_sizes.y - 1) {
                normal += compute_normal(i, r, u) * w_large;
            }
            if (iw > 0 && ih > 0) {
                normal += compute_normal(l, d, i) * w_large;
            }
            if (iw < this->vertex_sizes.x - 1 && ih > 0) {
                normal += compute_normal(d, rd, i) * w_small;
                normal += compute_normal(rd, r, i) * w_small;
            }

            normal = glm::normalize(normal);
        }

    // vbo
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(MeshVertex) * vertices.size(), vertices.data(), buffer_data_usage_vbo);
    glBindVertexArray(0);
}

size_t WaterSurface::Get1DIndex(int width, int height) const {
    return size_t(height) * size_t(this->vertex_sizes.x) + size_t(width);
}

void WaterSurface::FixedUpdate() {
    this->Simulate(simulation_steps_per_fixed_update_time);
    this->UpdateMeshVertices();
}

void WaterSurface::Simulate(unsigned times) {
    LocalToWorldPositions();
    IterateWaveParticles();
    ComputeObjectForces();
    IterateObjects();
    GenerateWaveParticles();
    RenderHeightFields();
    WorldToLocalPositions();
}

void WaterSurface::LocalToWorldPositions() {
    auto matrix = this->object->transform->ModelMat();
#pragma omp parallel for
    for (int i = 0; i < this->original_positions.size(); ++i) {
        Vec4 tmp = matrix * Vec4(this->original_positions.at(i), 1);
        this->water_vertices.at(i) = Vec3(tmp);
    }
}

void WaterSurface::WorldToLocalPositions() {
    auto matrix = this->object->transform->ModelMat();
#pragma omp parallel for
    for (Vec3 &water_vertex: this->water_vertices) {
        Vec4 tmp = glm::inverse(matrix) * Vec4(water_vertex, 1);
        water_vertex = Vec3(tmp);
    }
}

void WaterSurface::IterateWaveParticles() {
    std::cout << this->particles.size() << std::endl;
    std::vector<WaveParticle> new_particles;
    for (auto & particle : this->particles){
        float propagate_dist = glm::length(particle.position - particle.original_position);
        float distance = glm::radians(particle.dispersion_angle) * propagate_dist;
        if (distance > 0.25 * particle.radius){
            float prop = glm::cos(glm::radians(particle.dispersion_angle / 3));
            float hori = glm::sin(glm::radians(particle.dispersion_angle / 3));
            WaveParticle temp[3];
            for (auto & i : temp){
                i = particle;
                i.amplitude = particle.amplitude / 3;
                i.dispersion_angle = particle.dispersion_angle / 3;
            }
            temp[0].position = particle.original_position + propagate_dist * prop * particle.propagate + propagate_dist * hori * particle.horizontal;
            temp[2].position = particle.original_position + propagate_dist * prop * particle.propagate - propagate_dist * hori * particle.horizontal;
            for (auto & i : temp){
                i.original_position = i.position;
                i.propagate = glm::normalize(i.position - particle.original_position);
                new_particles.push_back(i);
            }
        }
        else if(particle.amplitude > 0.01f){
            new_particles.push_back(particle);
        }
    }
    this->particles = new_particles;
    for (auto & particle : this->particles) {
        particle.position += wave_speed * WaterSurface::fixed_delta_time * particle.propagate;
        if(reflect){
            if (particle.position.x > 0.5f * this->vertex_sizes.x * dx_local){
                particle.propagate = glm::reflect(particle.propagate, Vec3(-1, 0, 0));
            }
            if (particle.position.x < -0.5f * this->vertex_sizes.x * dx_local){
                particle.propagate = glm::reflect(particle.propagate, Vec3(1, 0, 0));
            }
            if (particle.position.z > 0.5f * this->vertex_sizes.y * dx_local){
                particle.propagate = glm::reflect(particle.propagate, Vec3(0, 0, -1));
            }
            if (particle.position.z < -0.5f * this->vertex_sizes.y * dx_local){
                particle.propagate = glm::reflect(particle.propagate, Vec3(0, 0, 1));
            }
        }
    }
}

void WaterSurface::ComputeObjectForces() {

}

void WaterSurface::IterateObjects() {

}

void WaterSurface::GenerateWaveParticles() {

}

void WaterSurface::RenderHeightFields() {
#pragma omp parallel for
    for (auto &water_vertice: this->water_vertices) {
        Vec2 x_pos = {water_vertice.x, water_vertice.z};
        for (auto &particle: this->particles) {
            Vec2 pos = {particle.position.x, particle.position.z};
            float length = glm::length(x_pos - pos);
            float longitude = glm::dot(particle.propagate, {(x_pos - pos).x, 0, (x_pos - pos).y});
            float a_i = particle.amplitude / 2;
            float W_i = glm::cos(pi * length / particle.radius) + 1;
            float B_i = rectangle_func(length / 2 / particle.radius);
            float D_i = a_i * W_i * B_i;
            Vec3 L_i = -glm::sin(pi * longitude / particle.radius) * rectangle_func(longitude / 2 / particle.radius) *
                       particle.propagate;
            Vec3 D_iL = D_i * L_i;
            water_vertice.y += D_i;
            water_vertice += D_iL;
        }
    }
}

float WaterSurface::rectangle_func(float x) {
    if (x >= -0.5f && x < 0.5f) {
        return 0.5f;
    } else {
        return 0.0f;
    }
}
