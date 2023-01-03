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
                                                                     particle_positions(limit),
                                                                     amplitude(limit),
                                                                     propagate(limit),
                                                                     horizontal(limit) {
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
                Vec3{glm::cos(i * glm::radians(90.0f) / 1000),
                     0.0f,
                     glm::sin(i * glm::radians(90.0f) / 1000)});
        Vec3 horizon_tmp = glm::normalize(glm::cross(propagate_tmp, Vec3{0.0f, 1.0f, 0.0f}));
        amplitude.at(i) = 0.5f;
        particle_positions.at(i) = position_tmp;
        propagate.at(i) = propagate_tmp;
        horizontal.at(i) = horizon_tmp;
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
    for (int i = 0; i < times; ++i) {
        LocalToWorldPositions();
        IterateWaveParticles();
        ComputeObjectForces();
        IterateObjects();
        GenerateWaveParticles();
        RenderHeightFields();
        WorldToLocalPositions();
    }
}

void WaterSurface::LocalToWorldPositions() {
    auto matrix = this->object->transform->ModelMat();
#pragma omp parallel for
    for (Vec3 &water_vertex: this->water_vertices) {
        Vec4 tmp = matrix * Vec4(water_vertex, 1);
        water_vertex = Vec3(tmp);
    }
    for (Vec3 &original_position: this->original_positions) {
        Vec4 tmp = matrix * Vec4(original_position, 1);
        original_position = Vec3(tmp);
    }
}

void WaterSurface::WorldToLocalPositions() {
    auto matrix = this->object->transform->ModelMat();
#pragma omp parallel for
    for (Vec3 &water_vertex: this->water_vertices) {
        Vec4 tmp = glm::inverse(matrix) * Vec4(water_vertex, 1);
        water_vertex = Vec3(tmp);
    }
    for (Vec3 &original_position: this->original_positions) {
        Vec4 tmp = glm::inverse(matrix) * Vec4(original_position, 1);
        original_position = Vec3(tmp);
    }
}

void WaterSurface::IterateWaveParticles() {
    for (int i = 0; i < this->limitation; i++) {
        particle_positions.at(i) += wave_speed / simulation_steps_per_fixed_update_time * propagate.at(i);
    }
}

void WaterSurface::ComputeObjectForces() {

}

void WaterSurface::IterateObjects() {

}

void WaterSurface::GenerateWaveParticles() {

}

void WaterSurface::RenderHeightFields() {

}
