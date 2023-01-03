//
// Created by nick on 23-1-2.
//

#include "water_surface.h"

WaterSurface::WaterSurface(int limit, UVec2 sizes, float dx_local) : Mesh(std::vector<MeshVertex>(sizes.x * sizes.y),
                                                                          std::vector<UVec3>(
                                                                                  (sizes.x - 1) * (sizes.y - 1) * 2),
                                                                          GL_STREAM_DRAW, GL_STATIC_DRAW,
                                                                          true), limitation(limit),
                                                                     local_or_world_pos(sizes.x * sizes.y),
                                                                     vertex_sizes(sizes), dx_local(dx_local) {
    float local_width = sizes.x * dx_local;
    float local_height = sizes.y * dx_local;
#pragma omp parallel for
    for (int height = 0; height < sizes.y; ++height) {
        for (int width = 0; width < sizes.x; ++width) {
            local_or_world_pos.at(this->GetIndex(width, height)) = Vec3(width * dx_local - 0.5f * local_width,
                                                                        0,
                                                                        height * dx_local - 0.5f * local_height);
        }
    }

    // initialize mesh vertices
    UpdateMeshVertices();

    // initialize mesh indices
#pragma omp parallel for
    for (int ih = 0; ih < sizes.y - 1; ++ih)
        for (int iw = 0; iw < sizes.x - 1; ++iw) {
            size_t i_indices = (size_t(ih) * size_t(sizes.x - 1) + size_t(iw)) << 1;

            auto i = GetIndex(iw, ih);
            auto r = GetIndex(iw + 1, ih);
            auto u = GetIndex(iw, ih + 1);
            auto ru = GetIndex(iw + 1, ih + 1);

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
    for (size_t i = 0; i < local_or_world_pos.size(); ++i)
        vertices[i].position = local_or_world_pos[i];

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

            auto i = GetIndex(iw, ih);
            auto l = GetIndex(iw - 1, ih);
            auto r = GetIndex(iw + 1, ih);
            auto u = GetIndex(iw, ih + 1);
            auto d = GetIndex(iw, ih - 1);
            auto lu = GetIndex(iw - 1, ih + 1);
            auto rd = GetIndex(iw + 1, ih - 1);
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

size_t WaterSurface::GetIndex(int width, int height) const {
    return size_t(height) * size_t(this->vertex_sizes.x) + size_t(width);
}

void WaterSurface::FixedUpdate() {
    this->UpdateMeshVertices();
}
