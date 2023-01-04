//
// Created by nick on 23-1-2.
//

#include "water_surface.h"
#include "object.h"
#include "transform.h"
#include <iostream>
#include <utility>

WaterSurface::WaterSurface(int limit, UVec2 sizes, float dx_local, std::shared_ptr<Mesh> other_mesh) :
        Mesh(std::vector<MeshVertex>(sizes.x * sizes.y), std::vector<UVec3>((sizes.x - 1) * (sizes.y - 1) * 2),
             GL_STREAM_DRAW, GL_STATIC_DRAW, true),
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
        Vec3 position_tmp = Vec3(0.25f * local_width, 0, 0.25f * local_height);
        Vec3 propagate_tmp = glm::normalize(
                Vec3{-1.0f,
                     0.0f,
                     -1.0f});
        Vec3 horizon_tmp = glm::normalize(glm::cross(propagate_tmp, Vec3{0.0f, 1.0f, 0.0f}));
        this->particles.at(i) = create_particle(position_tmp, propagate_tmp, 360, 5.0f);
//        this->particles.at(i).radius = 0.5f;
//        this->particles.at(i).amplitude = 2.0f;
//        this->particles.at(i).dispersion_angle = 360;
//        this->particles.at(i).surviving_time = 0.0f;
//        float prop = glm::cos(glm::radians(this->particles.at(i).dispersion_angle / 3));
//        float hori = glm::sin(glm::radians(this->particles.at(i).dispersion_angle / 3));
//        this->particles.at(i).position[0] = position_tmp;
//        this->particles.at(i).position[1] = position_tmp;
//        this->particles.at(i).position[2] = position_tmp;
//        this->particles.at(i).propagate[0] = propagate_tmp;
//        this->particles.at(i).propagate[1] = prop * propagate_tmp + hori * horizon_tmp;
//        this->particles.at(i).propagate[2] = prop * propagate_tmp - hori * horizon_tmp;
//        this->particles.at(i).horizontal[0] = horizon_tmp;
//        this->particles.at(i).horizontal[1] = glm::normalize(
//                glm::cross(this->particles.at(i).propagate[1], Vec3{0.0f, 1.0f, 0.0f}));
//        this->particles.at(i).horizontal[2] = glm::normalize(
//                glm::cross(this->particles.at(i).propagate[2], Vec3{0.0f, 1.0f, 0.0f}));
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

    sphere.velocity = {10, 0, 0};
    sphere.radius = 0.5f;
    sphere.radius_on_surface = 0.5f;
    sphere.old_radius_on_surface = 0.5f;
    sphere.center = other_mesh->object->transform->position;
    sphere.old_center = sphere.center;
    sphere.mesh = other_mesh;
    density = 4.0f;
    drag_coef = 0.1f;
    lift_coef = 1.0f;

    for(int i = 0; i < 8; ++i){
        float degree = 360.0f / 8;
        directions.emplace_back(glm::cos(glm::radians(i * degree)), 0, glm::sin(glm::radians(i * degree)));
    }

    timer = 250;
    u = std::uniform_real_distribution<float>(-0.4f*local_width, 0.4f*local_width);
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
#pragma omp parallel for
    for (auto &particle: this->particles) {
        float propagate_dist = particle.surviving_time * wave_speed;
        float distance = glm::radians(particle.dispersion_angle) * propagate_dist;
        if (glm::abs(particle.amplitude) > 5e-4) {
            if (distance > 0.2 * particle.radius) {
                WaveParticle temp[3];
                for (int i = 0; i < 3; ++i) {
//                    temp[i].amplitude = particle.amplitude / 3;
//                    temp[i].dispersion_angle = particle.dispersion_angle / 3;
//                    temp[i].radius = particle.radius;
//                    temp[i].surviving_time = 0.0f;
//                    temp[i].position[0] = particle.position[i];
//                    temp[i].position[1] = particle.position[i];
//                    temp[i].position[2] = particle.position[i];
//                    float prop = glm::cos(glm::radians(temp[i].dispersion_angle / 3));
//                    float hori = glm::sin(glm::radians(temp[i].dispersion_angle / 3));
//                    Vec3 propagate_tmp = particle.propagate[i];
//                    Vec3 horizon_tmp = particle.horizontal[i];
//                    temp[i].propagate[0] = propagate_tmp;
//                    temp[i].propagate[1] = prop * propagate_tmp + hori * horizon_tmp;
//                    temp[i].propagate[2] = prop * propagate_tmp - hori * horizon_tmp;
//                    temp[i].horizontal[0] = horizon_tmp;
//                    temp[i].horizontal[1] = glm::normalize(glm::cross(temp[i].propagate[1], Vec3{0.0f, 1.0f, 0.0f}));
//                    temp[i].horizontal[2] = glm::normalize(glm::cross(temp[i].propagate[2], Vec3{0.0f, 1.0f, 0.0f}));
                    temp[i] = create_particle(particle.position[i], particle.propagate[i],
                                              particle.dispersion_angle / 3, particle.amplitude / 3);
                    new_particles.push_back(temp[i]);
                }
            } else {
                new_particles.push_back(particle);
            }
        }
    }
    this->particles = new_particles;
    Vec3 center_of_sphere = {sphere.center.x, 0, sphere.center.z};
#pragma omp parallel for
    for (auto &particle: this->particles) {
        particle.amplitude *= 0.99;
        particle.surviving_time +=
                WaterSurface::fixed_delta_time * WaterSurface::simulation_steps_per_fixed_update_time;
        particle.position[0] +=
                wave_speed * WaterSurface::fixed_delta_time * WaterSurface::simulation_steps_per_fixed_update_time *
                particle.propagate[0];
        particle.position[1] +=
                wave_speed * WaterSurface::fixed_delta_time * WaterSurface::simulation_steps_per_fixed_update_time *
                particle.propagate[1];
        particle.position[2] +=
                wave_speed * WaterSurface::fixed_delta_time * WaterSurface::simulation_steps_per_fixed_update_time *
                particle.propagate[2];
//        std::cout << particle.position[0].x << " " << particle.position[0].y << " "<< particle.position[0].z << "\n";
        if (reflect) {
            if(glm::length(particle.position[0] - center_of_sphere) < 0.45){
                Vec3 normal = glm::normalize(particle.position[0] - center_of_sphere);
                for (int i = 0; i < 3; ++i){
                    if(glm::dot(glm::normalize(particle.propagate[i]), normal) < -0.99){
                        particle.propagate[i] = -particle.propagate[i];
                        particle.horizontal[i] = -particle.horizontal[i];
                    }
                    else{
                        particle.propagate[i] = glm::reflect(particle.propagate[i], normal);
                        particle.horizontal[i] = glm::reflect(particle.horizontal[i], normal);
                    }
                }
            }
            for (int i = 0; i < 3; ++i) {
                if (particle.position[i].x > 0.5f * this->vertex_sizes.x * dx_local) {
                    particle.propagate[i] = glm::reflect(particle.propagate[i], Vec3(-1, 0, 0));
                    particle.horizontal[i] = glm::reflect(particle.horizontal[i], Vec3(-1, 0, 0));
                }
                if (particle.position[i].x < -0.5f * this->vertex_sizes.x * dx_local) {
                    particle.propagate[i] = glm::reflect(particle.propagate[i], Vec3(1, 0, 0));
                    particle.horizontal[i] = glm::reflect(particle.horizontal[i], Vec3(1, 0, 0));
                }
                if (particle.position[i].z > 0.5f * this->vertex_sizes.y * dx_local) {
                    particle.propagate[i] = glm::reflect(particle.propagate[i], Vec3(0, 0, -1));
                    particle.horizontal[i] = glm::reflect(particle.horizontal[i], Vec3(0, 0, -1));
                }
                if (particle.position[i].z < -0.5f * this->vertex_sizes.y * dx_local) {
                    particle.propagate[i] = glm::reflect(particle.propagate[i], Vec3(0, 0, 1));
                    particle.horizontal[i] = glm::reflect(particle.horizontal[i], Vec3(0, 0, 1));
                }
            }
        }
    }
}

void WaterSurface::ComputeObjectForces() {
    sphere.acceleration = -gravity;
    float height = object->transform->position.y;
    Vec2 x_pos = {sphere.center.x, sphere.center.z};
    Vec3 center_of_sphere = {sphere.center.x, 0, sphere.center.z};
    Vec3 local_velo = {sphere.velocity.x, 0, sphere.velocity.z};
    Vec3 normal = {0, 0, 0};
    for (auto &particle: this->particles) {
        Vec2 pos = {particle.position[0].x, particle.position[0].z};
        float length = glm::length(x_pos - pos);
        if (length > particle.radius) {
            continue;
        }
        float a_i = particle.amplitude / 2;
        float W_i = glm::cos(pi * length / particle.radius) + 1;
        float B_i = rectangle_func(length / 2 / particle.radius);
        float D_i = a_i * W_i * B_i;
        height += D_i;
        local_velo -= particle.propagate[0];
    }
    float h = height - sphere.center.y + sphere.radius;
    float area = 0;
    if (h > 1) {
        h = 1;
        area = pi * sphere.radius * sphere.radius;
    }
    if (h < 0) {
        h = 0;
        area = 0;
    }
    if (h > sphere.radius) {
        float i = h - sphere.radius;
        float another = glm::sqrt(sphere.radius * sphere.radius - i * i);
        float cos = i / sphere.radius;
        float degree = glm::acos(cos);
        sphere.radius_on_surface = another;
        area += (2 * pi - 2 * degree) / (2 * pi) * pi * sphere.radius * sphere.radius;
        area += i * another;
    } else {
        float i = sphere.radius - h;
        float another = glm::sqrt(sphere.radius * sphere.radius - i * i);
        float cos = i / sphere.radius;
        float degree = glm::acos(cos);
        sphere.radius_on_surface = another;
        area += (2 * degree) / (2 * pi) * pi * sphere.radius * sphere.radius;
        area -= i * another;
    }
    float volume = pi * h * h * (sphere.radius - h / 3);
    Vec3 f_drag = -0.5f * density * drag_coef * area * glm::length(local_velo) * local_velo;
    Vec3 f_lift = -0.5f * density * lift_coef * area * glm::length(local_velo) *
                  glm::cross(local_velo, glm::cross(normal, local_velo));
    sphere.acceleration += density * gravity * volume;
    sphere.acceleration += f_drag;
    sphere.acceleration += f_lift;
    sphere.velocity += sphere.acceleration * Time::fixed_delta_time;
    if (glm::length(sphere.velocity) > 0.5f){
        sphere.velocity = 0.5f * glm::normalize(sphere.velocity);
    }
}

void WaterSurface::IterateObjects() {
    Mat4 matrix = sphere.mesh->object->transform->ModelMat();
    if (sphere.center.x - sphere.radius < -0.5f * this->vertex_sizes.x * dx_local ||
        sphere.center.x + sphere.radius > 0.5f * this->vertex_sizes.x * dx_local) {
        sphere.velocity.x = 0.0f;
    }
    if (sphere.center.z - sphere.radius < -0.5f * this->vertex_sizes.y * dx_local ||
        sphere.center.z + sphere.radius > 0.5f * this->vertex_sizes.y * dx_local) {
        sphere.velocity.z = 0.0f;
    }
    sphere.center += Time::fixed_delta_time * sphere.velocity;
    for (auto &vertice: sphere.mesh->vertices) {
        Vec3 tmp = Vec3(matrix * Vec4(vertice.position, 1));
        tmp += Time::fixed_delta_time * sphere.velocity;
        vertice.position = Vec3(glm::inverse(matrix) * Vec4(tmp, 1));
    }
    glBindVertexArray(sphere.mesh->vao);
    glBindBuffer(GL_ARRAY_BUFFER, sphere.mesh->vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(MeshVertex) * sphere.mesh->vertices.size(), sphere.mesh->vertices.data(),
                 sphere.mesh->buffer_data_usage_vbo);
    glBindVertexArray(0);
}

void WaterSurface::GenerateWaveParticles() {
    Vec3 current_center = {sphere.center.x, 0, sphere.center.z};
    Vec3 old_center = {sphere.old_center.x, 0, sphere.old_center.z};
    std::cout << sphere.radius_on_surface << " " << sphere.old_radius_on_surface << '\n';
    for (auto direction: directions){
        Vec3 current_pos = current_center + sphere.radius_on_surface * direction;
        Vec3 old_pos = old_center + sphere.old_radius_on_surface * direction;
        float amplitude = 0.5f * glm::dot(current_pos - old_pos, direction);
//        if (glm::dot(current_pos - old_pos, direction) < 0) {
//            amplitude = -amplitude;
//        }
        WaveParticle new_particle = create_particle(current_pos, direction, 180, amplitude);
        this->particles.push_back(new_particle);
    }
    sphere.old_radius_on_surface = sphere.radius_on_surface;
    sphere.old_center = sphere.center;
    if (timer == 0){
        Vec3 pos = Vec3(u(e), 0, u(e));
        Vec3 dir = Vec3(1, 0, 0);
        WaveParticle new_particle = create_particle(pos, dir, 360, 5.0f);
        this->particles.push_back(new_particle);
        timer = 250;
    }
    timer--;
}

void WaterSurface::RenderHeightFields() {
#pragma omp parallel for
    for (auto &water_vertice: this->water_vertices) {
        Vec2 x_pos = {water_vertice.x, water_vertice.z};
        for (auto &particle: this->particles) {
            Vec2 pos = {particle.position[0].x, particle.position[0].z};
            float length = glm::length(x_pos - pos);
            if (length > particle.radius) {
                continue;
            }
            float longitude = glm::dot(particle.propagate[0], {(x_pos - pos).x, 0, (x_pos - pos).y});
            float a_i = particle.amplitude / 2;
            float W_i = glm::cos(pi * length / particle.radius) + 1;
            float B_i = rectangle_func(length / 2 / particle.radius);
            float D_i = a_i * W_i * B_i;
            Vec3 L_i = -glm::sin(pi * longitude / particle.radius) * rectangle_func(longitude / 2 / particle.radius) *
                       particle.propagate[0];
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

WaveParticle WaterSurface::create_particle(Vec3 position, Vec3 propagate, float dispersion_angle, float amplitude) {
    WaveParticle new_particle{};
    new_particle.position[0] = position;
    new_particle.position[1] = position;
    new_particle.position[2] = position;
    new_particle.radius = 0.5f;
    new_particle.dispersion_angle = dispersion_angle;
    new_particle.amplitude = amplitude;
    new_particle.surviving_time = 0.0f;
    Vec3 propagate_tmp = glm::normalize(propagate);
    Vec3 horizon_tmp = glm::normalize(glm::cross(propagate_tmp, Vec3{0.0f, 1.0f, 0.0f}));
    float prop = glm::cos(glm::radians(dispersion_angle / 3));
    float hori = glm::sin(glm::radians(dispersion_angle / 3));
    new_particle.propagate[0] = propagate_tmp;
    new_particle.propagate[1] = prop * propagate_tmp + hori * horizon_tmp;
    new_particle.propagate[2] = prop * propagate_tmp - hori * horizon_tmp;
    new_particle.horizontal[0] = horizon_tmp;
    new_particle.horizontal[1] = glm::normalize(glm::cross(new_particle.propagate[1], Vec3{0.0f, 1.0f, 0.0f}));
    new_particle.horizontal[2] = glm::normalize(glm::cross(new_particle.propagate[2], Vec3{0.0f, 1.0f, 0.0f}));
    return new_particle;
}
