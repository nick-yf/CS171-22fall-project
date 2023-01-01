#include "cloth.h"
#include "object.h"
#include "transform.h"

///////////////////
/// constructor ///
///////////////////

RectCloth::RectCloth(Float cloth_weight,
                     const UVec2 &mass_dim,
                     Float dx_local,
                     Float stiffness, Float damping_ratio) : Mesh(std::vector<MeshVertex>(mass_dim.x * mass_dim.y),
                                                                  std::vector<UVec3>((mass_dim.y - 1) * (mass_dim.x - 1) * 2),
                                                                  GL_STREAM_DRAW, GL_STATIC_DRAW,
                                                                  true),
                                                             // private
                                                             mass_dim(mass_dim),
                                                             mass_weight(cloth_weight / Float(mass_dim.x * mass_dim.y)),
                                                             dx_local(dx_local),
                                                             stiffness(stiffness),
                                                             damping_ratio(damping_ratio),
                                                             is_fixed_masses(mass_dim.x * mass_dim.y),
                                                             local_or_world_positions(mass_dim.x * mass_dim.y),
                                                             world_velocities(mass_dim.x * mass_dim.y),
                                                             world_accelerations(mass_dim.x * mass_dim.y)
{

    // initialize local positions
    const auto local_width = Float(mass_dim.x) * dx_local;
    const auto local_height = Float(mass_dim.y) * dx_local;

#pragma omp parallel for collapse(2)
    for (int ih = 0; ih < mass_dim.y; ++ih)
        for (int iw = 0; iw < mass_dim.x; ++iw)
            local_or_world_positions[Get1DIndex(iw, ih)] = Vec3(Float(iw) * dx_local - local_width * Float(0.5),
                                                                Float(ih) * dx_local - local_height * Float(0.5),
                                                                0);

    // initialize mesh vertices
    UpdateMeshVertices();

    // initialize mesh indices
#pragma omp parallel for collapse(2)
    for (int ih = 0; ih < mass_dim.y - 1; ++ih)
        for (int iw = 0; iw < mass_dim.x - 1; ++iw)
        {
            size_t i_indices = (size_t(ih) * size_t(mass_dim.x - 1) + size_t(iw)) << 1;

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

//////////////////
/// interfaces ///
//////////////////

bool RectCloth::SetMassFixedOrNot(int iw, int ih, bool fixed_or_not)
{
    iw = iw < 0 ? int(mass_dim.x) + iw : iw;
    ih = ih < 0 ? int(mass_dim.y) + ih : ih;

    size_t idx;
    if (!Get1DIndex(iw, ih, idx))
        return false;

    is_fixed_masses[idx] = fixed_or_not;
    return true;
}

/*override*/ void RectCloth::FixedUpdate()
{
    // simulate
    Simulate(simulation_steps_per_fixed_update_time);

    // update mesh vertices
    UpdateMeshVertices();
}

/////////////////////////
/// force computation ///
/////////////////////////

Vec3 RectCloth::ComputeHookeForce(int iw_this, int ih_this,
                                  int iw_that, int ih_that,
                                  Float dx_world) const
{

    /*! TODO: implement this: compute the force according to Hooke's law
     *                        applied to mass(iw_this, ih_this)
     *                                by mass(iw_that, ih_that)
     *                        `dx_world` is "the zero-force distance" in world coordinate
     *
     *        note: for invalid `iw` or `ih`, you may simply return { 0, 0, 0 }
     */
    float k = this->stiffness;
    if (iw_this < 0 || iw_this >= this->mass_dim.x || ih_this < 0 || ih_this >= this->mass_dim.y || iw_that < 0 || iw_that >= this->mass_dim.x || ih_that < 0 || ih_that >= this->mass_dim.y)
    {
        return {0, 0, 0};
    }
    size_t here = this->Get1DIndex(iw_this, ih_this);
    size_t that = this->Get1DIndex(iw_that, ih_that);
    Vec3 p = this->local_or_world_positions[here];
    Vec3 q = this->local_or_world_positions[that];
    Vec3 L = k * (dx_world - glm::length(p - q)) * glm::normalize(p - q);
    return L;
}

Vec3 RectCloth::ComputeSpringForce(int iw, int ih) const
{

    const Vec3 scale = object->transform->scale;

    /*! TODO: implement this: compute the total spring force applied to mass(iw, ih)
     *                        by some other neighboring masses
     *
     *        note: you MUST consider structural, shear, and bending springs
     *              you MUST consider scaling of "the zero-force distance"
     *              you may find ComputeHookeForce() helpful
     *              for invalid `iw` or `ih`, you may simply return { 0, 0, 0 }
     *              for "fixed masses", you may also simply return { 0, 0, 0 }
     */
    Vec3 force(0.0f);
    // structure
    force += this->ComputeHookeForce(iw, ih, iw - 1, ih, this->dx_local);
    force += this->ComputeHookeForce(iw, ih, iw, ih - 1, this->dx_local);
    force += this->ComputeHookeForce(iw, ih, iw + 1, ih, this->dx_local);
    force += this->ComputeHookeForce(iw, ih, iw, ih + 1, this->dx_local);
    // shear
    force += this->ComputeHookeForce(iw, ih, iw - 1, ih - 1, std::numbers::sqrt2 * this->dx_local);
    force += this->ComputeHookeForce(iw, ih, iw - 1, ih + 1, std::numbers::sqrt2 * this->dx_local);
    force += this->ComputeHookeForce(iw, ih, iw + 1, ih - 1, std::numbers::sqrt2 * this->dx_local);
    force += this->ComputeHookeForce(iw, ih, iw + 1, ih + 1, std::numbers::sqrt2 * this->dx_local);
    // bending
    force += this->ComputeHookeForce(iw, ih, iw, ih - 2, 2 * this->dx_local);
    force += this->ComputeHookeForce(iw, ih, iw - 2, ih, 2 * this->dx_local);
    force += this->ComputeHookeForce(iw, ih, iw, ih + 2, 2 * this->dx_local);
    force += this->ComputeHookeForce(iw, ih, iw + 2, ih, 2 * this->dx_local);

    force -= this->damping_ratio * this->world_velocities[this->Get1DIndex(iw, ih)];

    return force;
}

Vec3 RectCloth::ComputeWindForce(int iw, int ih, Vec3 wind_speed) const
{

    const Vec3 scale = object->transform->scale;
    const Mat4 model_matrix = object->transform->ModelMat();

    /*! TODO: implement this: compute the total spring force applied to mass(iw, ih)
     *                        by some other neighboring masses
     *
     *        note: you MUST consider structural, shear, and bending springs
     *              you MUST consider scaling of "the zero-force distance"
     *              you may find ComputeHookeForce() helpful
     *              for invalid `iw` or `ih`, you may simply return { 0, 0, 0 }
     *              for "fixed masses", you may also simply return { 0, 0, 0 }
     */
    size_t here = this->Get1DIndex(iw, ih);
    Vec3 normal = Mat3(glm::transpose(glm::inverse(model_matrix))) * this->vertices[here].normal;
    Vec3 force = 0.001f * glm::dot(normal, (wind_speed - this->world_velocities[here])) * normal;
    return force;
}

///////////////////////////
/// simulation pipeline ///
///////////////////////////

void RectCloth::LocalToWorldPositions()
{

    const Mat4 model_matrix = object->transform->ModelMat();

    /*! TODO: implement this: transform mass positions from local coordinate to world coordinate
     *
     *        note: you may find `class Transform` in `transform.h` helpful
     */
    for (auto &local_or_world_position : this->local_or_world_positions)
    {
        Vec4 local = Vec4(local_or_world_position, 0);
        Vec4 global = model_matrix * local;
        local_or_world_position = Vec3(global);
    }
}

void RectCloth::ComputeAccelerations()
{

    /*! TODO: implement this: compute accelerations for each mass
     *
     *        note: you may find ComputeSpringForce() helpful
     *              you may store the results into `world_accelerations`
     */
    std::uniform_real_distribution<float> u(0.0f, 1.0f);
    Vec3 wind(u(e), 0.0f, u(e));
    wind = 10.0f * glm::normalize(wind);
    for (int j = 0; j < this->mass_dim.y; ++j)
    {
        for (int i = 0; i < this->mass_dim.x; ++i)
        {
            if (this->is_fixed_masses[Get1DIndex(i, j)])
                continue;
            // Vec3 vector = this->local_or_world_positions[Get1DIndex(i, j)] - Vec3(0, -2, 1);
            // float dist = glm::length(vector);
            world_accelerations[Get1DIndex(i, j)] = (this->ComputeSpringForce(i, j) + this->ComputeWindForce(i, j, wind)) / this->mass_weight - gravity;
            // float w = glm::dot(world_accelerations[Get1DIndex(i, j)], glm::normalize(vector));
            // float pred2sphere = glm::dot(world_velocities[Get1DIndex(i, j)], glm::normalize(vector));
            // if (dist <= 0.55f && w <= 0)
            // {
            //     Vec3 force_by_sphere = w * glm::normalize(vector);
            //     world_accelerations[Get1DIndex(i, j)] -= force_by_sphere;
            //     world_velocities[Get1DIndex(i, j)] -= pred2sphere * glm::normalize(vector);
            //     local_or_world_positions[Get1DIndex(i, j)] = 0.55f * glm::normalize(vector) + Vec3(0, -2, 1);
            // }
        }
    }
}

void RectCloth::ComputeVelocities()
{
    /*! TODO: implement this: update velocities for each mass
     *
     *        note: you may store the results into `world_velocities`
     *              you may use `this->fixed_delta_time` instead of `Time::fixed_delta_time`, why?
     */
    for (int j = 0; j < this->mass_dim.y; ++j)
    {
        for (int i = 0; i < this->mass_dim.x; ++i)
        {
            if (this->is_fixed_masses[Get1DIndex(i, j)])
                continue;
            world_velocities[Get1DIndex(i, j)] += world_accelerations[Get1DIndex(i, j)] * this->fixed_delta_time;
            Vec3 vector = this->local_or_world_positions[Get1DIndex(i, j)] - Vec3(0, -2, 1);
            float dist = glm::length(vector);
            float pred2sphere = glm::dot(world_velocities[Get1DIndex(i, j)], glm::normalize(vector));
            if (dist <= 0.52f)
            {
                world_velocities[Get1DIndex(i, j)] -= pred2sphere * glm::normalize(vector);
                local_or_world_positions[Get1DIndex(i, j)] = 0.52f * glm::normalize(vector) + Vec3(0, -2, 1);
            }
        }
    }
}

void RectCloth::ComputePositions()
{

    /*! TODO: implement this: update positions for each mass
     *
     *        note: you may store the results into `local_or_world_positions`
     *              you may use `this->fixed_delta_time` instead of `Time::fixed_delta_time`, why?
     */
    for (int j = 0; j < this->mass_dim.y; ++j)
    {
        for (int i = 0; i < this->mass_dim.x; ++i)
        {
            if (this->is_fixed_masses[Get1DIndex(i, j)])
                continue;
            local_or_world_positions[Get1DIndex(i, j)] += world_velocities[Get1DIndex(i, j)] * this->fixed_delta_time;
        }
    }
}

void RectCloth::WorldToLocalPositions()
{

    const Mat4 model_matrix = object->transform->ModelMat();

    /*! TODO: implement this: transform mass positions from world coordinate to local coordinate
     *
     *        note: you may find `class Transform` in `transform.h` helpful
     */
    for (auto &local_or_world_position : this->local_or_world_positions)
    {
        Vec4 global = Vec4(local_or_world_position, 0);
        Vec4 local = glm::inverse(model_matrix) * global;
        local_or_world_position = Vec3(local);
    }
}

void RectCloth::Simulate(unsigned num_steps)
{
    for (unsigned i = 0; i < num_steps; ++i)
    {
        LocalToWorldPositions();
        ComputeAccelerations();
        ComputeVelocities();
        ComputePositions();
        WorldToLocalPositions();
    }
}

/////////////////
/// rendering ///
/////////////////

void RectCloth::UpdateMeshVertices()
{

    // set vertex positions
    for (size_t i = 0; i < local_or_world_positions.size(); ++i)
        vertices[i].position = local_or_world_positions[i];

    // reset vertex normals
    auto compute_normal = [&](auto v1, auto v2, auto v3)
    {
        return glm::normalize(glm::cross(vertices[v2].position - vertices[v1].position, vertices[v3].position - vertices[v1].position));
    };

#pragma omp parallel for collapse(2)
    for (int ih = 0; ih < mass_dim.y; ++ih)
        for (int iw = 0; iw < mass_dim.x; ++iw)
        {
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

            if (iw > 0 && ih < mass_dim.y - 1)
            {
                normal += compute_normal(l, i, lu) * w_small;
                normal += compute_normal(i, u, lu) * w_small;
            }
            if (iw < mass_dim.x - 1 && ih < mass_dim.y - 1)
            {
                normal += compute_normal(i, r, u) * w_large;
            }
            if (iw > 0 && ih > 0)
            {
                normal += compute_normal(l, d, i) * w_large;
            }
            if (iw < mass_dim.x - 1 && ih > 0)
            {
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

//////////////////////////
/// supporting methods ///
//////////////////////////

size_t RectCloth::Get1DIndex(int iw, int ih) const
{
    return size_t(ih) * size_t(mass_dim.x) + size_t(iw);
}

bool RectCloth::Get1DIndex(int iw, int ih, size_t &idx) const
{
    if (iw < 0 || ih < 0 || iw >= mass_dim.x || ih >= mass_dim.y)
        return false;
    idx = size_t(ih) * size_t(mass_dim.x) + size_t(iw);
    return true;
}

void RectCloth::fixPointByCursor(Vec3 origin, Vec3 dir, Vec3 forward, bool setOrNot)
{
    if (!setOrNot)
    {
        if (this->control_point_index != -1)
        {
            this->is_fixed_masses[this->control_point_index] = false;
        }
        this->control_point_index = -1;
    }
    else
    {
        this->LocalToWorldPositions();
        if (this->control_point_index == -1)
        {
            float dis = 1e9;
            for (int i = 0; i < this->local_or_world_positions.size(); i++)
            {
                if (!this->is_fixed_masses[i])
                {
                    Vec3 vector = this->local_or_world_positions[i] - origin;
                    Vec3 vertical = vector - glm::dot(vector, dir) * dir;
                    if (glm::length(vertical) < this->dx_local / 2 && glm::dot(vector, dir) < dis)
                    {
                        this->control_point_index = i;
                        dis = glm::dot(vector, dir);
                    }
                }
            }
            if (this->control_point_index != -1)
            {
                this->is_fixed_masses[this->control_point_index] = true;
            }
        }
        float t = glm::dot(origin - this->local_or_world_positions[this->control_point_index], forward) / (-glm::dot(dir, forward));
        Vec3 P = origin + t * dir;
        this->local_or_world_positions[this->control_point_index] = P;
        this->WorldToLocalPositions();
    }
}
