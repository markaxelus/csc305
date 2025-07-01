// C++ include
#include <iostream>
#include <string>
#include <vector>
#include <limits>
#include <cmath>
#include <algorithm>
#include <cstdlib>

// Eigen include (assuming utils.h brings in Eigen)
#include <Eigen/Dense>

// Utilities for the Assignment
#include "utils.h"

// Image writing library
#define STB_IMAGE_WRITE_IMPLEMENTATION // Do not include this line twice in your project!
#include "stb_image_write.h"

// Shortcut to avoid Eigen:: everywhere, DO NOT USE IN .h
using namespace Eigen;

////////////////////////////////////////////////////////////////////////////////
// Scene setup, global variables
////////////////////////////////////////////////////////////////////////////////

const std::string filename("raytrace.png");

// Camera settings
const double focal_length = 10;
const double field_of_view = 0.7854; // 45 degrees
const double image_z = 5;
const bool is_perspective = false;
const Vector3d camera_position(0, 0, 5);
const double camera_aperture = 0.05;

// Maximum number of recursive calls
const int max_bounce = 5;

// Objects
std::vector<Vector3d> sphere_centers;
std::vector<double> sphere_radii;
std::vector<Matrix3d> parallelograms;

// Material for the object, same material for all objects
const Vector4d obj_ambient_color(0.5, 0.1, 0.1, 0);
const Vector4d obj_diffuse_color(0.5, 0.5, 0.5, 0);
const Vector4d obj_specular_color(0.2, 0.2, 0.2, 0);
const double obj_specular_exponent = 256.0;
const Vector4d obj_reflection_color(0.7, 0.7, 0.7, 0);
const Vector4d obj_refraction_color(0.7, 0.7, 0.7, 0);

// Precomputed (or otherwise) gradient vectors at each grid node
const int grid_size = 20;
std::vector<std::vector<Vector2d>> grid;

// Lights
std::vector<Vector3d> light_positions;
std::vector<Vector4d> light_colors;
// Ambient light
const Vector4d ambient_light(0.2, 0.2, 0.2, 0);

// Fills the different arrays
void setup_scene()
{
    grid.resize(grid_size + 1);
    for (int i = 0; i < grid_size + 1; ++i)
    {
        grid[i].resize(grid_size + 1);
        for (int j = 0; j < grid_size + 1; ++j)
            grid[i][j] = Vector2d::Random().normalized();
    }

    // Spheres
    sphere_centers.emplace_back(10, 0, 1);
    sphere_radii.emplace_back(1);

    sphere_centers.emplace_back(7, 0.05, -1);
    sphere_radii.emplace_back(1);

    sphere_centers.emplace_back(4, 0.1, 1);
    sphere_radii.emplace_back(1);

    sphere_centers.emplace_back(1, 0.2, -1);
    sphere_radii.emplace_back(1);

    sphere_centers.emplace_back(-2, 0.4, 1);
    sphere_radii.emplace_back(1);

    sphere_centers.emplace_back(-5, 0.8, -1);
    sphere_radii.emplace_back(1);

    sphere_centers.emplace_back(-8, 1.6, 1);
    sphere_radii.emplace_back(1);

    // parallelograms
    parallelograms.emplace_back();
    parallelograms.back() << -100, 100, -100,
                             -1.25, 0,   -1.2,
                             -100, -100, 100;

    // Lights
    light_positions.emplace_back(8, 8, 0);
    light_colors.emplace_back(16, 16, 16, 0);

    light_positions.emplace_back(6, -8, 0);
    light_colors.emplace_back(16, 16, 16, 0);

    light_positions.emplace_back(4, 8, 0);
    light_colors.emplace_back(16, 16, 16, 0);

    light_positions.emplace_back(2, -8, 0);
    light_colors.emplace_back(16, 16, 16, 0);

    light_positions.emplace_back(0, 8, 0);
    light_colors.emplace_back(16, 16, 16, 0);

    light_positions.emplace_back(-2, -8, 0);
    light_colors.emplace_back(16, 16, 16, 0);

    light_positions.emplace_back(-4, 8, 0);
    light_colors.emplace_back(16, 16, 16, 0);
}

// We need to make this function visible
Vector4d shoot_ray(const Vector3d &ray_origin, const Vector3d &ray_direction, int max_bounce);

////////////////////////////////////////////////////////////////////////////////
// Perlin noise code
////////////////////////////////////////////////////////////////////////////////

// Function to linearly interpolate between a0 and a1
// Weight w should be in the range [0.0, 1.0]
double lerp(double a0, double a1, double w)
{
    assert(w >= 0);
    assert(w <= 1);
    // Linear interpolation
    return a0 + w * (a1 - a0);
}

// Computes the dot product of the distance and gradient vectors.
double dotGridGradient(int ix, int iy, double x, double y)
{
    // Get gradient from grid
    const Vector2d &gradient = grid[ix][iy];
    // Compute distance vector
    double dx = x - static_cast<double>(ix);
    double dy = y - static_cast<double>(iy);
    // Return dot product
    return gradient.x() * dx + gradient.y() * dy;
}

// Compute Perlin noise at coordinates x, y
double perlin(double x, double y)
{
    // Determine grid cell coordinates x0, y0
    int x0 = static_cast<int>(std::floor(x));
    int x1 = x0 + 1;
    int y0 = static_cast<int>(std::floor(y));
    int y1 = y0 + 1;

    // Determine interpolation weights
    double sx = x - static_cast<double>(x0);
    double sy = y - static_cast<double>(y0);

    // Interpolate between grid point gradients
    double n0 = dotGridGradient(x0, y0, x, y);
    double n1 = dotGridGradient(x1, y0, x, y);
    double ix0 = lerp(n0, n1, sx);

    n0 = dotGridGradient(x0, y1, x, y);
    n1 = dotGridGradient(x1, y1, x, y);
    double ix1 = lerp(n0, n1, sx);

    double value = lerp(ix0, ix1, sy);
    return value;
}

Vector4d procedural_texture(const double tu, const double tv)
{
    assert(tu >= 0 && tu <= 1);
    assert(tv >= 0 && tv <= 1);

    // Use Perlin noise for texture
    const double color = (perlin(tu * grid_size, tv * grid_size) + 1.0) * 0.5;
    return Vector4d(0, color, 0, 0);
}

////////////////////////////////////////////////////////////////////////////////
// Intersection code
////////////////////////////////////////////////////////////////////////////////

// Compute the intersection between a ray and a sphere, return -1 if no intersection
double ray_sphere_intersection(const Vector3d &ray_origin, const Vector3d &ray_direction,
                               int index, Vector3d &p, Vector3d &N)
{
    const Vector3d &center = sphere_centers[index];
    double radius = sphere_radii[index];

    Vector3d oc = ray_origin - center;
    double b = ray_direction.dot(oc);
    double c = oc.dot(oc) - radius * radius;
    double discriminant = b * b - c;
    if (discriminant < 0) return -1;

    double sqrt_disc = std::sqrt(discriminant);
    double t1 = -b - sqrt_disc;
    double t2 = -b + sqrt_disc;
    double t = (t1 > 1e-4 ? t1 : (t2 > 1e-4 ? t2 : -1));
    if (t < 0) return -1;

    p = ray_origin + t * ray_direction;
    N = (p - center).normalized();
    return t;
}

// Compute the intersection between a ray and a parallelogram, return -1 if no intersection
double ray_parallelogram_intersection(const Vector3d &ray_origin, const Vector3d &ray_direction,
                                      int index, Vector3d &p, Vector3d &N)
{
    const Vector3d &O0 = parallelograms[index].col(0);
    Vector3d u = parallelograms[index].col(1) - O0;
    Vector3d v = parallelograms[index].col(2) - O0;

    // Plane normal
    Vector3d normal = u.cross(v).normalized();
    double denom = ray_direction.dot(normal);
    if (std::fabs(denom) < 1e-6) return -1; // Parallel

    double t = (O0 - ray_origin).dot(normal) / denom;
    if (t < 1e-4) return -1;

    Vector3d P = ray_origin + t * ray_direction;
    Vector3d w = P - O0;

    // Solve for barycentric coordinates
    double uu = u.dot(u), uv = u.dot(v), vv = v.dot(v);
    double wu = w.dot(u), wv = w.dot(v);
    double denom2 = uv * uv - uu * vv;
    double s = (uv * wv - vv * wu) / denom2;
    double r = (uv * wu - uu * wv) / denom2;

    if (s < 0 || s > 1 || r < 0 || r > 1) return -1;

    p = P;
    N = normal;
    return t;
}

// Finds the closest intersecting object, returns its index
// In case of intersection, writes into p and N (intersection point and normal)
int find_nearest_object(const Vector3d &ray_origin, const Vector3d &ray_direction,
                        Vector3d &p, Vector3d &N)
{
    int closest_index = -1;
    double closest_t = std::numeric_limits<double>::max();
    Vector3d tmp_p, tmp_N;

    for (int i = 0; i < static_cast<int>(sphere_centers.size()); ++i)
    {
        double t = ray_sphere_intersection(ray_origin, ray_direction, i, tmp_p, tmp_N);
        if (t >= 0 && t < closest_t)
        {
            closest_t = t;
            closest_index = i;
            p = tmp_p;
            N = tmp_N;
        }
    }

    for (int i = 0; i < static_cast<int>(parallelograms.size()); ++i)
    {
        double t = ray_parallelogram_intersection(ray_origin, ray_direction, i, tmp_p, tmp_N);
        if (t >= 0 && t < closest_t)
        {
            closest_t = t;
            closest_index = static_cast<int>(sphere_centers.size()) + i;
            p = tmp_p;
            N = tmp_N;
        }
    }

    return closest_index;
}

////////////////////////////////////////////////////////////////////////////////
// Raytracer code
////////////////////////////////////////////////////////////////////////////////

// Checks if the light is visible (shadow rays)
bool is_light_visible(const Vector3d &ray_origin, const Vector3d &ray_direction,
                      const Vector3d &light_position)
{
    double max_dist = (light_position - ray_origin).norm();
    Vector3d dir = (light_position - ray_origin).normalized();
    Vector3d p_tmp, n_tmp;
    int obj = find_nearest_object(ray_origin + dir * 1e-4, dir, p_tmp, n_tmp);
    if (obj >= 0)
    {
        double dist = (p_tmp - ray_origin).norm();
        if (dist < max_dist) return false;
    }
    return true;
}

Vector4d shoot_ray(const Vector3d &ray_origin, const Vector3d &ray_direction, int depth)
{
    Vector3d p, N;
    int obj_index = find_nearest_object(ray_origin, ray_direction, p, N);
    if (obj_index < 0)
    {
        return Vector4d(0, 0, 0, 0);
    }

    // Ambient contribution
    Vector4d ambient = obj_ambient_color.cwiseProduct(ambient_light);

    // Direct lighting
    Vector4d lights_color_accum(0, 0, 0, 0);
    for (size_t i = 0; i < light_positions.size(); ++i)
    {
        Vector3d L = (light_positions[i] - p).normalized();
        if (!is_light_visible(p, L, light_positions[i])) continue;

        // Diffuse shading
        Vector4d diff_col = obj_diffuse_color;
        if (obj_index == 4)
        {
            // Procedural texture for one sphere
            Vector3d c = p - sphere_centers[obj_index];
            double tu = std::acos(c.z() / sphere_radii[obj_index]) / M_PI;
            double tv = (M_PI + std::atan2(c.y(), c.x())) / (2 * M_PI);
            tu = std::clamp(tu, 0.0, 1.0);
            tv = std::clamp(tv, 0.0, 1.0);
            diff_col = procedural_texture(tu, tv);
        }
        double diff_intensity = std::max(N.dot(L), 0.0);
        Vector4d diffuse = diff_col * diff_intensity;

        // Specular shading (Blinn-Phong)
        Vector3d V = -ray_direction.normalized();
        Vector3d H = (L + V).normalized();
        double spec_intensity = std::pow(std::max(N.dot(H), 0.0), obj_specular_exponent);
        Vector4d specular = obj_specular_color * spec_intensity;

        // Attenuate by distance
        double dist2 = (light_positions[i] - p).squaredNorm();
        lights_color_accum += (diffuse + specular).cwiseProduct(light_colors[i]) / dist2;
    }

    // Reflection
    Vector4d reflection_color(0, 0, 0, 0);
    Vector4d refl_col = (obj_index == 4 ? Vector4d(0.5, 0.5, 0.5, 0) : obj_reflection_color);
    if (depth > 0)
    {
        Vector3d R = ray_direction - 2 * ray_direction.dot(N) * N;
        Vector3d origin_reflect = p + R * 1e-4;
        Vector4d trace_reflect = shoot_ray(origin_reflect, R.normalized(), depth - 1);
        reflection_color = refl_col.cwiseProduct(trace_reflect);
    }

    // Refraction
    Vector4d refraction_color(0, 0, 0, 0);
    if (depth > 0)
    {
        double ior1 = 1.0, ior2 = 1.5;
        Vector3d n_refract = N;
        double cosi = std::clamp(ray_direction.dot(N), -1.0, 1.0);
        double eta = ior1 / ior2;
        if (cosi > 0)
        {
            // Inside to outside
            std::swap(ior1, ior2);
            eta = ior1 / ior2;
            n_refract = -N;
        }
        double k = 1 - eta * eta * (1 - cosi * cosi);
        if (k >= 0)
        {
            Vector3d T = eta * ray_direction + (eta * cosi - std::sqrt(k)) * n_refract;
            Vector3d origin_refract = p + T * 1e-4;
            Vector4d trace_refract = shoot_ray(origin_refract, T.normalized(), depth - 1);
            refraction_color = obj_refraction_color.cwiseProduct(trace_refract);
        }
    }

    Vector4d C = ambient + lights_color_accum + reflection_color + refraction_color;
    C(3) = 1.0;
    return C;
}

////////////////////////////////////////////////////////////////////////////////

void raytrace_scene()a
{
    std::cout << "Simple ray tracer." << std::endl;

    int w = 800;
    int h = 400;
    MatrixXd R = MatrixXd::Zero(w, h);
    MatrixXd G = MatrixXd::Zero(w, h);
    MatrixXd B = MatrixXd::Zero(w, h);
    MatrixXd A = MatrixXd::Zero(w, h);

    double aspect_ratio = static_cast<double>(w) / h;
    // Compute image plane size
    double half_height = std::tan(field_of_view * 0.5) * image_z;
    double half_width = aspect_ratio * half_height;
    double image_y = half_height;
    double image_x = half_width;

    // The pixel grid is at distance 'image_z'
    const Vector3d image_origin(-image_x, image_y, -image_z);
    const Vector3d x_displacement(2.0 * image_x / w, 0, 0);
    const Vector3d y_displacement(0, -2.0 * image_y / h, 0);

    for (int i = 0; i < w; ++i)
    {
        for (int j = 0; j < h; ++j)
        {
            Vector3d pixel_center = image_origin + (i + 0.5) * x_displacement + (j + 0.5) * y_displacement;

            Vector3d ray_origin, ray_direction;
            if (is_perspective)
            {
                // Depth of field perspective camera
                Vector3d dir = pixel_center.normalized();
                Vector3d focus_point = dir * focal_length;

                // Sample lens
                double lens_radius = camera_aperture * 0.5;
                double rx, ry;
                do {
                    rx = 2.0 * (std::rand() / static_cast<double>(RAND_MAX)) - 1.0;
                    ry = 2.0 * (std::rand() / static_cast<double>(RAND_MAX)) - 1.0;
                } while (rx * rx + ry * ry > 1.0);
                Vector3d offset(rx * lens_radius, ry * lens_radius, 0);

                ray_origin = camera_position + offset;
                ray_direction = (focus_point - offset).normalized();
            }
            else
            {
                // Orthographic camera
                ray_origin = camera_position + Vector3d(pixel_center.x(), pixel_center.y(), 0);
                ray_direction = Vector3d(0, 0, -1);
            }

            Vector4d C = shoot_ray(ray_origin, ray_direction.normalized(), max_bounce);
            R(i, j) = C.x();
            G(i, j) = C.y();
            B(i, j) = C.z();
            A(i, j) = C.w();
        }
    }

    // Save to png
    write_matrix_to_png(R, G, B, A, filename);
}

////////////////////////////////////////////////////////////////////////////////

int main(int argc, char *argv[])
{
    std::srand(42); // Fixed seed for reproducibility
    setup_scene();
    raytrace_scene();
    return 0;
}
