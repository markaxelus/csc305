////////////////////////////////////////////////////////////////////////////////
// C++ include
#include <iostream>
#include <string>
#include <vector>
#include <limits>

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
const bool is_perspective = true;
const Vector3d camera_position(0, 0, 5);

double ep = .0001;

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
std::vector<std::vector<Vector2d> > grid;

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
        -1.25, 0, -1.2,
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

//We need to make this function visible
Vector4d shoot_ray(const Vector3d &ray_origin, const Vector3d &ray_direction, int max_bounce);

// For Sphere
Vector3d getSphere(Vector3d c, Vector3d e, Vector3d d, double R, double &t)
{
    double A = d.dot(d);
    double B = (2 * d).dot(e - c);
    double C = (e - c).dot(e - c) - (R * R);

    double discriminant = (B * B) - (4 * A * C);

    if (discriminant == 0)
    {
        // One intersection point
        t = (-B + sqrt(discriminant)) / (2 * A);
    }
    else if (discriminant > 0)
    {
        // More than one intersection point
        t = (-B + sqrt(discriminant)) / (2 * A);
        double t2 = (-B - sqrt(discriminant)) / (2 * A);
        if (t < 0)
        {
            t = t2;
        }
        else if (t > 0 && t2 > 0)
        {
            t = fmin(t, t2);
        }
    }

    return (e + (t * d));
}

bool raySphere(Vector3d c, Vector3d e, Vector3d d, double R)
{
    double A = d.dot(d);
    double B = (2 * d).dot(e - c);
    double C = (e - c).dot(e - c) - (R * R);

    double discriminant = (B * B) - (4 * A * C);

    if (discriminant < 0)
    {
        return false;
    }
    else if (discriminant == 0)
    {
        // One intersection point
        double t = (-B + sqrt(discriminant)) / (2 * A);

        if (t < 0)
        {
            return false;
        }
    }
    else if (discriminant > 0)
    {
        // More than one intersection point
        double t = (-B + sqrt(discriminant)) / (2 * A);
        double t2 = (-B - sqrt(discriminant)) / (2 * A);

        if (t < 0 && t2 < 0)
        {
            return false;
        }
    }
    return true;
}

// For Parallelogram
Vector3d getPoint(Vector3d u_vector, Vector3d v_vector, Vector3d d_vector, Vector3d a_vector, Vector3d e_vector, double &t)
{
    // Make matrix A 
    Matrix3d A;
    A << -u_vector, -v_vector, d_vector;
    Vector3d ae_vector = a_vector - e_vector;
    Vector3d solution_vector = A.colPivHouseholderQr().solve(ae_vector);
    t = solution_vector(2);

    return a_vector + (solution_vector(0) * u_vector) + (solution_vector(1) * v_vector);
}

bool rayMatrix(Vector3d u_vector, Vector3d v_vector, Vector3d d_vector, Vector3d a_vector, Vector3d e_vector)
{
    // Make matrix A 
    Matrix3d A;
    A << -u_vector, -v_vector, d_vector;
    Vector3d ae_vector = a_vector - e_vector;
    Vector3d solution_vector = A.colPivHouseholderQr().solve(ae_vector);

    // Check t
    if (solution_vector(2) < 0)
    {
        return false;
    }

    // Check u
    if (solution_vector(0) < 0 || solution_vector(0) > 1)
    {
        return false;
    }

    // Check 0 <= v <= 1
    if (solution_vector(1) < 0 || solution_vector(1) > 1)
    {
        return false;
    }

    return true;
}

////////////////////////////////////////////////////////////////////////////////
// Perlin noise code
////////////////////////////////////////////////////////////////////////////////

// Function to linearly interpolate between a0 and a1
// Weight w should be in the range [0.0, 1.0]
double lerp(double a0, double a1, double w)
{
    assert(w >= 0);
    assert(w <= 1);
    // Smooth interpolation using cubic function
    // Using (3.0 - w * 2.0) * w * w for smoother transitions
    return a0 + (a1 - a0) * ((3.0 - w * 2.0) * w * w);
}

// Computes the dot product of the distance and gradient vectors.
double dotGridGradient(int ix, int iy, double x, double y)
{
    // Compute the distance vector from grid point to the point
    const double dx = x - static_cast<double>(ix);
    const double dy = y - static_cast<double>(iy);

    // Compute and return the dot-product with the gradient vector
    return (dx * grid[iy][ix][0] + dy * grid[iy][ix][1]);
}

// Compute Perlin noise at coordinates x, y
double perlin(double x, double y)
{
    // Determine grid cell coordinates
    const int x0 = static_cast<int>(std::floor(x));
    const int x1 = x0 + 1;
    const int y0 = static_cast<int>(std::floor(y));
    const int y1 = y0 + 1;

    // Determine interpolation weights
    // Could also use higher order polynomial/s-curve here
    const double sx = x - static_cast<double>(x0);
    const double sy = y - static_cast<double>(y0);

    // Interpolate between grid point gradients
    const double n0 = dotGridGradient(x0, y0, x, y);
    const double n1 = dotGridGradient(x1, y0, x, y);

    const double ix0 = lerp(n0, n1, sx);

    const double n2 = dotGridGradient(x0, y1, x, y);
    const double n3 = dotGridGradient(x1, y1, x, y);

    const double ix1 = lerp(n2, n3, sx);

    return lerp(ix0, ix1, sy);
}

Vector4d procedural_texture(const double tu, const double tv)
{
    assert(tu >= 0);
    assert(tv >= 0);

    assert(tu <= 1);
    assert(tv <= 1);

    // TODO: uncomment these lines once you implement the perlin noise
    const double color = (perlin(tu * grid_size, tv * grid_size) + 1) / 2;
    return Vector4d(0, color, 0, 0);
}

////////////////////////////////////////////////////////////////////////////////
// Intersection code
////////////////////////////////////////////////////////////////////////////////

// Compute the intersection between a ray and a sphere, return -1 if no intersection
double ray_sphere_intersection(const Vector3d &ray_origin, const Vector3d &ray_direction, int index, Vector3d &p, Vector3d &N)
{
    const Vector3d &sphere_center = sphere_centers[index];
    const double sphere_radius = sphere_radii[index];

    // Compute quadratic equation coefficients
    const double A = ray_direction.dot(ray_direction);
    const Vector3d origin_to_center = ray_origin - sphere_center;
    const double B = 2.0 * ray_direction.dot(origin_to_center);
    const double C = origin_to_center.dot(origin_to_center) - sphere_radius * sphere_radius;

    // Compute discriminant
    const double discriminant = B * B - 4.0 * A * C;

    // No intersection if discriminant is negative
    if (discriminant < 0)
    {
        return -1;
    }

    // Compute intersection distances
    const double sqrt_discriminant = std::sqrt(discriminant);
    const double t1 = (-B - sqrt_discriminant) / (2.0 * A);
    const double t2 = (-B + sqrt_discriminant) / (2.0 * A);

    // Get the nearest positive intersection
    double t = -1;
    if (t1 > 0 && t2 > 0)
    {
        t = std::min(t1, t2);
    }
    else if (t1 > 0)
    {
        t = t1;
    }
    else if (t2 > 0)
    {
        t = t2;
    }

    if (t < 0)
    {
        return -1;
    }

    // Compute intersection point and normal
    p = ray_origin + t * ray_direction;
    N = (p - sphere_center).normalized();

    return t;
}

// Compute the intersection between a ray and a paralleogram, return -1 if no intersection
double ray_parallelogram_intersection(const Vector3d &ray_origin, const Vector3d &ray_direction, int index, Vector3d &p, Vector3d &N)
{
    const Vector3d &pgram_origin = parallelograms[index].col(0);
    const Vector3d &edge1 = parallelograms[index].col(1) - pgram_origin;
    const Vector3d &edge2 = parallelograms[index].col(2) - pgram_origin;

    // Compute the normal of the parallelogram
    N = edge2.cross(edge1).normalized();

    // Check if ray is parallel to the parallelogram
    const double denominator = ray_direction.dot(N);
    if (std::abs(denominator) < 1e-10)
    {
        return -1;
    }

    // Compute intersection point with the plane
    const double t = (pgram_origin - ray_origin).dot(N) / denominator;
    if (t < 0)
    {
        return -1;
    }

    // Compute intersection point
    p = ray_origin + t * ray_direction;

    // Check if point is inside parallelogram using barycentric coordinates
    const Vector3d v = p - pgram_origin;
    const double dot11 = edge1.dot(edge1);
    const double dot12 = edge1.dot(edge2);
    const double dot22 = edge2.dot(edge2);
    const double dotv1 = v.dot(edge1);
    const double dotv2 = v.dot(edge2);

    const double denom = dot11 * dot22 - dot12 * dot12;
    const double u = (dot22 * dotv1 - dot12 * dotv2) / denom;
    const double v_coord = (dot11 * dotv2 - dot12 * dotv1) / denom;

    if (u >= 0 && u <= 1 && v_coord >= 0 && v_coord <= 1)
    {
        return t;
    }

    return -1;
}

//Finds the closest intersecting object returns its index
//In case of intersection it writes into p and N (intersection point and normals)
int find_nearest_object(const Vector3d &ray_origin, const Vector3d &ray_direction, Vector3d &p, Vector3d &N)
{
    // Find the object in the scene that intersects the ray first
    // we store the index and the 'closest_t' to their expected values
    int closest_index = -1;
    double closest_t = std::numeric_limits<double>::max(); // closest t is "+ infinity"

    Vector3d tmp_p, tmp_N;
    for (int i = 0; i < sphere_centers.size(); ++i)
    {
        //returns t and writes on tmp_p and tmp_N
        const double t = ray_sphere_intersection(ray_origin, ray_direction, i, tmp_p, tmp_N);
        //We have intersection
        if (t >= 0)
        {
            //The point is before our current closest t
            if (t < closest_t)
            {
                closest_index = i;
                closest_t = t;
                p = tmp_p;
                N = tmp_N;
            }
        }
    }

    for (int i = 0; i < parallelograms.size(); ++i)
    {
        //returns t and writes on tmp_p and tmp_N
        const double t = ray_parallelogram_intersection(ray_origin, ray_direction, i, tmp_p, tmp_N);
        //We have intersection
        if (t >= 0)
        {
            //The point is before our current closest t
            if (t < closest_t)
            {
                closest_index = sphere_centers.size() + i;
                closest_t = t;
                p = tmp_p;
                N = tmp_N;
            }
        }
    }

    return closest_index;
}

////////////////////////////////////////////////////////////////////////////////
// Raytracer code
////////////////////////////////////////////////////////////////////////////////

// Checks if the light is visible
bool is_light_visible(const Vector3d &ray_origin, const Vector3d &ray_direction, const Vector3d &light_position)
{
    Vector3d p, N;
    const Vector3d light_dir = (light_position - ray_origin).normalized();
    const double light_distance = (light_position - ray_origin).norm();

    // Find any object between the point and the light
    const int nearest_object = find_nearest_object(ray_origin, light_dir, p, N);

    // No object found
    if (nearest_object < 0)
    {
        return true;
    }

    // Check if the intersection point is between the point and the light
    const double intersection_distance = (p - ray_origin).norm();
    return intersection_distance > light_distance - ep;
}

Vector4d shoot_ray(const Vector3d &ray_origin, const Vector3d &ray_direction, int max_bounce)
{
    //Intersection point and normal, these are output of find_nearest_object
    Vector3d p, N;

    const int nearest_object = find_nearest_object(ray_origin, ray_direction, p, N);

    if (nearest_object < 0)
    {
        // Return a transparent color
        return Vector4d(0, 0, 0, 0);
    }

    // Ambient light contribution
    const Vector4d ambient_color = obj_ambient_color.array() * ambient_light.array();

    // Punctual lights contribution (direct lighting)
    Vector4d lights_color(0, 0, 0, 0);
    for (int i = 0; i < light_positions.size(); ++i)
    {
        const Vector3d &light_position = light_positions[i];
        const Vector4d &light_color = light_colors[i];

        const Vector3d Li = (light_position - p).normalized();

        // TODO: Shoot a shadow ray to determine if the light should affect the intersection point and call is_light_visible
        if (!is_light_visible(p, Li, light_position))
        {
            continue;
        }

        Vector4d diff_color = obj_diffuse_color;

        if (nearest_object == 4)
        {
            // Compute UV coodinates for the point on the sphere
            const double x = p(0) - sphere_centers[nearest_object][0];
            const double y = p(1) - sphere_centers[nearest_object][1];
            const double z = p(2) - sphere_centers[nearest_object][2];
            const double tu = acos(z / sphere_radii[nearest_object]) / 3.1415;
            const double tv = (3.1415 + atan2(y, x)) / (2 * 3.1415);

            diff_color = procedural_texture(tu, tv);
        }

        // TODO: Add shading parameters

        // Diffuse contribution
        const Vector4d diffuse = diff_color * std::max(Li.dot(N), 0.0);

        // Specular contribution, use obj_specular_color
        const Vector3d v = (ray_origin - p).normalized();
        const Vector3d h = ((v + Li) / ((v + Li).norm())).normalized();
        const Vector4d specular = obj_specular_color * pow(std::max(h.dot(N), 0.0), obj_specular_exponent);

        // Attenuate lights according to the squared distance to the lights
        const Vector3d D = light_position - p;
        lights_color += (diffuse + specular).cwiseProduct(light_color) / D.squaredNorm();
    }

    Vector4d refl_color = obj_reflection_color;
    if (nearest_object == 4)
    {
        refl_color = Vector4d(0.5, 0.5, 0.5, 0);
    }
    // TODO: Compute the color of the reflected ray and add its contribution to the current point color.
    // use refl_color
    Vector4d reflection_color(0, 0, 0, 0);

    if (max_bounce != 0)
    {
        const Vector3d v = (ray_origin - p).normalized();
        const Vector3d r = ((2 * N * (N.dot(v))) - v).normalized();
        const Vector3d IMFUCKINGSTUPID = p + ep * r;

        reflection_color = (refl_color.cwiseProduct(shoot_ray(IMFUCKINGSTUPID, r, max_bounce - 1)));
    }

    // TODO: Compute the color of the refracted ray and add its contribution to the current point color.
    //       Make sure to check for total internal reflection before shooting a new ray.
    Vector4d refraction_color(0, 0, 0, 0);

    // Rendering equation
    Vector4d C = ambient_color + lights_color + reflection_color + refraction_color;

    // Set alpha to 1
    C(3) = 1;

    return C;
}

////////////////////////////////////////////////////////////////////////////////

void raytrace_scene()
{
    std::cout << "Simple ray tracer." << std::endl;

    const int w = 800;
    const int h = 400;
    MatrixXd R = MatrixXd::Zero(w, h);
    MatrixXd G = MatrixXd::Zero(w, h);
    MatrixXd B = MatrixXd::Zero(w, h);
    MatrixXd A = MatrixXd::Zero(w, h); // Store the alpha mask

    // The camera always points in the direction -z
    // The sensor grid is at a distance 'focal_length' from the camera center,
    // and covers an viewing angle given by 'field_of_view'.
    const double aspect_ratio = static_cast<double>(w) / static_cast<double>(h);
    const double image_y = focal_length * std::tan(field_of_view / 2.0);
    const double image_x = image_y * aspect_ratio;

    // The pixel grid through which we shoot rays is at a distance 'focal_length'
    const Vector3d image_origin(-image_x, image_y, -image_z);
    const Vector3d x_displacement(2.0 * image_x / w, 0, 0);
    const Vector3d y_displacement(0, -2.0 * image_y / h, 0);

    // Compute ray directions for each pixel
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < w; ++i)
    {
        for (int j = 0; j < h; ++j)
        {
            const Vector3d pixel_center = image_origin + (i + 0.5) * x_displacement + (j + 0.5) * y_displacement;

            // Prepare the ray
            Vector3d ray_origin;
            Vector3d ray_direction;

            if (is_perspective)
            {
                ray_origin = camera_position;
                ray_direction = (pixel_center - camera_position).normalized();
            }
            else
            {
                // Orthographic camera
                ray_origin = camera_position + Vector3d(pixel_center[0], pixel_center[1], 0);
                ray_direction = Vector3d(0, 0, -1);
            }

            const Vector4d C = shoot_ray(ray_origin, ray_direction, max_bounce);
            R(i, j) = C(0);
            G(i, j) = C(1);
            B(i, j) = C(2);
            A(i, j) = C(3);
        }
    }

    // Save to png
    write_matrix_to_png(R, G, B, A, filename);
}

////////////////////////////////////////////////////////////////////////////////

int main(int argc, char *argv[])
{
    setup_scene();

    raytrace_scene();
    return 0;
}