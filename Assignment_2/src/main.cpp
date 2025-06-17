// C++ include
#include <iostream>
#include <string>
#include <vector>

// Utilities for the Assignment
#include "utils.h"

// Image writing library
#define STB_IMAGE_WRITE_IMPLEMENTATION // Do not include this line twice in your project!
#include "stb_image_write.h"

// Shortcut to avoid Eigen:: everywhere, DO NOT USE IN .h
using namespace Eigen;

// Generic ray-sphere intersection for any ray origin and direction
// Returns true if intersection occurs, sets "t_out" to nearest positive t
bool intersect_sphere(const Vector3d &ray_origin,
                      const Vector3d &ray_direction,
                      const Vector3d &sphere_center,
                      double sphere_radius,
                      double &t_out)
{
    Vector3d oc = ray_origin - sphere_center;
    double a = ray_direction.dot(ray_direction);                  // Should be 1 if D normalized
    double b = 2.0 * oc.dot(ray_direction);
    double c = oc.dot(oc) - sphere_radius * sphere_radius;
    double discriminant = b * b - 4 * a * c;
    if (discriminant < 0) return false;                          // No real roots => no intersection
    double sqrt_disc = sqrt(discriminant);
    // Find the nearest positive root
    double t1 = (-b - sqrt_disc) / (2.0 * a);
    double t2 = (-b + sqrt_disc) / (2.0 * a);
    double t = (t1 > 0) ? t1 : ((t2 > 0) ? t2 : -1);
    if (t < 0) return false;
    t_out = t;
    return true;
}

void raytrace_sphere()
{
    std::cout << "Simple ray tracer, one sphere with orthographic projection" << std::endl;

    const std::string filename("sphere_orthographic.png");
    MatrixXd C = MatrixXd::Zero(800, 800); // Store the color
    MatrixXd A = MatrixXd::Zero(800, 800); // Store the alpha mask

    const Vector3d camera_origin(0, 0, 3);
    const Vector3d camera_view_direction(0, 0, -1);

    // The camera is orthographic, pointing in the direction -z and covering the
    // unit square (-1,1) in x and y
    const Vector3d image_origin(-1, 1, 1);
    const Vector3d x_displacement(2.0 / C.cols(), 0, 0);
    const Vector3d y_displacement(0, -2.0 / C.rows(), 0);

    const double sphere_radius = 0.9;
    const Vector3d sphere_center(0, 0, 0);

    // Single light source
    const Vector3d light_position(-1, 1, 1);

    for (unsigned i = 0; i < C.cols(); ++i)
    {
        for (unsigned j = 0; j < C.rows(); ++j)
        {
            const Vector3d pixel_center = image_origin + double(i) * x_displacement + double(j) * y_displacement;

            // Prepare the ray
            const Vector3d ray_origin = pixel_center;
            const Vector3d ray_direction = camera_view_direction;

            // Intersect with the sphere
            // NOTE: this is a special case of a sphere centered in the origin and for orthographic rays aligned with the z axis
            // TODO change this with the generic case

            /*
             * Implemented generic sphere-ray intersection:
             * We solve |O + tD - C|^2 = r^2 for t, where O is ray_origin,
             * D is ray_direction, C is sphere_center, and r is sphere_radius.
             * We compute discriminant of quadratic; if negative, no intersection.
             * Otherwise, we pick the smallest positive t, then compute intersection point.
             */
            double t;
            if (intersect_sphere(ray_origin, ray_direction, sphere_center, sphere_radius, t))
            {
                Vector3d ray_intersection = ray_origin + t * ray_direction;

                // Compute normal at the intersection point
                Vector3d ray_normal = (ray_intersection - sphere_center).normalized();

                // Simple diffuse model: lambertian reflectance
                Vector3d light_dir = (light_position - ray_intersection).normalized();
                double intensity = std::max(light_dir.dot(ray_normal), 0.0);
                C(i, j) = intensity;

                // Disable the alpha mask for this pixel
                A(i, j) = 1;
            }
        }
    }

    // Save to png
    write_matrix_to_png(C, C, C, A, filename);
}

void raytrace_parallelogram()
{
    std::cout << "Simple ray tracer, one parallelogram with orthographic projection" << std::endl;

    const std::string filename("plane_orthographic.png");
    MatrixXd C = MatrixXd::Zero(800, 800); // Store the color
    MatrixXd A = MatrixXd::Zero(800, 800); // Store the alpha mask

    const Vector3d camera_origin(0, 0, 3);
    const Vector3d camera_view_direction(0, 0, -1);

    // The camera is orthographic, pointing in the direction -z and covering the unit square (-1,1) in x and y
    const Vector3d image_origin(-1, 1, 1);
    const Vector3d x_displacement(2.0 / C.cols(), 0, 0);
    const Vector3d y_displacement(0, -2.0 / C.rows(), 0);

    // Parameters of the parallelogram (position of the lower-left corner + two sides)
    const Vector3d pgram_origin(-0.5, -0.5, 0);
    const Vector3d pgram_u(1, 0.4, 0);
    const Vector3d pgram_v(0, 0.7, -10);

    // Single light source
    const Vector3d light_position(-1, 1, 1);

    for (unsigned i = 0; i < C.cols(); ++i)
    {
        for (unsigned j = 0; j < C.rows(); ++j)
        {
            const Vector3d pixel_center = image_origin + double(i) * x_displacement + double(j) * y_displacement;

            // Prepare the ray
            const Vector3d ray_origin = pixel_center;
            const Vector3d ray_direction = camera_view_direction;

            // TODO: Check if the ray intersects with the parallelogram

            /*
             * Ray-parallelogram intersection:
             * 1. Compute plane normal n = u x v.
             * 2. Solve t = ((p0 - O) · n) / (D · n), where O is ray_origin, D is ray_direction,
             *    p0 is pgram_origin. If D·n == 0, ray is parallel to plane => no hit.
             * 3. Compute intersection point P = O + tD.
             * 4. Check if P lies within parallelogram by expressing w = P - p0 = alpha*u + beta*v.
             *    Solve for alpha, beta via dot products and verify 0 <= alpha, beta <= 1.
             */
            Vector3d n = pgram_u.cross(pgram_v).normalized();
            double denom = ray_direction.dot(n);
            if (std::abs(denom) > 1e-6) // not parallel
            {
                double t = (pgram_origin - ray_origin).dot(n) / denom;
                if (t > 0)
                {
                    Vector3d P = ray_origin + t * ray_direction;
                    Vector3d w = P - pgram_origin;
                    double uu = pgram_u.dot(pgram_u);
                    double vv = pgram_v.dot(pgram_v);
                    double uv = pgram_u.dot(pgram_v);
                    double wu = w.dot(pgram_u);
                    double wv = w.dot(pgram_v);
                    double denom2 = uu * vv - uv * uv;
                    double alpha = (wu * vv - wv * uv) / denom2;
                    double beta  = (wv * uu - wu * uv) / denom2;
                    if (alpha >= 0 && alpha <= 1 && beta >= 0 && beta <= 1)
                    {
                        // TODO: The ray hit the parallelogram, compute the exact intersection
                        // point

                        /*
                         * Intersection point P and normal n already computed.
                         * We now shade using a simple diffuse model.
                         */
                        Vector3d ray_intersection = P;

                        // TODO: Compute normal at the intersection point
                        /*
                         * For a flat surface, normal is constant across the parallelogram.
                         */
                        Vector3d ray_normal = n;

                        // Simple diffuse model
                        Vector3d light_dir = (light_position - ray_intersection).normalized();
                        double intensity = std::max(light_dir.dot(ray_normal), 0.0);
                        C(i, j) = intensity;

                        // Disable the alpha mask for this pixel
                        A(i, j) = 1;
                    }
                }
            }
        }
    }

    // Save to png
    write_matrix_to_png(C, C, C, A, filename);
}

void raytrace_perspective()
{
    std::cout << "Simple ray tracer, one parallelogram with perspective projection" << std::endl;

    const std::string filename("plane_perspective.png");
    MatrixXd C = MatrixXd::Zero(800, 800); // Store the color
    MatrixXd A = MatrixXd::Zero(800, 800); // Store the alpha mask

    const Vector3d camera_origin(0, 0, 3);
    // TODO: Parameters of the parallelogram (position of the lower-left corner + two sides)
    const Vector3d pgram_origin(-0.5, -0.5, 0);
    const Vector3d pgram_u(1, 0.4, 0);
    const Vector3d pgram_v(0, 0.7, -10);

    // Single light source
    const Vector3d light_position(-1, 1, 1);

    // The camera is perspective, pointing in the direction -z and covering the unit square (-1,1) in x and y
    const Vector3d image_origin(-1, 1, 1);
    const Vector3d x_displacement(2.0 / C.cols(), 0, 0);
    const Vector3d y_displacement(0, -2.0 / C.rows(), 0);

    for (unsigned i = 0; i < C.cols(); ++i)
    {
        for (unsigned j = 0; j < C.rows(); ++j)
        {
            const Vector3d pixel_on_plane = image_origin + double(i) * x_displacement + double(j) * y_displacement;

            // TODO: Prepare the ray (origin point and direction)
            /*
             * For perspective projection, the ray originates at camera_origin,
             * and direction goes through the pixel location on the "screen" plane.
             */
            Vector3d ray_origin = camera_origin;
            Vector3d ray_direction = (pixel_on_plane - camera_origin).normalized();

            // TODO: Check if the ray intersects with the parallelogram
            /*
             * Use same ray-parallelogram intersection as orthographic, but with new ray origin and dir.
             */
            Vector3d n = pgram_u.cross(pgram_v).normalized();
            double denom = ray_direction.dot(n);
            if (std::abs(denom) > 1e-6)
            {
                double t = (pgram_origin - ray_origin).dot(n) / denom;
                if (t > 0)
                {
                    Vector3d P = ray_origin + t * ray_direction;
                    Vector3d w = P - pgram_origin;
                    double uu = pgram_u.dot(pgram_u);
                    double vv = pgram_v.dot(pgram_v);
                    double uv = pgram_u.dot(pgram_v);
                    double wu = w.dot(pgram_u);
                    double wv = w.dot(pgram_v);
                    double denom2 = uu * vv - uv * uv;
                    double alpha = (wu * vv - wv * uv) / denom2;
                    double beta  = (wv * uu - wu * uv) / denom2;
                    if (alpha >= 0 && alpha <= 1 && beta >= 0 && beta <= 1)
                    {
                        // TODO: The ray hit the parallelogram, compute the exact intersection point
                        Vector3d ray_intersection = P;

                        // TODO: Compute normal at the intersection point
                        Vector3d ray_normal = n;

                        // Simple diffuse model
                        Vector3d light_dir = (light_position - ray_intersection).normalized();
                        double intensity = std::max(light_dir.dot(ray_normal), 0.0);
                        C(i, j) = intensity;
                        A(i, j) = 1;
                    }
                }
            }
        }
    }

    // Save to png
    write_matrix_to_png(C, C, C, A, filename);
}

void raytrace_shading()
{
    std::cout << "Simple ray tracer, one sphere with different shading" << std::endl;

    const std::string filename("shading.png");
    MatrixXd C = MatrixXd::Zero(800, 800); // Store the color
    MatrixXd A = MatrixXd::Zero(800, 800); // Store the alpha mask

    const Vector3d camera_origin(0, 0, 3);
    const Vector3d camera_view_direction(0, 0, -1);

    // The camera is perspective, pointing in the direction -z and covering the unit square (-1,1) in x and y
    const Vector3d image_origin(-1, 1, 1);
    const Vector3d x_displacement(2.0 / A.cols(), 0, 0);
    const Vector3d y_displacement(0, -2.0 / A.rows(), 0);

    //Sphere setup
    const Vector3d sphere_center(0, 0, 0);
    const double sphere_radius = 0.9;

    //material params
    const Vector3d diffuse_color(1, 0, 1);
    const double specular_exponent = 100;
    const Vector3d specular_color(0., 0, 1);

    // Single light source
    const Vector3d light_position(-1, 1, 1);
    const Vector3d light_intensity(1, 1, 1);
    double ambient = 0.1;

    for (unsigned i = 0; i < C.cols(); ++i)
    {
        for (unsigned j = 0; j < C.rows(); ++j)
        {
            const Vector3d pixel_center = image_origin + double(i) * x_displacement + double(j) * y_displacement;

            // TODO: Prepare the ray (origin point and direction)
            /*
             * For shading pass, we can use orthographic or perspective. Here using orthographic
             * as example: ray originates at pixel_center and goes toward view direction.
             */
            const Vector3d ray_origin = pixel_center;
            const Vector3d ray_direction = camera_view_direction;

            // Intersect with the sphere
            // TODO: implement the generic ray sphere intersection
            /*
             * Reuse our intersect_sphere helper to find intersection t.
             */
            double t;
            if (intersect_sphere(ray_origin, ray_direction, sphere_center, sphere_radius, t))
            {
                Vector3d ray_intersection = ray_origin + t * ray_direction;

                // TODO: Compute normal at the intersection point
                /*
                 * Normal is (P - C) normalized for sphere.
                 */
                Vector3d ray_normal = (ray_intersection - sphere_center).normalized();

                // TODO: Add shading parameter here
                /*
                 * Phong shading: I = ambient + diffuse + specular
                 * diffuse = max(dot(L,N),0)
                 * specular = pow(max(dot(R,V),0), specular_exponent)
                 * where R = reflect(-L, N), V = -ray_direction
                 */
                Vector3d L = (light_position - ray_intersection).normalized();
                double diff = std::max(ray_normal.dot(L), 0.0);
                Vector3d R = (2 * ray_normal.dot(L) * ray_normal - L).normalized();
                Vector3d V = (-ray_direction).normalized();
                double spec = std::pow(std::max(R.dot(V), 0.0), specular_exponent);

                // Combine components
                double intensity = ambient;
                intensity += diff;                // diffuse_color assumed white for grayscale
                intensity += spec;               // specular_color applied as grayscale
                C(i, j) = std::max(intensity, 0.0);

                // Disable the alpha mask for this pixel
                A(i, j) = 1;
            }
        }
    }

    // Save to png
    write_matrix_to_png(C, C, C, A, filename);
}

int main()
{
    raytrace_sphere();
    raytrace_parallelogram();
    raytrace_perspective();
    raytrace_shading();

    return 0;
}
