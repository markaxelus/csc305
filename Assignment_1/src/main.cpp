////////////////////////////////////////////////////////////////////////////////
#include <algorithm>
#include <complex>
#include <fstream>
#include <iostream>
#include <numeric>
#include <vector>

#include <Eigen/Dense>
// Shortcut to avoid  everywhere, DO NOT USE IN .h
using namespace Eigen;
////////////////////////////////////////////////////////////////////////////////

const std::string root_path = DATA_DIR;

// Computes the determinant of the matrix whose columns are the vector u and v
double inline det(const Vector2d &u, const Vector2d &v)
{
    // TODO
    /* det = (ux)(vy) - (uy)(vx) */
    return u.x() * v.y() - u.y() * v.x();
}

/* Compute orientation */
inline double orient(const Vector2d &a, const Vector2d &b, const Vector2d &c) {
    /* If:
        1. det > 0; direction = counter-clockwise (left-side of AB)
        2. det < 0; directon = clockwise (right-side of AB)
        3. det = 0; direction = collinear (on the same line as AB)
    */
    return det(b - a, c - a);
}

/* Checks if a point q lies between p and r */
bool on_segment(const Vector2d &p, const Vector2d &q, const Vector2d &r) {
    /* Must check:
        1. p.x <= q.x <= r.x
        2. p.y <= q.y <= r.y
        NOTE : p is not guaranteed to be left endpoint,
               r is not guaranteed to be right endpoint.
    */
    return std::min(p.x(), r.x()) <= q.x() && q.x() <= std::max(p.x(), r.x()) &&
           std::min(p.y(), r.y()) <= q.y() && q.y() <= std::max(p.y(), r.y()) 
}

// Return true iff [a,b] intersects [c,d]
bool intersect_segment(const Vector2d &a, const Vector2d &b, const Vector2d &c, const Vector2d &d)
{
    /* Given segment AB, CD, they intersect iff
        1.  CD lies on opposite sides of AB
        2.  AB lies on opposite sides of CD
        i.e(
            orient(a,b,c) * orient(a,b,d) < 0 &&
            orient(c,d,a) * orient(c,d,b) < 0
        )

        Edge Case:
        If any orientation is 0 (collinear)
        check if any points overlaps on the other segment which guarantees intersection
            - C/D lies on AB
            - A/B lies on CD
        
    */
    // TODO

    double o1 = orient(a,b,c);
    double o2 = orient(a,b,d);
    double o3 = orient(c,d,a);
    double o4 = orient(c,d,b);

    if (o1*o2 < 0 && o3*o4 < 0) {
        return true;
    }

    if (o1 == 0 && on_segment(a,c,b)) return true;
    if (o2 == 0 && on_segment(a,d,b)) return true;
    if (o3 == 0 && on_segment(c,a,d)) return true;
    if (o4 == 0 && on_segment(c,b,d)) return true;
    
    return false;
}

////////////////////////////////////////////////////////////////////////////////

bool is_inside(const std::vector<Vector2d> &poly, const Vector2d &query)
{
    // 1. Compute bounding box and set coordinate of a point outside the polygon
    // TODO
    /* Init: x=0, y=0  
        
    */
    // Vector2d outside(0, 0);
    double minX = poly[0].x(), maxX = poly[0].x();
    double minY = poly[0].y(), maxY = poly[0].y();

    for (auto &point : poly) {
        minX = std::min(minX, point.x());         
        maxX = std::max(maxX, point.x());   
        minY = std::min(minY, point.y());   
        maxY = std::max(maxY, point.y());           
    }
    /* We do not want the ray to be trapped within the boundaries
        Choose a point that is guaranteed to be outside of the polygon
    */
    Vector2d outside(maxX + 1.0, maxY + 1.0);
    // 2. Cast a ray from the query point to the 'outside' point, count number of intersections
    /* Even-Odd Rule
        1. Odd number of crossings -> point inside of polygon
        2. Even number of crossings -> point outside of polygon
    */
    int count = 0;
    for (size_t i = 0; i < poly.size(); i++) {
        Vector2d A = poly[i];
        Vector2d B = poly[(i+1) % poly.size()]; // Wraps around back to 0 when it reaches the end
        /* Checks if A/B crosses the ray query -> outside 
            When checking orient we want 
            orient(query, outside, A/B);
            to check if it crosses outside which is guaranteed to NOT be in the polygon
        */
        if (intersect_segment(query, outside, A, B)) {
            count++;
        }
    }

    return (count % 2 == 1);
}

////////////////////////////////////////////////////////////////////////////////

std::vector<Vector2d> load_xyz(const std::string &filename)
{
    std::vector<Vector2d> points;
    std::ifstream in(filename);
    // TODO
    return points;
}

void save_xyz(const std::string &filename, const std::vector<Vector2d> &points)
{
    // TODO
}

std::vector<Vector2d> load_obj(const std::string &filename)
{
    std::ifstream in(filename);
    std::vector<Vector2d> points;
    std::vector<Vector2d> poly;
    char key;
    while (in >> key)
    {
        if (key == 'v')
        {
            double x, y, z;
            in >> x >> y >> z;
            points.push_back(Vector2d(x, y));
        }
        else if (key == 'f')
        {
            std::string line;
            std::getline(in, line);
            std::istringstream ss(line);
            int id;
            while (ss >> id)
            {
                poly.push_back(points[id - 1]);
            }
        }
    }
    return poly;
}

////////////////////////////////////////////////////////////////////////////////

int main(int argc, char *argv[])
{
    const std::string points_path = root_path + "/points.xyz";
    const std::string poly_path = root_path + "/polygon.obj";

    std::vector<Vector2d> points = load_xyz(points_path);

    ////////////////////////////////////////////////////////////////////////////////
    //Point in polygon
    std::vector<Vector2d> poly = load_obj(poly_path);
    std::vector<Vector2d> result;
    for (size_t i = 0; i < points.size(); ++i)
    {
        if (is_inside(poly, points[i]))
        {
            result.push_back(points[i]);
        }
    }
    save_xyz("output.xyz", result);

    return 0;
}
