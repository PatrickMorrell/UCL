# from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fsolve
import math

radius_earth = 6371000
# scale = 1000000
scale = 1
radius_earth /= scale

earth = ([0, 0, 0], radius_earth)

# Coordinates are Cartesian Earth-centred Earth-fixed (ECEF) in metres
satellite_1 = [10053355.10, 15056487.96, 25504822.96]
satellite_2 = [20655550.99, 5147316.42, 16608524.31]
satellite_3 = [21764223.92, -6594586.14, 15678237.26]
satellite_4 = [10509172.26, -14025870.89, 25122346.95]

satellites = [satellite_1, satellite_2, satellite_3, satellite_4]
colours = ["red", "blue", "green", "orange"]

# Pseudo-ranges
pseudo_ranges_test = [26405306.83, 20820187.04, 21625650.05, 25713099.1]
pseudo_ranges_normal = [26089983.46, 21098423.3, 21918425.56, 25315224.14]
pseudo_ranges_scintillation = [26089978.28, 21098420.42, 21918424.29, 25315222.1]

ax = plt.axes(projection="3d")


def get_spheres(pseudo_ranges):
    satellite_spheres = []
    for i in range(4):
        satellite = satellites[i]
        satellite_params = []
        coordinates = []
        for j in range(3):
            r = satellite[j]
            coordinates.append(round(r / scale, 2))

        satellite_params.append(coordinates)
        satellite_params.append(round(pseudo_ranges[i] / scale, 2))
        satellite_spheres.append(satellite_params)
    return satellite_spheres


def pseudorange_residual(vars, pseudo_ranges):
    spheres = get_spheres(pseudo_ranges)
    x, y, z, b = vars
    eqs = []

    for (sx, sy, sz), rho in spheres:
        eqs.append(np.sqrt((x - sx) ** 2 + (y - sy) ** 2 + (z - sz) ** 2) + b - rho)

    return eqs


def ecef_to_geodetic(x, y, z):
    # Convert ECEF to longitude, latitude and height

    # WGS84 constants
    axis_major = 6378137.0
    eccentricity = 0.081819191

    A = math.sqrt(1 - eccentricity**2)  # temparory variable for simplification
    B = math.sqrt(x**2 + y**2)  # temparory variable

    zeta = math.atan(z / (A * B))
    C = eccentricity**2 * axis_major * (math.sin(zeta)) ** 3  # temparory variable
    D = eccentricity**2 * axis_major * (math.cos(zeta)) ** 3  # temparory variable

    lat = math.atan((A * z + C) / (A * (B - D)))  # latitude in radians
    long = math.atan2(y, x)  # longitude in radians

    E = math.sqrt(1 - eccentricity**2 * (math.sin(lat)) ** 2)  # temparory variable
    height = (B / (math.cos(lat))) - (axis_major / E)  # height in m

    # Convert radians to degrees
    lat_deg = math.degrees(lat)
    long_deg = math.degrees(long)

    return lat_deg, long_deg, height


def find_reciever(pseudo_ranges):
    # Initial guess
    guess = [0, 0, 0, 0]

    solution = fsolve(pseudorange_residual, guess, args=(pseudo_ranges,))

    x, y, z, bias = solution
    # x = 3691798.493765
    # y = -7779.420718
    # z = 5183700.241414
    ax.scatter(x, y, z, color="purple", s=50)

    print(f"Receiver position:")
    print(f"x = {x:.6f}")
    print(f"y = {y:.6f}")
    print(f"z = {z:.6f}")
    print(f"clock bias = {bias:.6f}")

    print(f"Receiver coordinates:")
    lat, long, h = ecef_to_geodetic(x, y, z)
    return lat, long, h


def plot_sphere(x, y, z, r, colour):

    # Create sphere surface
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)

    xs = x + r * np.outer(np.cos(u), np.sin(v))
    ys = y + r * np.outer(np.sin(u), np.sin(v))
    zs = z + r * np.outer(np.ones_like(u), np.cos(v))

    # Plot sphere
    ax.plot_surface(xs, ys, zs, color=colour, alpha=0.4)


def intersection_circle(centre1, radius1, centre2, radius2, n=200):
    # When two spheres intersect, their intersection is a circle. The function returns n points sampled around that circle.

    centre1 = np.array(centre1, dtype=float)
    centre2 = np.array(centre2, dtype=float)

    dvec = centre2 - centre1 # vector pointing from one centre to another
    d = np.linalg.norm(dvec) # distance between centres

    # No intersection
    if d > radius1 + radius2 or d < abs(radius1 - radius2) or d == 0:
        return None

    ex = dvec / d

    a = (radius1**2 - radius2**2 + d**2) / (2 * d) # distance from centre1 to circle center
    radius_intersect = np.sqrt(max(radius1**2 - a**2, 0)) # radius of intersecting circle

    p = centre1 + a * ex

    # Create basis vectors in plane of circle
    temp = np.array([1, 0, 0]) if abs(ex[0]) < 0.9 else np.array([0, 1, 0])
    ey = np.cross(ex, temp)
    ey /= np.linalg.norm(ey)
    ez = np.cross(ex, ey)

    t = np.linspace(0, 2 * np.pi, n)

    circle = p[:, None] + radius_intersect * (np.cos(t) * ey[:, None] + np.sin(t) * ez[:, None])

    return circle


def plot_sphere_earth_intersect(pseudo_ranges):
    spheres = get_spheres(pseudo_ranges)
    for i in range(4):
        c, r = spheres[i]
        circle = intersection_circle(earth[0], earth[1], c, r)

        if circle is not None:
            ax.plot(circle[0], circle[1], circle[2], lw=1, color=colours[i])


def plot_satellites():
    for i in range(4):
        c = satellites[i]
        ax.scatter(c[0], c[1], c[2], color=colours[i], s=50)


print("TEST")
# find_reciever(pseudo_ranges_test)
print("NORMAL")
lat_n, long_n, h_n = find_reciever(pseudo_ranges_normal)
print("SCINTILLATION")
lat_s, long_s, h_s = find_reciever(pseudo_ranges_scintillation)

print("latitude error:", lat_n - lat_s, "degrees")
print("longitude error:", long_n - long_s, "degrees")
print("height error:", h_n - h_s, "m")

plot_sphere(0, 0, 0, radius_earth, "skyblue")
plot_satellites()

# plot_sphere_earth_intersect(pseudo_ranges_test)
plot_sphere_earth_intersect(pseudo_ranges_normal)
# plot_sphere_earth_intersect(pseudo_ranges_scintillation)

ax.set_box_aspect([1, 1, 1])
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")
plt.show()
