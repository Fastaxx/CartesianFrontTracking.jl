using CartesianFrontTracking

# Pour une grille 1D
x = collect(-5:0.5:5)
mesh1D = Mesh([x])

# Pour une grille 2D
y = collect(-5:0.5:5)
mesh2D = Mesh([x, y])

# Pour une grille 3D
z = collect(-5:0.5:5)
mesh3D = Mesh([x, y, z])

# Générer les points du maillage
points1D = generate_mesh(mesh1D)
points2D = generate_mesh(mesh2D)
points3D = generate_mesh(mesh3D)