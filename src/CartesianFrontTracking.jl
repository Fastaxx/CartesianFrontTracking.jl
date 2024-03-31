module CartesianFrontTracking

using LinearAlgebra
using ColorSchemes
using Roots
using Plots
Plots.default(show = true)

# Write your package code here.

struct Mesh
    grids::Array{Vector{Float64}, 1}
end

# Pour une grille 1D
x = collect(-5:0.5:5)
mesh1D = Mesh([x])

# Pour une grille 2D
y = collect(-5:0.5:5)
mesh2D = Mesh([x, y])

# Pour une grille 3D
z = collect(-5:0.5:5)
mesh3D = Mesh([x, y, z])

# Extraire les coordonnées x, y et z du maillage
x_coords = mesh3D.grids[1]
y_coords = mesh3D.grids[2]
z_coords = mesh3D.grids[3]

# Générer tous les points du maillage
points = []
for x in x_coords
    for y in y_coords
        for z in z_coords
            push!(points, (x, y, z))
        end
    end
end

"""
# Extraire les coordonnées x, y et z de tous les points
x = [p[1] for p in points]
y = [p[2] for p in points]
z = [p[3] for p in points]

# Créer un nuage de points
scatter(x, y, z, xlabel="x", ylabel="y", zlabel="z", title="3D Mesh")
#readline()
"""

# Définir la fonction de distance signée 
function signed_distance(point::Vector{Float64}, r::Float64)
    x, y, z = point
    R = r
    a = 1.0
    return x^2 + y^2 + z^2 - r^2
end

# Créer une sphère de rayon r
r = 2.0
sphere = point -> signed_distance(point, r)

# Calculer les valeurs de la sdf aux sommets du maillage
phi = [sphere([x[i], y[i], z[i]]) for i in 1:length(x)]

"""
# Convertir les valeurs de la sdf en couleurs
colors = [get(ColorSchemes.rainbow, p) for p in phi]

# Créer un nuage de points où la couleur de chaque point est déterminée par la valeur de la sdf
p = scatter(x, y, z, color=colors, xlabel="x", ylabel="y", zlabel="z", title="Signed Distance Function")
surface!(p, [0,1], [0,1], (x,y)->0, opacity=0, color=:rainbow, colorbar_title="SDF Value")
readline()
"""

@show mesh3D

# Obtenir les arêtes du maillage
function obtenir_aretes(maillage)
    x, y, z = maillage
    aretes = []
    for i in 1:length(x)-1
        for j in 1:length(y)-1
            for k in 1:length(z)-1
                push!(aretes, [[x[i], y[j], z[k]], [x[i+1], y[j], z[k]]])
                push!(aretes, [[x[i], y[j], z[k]], [x[i], y[j+1], z[k]]])
                push!(aretes, [[x[i], y[j], z[k]], [x[i], y[j], z[k+1]]])
            end
        end
    end
    return aretes
end

# Obtenir toutes les arêtes du maillage
aretes = obtenir_aretes(mesh3D.grids)

# Créer une liste pour stocker les points P où Phi=0
points_P = []

# Calculer le point P où Phi=0 pour chaque arête
for arete in aretes
    # Paramétriser l'arête
    f(t) = sphere(arete[1] + t * (arete[2] - arete[1]))
    try
        # Utiliser la méthode de la bissection pour trouver le point P
        t0 = find_zero(f, (0.0, 1.0), Bisection())
        x0 = arete[1] + t0 * (arete[2] - arete[1])
        #println("Le point P où Phi=0 sur l'arête entre les points $(arete[1]) et $(arete[2]) est $x0")
        
        # Ajouter le point P à la liste des points P
        push!(points_P, x0)
    catch e
        #println("Aucun point P trouvé sur l'arête entre les points $(arete[1]) et $(arete[2])")
    end
end

# Créer une figure 3D
p = plot(legend = false, xlims = (-5, 5), ylims = (-5, 5), zlims = (-5, 5))

# Ajouter chaque point P à la figure
for point in points_P
    scatter!(p, [point[1]], [point[2]], [point[3]], color = :red, markersize = 4)
end

# Afficher la figure
display(p)
readline()

#Etat Initial : Les zéros ont été repérés dans l'espace 3D : (xp,yp,zp). 
#Les zéros sont les points où la fonction Phi s'annule.

# Calcul de la longueur d'arc
function arc_length(point1, point2)
    return norm(point2 - point1)
end

# Calculer la longueur d'arc entre chaque paire de points P
arc_lengths = [arc_length(points_P[i], points_P[i+1]) for i in 1:length(points_P)-1]

# Calcul de la normale à la surface
using ForwardDiff

# Définir la fonction Phi
function Phi(point)
    # Remplacer par la définition réelle de votre fonction
    x, y, z = point
    return x^2 + y^2 + z^2 - r^2
end

# Calculer le gradient de Phi en un point
function gradient_Phi(point)
    return ForwardDiff.gradient(Phi, point)
end

# Calculer le vecteur normal en un point
function normal_vector(point)
    # Calculer le gradient de Phi au point
    gradient = gradient_Phi(point)
    
    # Le vecteur normal est le gradient normalisé
    normal = gradient / norm(gradient)
    
    return normal
end

# Calculer le vecteur normal aux points P
normals = [normal_vector(p) for p in points_P]

"""
# Créer une nouvelle figure
plot()

# Ajouter la sphère à la figure
# Définir les coordonnées de la sphère
theta = 0:0.01:2*pi
phi = 0:0.01:pi
x = r * cos.(theta) * sin.(phi)'
y = r * sin.(theta) * sin.(phi)'
z = r * ones(length(theta)) * cos.(phi)'

# Ajouter la sphère à la figure
surface(x, y, z, alpha=0.5, color=:blue)

# Ajouter les vecteurs normaux à la figure
for (point, normal) in zip(points_P, normals)
    point = collect(point)
    normal = collect(normal)
    quiver!(point[1:1], point[2:2], point[3:3], quiver=(normal[1:1], normal[2:2], normal[3:3]), color=:red)
end

# Afficher la figure
display(plot)
readline()
"""

# Calculer le vecteur tangentiel à la surface
function tangent_vector(normal)
    # Générer un vecteur aléatoire
    random_vector = rand(3)
    
    # Calculer le vecteur tangentiel
    tangent = cross(normal, random_vector)
    
    return tangent
end

# Calculer le vecteur tangentiel aux points P
tangents = [tangent_vector(n) for n in normals]

"""
# Créer une nouvelle figure
plot()

# Ajouter la sphère à la figure
# Définir les coordonnées de la sphère
theta = 0:0.01:2*pi
phi = 0:0.01:pi
x = r * cos.(theta) * sin.(phi)'
y = r * sin.(theta) * sin.(phi)'
z = r * ones(length(theta)) * cos.(phi)'
surface(x, y, z, alpha=0.5, color=:blue)

# Ajouter les vecteurs normaux et tangentiels à la figure
for (point, normal, tangent) in zip(points_P, normals, tangents)
    point = collect(point)
    normal = collect(normal)
    tangent = collect(tangent)
    quiver!(point[1:1], point[2:2], point[3:3], quiver=(normal[1:1], normal[2:2], normal[3:3]), color=:red)
    quiver!(point[1:1], point[2:2], point[3:3], quiver=(tangent[1:1], tangent[2:2], tangent[3:3]), color=:green)
end

# Afficher la figure
display(plot)
readline()
"""

# Calcul de la courbure
# https://u.cs.biu.ac.il/~katzmik/goldman05.pdf
function curvature(point)
    # Calculer le gradient de Phi au point
    gradient = ForwardDiff.gradient(Phi, point)
    
    # Calculer la hessienne de Phi au point
    hessian = ForwardDiff.hessian(Phi, point)
    adjoint_hessian = det(hessian) * inv(hessian)
    
    # Calculer la courbure gaussienne
    gaussian_curvature = gradient' * adjoint_hessian * gradient / norm(gradient)^4

    # Calculer la courbure moyenne
    mean_curvature = (gradient' * hessian * gradient - norm(gradient)^2 * tr(hessian))/(2*norm(gradient)^3)
    
    return gaussian_curvature, mean_curvature
end

# Calculer la courbure aux points P
gaussian_curvature = [curvature(p)[1] for p in points_P]
mean_curvature = [curvature(p)[2] for p in points_P]

# Faces 
# Calculer les faces du maillage
function cell_faces(maillage)
    x, y, z = maillage
    faces = []
    for i in 1:length(x)-1
        for j in 1:length(y)-1
            for k in 1:length(z)-1
                # Les faces de la cellule sont définies par les points de grille (x[i], y[j], z[k]) et (x[i+1], y[j+1], z[k+1])
                # Il y a six faces pour chaque cellule : deux dans chaque direction
                push!(faces, [(x[i], y[j], z[k]), (x[i+1], y[j], z[k])]) # Faces en x
                push!(faces, [(x[i], y[j], z[k]), (x[i], y[j+1], z[k])]) # Faces en y
                push!(faces, [(x[i], y[j], z[k]), (x[i], y[j], z[k+1])]) # Faces en z
                push!(faces, [(x[i+1], y[j+1], z[k+1]), (x[i], y[j+1], z[k+1])]) # Faces en x
                push!(faces, [(x[i+1], y[j+1], z[k+1]), (x[i+1], y[j], z[k+1])]) # Faces en y
                push!(faces, [(x[i+1], y[j+1], z[k+1]), (x[i+1], y[j+1], z[k])]) # Faces en z
            end
        end
    end
    return faces
end

# Repérer les faces de chaque cellule
faces = cell_faces(mesh3D.grids)

using Statistics

# Calculer le centre d'une face
function face_center(face)
    x = mean([f[1] for f in face])
    y = mean([f[2] for f in face])
    z = mean([f[3] for f in face])
    return [x, y, z]
end

# Calculer les centres des faces
face_centers = [face_center(f) for f in faces]

# Initialiser un champ de vitesse face-centered
function velocity_field(point, t)
    # Définir un champ de vitesse variable dans l'espace et le temps
    return [sin(point[1] + t), cos(point[2]), sin(point[1])*cos(point[2])]
end

# Obtenir les coordonnées x, y, z et les composantes u, v, w du champ de vitesse
x = [center[1] for center in face_centers]
y = [center[2] for center in face_centers]
z = [center[3] for center in face_centers]
u = [velocity_field(center, 0.0)[1] for center in face_centers]
v = [velocity_field(center, 0.0)[2] for center in face_centers]
w = [velocity_field(center, 0.0)[3] for center in face_centers]

"""
# Créer un graphique de vecteurs
quiver(x, y, z, quiver=(u, v, w))

# Afficher le graphique
display(plot)
readline()
"""

# Interpolation bilinéaire du champ de vitesse face centered aux points P
function bilinear_interpolation(point, face_centers, velocity_field, t)
    # Trouver les faces les plus proches du point
    distances = [norm(point - center) for center in face_centers]
    closest_faces = sortperm(distances)[1:4]
    
    # Calculer les coordonnées barycentriques
    x, y, z = point
    x1, y1, z1 = face_centers[closest_faces[1]]
    x2, y2, z2 = face_centers[closest_faces[2]]
    x3, y3, z3 = face_centers[closest_faces[3]]
    x4, y4, z4 = face_centers[closest_faces[4]]
    
    # Calculer les aires des triangles formés par le point et les centres des faces
    A1 = 0.5 * norm(cross([x - x1, y - y1, z - z1], [x - x2, y - y2, z - z2]))
    A2 = 0.5 * norm(cross([x - x2, y - y2, z - z2], [x - x3, y - y3, z - z3]))
    A3 = 0.5 * norm(cross([x - x3, y - y3, z - z3], [x - x4, y - y4, z - z4]))
    A4 = 0.5 * norm(cross([x - x4, y - y4, z - z4], [x - x1, y - y1, z - z1]))
    
    # Calculer les coordonnées barycentriques
    b1 = A1 / (A1 + A2 + A3 + A4)
    b2 = A2 / (A1 + A2 + A3 + A4)
    b3 = A3 / (A1 + A2 + A3 + A4)
    b4 = A4 / (A1 + A2 + A3 + A4)
    
    # Interpoler le champ de vitesse
    u1, v1, w1 = velocity_field(face_centers[closest_faces[1]], t)
    u2, v2, w2 = velocity_field(face_centers[closest_faces[2]], t)
    u3, v3, w3 = velocity_field(face_centers[closest_faces[3]], t)
    u4, v4, w4 = velocity_field(face_centers[closest_faces[4]], t)

    u = b1 * u1 + b2 * u2 + b3 * u3 + b4 * u4
    v = b1 * v1 + b2 * v2 + b3 * v3 + b4 * v4
    w = b1 * w1 + b2 * w2 + b3 * w3 + b4 * w4

    return [u, v, w]
end

# Interpoler le champ de vitesse aux points P
velocities = [bilinear_interpolation(p, face_centers, velocity_field, 0.0) for p in points_P]

# Extraire les composantes u, v, w des vitesses et les coordonnées x, y, z des points
u = [vel[1] for vel in velocities]
v = [vel[2] for vel in velocities]
w = [vel[3] for vel in velocities]
x = [point[1] for point in points_P]
y = [point[2] for point in points_P]
z = [point[3] for point in points_P]

"""
# Créer un diagramme de quiver (flèches) pour le champ de vitesse en 3D
quiver(x, y, z, quiver=(u, v, w))

# Afficher le diagramme
display(plot)
readline()
"""

## Déplacer les points P le long du champ de vitesse
# Schéma d'intégration d'Euler
function euler_move(point, velocity, dt)
    # Déplacer le point le long du champ de vitesse
    new_point = point + velocity * dt
    
    return new_point
end

# Schéma d'intégration de Runge-Kutta
function runge_kutta_move(point, velocity, dt)
    # Calculer les quatre "étapes" de Runge-Kutta
    k1 = dt * velocity
    k2 = dt * (velocity + k1 / 2)
    k3 = dt * (velocity + k2 / 2)
    k4 = dt * (velocity + k3)

    # Mettre à jour le point en utilisant une combinaison pondérée des quatre étapes
    new_point = point + (k1 + 2*k2 + 2*k3 + k4) / 6

    return new_point
end

# Définir les paramètres de la simulation
dt = 1.0
t0 = 0.0

# Boucle sur les pas de temps
for step in 1:100
    t0 = 0.0 + step * dt
    # Mettre à jour les vitesses à chaque pas de temps
    velocities = [bilinear_interpolation(p, face_centers, velocity_field, t0) for p in points_P]
    # Déplacer chaque point
    for i in eachindex(points_P)
        points_P[i] = runge_kutta_move(points_P[i], velocities[i], dt)
    end
    
    # Créer une figure 3D pour cette étape de temps
    p = plot(legend = false, xlims = (-5, 5), ylims = (-5, 5), zlims = (-5, 5))
    for point in points_P
        scatter!(p, [point[1]], [point[2]], [point[3]], color = :red, markersize = 4)
    end
    display(p)
end

# Surveiller les propriétés du front
function monitor_front_properties(points_prev, points_curr)
    # Initialiser les différences de propriétés du front
    normal_diffs = []
    curvature_diffs = []
    arc_length_diffs = []
    
    # Calculer les différences de propriétés du front pour chaque point
    for i in 1:length(points_curr)
        point_prev = points_prev[i]
        point_curr = points_curr[i]
        
        normal_prev = normal_vector(point_prev)
        normal_curr = normal_vector(point_curr)
        normal_diffs = append!(normal_diffs, norm(normal_curr - normal_prev))
        
        curvature_prev = curvature(point_prev)
        curvature_curr = curvature(point_curr)
        curvature_diffs = append!(curvature_diffs, abs(curvature_curr - curvature_prev))
        
        if i > 1
            arc_length_prev = arc_length(points_prev[i-1], point_prev)
            arc_length_curr = arc_length(points_curr[i-1], point_curr)
            arc_length_diffs = append!(arc_length_diffs, abs(arc_length_curr - arc_length_prev))
        end
    end
    
    # Retourner les différences de propriétés du front
    return normal_diffs, curvature_diffs, arc_length_diffs
end

# Seuils pour la rééchantillonnage
const THRESHOLD_NORMAL_DIFF = 0.1
const THRESHOLD_CURVATURE_DIFF = 0.1
const THRESHOLD_ARC_LENGTH_DIFF = 0.1

# Vérifier si le rééchantillonnage est nécessaire
function check_resampling_needed(points_prev, points_curr)
    # Surveiller les propriétés du front
    normal_diffs, curvature_diffs, arc_length_diffs = monitor_front_properties(points_prev, points_curr)
    
    # Vérifier si les différences maximales dépassent les seuils
    if maximum(normal_diffs) > THRESHOLD_NORMAL_DIFF || maximum(curvature_diffs) > THRESHOLD_CURVATURE_DIFF || maximum(arc_length_diffs) > THRESHOLD_ARC_LENGTH_DIFF
        return true
    else
        return false
    end
end

# Seuils pour l'ajout et la suppression de points
const THRESHOLD_ADD_POINT = 0.2
const THRESHOLD_REMOVE_POINT = 0.05

# Rééchantillonner le front
function resample_front(points)
    # Surveiller les propriétés du front
    normal_diffs, curvature_diffs, arc_length_diffs = monitor_front_properties(points[1:end-1], points[2:end])
    
    # Initialiser les nouveaux points
    new_points = [points[1]]
    
    # Parcourir les points du front
    for i in 2:length(points)-1
        # Si la différence de courbure ou de normale est grande, ajouter un point
        if curvature_diffs[i-1] > THRESHOLD_ADD_POINT || normal_diffs[i-1] > THRESHOLD_ADD_POINT
            new_point = (points[i-1] + points[i]) / 2
            append!(new_points, new_point)
        end
        
        # Si la différence de courbure ou de normale est petite et la longueur d'arc est grande, supprimer un point
        if curvature_diffs[i-1] < THRESHOLD_REMOVE_POINT && normal_diffs[i-1] < THRESHOLD_REMOVE_POINT && arc_length_diffs[i-1] > THRESHOLD_REMOVE_POINT
            continue
        end
        
        append!(new_points, points[i])
    end
    
    # Ajouter le dernier point
    append!(new_points, points[end])
    
    # Retourner les nouveaux points
    return new_points
end

"""
# Définir les sommets du cube
sommet1 = [x1, y1, z1]
sommet2 = [x2, y2, z2]
sommet3 = [x3, y3, z3]
sommet4 = [x4, y4, z4]
sommet5 = [x5, y5, z5]
sommet6 = [x6, y6, z6]
sommet7 = [x7, y7, z7]
sommet8 = [x8, y8, z8]

# Calculer les centres des faces
centre_face1 = (sommet1 + sommet2 + sommet3 + sommet4) / 4
centre_face2 = (sommet5 + sommet6 + sommet7 + sommet8) / 4
centre_face3 = (sommet1 + sommet2 + sommet5 + sommet6) / 4
centre_face4 = (sommet3 + sommet4 + sommet7 + sommet8) / 4
centre_face5 = (sommet1 + sommet4 + sommet5 + sommet8) / 4
centre_face6 = (sommet2 + sommet3 + sommet6 + sommet7) / 4

# Calculer les centres des arêtes
centre_arete1 = (sommet1 + sommet2) / 2
centre_arete2 = (sommet1 + sommet4) / 2
centre_arete3 = (sommet1 + sommet5) / 2
centre_arete4 = (sommet2 + sommet3) / 2
centre_arete5 = (sommet2 + sommet6) / 2
centre_arete6 = (sommet3 + sommet4) / 2
centre_arete7 = (sommet3 + sommet7) / 2
centre_arete8 = (sommet4 + sommet8) / 2
centre_arete9 = (sommet5 + sommet6) / 2
centre_arete10 = (sommet5 + sommet8) / 2
centre_arete11 = (sommet6 + sommet7) / 2
centre_arete12 = (sommet7 + sommet8) / 2

# Calculer le centre de la cellule
centre_cellule = (sommet1 + sommet2 + sommet3 + sommet4 + sommet5 + sommet6 + sommet7 + sommet8) / 8
"""

end