module CartesianFrontTracking
using LinearAlgebra
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

using Plots
using IterTools
Plots.default(show = true)

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

# Extraire les coordonnées x, y et z de tous les points
x = [p[1] for p in points]
y = [p[2] for p in points]
z = [p[3] for p in points]

# Créer un nuage de points
scatter(x, y, z, xlabel="x", ylabel="y", zlabel="z", title="3D Mesh")
#readline()

# Définir la fonction de distance signée pour une sphère
function signed_distance_sphere(point::Vector{Float64}, r::Float64)
    return norm(point) - r
end

# Créer une sphère de rayon r
r =2.0
sphere = point -> signed_distance_sphere(point, r)

# Calculer les valeurs de la sdf aux sommets du maillage
phi = [sphere([x[i], y[i], z[i]]) for i in 1:length(x)]

using ColorSchemes

# Convertir les valeurs de la sdf en couleurs
colors = [get(ColorSchemes.rainbow, p) for p in phi]

# Créer un nuage de points où la couleur de chaque point est déterminée par la valeur de la sdf
p = scatter(x, y, z, color=colors, xlabel="x", ylabel="y", zlabel="z", title="Signed Distance Function")
surface!(p, [0,1], [0,1], (x,y)->0, opacity=0, color=:rainbow, colorbar_title="SDF Value")
readline()

@show mesh3D

using Roots

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

# Définir la fonction Phi
sphere = point -> signed_distance_sphere(point, r)

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
        println("Le point P où Phi=0 sur l'arête entre les points $(arete[1]) et $(arete[2]) est $x0")
        
        # Ajouter le point P à la liste des points P
        push!(points_P, x0)
    catch e
        println("Aucun point P trouvé sur l'arête entre les points $(arete[1]) et $(arete[2])")
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

function move_point(point, time)
    # Définir un vecteur de déplacement
    displacement = [sin(time), cos(time), sin(time)*cos(time)]
    
    # Ajouter le vecteur de déplacement au point
    new_point = point + displacement
    
    return new_point
end

dt = 0.01
t0 = 0.0

# Boucle sur les pas de temps
for step in 1:100
    t0 = 0.0 + step * dt
    # Déplacer chaque point
    for i in eachindex(points_P)
        points_P[i] = move_point(points_P[i], t0)
    end
    
    # Créer une figure 3D pour cette étape de temps
    p = plot(legend = false, xlims = (-5, 5), ylims = (-5, 5), zlims = (-5, 5))
    for point in points_P
        scatter!(p, [point[1]], [point[2]], [point[3]], color = :red, markersize = 4)
    end
    
    savefig(p, "step_$(lpad(step, 3, '0')).png")

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