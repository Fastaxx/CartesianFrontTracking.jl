function plot_mesh_points(points)
    # Vérifier la dimension des points
    dim = length(points[1])

    if dim == 1
        # Cas 1D
        x = [p[1] for p in points]
        plot(x, title="1D Mesh")
    elseif dim == 2
        # Cas 2D
        x = [p[1] for p in points]
        y = [p[2] for p in points]
        plot(x, y, title="2D Mesh")
    elseif dim == 3
        # Cas 3D
        x = [p[1] for p in points]
        y = [p[2] for p in points]
        z = [p[3] for p in points]
        scatter(x, y, z, xlabel="x", ylabel="y", zlabel="z", title="3D Mesh")
    else
        error("Unsupported mesh dimension: $dim")
    end
end

function plot_sdf_values(coords, phi)
    # Convertir les valeurs de la sdf en couleurs
    colors = [get(ColorSchemes.rainbow, p) for p in phi]

    # Vérifier la dimension des coordonnées
    dim = length(coords)

    if dim == 1
        # Cas 1D
        plot(coords[1], phi, color=colors, xlabel="x", title="Signed Distance Function")
    elseif dim == 2
        # Cas 2D
        scatter(coords[1], coords[2], color=colors, xlabel="x", ylabel="y", title="Signed Distance Function")
    elseif dim == 3
        # Cas 3D
        p = scatter(coords[1], coords[2], coords[3], color=colors, xlabel="x", ylabel="y", zlabel="z", title="Signed Distance Function")
        surface!(p, [0,1], [0,1], (x,y)->0, opacity=0, color=:rainbow, colorbar_title="SDF Value")
    else
        error("Unsupported coordinate dimension: $dim")
    end
end

function plot_zero_points(points_P)
    # Créer une figure
    p = plot(legend = false)

    # Ajouter chaque point P à la figure
    for point in points_P
        if length(point) == 1
            # Cas 1D
            scatter!(p, [point[1]], color = :red, markersize = 4)
        elseif length(point) == 2
            # Cas 2D
            scatter!(p, [point[1]], [point[2]], color = :red, markersize = 4)
        elseif length(point) == 3
            # Cas 3D
            scatter!(p, [point[1]], [point[2]], [point[3]], color = :red, markersize = 4)
        else
            error("Unsupported point dimension: $(length(point))")
        end
    end

    # Afficher la figure
    display(p)
end