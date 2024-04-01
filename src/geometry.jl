function compute_sdf_values(SDF, coords)
    # Vérifier la dimension des coordonnées
    dim = length(coords)

    if dim == 1
        # Cas 1D
        phi = [SDF([coords[1][i]]) for i in 1:length(coords[1])]
    elseif dim == 2
        # Cas 2D
        phi = [SDF([coords[1][i], coords[2][i]]) for i in 1:length(coords[1])]
    elseif dim == 3
        # Cas 3D
        phi = [SDF([coords[1][i], coords[2][i], coords[3][i]]) for i in 1:length(coords[1])]
    else
        error("Unsupported coordinate dimension: $dim")
    end

    return phi
end

function obtenir_aretes(maillage)
    # Vérifier la dimension du maillage
    dim = length(maillage)

    aretes = []
    if dim == 1
        # Cas 1D
        x = maillage[1]
        for i in 1:length(x)-1
            push!(aretes, [x[i]])
            push!(aretes, [x[i+1]])
        end
    elseif dim == 2
        # Cas 2D
        x, y = maillage
        for i in 1:length(x)-1
            for j in 1:length(y)-1
                push!(aretes, [[x[i], y[j]], [x[i+1], y[j]]])
                push!(aretes, [[x[i], y[j]], [x[i], y[j+1]]])
            end
        end
    elseif dim == 3
        # Cas 3D
        x, y, z = maillage
        for i in 1:length(x)-1
            for j in 1:length(y)-1
                for k in 1:length(z)-1
                    push!(aretes, [[x[i], y[j], z[k]], [x[i+1], y[j], z[k]]])
                    push!(aretes, [[x[i], y[j], z[k]], [x[i], y[j+1], z[k]]])
                    push!(aretes, [[x[i], y[j], z[k]], [x[i], y[j], z[k+1]]])
                end
            end
        end
    else
        error("Unsupported mesh dimension: $dim")
    end

    return aretes
end

function compute_zero_points(SDF, aretes)
    # Créer une liste pour stocker les points P où Phi=0
    points_P = []

    # Calculer le point P où Phi=0 pour chaque arête
    for arete in aretes
        # Paramétriser l'arête
        f(t) = SDF(arete[1] + t * (arete[2] - arete[1]))
        try
            # Utiliser la méthode de la bissection pour trouver le point P
            t0 = find_zero(f, (0.0, 1.0), Bisection())
            x0 = arete[1] + t0 * (arete[2] - arete[1])
            
            # Ajouter le point P à la liste des points P
            push!(points_P, x0)
        catch e
            # Aucun point P trouvé sur l'arête
        end
    end

    return points_P
end