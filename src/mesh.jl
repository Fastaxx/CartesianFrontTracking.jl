struct Mesh
    grids::Array{Vector{Float64}, 1}
end

Mesh(grids::Array{Vector{Float64}, 1}) = Mesh(grids)

function generate_mesh(mesh)
    # Extraire les grilles du maillage
    grids = mesh.grids

    # Générer tous les points du maillage
    points = []

    if length(grids) == 1
        # Maillage 1D
        for x in grids[1]
            push!(points, (x,))
        end
    elseif length(grids) == 2
        # Maillage 2D
        for x in grids[1]
            for y in grids[2]
                push!(points, (x, y))
            end
        end
    elseif length(grids) == 3
        # Maillage 3D
        for x in grids[1]
            for y in grids[2]
                for z in grids[3]
                    push!(points, (x, y, z))
                end
            end
        end
    else
        error("Unsupported mesh dimension: $(length(grids))")
    end

    return points
end