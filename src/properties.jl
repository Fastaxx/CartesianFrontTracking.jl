function compute_arc_lengths(points_P)
    # Calcul de la longueur d'arc entre chaque paire de points P
    arc_lengths = [arc_length(points_P[i], points_P[i+1]) for i in 1:length(points_P)-1]
    return arc_lengths
end

# Calcul de la longueur d'arc
function arc_length(point1, point2)
    return norm(point2 - point1)
end


# Calculer le gradient de Phi en un point
function gradient_Phi(Phi, point)
    return ForwardDiff.gradient(Phi, point)
end

# Calculer le vecteur normal en un point
function normal_vector(Phi, point)
    # Calculer le gradient de Phi au point
    gradient = gradient_Phi(Phi, point)
    
    # Le vecteur normal est le gradient normalisé
    normal = gradient / norm(gradient)
    
    return normal
end

function compute_normals(points_P)
    # Calculer le vecteur normal pour chaque point P
    normals = [normal_vector(Phi, p) for p in points_P]
    return normals
end