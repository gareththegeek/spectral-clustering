import numpy as np

# Parse .obj file to get an array of vertices and an array of face indices
def get_mesh_data(filename):

    count = 0

    found_faces = False

    vertices = []
    faces = []

    print("Parsing obj file: '" + filename + "'")

    with open(filename) as f:
        for line in f:
            if (line.startswith("v ")):
                line = line.replace("v ", "")
                vertex = np.array(line.split(" ")).astype(np.float)
                vertices.append(vertex)
                faces.append([])
            elif (line.startswith("f ")):
                found_faces = True
                line = line.replace("f ", "")
                face = line.split(" ")
                indices = list(map(lambda x: int(x.split("/")[0]), face))
                for i in range(len(indices)):
                    a = indices[i]-1
                    for j in range(len(indices)):
                        b = indices[j]-1
                        if (i != j and not b in faces[a]):
                            faces[a].append(b)
            else:
                if (found_faces):
                    count += 1
                    print("Processed " + str(count) + " meshes")
                    
                    found_faces = False

    print("\nParsed obj file")
    print("\nVertices:")
    print(vertices)
    print("\nFaces:")
    print(faces)

    return vertices, faces

    # Build a adjacency matrix by using face indices
def build_adjacency_matrix(faces):

    print("\nBuilding adjacency matrix")
    matrix = np.zeros((len(faces), len(faces)), dtype=int)
    for i in range(len(faces)):
        #matrix[i, i] = len(faces[i])
        for index in faces[i]:
            matrix[i, index] += 1
            matrix[index, i] += 1

    print("\nBuilt " + str(matrix.shape[0]) + " x " + str(matrix.shape[1]) + " matrix")
    print(matrix)

    assert np.isfinite(matrix).all()

    return matrix