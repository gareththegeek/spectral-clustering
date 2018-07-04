# Build an adjacency matrix by using face data stored within .obj file

import numpy as np

file = "data/faces.obj"
count = 0

found_faces = False
#current_vertices = []
#current_faces = []

vertices = []
faces = []

print("Parsing obj file: '" + file + "'")

with open(file) as f:
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
print("Vertices:")
print(vertices)
print("Faces:")
print(faces)

print("\nBuilding adjacency matrix")
matrix = np.zeros((len(faces), len(faces)), dtype=int)
for i in range(len(faces)):
    matrix[i, i] = len(faces[i])
    for index in faces[i]:
        matrix[i, index] = -1
        matrix[index, i] = -1

print("\nBuilt " + str(matrix.shape[0]) + " x " + str(matrix.shape[1]) + " matrix")
print(matrix)

print("\nDone")