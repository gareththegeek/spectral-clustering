import numpy as np

count = 0

found_faces = False
#current_vertices = []
#current_faces = []

vertices = []
faces = []

with open("data/faces.csv") as f:
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

print(vertices)
print(faces)

