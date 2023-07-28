sep = 2
nb = 8
i = 0
for j = 1:sep 
    print("x sep ",1 ,"y sep ",2 ,"z sep ",3 , "\n")
    print("x sep ",1 ,"y sep ",2 ,"z sep ",3 , "\n")
    print("x sep ",1 ,"y sep ",2 ,"z sep ",3 , "\n")
end

# First baby cube [1:cellsA[1]/2, 1:cellsA[2]/2, cellsA[3]/2:end]
# Second baby cube: [1:cellsA[1]/2, 1:cellsA[2]/2, cellsA[3]/2+1:end]
# Third baby cube: [1:cellsA[1]/2, cellsA[2]/2+1:end, 1:cellsA[3]/2]
# Fourth baby cube: [1:cellsA[1]/2, cellsA[2]/2+1:end, cellsA[3]/2+1:end]

# Fifth baby cube: [cellsA[1]/2+1:end, 1:cellsA[2]/2, 1:cellsA[3]/2]
# Sixth baby cube: [cellsA[1]/2+1:end, cellsA[2]/2+1:end, 1:cellsA[3]/2]
# Seventh baby cube: [cellsA[1]/2+1:end, 1:cellsA[2]/2, cellsA[3]/2+1:end]
# Sixth baby cube: [cellsA[1]/2+1:end, cellsA[2]/2+1:end, 1:cellsA[3]/2]

