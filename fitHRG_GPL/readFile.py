def swap(array):
    dummy = array[0]
    array[0] = array[1]
    array[1] = dummy

f = open(f"Norway_euclid_best-dendro.hrg", "r")
lines = f.readlines()
d = {}
score = {}
finalScore = {}
i = 0


names = {}
for i in range(len(lines)+1):
    template = "Zone{}"
    names[i] = template.format(i)

#print("iNames =", names)
iNames = dict([(value, key) for key, value in names.items()])
print("iNames = ", iNames)

zone = []
n = len(names) - 1
    
for line in lines:
    dLine = line.split(" ")
    dummy = [[int(dLine[7]), 0 if dLine[8] == '(D)' else 1], [int(dLine[4]), 0 if dLine[5] == '(D)' else 1]]
    d[int(dLine[1])] = dummy
    score[int(dLine[1])] = float(dLine[17])

dDummy = d.copy()
final = [0]   
i = 0
while len(dDummy) != 0:
    aS = dDummy.pop(final[i])
    for a in aS:
        if a[0] not in final and a[1] == 0:
            final.append(a[0])
    
    i += 1;    



index = 1
curr = 10**len(str(n))
result = {}
for i in range(len(final)):
    aS = d.get(final[i]),
    
    test = []
    for a in aS[0]:
        if(a[1] == 0):
            test.append(index)
            index += 1
        else:
            test.append(str(curr))
            zone.append(names[a[0]])
            curr += 1
            
    if (isinstance(test[0], int)) and (not isinstance(test[1], int)):
        swap(test)
        
    finalScore[i] = score[final[i]]
    result[i] = test    


print("test = ", finalScore)
print("names = ", zone)
print("d = ", result)

