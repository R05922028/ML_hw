import sys

filename = sys.argv[1]
fin = open(filename,'r')

string = fin.read()

data = string.split()
fin.close()

resultList = []
for item in data:
	if not item in resultList:
		resultList.append(item)

answerList = []
for word in resultList:
	answerList.append(data.count(word))

fout = open('Q1.txt','w')
index = 0
for i in range(len(resultList)-1):
	fout.write(resultList[i])
	fout.write(' ')
	fout.write(str(index))
	fout.write(' ')
	fout.write(str(answerList[i]))
	fout.write('\n')
	index = index + 1
fout.write(resultList[len(resultList)-1])
fout.write(' ')
fout.write(str(index))
fout.write(' ')
fout.write(str(answerList[len(resultList)-1]))
fout.close()
