import os,zipfile
def compre(file,zip_name):
    zp = zipfile.ZipFile(zip_name, 'a', zipfile.ZIP_DEFLATED)
    zp.write(file)
    zp.close()
def create__file(file_path,msg):
    f=open(file_path,"w")
    f.write(msg)
    f.close
for i in range(10):
    name=str(i)+".txt"
    path = os.path.join("./test","round1",name)
    print(path)
    create__file(path,str(i))
    compre(path,"test.zip")

for i in range(10):
    name=str(i)+".txt"
    path = os.path.join("./test","round2",name)
    print(path)
    create__file(path,str(i))
    compre(path,"test.zip")
print("done!")