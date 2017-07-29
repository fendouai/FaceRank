from PIL import Image
import os

list = os.listdir("./test_face")
print(list)

for image in list:
    name_len=len(image)
    name=image[0:name_len-3]
    print(name)
    im=Image.open("./test_face/"+image)
    out = im.resize((128, 128))
    #out.show()
    out.save("./test_resize/"+name+"jpg")

