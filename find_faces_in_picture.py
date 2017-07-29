from PIL import Image
import face_recognition
import os
print("h")
def find_and_save_face(web_file,face_file):
    # Load the jpg file into a numpy array
    image = face_recognition.load_image_file(web_file)
    print(image.dtype)
    # Find all the faces in the image
    face_locations = face_recognition.face_locations(image)

    print("I found {} face(s) in this photograph.".format(len(face_locations)))

    for face_location in face_locations:

        # Print the location of each face in this image
        top, right, bottom, left = face_location
        print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))

        # You can access the actual face itself like this:
        face_image = image[top:bottom, left:right]
        pil_image = Image.fromarray(face_image)
        pil_image.save(face_file)
print("h")
list = os.listdir("web_image/")
print(list)

for image in list:
    id_tag = image.find(".")
    name=image[0:id_tag]
    print(name)

    web_file = "./web_image/" +image
    face_file="./face_image/"+name+".jpg"

    im=Image.open("./web_image/"+image)
    try:
        find_and_save_face(web_file, face_file)
    except:
        print("fail")
