# 2024Aug16
# Face Recognition Diffusion Model (FRDM) Project
# This is initial simple test of facenet on LFW dataset

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
from torchvision.datasets import LFWPeople

test_case = 1   # 0 or 1 (different or same)

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


if __name__ == '__main__':
    print_hi('FRDM')
    lfw_ds = LFWPeople(root='./', download=True)

    # Initialize MTCNN for face detection
    mtcnn = MTCNN()

    # Load pre-trained Inception ResNet model
    resnet = InceptionResnetV1(pretrained='casia-webface').eval()

    # Load two face images to be verified
    if test_case==0:
        # case1: different persons
        image1_path = './lfw-py/lfw_funneled/Aaron_Eckhart/Aaron_Eckhart_0001.jpg'
        image2_path = './lfw-py/lfw_funneled/Aaron_Guiel/Aaron_Guiel_0001.jpg'
    if test_case == 1:
        # case2: same person
        image1_path = './lfw-py/lfw_funneled/Abel_Pacheco/Abel_Pacheco_0001.jpg'
        image2_path = './lfw-py/lfw_funneled/Abel_Pacheco/Abel_Pacheco_0003.jpg'

    img1 = Image.open(image1_path)
    img2 = Image.open(image2_path)
    # Detect faces and extract embeddings
    faces1, _ = mtcnn.detect(img1)
    faces2, _ = mtcnn.detect(img2)

    if faces1 is not None and faces2 is not None:
        aligned1 = mtcnn(img1).unsqueeze(0)
        aligned2 = mtcnn(img2).unsqueeze(0)
        embeddings1 = resnet(aligned1).detach()
        embeddings2 = resnet(aligned2).detach()

        # Calculate the Euclidean distance between embeddings
        distance = (embeddings1 - embeddings2).norm().item()
        print(image1_path)
        print(image2_path)
        if distance < 1.0:  # You can adjust the threshold for verification
            print("Same person")
        else:
            print("Different persons")

