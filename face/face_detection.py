'''import cv2

# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# Read the input image
img = cv2.imread('vk.jpeg')
# Convert into grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Detect faces
faces = face_cascade.detectMultiScale(gray, 1.1, 4)
# Draw rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0,255, 0), 2)
# Display the output
cv2.imshow('img', img)
cv2.waitKey()'''
# Usage: python face-detect-haar.py [optional.jpg]

import cv2
import sys

# Load the cascade classifiers
frontalFaceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
profileFaceCascade = cv2.CascadeClassifier("haarcascade_profileface.xml")

# Color in BGR
blue = (247, 173, 62)
green = (120, 217, 30)

# Thickness of rectangle drawn around faces
thickness = 2

def drawRectangle(image, color, faces):
	for (x, y, w, h) in faces:
		barLength = int(h / 8)
		barWidth = w
		cv2.rectangle(image, (x, y-barLength), (x+barWidth, y), color, -1)
		cv2.rectangle(image, (x, y-barLength), (x+barWidth, y), color, thickness)
		cv2.rectangle(image, (x, y), (x+w, y+h), color, thickness)
	return image

def detectFace(grayscale, image, isWebcam):
	# Detects frontal faces in the image using the face cascade
	faces = frontalFaceCascade.detectMultiScale(
		grayscale,
		scaleFactor=1.05,
		minNeighbors=5,
		minSize=(30,30),
	)

	if not(isWebcam):
		# Detects profile faces in the image using the face cascade
		profileFaces = profileFaceCascade.detectMultiScale(
			grayscale,
			scaleFactor=1.05,
			minNeighbors=5,
			minSize=(30,30),
		)
		# Detect profile faces in the flipped image to detect profile faces facing right
		flipped = cv2.flip(grayscale, 1)
		profileFacesFlipped = profileFaceCascade.detectMultiScale(
			flipped,
			scaleFactor=1.05,
			minNeighbors=5,
			minSize=(30,30)
		)

	# Draw a rectangle around each detected frontal face
	image = drawRectangle(image, blue, faces)

	if not (isWebcam):
		# Draw a rectangle around each detected profile face
		image = drawRectangle(image, green, profileFaces)
		image = cv2.flip(image, 1)
		image = drawRectangle(image, green, profileFacesFlipped)
		image = cv2.flip(image, 1)

	return image

def useWebcam():
	video  = cv2.VideoCapture(0)
	while True:
		_, frame = video.read()

		# Convert frame to grayscale
		grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		frame = detectFace(grayscale, frame, True)

		# Flip the frame
		frame = cv2.flip(frame, 1)

		cv2.imshow("Face Detection", frame)
		if cv2.waitKey(1) > 0:
			break

	video.release()
	cv2.destroyAllWindows()

def useImage():
	# Read the image
	image =cv2.imread('vk.jpeg')

	# Convert image to grayscale
	grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	
	image = detectFace(grayscale, image, False)

	cv2.imshow("Face Detection", image)
	cv2.waitKey(0)

def main():
        #useImage()
        if len(sys.argv) == 1:
		useWebcam()
	elif len(sys.argv) == 2:
		useImage()
	else:
		print("Usage: python face-detect-haar.py [optional.jpg]")
		exit()

if __name__ == "__main__":
	main()
