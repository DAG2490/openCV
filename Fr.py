import cv2

# Set the file names
imagePath = 'test2.jpg' # You can change this to 'tes3.jpg' if that file is in your folder
cascPath = 'haarcascade_frontalface_default.xml'

# Load the image and the classifier
image = cv2.imread(imagePath)
faceCascade = cv2.CascadeClassifier(cascPath)

# Safety check: Stop if the image didn't load
if image is None:
    print(f"Error: Could not find {imagePath}!")
else:
    # Process the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(10,10))
    gray = clahe.apply(gray)

    # Detect faces
    faces = faceCascade.detectMultiScale(gray, 1.2, 5, minSize=(30,30))
    print(f"Found {len(faces)} faces!")

    # Draw a green box around any faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Show the final result
    cv2.imshow("Faces found", image)
    cv2.waitKey(0) 
    cv2.destroyAllWindows()