import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model # type: ignore
import cv2
import numpy as np

root = tk.Tk()
root.title("ALL Detection")

# sets size of the window
root.geometry("1000x600")

scores = []

# Ensure input image is properly preprocessed
def preprocess_image(image_path):
    # Load the image
    image = cv2.imread(image_path)
    
    # Resize the image to the required input size of the VGG16 model
    image = cv2.resize(image, (224, 224))
    
    # Normalize the image (from [0, 255] to [0, 1])
    image = image / 255.0
    
    # Expand dimensions to match the model input shape (1, 224, 224, 3)
    image = np.expand_dims(image, axis=0)
    
    return image

# Function to predict the label of an input image
def predict_image(image_path, model):
    # Preprocess the image
    image = preprocess_image(image_path)
    
    # Predict using the loaded model
    prediction = model.predict(image)
    
    # Convert prediction to class label (0 or 1)
    predicted_label = int(prediction[0][0] > 0.5)
    
    # Label mapping (0: Normal, 1: Blast)
    labels = {0: 'Normal', 1: 'Blast'}
    
    # return labels[predicted_label]
    return labels[predicted_label]    

model_list = ['vgg.keras', 'vgg19.keras', 'resnet101v2.keras', 'densenet201.keras', 'inceptionv3.keras', 'mobilenetv2.keras']
labels = ['vgg', 'vgg19', 'resnet101v2', 'densenet201', 'inceptionv3', 'mobilenetv2']


def check_results():
    res_list = []
    for i in range(6):
        # Load the model correctly
        model = load_model(model_list[i]) 
        predicted_label = predict_image(img_path, model)
        result = model_list[i][:-6]+" - "+predicted_label
        res_list.append(result)
    label31.config(text=res_list[0])
    label32.config(text=res_list[1])
    label33.config(text=res_list[2])
    label34.config(text=res_list[3])
    label35.config(text=res_list[4])
    label36.config(text=res_list[5])


def open_image():
    global img_path
    img_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png *.jpg *.jpeg *.gif *.bmp")])
    if img_path:
        img = Image.open(img_path)
        img = img.resize((400, 300))  # Resize the image if needed
        img = ImageTk.PhotoImage(img)
        label21.config(image=img)
        label21.image = img  # Keep reference to the image
        

def adaptive_sharpen(image):
    blurred = cv2.GaussianBlur(image, (9, 9), 10.0)
    sharpened = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)
    return sharpened

def conv_img():
    image = cv2.imread(img_path)
    image = cv2.resize(image, (400, 300))
    image = adaptive_sharpen(image)
    imageSn = Image.fromarray(image)
    imageS = ImageTk.PhotoImage(imageSn)
    label23.config(image=imageS).pack()

# heading
frame1 = ttk.Frame(root)
frame1.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

text_var = tk.StringVar()
text_var.set("Acute Lymphoblastic Leukemia Detection")

label = tk.Label(frame1,
    textvariable=text_var,anchor=tk.CENTER,height=3,width=100,bd=0,font=("Arial", 16, "bold"),cursor="hand2",padx=0, pady=5,justify=tk.CENTER,relief=tk.RAISED) 
label.pack() 


#frame to take images
frame2 = ttk.Frame(root)
frame2.grid(row=1, column=0, sticky="nsew", padx=10, pady=0)

frame21 = ttk.Frame(frame2)
frame21.grid(row=0, column=0, sticky="nsew", padx=30, pady=5)

# Set default image
default_image = Image.open("C:/Users/SATVIK/Downloads/project/project/default.jpg")  
default_photo = ImageTk.PhotoImage(default_image.resize((400, 300)))

label21 = tk.Label(frame21, image=default_photo)
label21.pack()

button = tk.Button(frame21, text="Choose Image", padx=5, pady=2, command=open_image)
button.pack(pady=10)

frame22 = ttk.Frame(frame2)
frame22.grid(row=0, column=1, sticky="nsew", padx=100, pady=5)

conv_image = Image.open("C:/Users/SATVIK/Downloads/project/project/convert.png")  
conv_photo = ImageTk.PhotoImage(conv_image.resize((100, 100)))
label22 = tk.Label(frame22, image=conv_photo)
label22.pack()

convert = tk.Button(frame22, text="Convert", command=conv_img)
convert.pack(pady=5)

convert = tk.Button(frame22, text="Find Results", command=check_results)
convert.pack(pady=5)

frame23 = ttk.Frame(frame2)
frame23.grid(row=0, column=2, sticky="nsew", padx=30, pady=5)

# Set default image
default_image1 = Image.open("C:/Users/SATVIK/Downloads/project/project/default.jpg")  
default_photo1 = ImageTk.PhotoImage(default_image1.resize((400, 300)))

label23 = tk.Label(frame23, image=default_photo1)
label23.pack()

button = tk.Button(frame23, text="Converted Image", padx=5, pady=2)
button.pack(pady=10)

#Results

frame3 = ttk.Frame(root)
frame3.grid(row=2, column=0, sticky="nsew", padx=0, pady=5)

label1 = tk.Label(frame3, text="Results", font=("Arial", 14, "bold"), padx=5, pady=3)
label1.pack()

label31 = tk.Label(frame3, text=labels[0], font=("Arial", 12, "bold"), padx=5, pady=3)
label31.pack()

label32 = tk.Label(frame3, text=labels[1], font=("Arial", 12, "bold"), padx=5, pady=3)
label32.pack()

label33 = tk.Label(frame3, text=labels[2], font=("Arial", 12, "bold"), padx=5, pady=3)
label33.pack()

label34 = tk.Label(frame3, text=labels[3], font=("Arial", 12, "bold"), padx=5, pady=3)
label34.pack()

label35 = tk.Label(frame3, text=labels[4], font=("Arial", 12, "bold"), padx=5, pady=3)
label35.pack()

label36 = tk.Label(frame3, text=labels[5], font=("Arial", 12, "bold"), padx=5, pady=3)
label36.pack()

root.mainloop()
