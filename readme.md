#  Face Detection Using OpenCV

This Project is built on Python to detect Faces for a Given input image. Using Haar-Cascade Classifier the faces were detected and using LBPH face recognizer the model was trained to store two classes of images.

## Dependencies

I suggest to run it on virtual environment to avoid previous dependencies.

```bash
pip install virtualenv
virtualenv [foldername]
source activate foldername/bin/activate
```
Extract the zip file and place it in filename folder.

Then install OpenCV libraries.

```bash
pip install opencv-python
pip install opencv-contrib-python
```

## File Structure

> After Extracting the zip file in [foldername]. This file structure Appears. 

### A typical top-level directory layout

    .
    ├── bin                   # Virtual Environment files
    ├── include               # Virtual Environment files
    ├── lib                   # Virtual Environment files
    ├── Q3                    # Project Root Folder
    └── README.md


> Contents of Project folder is down below.

### Project folder

    .
    ├── Test                    # Test files 
    │   ├── example.jpg
    │   ├── tomlookalike.jpg
    │   └── tom1.jpeg 
    ├── Train
    │   ├── 0                   # Contains 200 images
    │   │   ├──1.jpg
    │   │   └──...
    │   └── 1                   # Contains 214 images
    │       ├──1.jpg
    │       └──...
    ├── haarcascade_frontalface_default.xml   # Haar-cascade file
    ├── test.py                 # Run this to get output
    ├── train.py                # Contains logic for face detection and training using 
    │                             LBPH Recognizer
    └── trainData.yml           # Contains labels of trained data in yml format. 
            

 
## Usage

```python
python test.py Test/example.jpg
```
Add new images in your desired folder and run the above code accordingly.

## References
 [Face Recognition with OpenCV](https://docs.opencv.org/2.4/modules/contrib/doc/facerec/facerec_tutorial.html). 
