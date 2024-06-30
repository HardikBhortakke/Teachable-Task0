# Techligence - Teachable-machine 

## Descrpition:
This repo contains some machine learning models with GUI interface using Tkinter. Can be used in object classification task where in the object is unkonwn and with the help of GUI a dataset containing the new object can be generated and trained to be deployed within minutes.

## File Structure 

```
👨‍💻General Purpose Calculator 
 ┣ Templates                            
    ┗ 📄Index.html        //Contains Basic web Design          
 ┣ 📄app.py               //Contains Flask application                    
 ┣ 📄main.py              //Contains main file with Tkinter GUI          
 ┣ 📄models_imp.py        //Contains Models implementation(Data set generation, preprocessing, etc)
 ┣ 📄models.py            //Contains 3 models 
 ┣ 📄Models_Intution.md
 ┗ 📄README.md      
```

## Installations
Ensure that the following libraries are installed in your environment before running 
- Flask 
- Keras_applications
- Tkinter
- Tensorflow
- Opencv

## Using the program 

To run the Flask application, navigate to the directory and run:
```
flask run
```
This will give you a local host ip address copy and open in browser to view the contains of the website and interact.

To run the Tkinter based GUI applicaiton, navigate to the directory and run:
```
python3 main.py
```

NOTE: The Flask application does not show the video frames being captured and requires virtual camera which need to be integrated before use along with some additional changes.

The Tkinter application is complete and can be used as per required.

## For More details about the program visit Models_Intuition.md

## Mentors
- Mr. Bhavesh Sirvi
