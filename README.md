# 🐢 Shape Detection & Drawing with Turtle  

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)  
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green.svg)](https://opencv.org/)  
[![Tkinter](https://img.shields.io/badge/Tkinter-GUI-orange.svg)](https://docs.python.org/3/library/tkinter.html)  

This project combines **OpenCV**, **Tkinter**, and **Python Turtle Graphics** to:  
✅ Detect shapes from an input image  
✅ Display them in a GUI with clickable overlays  
✅ Draw the clicked shape in a **Turtle graphics window**  

---

## 📂 Project Structure
├── main.py # Main script
├── shapedetector.py # Shape detection utilities
├── centre_of_shape.py # Functions to calculate centroids
├── color_detector.py # (Optional) Detect shape colors
└── README.md # Documentation


## ⚙️ Installation & Requirements  

> 📝 Note: The table below is written in **Markdown table syntax**, not YAML.  
> It just looks similar to YAML because of the structured format.  

| Dependency      | Version / Command                |
|-----------------|----------------------------------|
| Python          | 3.8+                             |
| OpenCV          | `pip install opencv-python`      |
| imutils         | `pip install imutils`            |
| Pillow (PIL)    | `pip install pillow`             |

Install all at once:
```bash
pip install opencv-python imutils pillow
