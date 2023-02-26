# Import the required Libraries
from time import sleep
from tkinter import *
from tkinter import ttk, filedialog
from tkinter.filedialog import askdirectory
import customtkinter
import signal
import sys
from PIL import ImageTk, Image
import os
from TNAP_main import predict_image

array = []
result_array = [(0,0), (0,1), (1,0), (1,1)]
array_size = 0

def handler(signum, frame):
    print("STOP!")
    sys.exit(1)

signal.signal(signal.SIGINT, handler)

# create a frame with three widgets that are equally spaced using grid




def open_file(self):
    #    file = filedialog.askopenfile(mode='r', filetypes=[('JPG files', '*.jpg'), ('PNG files', '*.png'), 
    #                                                       ('tiff files', '*.tiff'), ('jfif files', '*.jfif')])
    # self.clear_packed_labels(self)
    
    # reset the size of the scrollable frame to the default size
    array_size = len(array)  
    self.clear_packed_frames(self)
    array.clear()
    files = filedialog.askopenfilenames(parent=self, title='Choose a file', filetypes=[('JPG files', '*.jpg'), ('PNG files', '*.png'), 
                                                       ('tiff files', '*.tiff'), ('jfif files', '*.jfif')])
    
    
    # print(len(files))
    for x in files:
        array.append(x)

    self.pack_array(self)
    # if file:
    #   content = file.read()
    #   file.close()
    #   print("%d characters in this file" % len(content))


def open_folder(self, _event=None):
    # set the postition of the scrollbar to the topmost position without using canvas
    #self.scrollable_frame.configure(scrollregion=(0, 0, 500, 500))
    # reset the scrollable frame
    array_size = len(array)
    Tk.update(self.scrollable_frame)
    self.clear_packed_frames()
    array.clear()
    folder = filedialog.askdirectory()
    for filename in os.listdir(folder):
        if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".tiff") or filename.endswith(".jfif"):
            array.append(os.path.join(folder, filename))
    
    self.pack_array(self)

# function to save a file in download folder
def save_file(self, _event=None):
    path = askdirectory(title='Select Folder') # shows dialog box and return the path
    # open up a file with directory path
    file = open(path + "/results.csv", "w")
    # write the results to the file
    for i in range(len(array)):
        words = array[i].split("\\")
        if words[len(words)-1] == array[i]:
            words = array[i].split("/")
        image_name = words[len(words)-1]

        file.write(image_name + "," + str(result_array[i][0]) + "," +  str(result_array[i][0]) + "\n")
    # close the file
    file.close()








class DataEntry(customtkinter.CTkFrame):
    def __init__(self, master, image, image_name, pred, **kwargs):
        super().__init__(master, **kwargs)

        # add widgets onto the frame...
        self.image_name = customtkinter.CTkLabel(self)

        words = image_name.split("\\")
        if words[len(words)-1] == image_name:
            words = image_name.split("/")
        image_name = words[len(words)-1]
        if(len(image_name) > 15):
            image_name = image_name[:15]+str("...")

        self.image_name.configure(text=image_name)
        self.image_name.grid(row=0, column=1)


        # this is the point where the backend will be connected to give the user the result
        self.label2 = customtkinter.CTkLabel(self)
        self.label2.configure(text=str(pred[0])+" image with "+str(pred[1])+" accuracy")
        self.label2.grid(row=0, column=2)


        self.my_image = customtkinter.CTkImage(light_image=image, size=(50, 50))
        
        self.label3 = customtkinter.CTkLabel(self, image=self.my_image)
        self.label3.configure(text="")
        self.label3.grid(row=0, column=0)

        self.grid_columnconfigure(0, weight=1, minsize=200)
        self.grid_columnconfigure(1, weight=1, minsize=240)
        self.grid_columnconfigure(2, weight=1, minsize=240)

        
class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        self.title("AI Art Predictor")
        self.minsize(600, 500)
        
        self.resizable(False,False)

        self.withdraw()
        self.update()
        self.deiconify()

        # self.configure(fg_color="dark blue", bg_color="blue")
        # make a scrollable frame
        # self.withdraw()
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure((0, 1, 2), weight=1)


        self.scrollable_frame = customtkinter.CTkScrollableFrame(master=self)
        self.scrollable_frame.grid(row=0, column=0, columnspan=3, padx=20, pady=(20, 0), sticky="nsew")
        #get the width and height of the scrollable frame

        #get the width and height of the scrollable frame
        self.scrollable_frame_width = self.scrollable_frame.winfo_width()
        self.scrollable_frame_height = self.scrollable_frame.winfo_height()

        self.scrollable_frame.configure(height=400)


        self.loading_frame = customtkinter.CTkFrame(master=self)
        self.loading_frame.grid(row=1, column=0, columnspan=3, padx=20, pady=(20, 0), sticky="nsew")
        self.loading_frame.configure(width=self.scrollable_frame_width, height=50)

        # #remove the loading frame
        # self.loading_frame.grid_remove()
        # self.update()
        # # add back loading frame
        # self.loading_frame.grid()
        # self.update()

        
        # file button
        self.button1 = customtkinter.CTkButton(master=self, command=lambda: open_file(self))
        self.button1.configure(text="Open Files")
        self.button1.grid(row=2, column=0, padx=30, pady=20, sticky="e")

        # folder button
        self.button2 = customtkinter.CTkButton(master=self, command=lambda: open_folder(self))
        self.button2.configure(text="Open Folder")
        self.button2.grid(row=2, column=1, padx=30, pady=20, sticky="w")

        # download results button
        self.button3 = customtkinter.CTkButton(master=self, command=lambda: save_file(self))
        self.button3.configure(text="Download CSV")
        self.button3.grid(row=2, column=2, padx=30, pady=20, sticky="w")

        #header of scrollable frame
        self.header = customtkinter.CTkLabel(master=self.scrollable_frame, text="      Image                         Result",
                                              font=("Arial", 25, "bold"))
        # self.header.grid(row=0, column=0, columnspan=2, padx=20, pady=(20, 0), sticky="nsew")
        self.header.pack(padx=50)

        # self.hide_loading_frame()
        # self.my_frame = DataEntry(master=self.scrollable_frame, width=300, height=200, corner_radius=0, fg_color="transparent")
        # self.my_frame.pack(padx=20, pady=20, fill="both", expand=True)

        #self.hide_loading_frame(self)


    def hide_loading_frame(self, _event=None):
        self.loading_frame.grid_remove()
        self.update()
        sleep(2)
        self.loading_frame.grid()
        self.update()
        sleep(1)
        self.loading_frame.grid_remove()
        self.update()

    def pack_array(self, _event=None):

        # during the function call will bring up the loading frame and hide it after the function is done
        # display loading frame when the function is running
        # self.loading_frame.grid()
        # self.update()

        for i in range(len(array)):
            my_image = Image.open(array[i])
            # call the data entry class and pack it

            result = predict_image(array[i])

            # call to external function to get the result using the image path
            # will use the result to display the result in the data entry class
            self.my_data = DataEntry(self.scrollable_frame, my_image, array[i], result, width=500, height=200, corner_radius=3, fg_color="transparent")
            self.my_data.pack(pady=20, anchor='w')

    def clear_packed_frames(self, _event=None):
        count = 0
        for widget in self.scrollable_frame.winfo_children():
            if widget.winfo_class() == "Frame" and count > 0:
                widget.destroy()
            count += 1

    


# run the main loop
if __name__ == "__main__":
    customtkinter.set_appearance_mode("dark")
    customtkinter.set_default_color_theme("dark-blue")
    app = App()
    app.mainloop()


# import tkinter as tk
# import time

# class Splash(tk.Toplevel):
#     def __init__(self, parent):
#         tk.Toplevel.__init__(self, parent)
#         self.title("Splash")

#         ## required to make window show before the program gets to the mainloop
#         self.update()

# class App(tk.Tk):
#     def __init__(self):
#         tk.Tk.__init__(self)
#         self.withdraw()
#         splash = Splash(self)

#         ## setup stuff goes here
#         self.title("Main Window")
#         ## simulate a delay while loading
#         time.sleep(6)

#         ## finished loading so destroy splash
#         splash.destroy()

#         ## show window again
#         self.deiconify()

# if __name__ == "__main__":
#     app = App()
#     app.mainloop()
