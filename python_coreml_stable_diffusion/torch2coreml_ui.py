#
# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

from python_coreml_stable_diffusion import torch2coreml

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        
import os
import traceback
import webbrowser
import subprocess
import multiprocessing
import tkinter as tk
from tkinter import *
from tkinter import filedialog
from tkinter.ttk import Progressbar
from tkinter import messagebox as mb
        
class PlaceholderEntry(tk.Entry):
    def __init__(self, master=None, placeholder='', cnf={}, fg='black',
                 fg_placeholder='grey50', *args, **kw):
        super().__init__(master=None, cnf={}, bg='white', *args, **kw)
        self.fg = fg
        self.fg_placeholder = fg_placeholder
        self.placeholder = placeholder
        self.bind('<FocusOut>', lambda event: self.fill_placeholder())
        self.bind('<FocusIn>', lambda event: self.clear_box())
        self.fill_placeholder()

    def clear_box(self):
        if not self.get() and super().get():
            self.config(fg=self.fg)
            self.delete(0, tk.END)

    def fill_placeholder(self):
        if not super().get():
            self.config(fg=self.fg_placeholder)
            self.insert(0, self.placeholder)
    
    def get(self):
        content = super().get()
        if content == self.placeholder:
            return ''
        return content
        
import time

def is_guernika_installed():
    return os.path.exists("/Applications/Guernika.app")

def is_coremlcompiler_installed():
    try:
        print(subprocess.check_output(['xcrun', '--version']))
    except:
        return False
    try:
        print(subprocess.check_output(['xcrun', 'coremlcompiler', 'version']))
    except:
        return False
    return True

if __name__ == "__main__":
    # Pyinstaller fix
    multiprocessing.freeze_support()
    
    coremlcompiler_installed = is_coremlcompiler_installed()
    
    window = Tk()
    window.title("Guernika Model Converter")
    window.geometry('+540+360')
        
    version_label = Label(window, text="Enter a model version identifier OR select a local model OR select a local CKPT\n\n\nModel version:", justify="left")
    version_label.grid(row=0, column=0, columnspan=5, padx=16, pady=(24, 0), sticky='W')
    version_entry = PlaceholderEntry(window, placeholder="CompVis/stable-diffusion-v1-4")
    version_entry.grid(row=1, column=0, columnspan=5, padx=16, pady=4, sticky='WE')
            
    ckpt_location = None
    model_location = None
    def select_local_model():
        global ckpt_location
        global model_location
        model_location = filedialog.askdirectory(parent=window, title='Please select a diffusion model')
        local_model_label.configure(text="")
        if model_location:
            local_model_label.configure(text=model_location)
            ckpt_label.configure(text="")
            ckpt_location = None
            version_entry.delete(0, END)
    local_model_label = Label(window)
    local_model_label["state"] = DISABLED
    local_model_label.grid(row=2, column=0, columnspan=3, padx=(16, 8), sticky='W')
    local_model_button = Button(window, text="Select local model", command=select_local_model)
    local_model_button.grid(row=2, column=3, columnspan=2, padx=(0,16), sticky='E')
    
    def convert_ckpt():
        global ckpt_location
        global model_location
        ckpt_location = filedialog.askopenfilename(parent=window, title = "Please select a CKPT", filetypes = (("Model checkpoint", "*.ckpt"),))
        ckpt_label.configure(text="")
        if ckpt_location:
            local_model_label.configure(text="")
            ckpt_label.configure(text=ckpt_location)
            model_location = None
            version_entry.delete(0, END)
    ckpt_label = Label(window, text="Defaults to v1-inference.yaml\nFor custom YAML, give it the same name and\nplace it on the same folder as the CKPT\n(model.yaml, model.ckpt)", justify="left")
    ckpt_label["state"] = DISABLED
    ckpt_label.grid(row=3, column=0, columnspan=3, padx=(16, 8), sticky='W')
    ckpt_button = Button(window, text="Select CKPT", command=convert_ckpt)
    ckpt_button.grid(row=3, column=3, columnspan=2, padx=(0,16), sticky='E')
    
    # convert modules
    def convert_unet_switched():
        chunk_unet_check["state"] = NORMAL if convert_unet_value.get() == 1 else DISABLED
    
    convert_options_label = Label(window, text="Convert modules:")
    convert_options_label.grid(row=4, column=0, columnspan=5, padx=16, pady=(12, 0), sticky='W')
    
    convert_unet_value = IntVar(window)
    convert_unet_value.set(1)
    convert_unet_check = Checkbutton(window, text='UNet',variable=convert_unet_value, onvalue=1, offvalue=0, command=convert_unet_switched)
    convert_unet_check.grid(row=5, column=1, padx=16, pady=4, sticky='W')
    chunk_unet_value = IntVar(window)
    chunk_unet_check = Checkbutton(window, text='Chunk UNet',variable=chunk_unet_value, onvalue=1, offvalue=0)
    chunk_unet_check.grid(row=5, column=2, padx=16, pady=4, sticky='W')
    convert_text_encoder_value = IntVar(window)
    convert_text_encoder_value.set(1)
    convert_text_encoder_check = Checkbutton(window, text='Text encoder',variable=convert_text_encoder_value, onvalue=1, offvalue=0)
    convert_text_encoder_check.grid(row=5, column=3, padx=16, pady=4, sticky='W')
        
    convert_encoder_value = IntVar(window)
    convert_encoder_value.set(1)
    convert_encoder_check = Checkbutton(window, text='Encoder',variable=convert_encoder_value, onvalue=1, offvalue=0)
    convert_encoder_check.grid(row=6, column=1, padx=16, pady=4, sticky='W')
    convert_decoder_value = IntVar(window)
    convert_decoder_value.set(1)
    convert_decoder_check = Checkbutton(window, text='Decoder',variable=convert_decoder_value, onvalue=1, offvalue=0)
    convert_decoder_check.grid(row=6, column=2, padx=16, pady=4, sticky='W')
    convert_safety_checker_value = IntVar(window)
    convert_safety_checker_value.set(1)
    convert_safety_checker_check = Checkbutton(window, text='Safety checker',variable=convert_safety_checker_value, onvalue=1, offvalue=0)
    convert_safety_checker_check.grid(row=6, column=3, padx=16, pady=4, sticky='W')
    
    # output size
    output_size_label = Label(window, text="Output size:")
    output_size_label.grid(row=7, column=0, columnspan=5, padx=16, pady=(12, 0), sticky='W')
        
    width_entry = PlaceholderEntry(window, placeholder="Width")
    width_entry.grid(row=8, column=1, padx=(16,0), pady=4, sticky='E')
    sizedivider_label = Label(window, text="X")
    sizedivider_label.grid(row=8, column=2, padx=6)
    height_entry = PlaceholderEntry(window, placeholder="Height")
    height_entry.grid(row=8, column=3, padx=(0,16), pady=4, sticky='W')
    
    # compute units
    selected_compute_unit = StringVar(window)
    selected_compute_unit.set('CPU_AND_NE')
    compute_units_label = Label(window, text="Compute units:")
    compute_units_label.grid(row=9, column=0, columnspan=5, padx=16, pady=(12, 0), sticky='W')
    
    cpu_and_ne_check = Radiobutton(window, text='CPU and NE', variable=selected_compute_unit, value='CPU_AND_NE')
    cpu_and_ne_check.select()
    cpu_and_ne_check.grid(row=10, column=1, padx=16, pady=4, sticky='W')
    cpu_and_gpu_check = Radiobutton(window, text='CPU and GPU', variable=selected_compute_unit, value='CPU_AND_GPU')
    cpu_and_gpu_check.grid(row=10, column=2, padx=16, pady=4, sticky='W')
    all_compute_units_check = Radiobutton(window, text='All', variable=selected_compute_unit, value='ALL')
    all_compute_units_check.grid(row=10, column=3, padx=16, pady=4, sticky='W')
    
            
    progress_label = Label(window, text="Converting model, this may take a while (15-20 minutes).")
    progressbar = Progressbar(window, orient=HORIZONTAL, mode="indeterminate", length=360)
    
    def add_convert_button():
        if not is_guernika_installed():
            convert_button.grid(row=18, column=3, padx=16, pady=(24, 24))
        else:
            convert_button.grid(row=18, column=1, columnspan=3, padx=16, pady=(24, 24))
    
    def show_converting(is_converting):
        if is_converting:
            progress_label.grid(row=16, column=0, columnspan=5, padx=16, pady=(16,4))
            progressbar.grid(row=17, column=0, columnspan=5, padx=16, pady=(4,24))
            progressbar.start()
            convert_button.grid_remove()
        else:
            progress_label.grid_remove()
            progressbar.grid_remove()
            progressbar.stop()
            add_convert_button()
            
        version_entry["state"] = DISABLED if is_converting else NORMAL
        local_model_button["state"] = DISABLED if is_converting else NORMAL
        ckpt_button["state"] = DISABLED if is_converting else NORMAL
        
        convert_unet_check["state"] = DISABLED if is_converting else NORMAL
        chunk_unet_check["state"] = DISABLED if is_converting else NORMAL
        convert_text_encoder_check["state"] = DISABLED if is_converting else NORMAL
        convert_encoder_check["state"] = DISABLED if is_converting else NORMAL
        convert_decoder_check["state"] = DISABLED if is_converting else NORMAL
        convert_safety_checker_check["state"] = DISABLED if is_converting else NORMAL
        window.update()
        
    def open_appstore():
        webbrowser.open("https://apps.apple.com/app/id1660407508")

    def convert_model():
        global ckpt_location
        global model_location
        global selected_compute_unit
        global coremlcompiler_installed
        
        if not coremlcompiler_installed:
            # check again
            coremlcompiler_installed = is_coremlcompiler_installed()
            if coremlcompiler_installed:
                xcode_label.grid_remove()
            else:
                mb.showerror(title = "Error", message = "CoreMLCompiler not available.\nMake sure you have Xcode installed and you run \"sudo xcode-select --switch /Applications/Xcode.app/Contents/Developer/\" on a Terminal.")
                return
        
        model_version = version_entry.get().strip()
        if ckpt_location:
            model_version = os.path.basename(os.path.normpath(ckpt_location))
            # remove extension
            model_version = os.path.splitext(model_version)[0]
        if model_location:
            model_version = os.path.basename(os.path.normpath(model_location))
        
        if not model_version:
            mb.showerror(title = "Error", message = "You need to choose a model to convert")
            return
        
        output_h = None
        output_w = None
                
        try:
            output_h_str = height_entry.get().strip()
            if output_h_str:
                output_h = int(output_h_str)
            output_w_str = width_entry.get().strip()
            if output_w_str:
                output_w = int(output_w_str)
        except:
            mb.showerror(title = "Error", message = "Invalid output size")
            return

        output_folder = filedialog.askdirectory(parent=window, title='Where do you want to save the model?')
        if not output_folder.strip():
            mb.showerror(title = "Error", message = "Invalid output folder")
            return
        
        args = Namespace(
            attention_implementation='ORIGINAL' if selected_compute_unit.get() == 'ALL' else 'SPLIT_EINSUM',
            bundle_resources_for_guernika=True,
            check_output_correctness=False,
            chunk_unet=chunk_unet_value.get() == 1,
            compute_unit=selected_compute_unit.get(),
            convert_safety_checker=convert_safety_checker_value.get() == 1,
            convert_text_encoder=convert_text_encoder_value.get() == 1,
            convert_unet=convert_unet_value.get() == 1,
            convert_vae_decoder=convert_decoder_value.get() == 1,
            convert_vae_encoder=convert_encoder_value.get() == 1,
            output_h=output_h,
            output_w=output_w,
            model_location=None if not model_location else model_location,
            model_version=model_version,
            checkpoint_path=None if not ckpt_location else ckpt_location,
            original_config_file=None,
            resources_dir_name=model_version,
            clean_up_mlpackages=True,
            o=output_folder,
            quantize_weights_to_8bits=False,
            text_encoder_merges_url='https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/merges.txt',
            text_encoder_vocabulary_url='https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/vocab.json'
        )
        show_converting(True)
        try:
            torch2coreml.main(args)
            mb.showinfo(title = "Success", message = "Model successfully converted")
        except:
            mb.showerror(title = "Error", message = f"An error occurred during conversion\n{traceback.format_exc()}")
            traceback.print_exc()
        finally:
            show_converting(False)
        
    if not is_guernika_installed():
        guernika_button = Button(window, text="Install Guernika", command=open_appstore)
        guernika_button.grid(row=18, column=1, padx=16, pady=(24, 24))
    convert_button = Button(window, text="Convert to Guernika", command=convert_model)
    add_convert_button()
            
    if not coremlcompiler_installed:
        xcode_label = Label(window, text="CoreMLCompiler not available!\nMake sure you have Xcode installed before starting conversion.")
        xcode_label.grid(row=20, column=0, columnspan=5, padx=16, pady=(0, 24))
    
    window.minsize(500, 50)
    window.resizable(False, False)
    window.grid_columnconfigure(0, weight=1)
    window.grid_columnconfigure(4, weight=1)
    window.mainloop()
