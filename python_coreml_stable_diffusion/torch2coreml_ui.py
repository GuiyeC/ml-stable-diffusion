#
# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

from python_coreml_stable_diffusion import torch2coreml

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        
import os
import json
from pathlib import Path
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
    
def fetch_preferences():
    try:
        application_support = Path.home().joinpath('Library', 'Application Support', 'GuernikaModelConverter')
        preferences_file = application_support.joinpath('preferences.json')
        with open(preferences_file) as f:
            raw_json = f.read()
        preferences = json.loads(raw_json)
    except:
        preferences = {}
    # register defaults
    if 'last_model_version' not in preferences:
        preferences['last_model_version'] = 'CompVis/stable-diffusion-v1-4'
    if 'convert_unet' not in preferences:
        preferences['convert_unet'] = True
    if 'chunk_unet' not in preferences:
        preferences['chunk_unet'] = False
    if 'controlnet_support' not in preferences:
        preferences['controlnet_support'] = True
    if 'convert_text_encoder' not in preferences:
        preferences['convert_text_encoder'] = True
    if 'convert_vae_encoder' not in preferences:
        preferences['convert_vae_encoder'] = True
    if 'convert_vae_decoder' not in preferences:
        preferences['convert_vae_decoder'] = True
    if 'convert_safety_checker' not in preferences:
        preferences['convert_safety_checker'] = False
    if 'compute_unit' not in preferences:
        preferences['compute_unit'] = 'CPU_AND_NE'
    if 'from_safetensors' not in preferences:
        preferences['from_safetensors'] = False
    if 'last_model_location' not in preferences:
        preferences['last_model_location'] = Path.home()
    if 'last_checkpoint_path' not in preferences:
        preferences['last_checkpoint_path'] = Path.home()
    if 'last_output_folder' not in preferences:
        preferences['last_output_folder'] = Path.home()
    
    return preferences

def save_last_checkpoint_path(preferences, path):
    preferences['last_checkpoint_path'] = path
    save_preferences(preferences)

def save_args_preferences(args):
    preferences = {
        'convert_unet': args.convert_unet,
        'chunk_unet': args.chunk_unet,
        'controlnet_support': args.controlnet_support,
        'convert_text_encoder': args.convert_text_encoder,
        'convert_vae_encoder': args.convert_vae_encoder,
        'convert_vae_decoder': args.convert_vae_decoder,
        'convert_safety_checker': args.convert_safety_checker,
        'compute_unit': args.compute_unit,
        'last_output_folder': args.o,
        'from_safetensors': args.from_safetensors
    }
    if args.model_location:
        preferences['last_model_location'] = args.model_location
    if args.checkpoint_path:
        preferences['last_checkpoint_path'] = args.checkpoint_path
    if not args.checkpoint_path and not args.model_location:
        preferences['last_model_version'] = args.model_version
    save_preferences(preferences)

def save_preferences(data):
    application_support = Path.home().joinpath('Library', 'Application Support', 'GuernikaModelConverter')
    application_support.mkdir(parents=True, exist_ok=True)
    preferences_file = application_support.joinpath('preferences.json')
    preferences_file.touch(exist_ok=True)
    with open(preferences_file, 'w+', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    # Pyinstaller fix
    multiprocessing.freeze_support()
    
    coremlcompiler_installed = is_coremlcompiler_installed()
    
    preferences = fetch_preferences()
    print("Preferences", preferences)
    
    window = Tk()
    window.title("Guernika Model Converter")
    window.geometry('+540+360')
    
    current_row=0
    version_label = Label(window, text="Enter a model version identifier OR select a local model OR select a local CKPT\n\n\nModel version:", justify="left")
    version_label.grid(row=current_row, column=0, columnspan=5, padx=16, pady=(24, 0), sticky='W')
    current_row += 1
    
    version_entry = PlaceholderEntry(window, placeholder=preferences['last_model_version'])
    version_entry.grid(row=current_row, column=0, columnspan=5, padx=16, pady=4, sticky='WE')
    current_row += 1
            
    ckpt_location = None
    model_location = None
    def select_local_model():
        global ckpt_location
        global model_location
        model_location = filedialog.askdirectory(parent=window, initialdir=preferences['last_model_location'], title='Please select a diffusion model')
        local_model_label.configure(text="")
        if model_location:
            save_last_checkpoint_path(preferences, model_location)
            local_model_label.configure(text=os.path.basename(os.path.normpath(model_location)))
            ckpt_label.configure(text="")
            ckpt_location = None
            version_entry.delete(0, END)
    safetensors_model_value = IntVar(window)
    safetensors_model_value.set(1 if preferences['from_safetensors'] else 0)
    safetensors_model_check = Checkbutton(window, text='Safetensors model',variable=safetensors_model_value, onvalue=1, offvalue=0)
    safetensors_model_check.grid(row=current_row, column=1, padx=16, pady=4, sticky='W')
    local_model_label = Label(window)
    local_model_label["state"] = DISABLED
    local_model_label.grid(row=current_row, column=2, columnspan=3, padx=(16, 8), sticky='W')
    local_model_button = Button(window, text="Select local model", command=select_local_model)
    local_model_button.grid(row=current_row, column=3, columnspan=2, padx=(0,16), sticky='E')
    current_row += 1
    
    def convert_ckpt():
        global ckpt_location
        global model_location
        ckpt_location = filedialog.askopenfilename(parent=window, initialdir=preferences['last_checkpoint_path'], title = "Please select a CKPT", filetypes = (("Model checkpoint", "*.ckpt"),))
        ckpt_label.configure(text="")
        if ckpt_location:
            local_model_label.configure(text="")
            ckpt_label.configure(text=ckpt_location)
            model_location = None
            version_entry.delete(0, END)
    ckpt_label = Label(window, text="Defaults to v1-inference.yaml\nFor custom YAML, give it the same name and\nplace it on the same folder as the CKPT\n(model.yaml, model.ckpt)", justify="left")
    ckpt_label["state"] = DISABLED
    ckpt_label.grid(row=current_row, column=0, columnspan=3, padx=(16, 8), sticky='W')
    ckpt_button = Button(window, text="Select CKPT", command=convert_ckpt)
    ckpt_button.grid(row=current_row, column=3, columnspan=2, padx=(0,16), sticky='E')
    current_row += 1
    
    # convert modules
    def convert_unet_switched():
        chunk_unet_check["state"] = NORMAL if convert_unet_value.get() == 1 else DISABLED
        controlnet_support_check["state"] = NORMAL if convert_unet_value.get() == 1 else DISABLED
    
    convert_options_label = Label(window, text="Convert modules:")
    convert_options_label.grid(row=current_row, column=0, columnspan=5, padx=16, pady=(12, 0), sticky='W')
    current_row += 1
    
    convert_unet_value = IntVar(window)
    convert_unet_value.set(1 if preferences['convert_unet'] else 0)
    convert_unet_check = Checkbutton(window, text='UNet',variable=convert_unet_value, onvalue=1, offvalue=0, command=convert_unet_switched)
    convert_unet_check.grid(row=current_row, column=1, padx=16, pady=4, sticky='W')
    chunk_unet_value = IntVar(window)
    chunk_unet_value.set(1 if preferences['chunk_unet'] else 0)
    chunk_unet_check = Checkbutton(window, text='Chunk UNet',variable=chunk_unet_value, onvalue=1, offvalue=0)
    chunk_unet_check["state"] = NORMAL if convert_unet_value.get() == 1 else DISABLED
    chunk_unet_check.grid(row=current_row, column=2, padx=16, pady=4, sticky='W')
    controlnet_support_value = IntVar(window)
    controlnet_support_value.set(1 if preferences['controlnet_support'] else 0)
    controlnet_support_check = Checkbutton(window, text='ControlNet support',variable=controlnet_support_value, onvalue=1, offvalue=0)
    controlnet_support_check["state"] = NORMAL if convert_unet_value.get() == 1 else DISABLED
    controlnet_support_check.grid(row=current_row, column=3, padx=16, pady=4, sticky='W')
    current_row += 1
        
    convert_text_encoder_value = IntVar(window)
    convert_text_encoder_value.set(1)
    convert_text_encoder_value.set(1 if preferences['convert_text_encoder'] else 0)
    convert_text_encoder_check = Checkbutton(window, text='Text encoder',variable=convert_text_encoder_value, onvalue=1, offvalue=0)
    convert_text_encoder_check.grid(row=current_row, column=1, padx=16, pady=4, sticky='W')
    convert_encoder_value = IntVar(window)
    convert_encoder_value.set(1)
    convert_encoder_value.set(1 if preferences['convert_vae_encoder'] else 0)
    convert_encoder_check = Checkbutton(window, text='Encoder',variable=convert_encoder_value, onvalue=1, offvalue=0)
    convert_encoder_check.grid(row=current_row, column=2, padx=16, pady=4, sticky='W')
    convert_decoder_value = IntVar(window)
    convert_decoder_value.set(1)
    convert_decoder_value.set(1 if preferences['convert_vae_decoder'] else 0)
    convert_decoder_check = Checkbutton(window, text='Decoder',variable=convert_decoder_value, onvalue=1, offvalue=0)
    convert_decoder_check.grid(row=current_row, column=3, padx=16, pady=4, sticky='W')
    current_row += 1
    
    convert_safety_checker_value = IntVar(window)
    convert_safety_checker_value.set(1)
    convert_safety_checker_value.set(1 if preferences['convert_safety_checker'] else 0)
    convert_safety_checker_check = Checkbutton(window, text='Safety checker',variable=convert_safety_checker_value, onvalue=1, offvalue=0)
    convert_safety_checker_check.grid(row=current_row, column=1, padx=16, pady=4, sticky='W')
    
    def select_none_action():
        convert_unet_check.deselect()
        convert_unet_switched()
        convert_text_encoder_check.deselect()
        convert_encoder_check.deselect()
        convert_decoder_check.deselect()
        convert_safety_checker_check.deselect()
    select_none_button = Button(window, text="Select none", command=select_none_action)
    select_none_button.grid(row=current_row, column=2, padx=16, pady=4, sticky='E')
    def select_all_action():
        convert_unet_check.select()
        convert_unet_switched()
        convert_text_encoder_check.select()
        convert_encoder_check.select()
        convert_decoder_check.select()
        convert_safety_checker_check.select()
    select_all_button = Button(window, text="Select all", command=select_all_action)
    select_all_button.grid(row=current_row, column=3, padx=16, pady=4, sticky='E')
    current_row += 1
    
    # controlnet
    controlnet_version_label = Label(window, text="ControlNet version (not needed to add support):", justify="left")
    controlnet_version_label.grid(row=current_row, column=0, columnspan=5, padx=16, pady=(24, 0), sticky='W')
    current_row += 1
    
    controlnet_version_entry = PlaceholderEntry(window, placeholder="Empty for no ControlNet (lllyasviel/sd-controlnet-depth)")
    controlnet_version_entry.grid(row=current_row, column=0, columnspan=5, padx=16, pady=4, sticky='WE')
    current_row += 1
    
    # output size
    output_size_label = Label(window, text="Output size:")
    output_size_label.grid(row=current_row, column=0, columnspan=5, padx=16, pady=(12, 0), sticky='W')
    current_row += 1
        
    width_entry = PlaceholderEntry(window, placeholder="Width")
    width_entry.grid(row=current_row, column=1, padx=(16,0), pady=4, sticky='E')
    sizedivider_label = Label(window, text="×")
    sizedivider_label.grid(row=current_row, column=2, padx=6)
    height_entry = PlaceholderEntry(window, placeholder="Height")
    height_entry.grid(row=current_row, column=3, padx=(0,16), pady=4, sticky='W')
    current_row += 1
    
    # compute units
    selected_compute_unit = StringVar(window)
    selected_compute_unit.set(preferences['compute_unit'])
    compute_units_label = Label(window, text="Compute units:")
    compute_units_label.grid(row=current_row, column=0, columnspan=5, padx=16, pady=(12, 0), sticky='W')
    current_row += 1
    
    cpu_and_ne_check = Radiobutton(window, text='CPU and NE', variable=selected_compute_unit, value='CPU_AND_NE')
    if selected_compute_unit.get() == 'CPU_AND_NE':
        cpu_and_ne_check.select()
    cpu_and_ne_check.grid(row=current_row, column=1, padx=16, pady=4, sticky='W')
    cpu_and_gpu_check = Radiobutton(window, text='CPU and GPU', variable=selected_compute_unit, value='CPU_AND_GPU')
    if selected_compute_unit.get() == 'CPU_AND_GPU':
        cpu_and_gpu_check.select()
    cpu_and_gpu_check.grid(row=current_row, column=2, padx=16, pady=4, sticky='W')
    all_compute_units_check = Radiobutton(window, text='All', variable=selected_compute_unit, value='ALL')
    if selected_compute_unit.get() == 'ALL':
        all_compute_units_check.select()
    all_compute_units_check.grid(row=current_row, column=3, padx=16, pady=4, sticky='W')
    current_row += 1
    
            
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
        safetensors_model_check["state"] = DISABLED if is_converting else NORMAL
        
        convert_unet_check["state"] = DISABLED if is_converting else NORMAL
        chunk_unet_check["state"] = DISABLED if is_converting else NORMAL if convert_unet_value.get() == 1 else DISABLED
        controlnet_support_check["state"] = DISABLED if is_converting else NORMAL if convert_unet_value.get() == 1 else DISABLED
        convert_text_encoder_check["state"] = DISABLED if is_converting else NORMAL
        convert_encoder_check["state"] = DISABLED if is_converting else NORMAL
        convert_decoder_check["state"] = DISABLED if is_converting else NORMAL
        convert_safety_checker_check["state"] = DISABLED if is_converting else NORMAL
        select_none_button["state"] = DISABLED if is_converting else NORMAL
        select_all_button["state"] = DISABLED if is_converting else NORMAL
                                
        controlnet_version_entry["state"] = DISABLED if is_converting else NORMAL
        width_entry["state"] = DISABLED if is_converting else NORMAL
        height_entry["state"] = DISABLED if is_converting else NORMAL
        
        cpu_and_ne_check["state"] = DISABLED if is_converting else NORMAL
        cpu_and_gpu_check["state"] = DISABLED if is_converting else NORMAL
        all_compute_units_check["state"] = DISABLED if is_converting else NORMAL
        
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
        if not model_version:
            model_version = preferences['last_model_version']
        if ckpt_location:
            model_version = os.path.basename(os.path.normpath(ckpt_location))
            # remove extension
            model_version = os.path.splitext(model_version)[0]
        if model_location:
            model_version = os.path.basename(os.path.normpath(model_location))
        
        if not model_version:
            mb.showerror(title = "Error", message = "You need to choose a model to convert")
            return
            
        controlnet_version = controlnet_version_entry.get().strip()
        
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
        
        output_folder = filedialog.askdirectory(parent=window, initialdir=preferences['last_output_folder'], title='Where do you want to save the model?')
        if not output_folder.strip():
            return
        
        args = Namespace(
            attention_implementation='ORIGINAL' if selected_compute_unit.get() == 'CPU_AND_GPU' else 'SPLIT_EINSUM',
            bundle_resources_for_guernika=True,
            check_output_correctness=False,
            chunk_unet=chunk_unet_value.get() == 1,
            controlnet_support=controlnet_support_value.get() == 1,
            convert_safety_checker=convert_safety_checker_value.get() == 1,
            convert_text_encoder=convert_text_encoder_value.get() == 1,
            convert_unet=convert_unet_value.get() == 1,
            convert_vae_decoder=convert_decoder_value.get() == 1,
            convert_vae_encoder=convert_encoder_value.get() == 1,
            controlnet_version=None if not controlnet_version else controlnet_version,
            output_h=output_h,
            output_w=output_w,
            compute_unit=selected_compute_unit.get(),
            model_location=None if not model_location else model_location,
            from_safetensors=safetensors_model_value.get() == 1,
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
        save_args_preferences(args)
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
