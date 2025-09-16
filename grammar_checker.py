import tkinter as tk
from tkinter import scrolledtext, filedialog, messagebox
from threading import Thread
import pyttsx3
import difflib
import pyperclip
from docx import Document
from ollama import chat
import re
from PIL import Image, ImageTk  # For UALR logo

# ---------------- Globals ---------------- #
engine = None
is_reading = False


# ---------------- Grammar Checker Logic ---------------- #
def run_correction(input_text, language="English", style="Default"):
    prompt = (
        f"Correct the grammar and spelling of the following {language} text. "
        f"Return only the corrected text in a {style.lower()} style.\n\nText: \"{input_text}\""
    )
    response = chat(
        model="gemma3",  # Change if you use a different Ollama model
        messages=[{"role": "user", "content": prompt}]
    )
    return response['message']['content'].strip()


# ---------------- Stats Functions ---------------- #
def count_words(text):
    words = re.findall(r"\w+", text)
    return len(words)


def flesch_kincaid(text):
    sentences = max(1, text.count('.') + text.count('!') + text.count('?'))
    words = count_words(text)
    syllables = sum(count_syllables(word) for word in re.findall(r"\w+", text))

    if words == 0:
        return 0
    return round(0.39 * (words / sentences) + 11.8 * (syllables / words) - 15.59, 2)


def count_syllables(word):
    word = word.lower()
    vowels = "aeiouy"
    count = 0
    prev_char_was_vowel = False
    for char in word:
        if char in vowels:
            if not prev_char_was_vowel:
                count += 1
            prev_char_was_vowel = True
        else:
            prev_char_was_vowel = False
    if word.endswith("e"):
        count = max(1, count - 1)
    return max(1, count)


# ---------------- GUI Functions ---------------- #
def correct_grammar_threaded():
    def task():
        input_text = input_box.get("1.0", tk.END).strip()
        if not input_text:
            return
        status_label.config(text="Processing...")
        try:
            lang = language_var.get()
            style = style_var.get()
            corrected_text = run_correction(input_text, language=lang, style=style)
        except Exception as e:
            corrected_text = f"Error: {e}"

        output_box.delete("1.0", tk.END)
        output_box.insert(tk.END, corrected_text)
        highlight_corrections(input_text, corrected_text)

        words = count_words(corrected_text)
        grade = flesch_kincaid(corrected_text)
        stats_label.config(text=f"Word Count: {words} | Grade Level: {grade}")
        status_label.config(text="")
    Thread(target=task).start()


def highlight_corrections(original, corrected):
    output_box.tag_delete("correction")
    output_box.tag_config("correction", foreground="red")
    original_words = original.split()
    corrected_words = corrected.split()
    s = difflib.SequenceMatcher(None, original_words, corrected_words)
    index = 0
    for tag, i1, i2, j1, j2 in s.get_opcodes():
        if tag != "equal":
            start_idx = f"1.0+{index}c"
            end_idx = f"1.0+{index + sum(len(w) + 1 for w in corrected_words[j1:j2])}c"
            output_box.tag_add("correction", start_idx, end_idx)
        index += sum(len(w) + 1 for w in corrected_words[j1:j2])


def clear_text():
    input_box.delete("1.0", tk.END)
    output_box.delete("1.0", tk.END)
    stats_label.config(text="Word Count: 0 | Grade Level: 0")


def copy_output():
    pyperclip.copy(output_box.get("1.0", tk.END).strip())
    messagebox.showinfo("Copied", "Corrected text copied to clipboard!")


def save_output():
    corrected = output_box.get("1.0", tk.END).strip()
    if not corrected:
        return
    file_path = filedialog.asksaveasfilename(defaultextension=".txt",
                                             filetypes=[("Text files", "*.txt"), ("Word files", "*.docx")])
    if file_path.endswith(".txt"):
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(corrected)
    elif file_path.endswith(".docx"):
        doc = Document()
        doc.add_paragraph(corrected)
        doc.save(file_path)


# ---------------- TTS Functions ---------------- #
def read_aloud():
    global engine, is_reading
    corrected = output_box.get("1.0", tk.END).strip()
    if corrected and not is_reading:
        engine = pyttsx3.init()
        is_reading = True
        Thread(target=lambda: tts_thread(corrected)).start()


def tts_thread(text):
    global engine, is_reading
    engine.say(text)
    engine.runAndWait()
    is_reading = False


def pause_reading():
    global engine
    if engine:
        engine.stop()


def stop_reading():
    global engine, is_reading
    if engine:
        engine.stop()
        engine = None
    is_reading = False


# ---------------- GUI Setup ---------------- #
root = tk.Tk()
root.title("Advanced Grammar Checker")
root.geometry("1100x650")

# Frames
top_frame = tk.Frame(root)
top_frame.pack(side=tk.TOP, fill=tk.X, pady=5)

text_frame = tk.Frame(root)
text_frame.pack(fill=tk.BOTH, expand=True)

bottom_frame = tk.Frame(root)
bottom_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=5)

# Input (Left)
input_label = tk.Label(text_frame, text="Input Text:")
input_label.grid(row=0, column=0, sticky="w")
input_box = scrolledtext.ScrolledText(text_frame, width=60, height=20)
input_box.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")

# Output (Right)
output_label = tk.Label(text_frame, text="Corrected Text:")
output_label.grid(row=0, column=1, sticky="w")
output_box = scrolledtext.ScrolledText(text_frame, width=60, height=20)
output_box.grid(row=1, column=1, padx=5, pady=5, sticky="nsew")

# Allow resizing
text_frame.columnconfigure(0, weight=1)
text_frame.columnconfigure(1, weight=1)

# Buttons
tk.Button(top_frame, text="Correct Grammar", command=correct_grammar_threaded).pack(side=tk.LEFT, padx=5)
tk.Button(top_frame, text="Clear", command=clear_text).pack(side=tk.LEFT, padx=5)
tk.Button(top_frame, text="Copy", command=copy_output).pack(side=tk.LEFT, padx=5)
tk.Button(top_frame, text="Save", command=save_output).pack(side=tk.LEFT, padx=5)

# TTS Controls (arranged vertically like media buttons)
tts_frame = tk.Frame(root)
tts_frame.pack(side=tk.LEFT, padx=20)
tts_frame.pack(pady=10)

tk.Button(tts_frame, text="▶", font=("Arial", 12), fg="blue" ,command=read_aloud).pack(pady=2)   # Play
tk.Button(tts_frame, text="⏸", font=("Arial", 12), fg="yellow", command=pause_reading).pack(pady=2)  # Pause
tk.Button(tts_frame, text="⏹", font=("Arial", 12), fg="red", command=stop_reading).pack(pady=2)  # Stop

# Language & Style dropdown
language_var = tk.StringVar(value="English")
style_var = tk.StringVar(value="Default")
tk.Label(top_frame, text="Language:").pack(side=tk.LEFT, padx=5)
tk.OptionMenu(top_frame, language_var, "English", "Spanish", "French", "German").pack(side=tk.LEFT)
tk.Label(top_frame, text="Style:").pack(side=tk.LEFT, padx=5)
tk.OptionMenu(top_frame, style_var, "Default", "Formal", "Academic", "Casual", "Professional").pack(side=tk.LEFT)

# Status + Stats
status_label = tk.Label(bottom_frame, text="", fg="blue")
status_label.pack(anchor="w")
stats_label = tk.Label(bottom_frame, text="Word Count: 0 | Grade Level: 0", fg="darkgreen")
stats_label.pack(anchor="w")

# UALR Logo (bottom right)
try:
    logo_img = Image.open("ualr_logo1.png")   # Ensure this file exists in the same folder
    logo_img = logo_img.resize((300, 100), Image.Resampling.LANCZOS)
    logo_photo = ImageTk.PhotoImage(logo_img)

    logo_label = tk.Label(bottom_frame, image=logo_photo)
    logo_label.image = logo_photo  # Keep reference
    logo_label.pack(side=tk.RIGHT, padx=10, anchor="e")
except Exception as e:
    print("Logo not loaded:", e)

root.mainloop()
