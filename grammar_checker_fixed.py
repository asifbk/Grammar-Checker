import tkinter as tk
from tkinter import scrolledtext, filedialog, messagebox, ttk
from threading import Thread
import difflib
import pyperclip
from docx import Document
try:
    from ollama import chat
except Exception:
    chat = None
import re
from PIL import Image, ImageTk
import platform
import tempfile
import os
import subprocess
import time
from datetime import datetime
import numpy as np
from collections import Counter
import csv

# ---------------- Globals ---------------- #
start_time = None
data_log = []

AVAILABLE_MODELS = ["llama3.2", "mistral", "deepseek-r1:7b", "gemma3"]


# --- Metrics Functions ---
def calculate_gleu(reference, hypothesis, n=4):
    def get_ngrams(text, n):
        words = text.split()
        ngrams = []
        for i in range(len(words) - n + 1):
            ngrams.append(tuple(words[i:i + n]))
        return Counter(ngrams)
    ref_ngrams = get_ngrams(reference, n)
    hyp_ngrams = get_ngrams(hypothesis, n)
    match = sum(min(hyp_ngrams[g], ref_ngrams[g]) for g in hyp_ngrams if g in ref_ngrams)
    total = sum(hyp_ngrams.values())
    return match / total if total else 0.0

def calculate_err(original, corrected):
    sm = difflib.SequenceMatcher(None, original.split(), corrected.split())
    edits = sum(max(i2-i1, j2-j1) for tag,i1,i2,j1,j2 in sm.get_opcodes() if tag!='equal')
    return (edits / max(1,len(original.split()))) * 100

def calculate_f05(p, r):
    b2 = 0.25
    return (1+b2)*p*r/((b2*p)+r) if p+r>0 else 0

def count_grammar_issues(t):
    issues=0
    if re.search(r'\bi\s+',t.lower()):issues+=1
    if re.search(r'\b(he|she|it)\s+(go|have|do)\b',t.lower()):issues+=1
    if re.search(r'\b(we|they|you)\s+(goes|has|does)\b',t.lower()):issues+=1
    return issues

def calculate_grade_score(o,c):
    s=0
    if len(re.split(r'[.!?]+',c))>=len(re.split(r'[.!?]+',o)):s+=.2
    wr=min(len(o.split()),len(c.split()))/max(1,max(len(o.split()),len(c.split())))
    s+=wr*.2
    if len(re.findall(r'[,.!?;:]',c))>=len(re.findall(r'[,.!?;:]',o)):s+=.2
    if len(re.findall(r'\b[A-Z][a-z]+\b',c))>=len(re.findall(r'\b[A-Z][a-z]+\b',o)):s+=.2
    if count_grammar_issues(c)<count_grammar_issues(o):s+=.2
    return min(s,1.0)

def calculate_comprehensive_score(o,c,r=None):
    if r is None:r=o
    oc,cc=Counter(o.split()),Counter(c.split())
    m=sum((oc&cc).values())
    p=m/max(1,len(c.split()));rec=m/max(1,len(o.split()))
    f05=calculate_f05(p,rec);grade=calculate_grade_score(o,c)
    gleu=calculate_gleu(r,c)
    comp=(f05*.4+gleu*.3+grade*.3)*100
    return{'f05':f05,'gleu':gleu,'err':calculate_err(o,c),'grade_score':grade,'composite_score':comp,'word_count':len(c.split())}

def get_installed_models():
    try:
        r=subprocess.run(["ollama","list"],capture_output=True,text=True,timeout=8)
        if r.returncode==0:
            return [l.split()[0] for l in r.stdout.strip().split('\n')[1:] if l.strip()]
    except Exception:pass
    return []

def refresh_model_list():
    def task():
        try:
            installed=get_installed_models()
            menu=model_menu["menu"];menu.delete(0,"end")
            for m in AVAILABLE_MODELS:
                lbl=m if any(m in i for i in installed) else f"{m} (Not installed)"
                menu.add_command(label=lbl,command=lambda v=lbl:model_var.set(v))
            root.after(0,lambda:status_label.config(text="Model list refreshed."))
        except Exception as e:
            root.after(0,lambda:status_label.config(text="Failed to refresh models."))
    Thread(target=task,daemon=True).start()

def run_correction(t,language="English",style="Default"):
    if chat is None:raise Exception("Ollama client not available.")
    prompt=f"""Correct ONLY the grammar, spelling and punctuation of the following {language} text. 
Return ONLY the corrected text without any introductory phrases like "Here is the corrected version" or explanations.
Maintain the {style.lower()} style.

Original text: \"{t}\"

Corrected text:"""
    m=model_var.get().split(" (Not installed)")[0]
    r=chat(model=m,messages=[{"role":"user","content":prompt}],options={"timeout":45})
    return r['message']['content'].strip()

def count_words(t):return len(re.findall(r'\w+',t))

def flesch_kincaid(t):
    s=max(1,t.count('.')+t.count('!')+t.count('?'))
    w=count_words(t)
    y=sum(count_syllables(wd) for wd in re.findall(r'\w+',t))
    return round(0.39*(w/s)+11.8*(y/w)-15.59,2) if w else 0

def count_syllables(w):
    w=w.lower();v="aeiouy";c=0;p=False
    for ch in w:
        if ch in v:
            if not p:c+=1
            p=True
        else:p=False
    if w.endswith('e'):c=max(1,c-1)
    return max(1,c)

# --------- Optimized Highlight Function ---------
def highlight_corrections(original, corrected):
    """Efficiently highlight only changed words in long texts."""
    output_box.tag_delete("correction")
    output_box.tag_config("correction", foreground="red")

    # Precompute corrected words with spans
    words = list(re.finditer(r'\S+', corrected))
    word_spans = [(m.start(), m.end()) for m in words]

    # Use character-level SequenceMatcher
    s = difflib.SequenceMatcher(None, original, corrected)
    for tag, i1, i2, j1, j2 in s.get_opcodes():
        if tag == 'equal':
            continue
        # Binary search optimization: only check words overlapping changed range
        for start, end in word_spans:
            if end <= j1:
                continue
            if start >= j2:
                break
            output_box.tag_add("correction", f"1.0+{start}c", f"1.0+{end}c")


def add_to_data_log(metrics, processing_time, language, style, model):
    """Add current correction to data log"""
    global data_log
    entry = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'words': metrics['word_count'],
        'time': f"{processing_time:.2f}",
        'f05': f"{metrics['f05']:.3f}",
        'err': f"{metrics['err']:.2f}%",
        'gleu': f"{metrics['gleu']:.3f}",
        'grade': f"{metrics['grade_score']:.3f}",
        'score': f"{metrics['composite_score']:.1f}%",
        'language': language,
        'style': style,
        'model': model
    }
    data_log.append(entry)
    update_data_table()

def update_data_table():
    """Update the data table with all logged entries"""
    # Clear existing data
    for row in data_tree.get_children():
        data_tree.delete(row)
    
    # Add all entries in reverse order (newest first)
    for entry in reversed(data_log):
        data_tree.insert("", "end", values=(
            entry['words'],
            entry['time'],
            entry['f05'],
            entry['err'],
            entry['gleu'],
            entry['grade'],
            entry['score'],
            entry['language'],
            entry['style'],
            entry['model']
        ))

def clear_data_log():
    """Clear all data from the log"""
    global data_log
    data_log = []
    update_data_table()
    messagebox.showinfo("Data Log", "All data cleared from log.")

def export_data_log():
    """Export data log to CSV file"""
    if not data_log:
        messagebox.showwarning("Export", "No data to export.")
        return
    
    filename = filedialog.asksaveasfilename(
        defaultextension=".csv",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        title="Export Data Log"
    )
    
    if filename:
        try:
            with open(filename, 'w', newline='', encoding='utf-8') as file:
                writer = csv.DictWriter(file, fieldnames=[
                    'timestamp', 'words', 'time', 'f05', 'err', 'gleu', 
                    'grade', 'score', 'language', 'style', 'model'
                ])
                writer.writeheader()
                writer.writerows(data_log)
            messagebox.showinfo("Export Successful", f"Data exported to {filename}")
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export data: {str(e)}")

def correct_grammar_threaded():
    def task():
        global start_time;start_time=time.time()
        text=input_box.get("1.0",tk.END).strip()
        if not text:
            messagebox.showwarning("Warning","Please enter some text.");return
        root.after(0,lambda:status_label.config(text="Processing..."));root.after(0,progress.start)
        try:
            lang = language_var.get()
            styl = style_var.get()
            model_used = model_var.get().split(" (Not installed)")[0]
            corr=run_correction(text, lang, styl)
            met=calculate_comprehensive_score(text,corr)
            processing_time = time.time()-start_time
            add_to_data_log(met, processing_time, lang, styl, model_used)
            root.after(0,update_output,corr,text,processing_time,met)
        except Exception as e:
            root.after(0,show_error,f"Correction failed: {e}",time.time()-start_time)
        finally:root.after(0,progress.stop)
    Thread(target=task,daemon=True).start()

def update_output(c,o,t,m):
    output_box.delete("1.0",tk.END);output_box.insert(tk.END,c)
    highlight_corrections(o, c)
    
    # Use superscript for F⁰·⁵
    stats_label.config(text=f"Words:{m['word_count']} | Time:{t:.2f}s | F⁰·⁵:{m['f05']:.3f} | ERR:{m['err']:.2f}% | GLEU:{m['gleu']:.3f} | Grade:{m['grade_score']:.3f} | Score:{m['composite_score']:.1f}%")
    status_label.config(text="Done.")

def show_error(m,t=0):messagebox.showerror("Error",m)

def clear_text():
    input_box.delete("1.0",tk.END);output_box.delete("1.0",tk.END)
    output_box.tag_delete("correction")
    stats_label.config(text="Words:0 | Time:0.00s | F⁰·⁵:0.000 | ERR:0.00% | GLEU:0.000 | Grade:0.000 | Score:0.0%")

def copy_output():
    t=output_box.get("1.0",tk.END).strip()
    if not t:messagebox.showwarning("Warning","No text to copy!");return
    pyperclip.copy(t);messagebox.showinfo("Copied","Copied!")

def save_output():
    t=output_box.get("1.0",tk.END).strip()
    if not t:return messagebox.showwarning("Warning","No text to save!")
    f=filedialog.asksaveasfilename(defaultextension=".txt",filetypes=[("Text","*.txt"),("Word","*.docx")])
    if not f:return
    try:
        if f.endswith('.txt'):open(f,'w',encoding='utf-8').write(t)
        else:doc=Document();doc.add_paragraph(t);doc.save(f)
        messagebox.showinfo("Saved","File saved.")
    except Exception as e:messagebox.showerror("Error",str(e))

def swap_text():
    i=input_box.get("1.0",tk.END).strip();o=output_box.get("1.0",tk.END).strip()
    input_box.delete("1.0",tk.END);input_box.insert(tk.END,o)
    output_box.delete("1.0",tk.END);output_box.insert(tk.END,i)
    output_box.tag_delete("correction")

def load_file():
    f=filedialog.askopenfilename(filetypes=[("Text","*.txt"),("Word","*.docx"),("All","*.*")])
    if not f:return
    try:
        if f.endswith('.docx'):
            doc=Document(f);t='\n'.join(p.text for p in doc.paragraphs)
        else:t=open(f,'r',encoding='utf-8').read()
        input_box.delete("1.0",tk.END);input_box.insert(tk.END,t)
    except Exception as e:messagebox.showerror("Error",str(e))

# Create main window
root=tk.Tk();root.title("UALR Grammar Checker with Data Logger");root.geometry("1400x750")

# Create notebook for tabs
notebook = ttk.Notebook(root)
notebook.pack(fill='both', expand=True, padx=10, pady=10)

# Tab 1: Grammar Checker
tab1 = ttk.Frame(notebook)
notebook.add(tab1, text='Grammar Checker')

# Tab 2: Data Logger
tab2 = ttk.Frame(notebook)
notebook.add(tab2, text='Data Logger')

# Move all existing GUI elements to tab1
top_frame=tk.Frame(tab1);top_frame.pack(fill=tk.X)
controls_frame=tk.Frame(tab1);controls_frame.pack(fill=tk.X)
text_frame=tk.Frame(tab1);text_frame.pack(fill=tk.BOTH,expand=True)
bottom_frame=tk.Frame(tab1);bottom_frame.pack(fill=tk.X)

model_var=tk.StringVar(value=AVAILABLE_MODELS[0])
tk.Label(controls_frame,text="AI Model:").pack(side=tk.LEFT,padx=5)
model_menu=tk.OptionMenu(controls_frame,model_var,*AVAILABLE_MODELS)
model_menu.pack(side=tk.LEFT,padx=5)

tk.Button(controls_frame,text="🔄 Refresh Models",command=refresh_model_list).pack(side=tk.LEFT,padx=5)

def force_models_installed():
    model_var.set(AVAILABLE_MODELS[0]);messagebox.showinfo("Manual Override","Model set.")

tk.Button(controls_frame,text="🔧 Manual Override",command=force_models_installed).pack(side=tk.LEFT,padx=5)

language_var=tk.StringVar(value="English");style_var=tk.StringVar(value="Default")
tk.Label(controls_frame,text="Language:").pack(side=tk.LEFT,padx=5)
tk.OptionMenu(controls_frame,language_var,"English","Spanish","French","German","Italian","Portuguese","Dutch","Russian","Japanese","Chinese","Korean","Arabic","Hindi","Bengali").pack(side=tk.LEFT)
tk.Label(controls_frame,text="Style:").pack(side=tk.LEFT,padx=5)
tk.OptionMenu(controls_frame,style_var,"Default","Formal","Academic","Casual","Professional").pack(side=tk.LEFT)

input_label=tk.Label(text_frame,text="Input Text:");input_label.grid(row=0,column=0,sticky="w")
input_box=scrolledtext.ScrolledText(text_frame,width=60,height=20);input_box.grid(row=1,column=0,padx=5,pady=5,sticky="nsew")
output_label=tk.Label(text_frame,text="AI Corrected Text:");output_label.grid(row=0,column=1,sticky="w")
output_box=scrolledtext.ScrolledText(text_frame,width=60,height=20);output_box.grid(row=1,column=1,padx=5,pady=5,sticky="nsew")
text_frame.columnconfigure(0,weight=1);text_frame.columnconfigure(1,weight=1)

# Buttons without TTS
tk.Button(top_frame,text="Correct Grammar",command=correct_grammar_threaded).pack(side=tk.LEFT,padx=5)
tk.Button(top_frame,text="Clear",command=clear_text).pack(side=tk.LEFT,padx=5)
tk.Button(top_frame,text="Swap Text",command=swap_text).pack(side=tk.LEFT,padx=5)
tk.Button(top_frame,text="Copy",command=copy_output).pack(side=tk.LEFT,padx=5)
tk.Button(top_frame,text="Save",command=save_output).pack(side=tk.LEFT,padx=5)
tk.Button(top_frame,text="Load File",command=load_file).pack(side=tk.LEFT,padx=5)

progress=ttk.Progressbar(bottom_frame,mode='indeterminate');progress.pack(fill=tk.X,padx=5,pady=2)
status_label=tk.Label(bottom_frame,text="Ready. Click 'Refresh Models' if needed.",fg="blue");status_label.pack(anchor="w")

# Use superscript for F⁰·⁵ in initial stats
stats_label=tk.Label(bottom_frame,text="Words:0 | Time:0.00s | F⁰·⁵:0.000 | ERR:0.00% | GLEU:0.000 | Grade:0.000 | Score:0.0%",fg="darkgreen",font=("Cambria",10,"bold"));stats_label.pack(anchor="w")

try:
    img=Image.open("ualr_logo1.png");img=img.resize((300,100));photo=ImageTk.PhotoImage(img)
    lbl=tk.Label(bottom_frame,image=photo);lbl.image=photo;lbl.pack(side=tk.RIGHT,padx=10,anchor="e")
except Exception as e:print("Logo not loaded",e)

# Data Logger Tab (Tab2) setup
data_log_frame = tk.Frame(tab2)
data_log_frame.pack(fill='both', expand=True, padx=10, pady=10)

# Data log controls
data_controls = tk.Frame(data_log_frame)
data_controls.pack(fill='x', pady=(0, 10))

tk.Button(data_controls, text="Clear Data", command=clear_data_log).pack(side=tk.LEFT, padx=5)
tk.Button(data_controls, text="Export to CSV", command=export_data_log).pack(side=tk.LEFT, padx=5)

# Data table
columns = ('Words', 'Time', 'F⁰·⁵', 'ERR', 'GLEU', 'Grade', 'Score', 'Language', 'Style', 'Model')
data_tree = ttk.Treeview(data_log_frame, columns=columns, show='headings', height=15)

# Define headings
data_tree.heading('Words', text='No of Words')
data_tree.heading('Time', text='Time (s)')
data_tree.heading('F⁰·⁵', text='F⁰·⁵')
data_tree.heading('ERR', text='ERR (%)')
data_tree.heading('GLEU', text='GLEU')
data_tree.heading('Grade', text='Grade')
data_tree.heading('Score', text='Score (%)')
data_tree.heading('Language', text='Language')
data_tree.heading('Style', text='Writing Style')
data_tree.heading('Model', text='LLM Model')

# Set column widths
data_tree.column('Words', width=80)
data_tree.column('Time', width=80)
data_tree.column('F⁰·⁵', width=80)
data_tree.column('ERR', width=80)
data_tree.column('GLEU', width=80)
data_tree.column('Grade', width=80)
data_tree.column('Score', width=80)
data_tree.column('Language', width=100)
data_tree.column('Style', width=100)
data_tree.column('Model', width=120)

# Add scrollbar to data table
data_scrollbar = ttk.Scrollbar(data_log_frame, orient=tk.VERTICAL, command=data_tree.yview)
data_tree.configure(yscrollcommand=data_scrollbar.set)
data_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
data_tree.pack(side=tk.LEFT, fill='both', expand=True)

# Status label for data tab
data_status = tk.Label(
    data_log_frame, 
    text="Data log will automatically record each grammar correction.", 
    fg="blue"
)
data_status.pack(anchor='w', pady=(10, 0))


# --- Metric Definitions Section ---
metrics_frame = tk.LabelFrame(data_log_frame, text="Metric Definitions", padx=10, pady=10)
metrics_frame.pack(fill='x', pady=(10, 5))

# Text widget for formatting, color, and alignment
metrics_text_widget = tk.Text(
    metrics_frame,
    wrap="word",
    height=45,
    font=("Cambria", 10),
    relief="flat",
    bg=root.cget("bg"),
    padx=12,
    pady=8
)

# Define text tags for formatting
metrics_text_widget.tag_configure("title", foreground="#003366", font=("Cambria", 10, "bold"))
metrics_text_widget.tag_configure("formula", foreground="#444444", font=("Consolas", 10, "italic"))
metrics_text_widget.tag_configure("body", justify="left", lmargin1=20, lmargin2=35)

# --- Metric Descriptions with Equations ---
metrics_text_widget.insert("end", "• F⁰·⁵ Score (F0.5)\n", "title")
metrics_text_widget.insert(
    "end",
    "   Equation: F₀․₅ = (1 + 0.5²) × (P × R) / ((0.5² × P) + R)\n",
    "formula"
)
metrics_text_widget.insert(
    "end",
    "   A precision-weighted F-score that gives twice as much importance to precision as recall.\n"
    "   It measures how accurately and completely the corrected text matches the intended meaning.\n\n",
    "body"
)

metrics_text_widget.insert("end", "• ERR (%) – Error Rate Reduction\n", "title")
metrics_text_widget.insert(
    "end",
    "   Equation: ERR = (Edits / Total_Words) × 100\n",
    "formula"
)
metrics_text_widget.insert(
    "end",
    "   Represents the percentage of edits relative to the total number of words.\n"
    "   A lower ERR indicates fewer unnecessary or incorrect changes.\n\n",
    "body"
)

metrics_text_widget.insert("end", "• GLEU – Google’s Grammar/Translation Edit Metric\n", "title")
metrics_text_widget.insert(
    "end",
    "   Equation: GLEU = (Matching n-grams) / (Total n-grams)\n",
    "formula"
)
metrics_text_widget.insert(
    "end",
    "   Compares n-gram overlap between the corrected and reference text.\n"
    "   Higher GLEU values indicate better grammatical consistency and fluency.\n\n",
    "body"
)

metrics_text_widget.insert("end", "• Grade Score\n", "title")
metrics_text_widget.insert(
    "end",
    "   Evaluates sentence structure, punctuation, capitalization, and grammar improvements.\n"
    "   A higher score reflects more polished and natural writing.\n\n",
    "body"
)

metrics_text_widget.insert("end", "• Composite Score (%)\n", "title")
metrics_text_widget.insert(
    "end",
    "   Equation: Composite = (0.4 × F₀․₅) + (0.3 × GLEU) + (0.3 × Grade)\n",
    "formula"
)
metrics_text_widget.insert(
    "end",
    "   Represents the overall performance by combining accuracy (F-score), fluency (GLEU),\n"
    "   and stylistic refinement (Grade). A higher composite score indicates stronger correction quality.\n",
    "body"
)

metrics_text_widget.configure(state="disabled")
metrics_text_widget.pack(fill="x", expand=True)



root.mainloop()