d = 5

def click(button_text):
    current = entry.get()
    if button_text == "=":
        try:
            result = str(eval(current))  # Evaluate the expression
            entry.delete(0, tk.END)
            entry.insert(tk.END, result)
        except Exception:
            entry.delete(0, tk.END)
            entry.insert(tk.END, "Error")
    elif button_text == "C":
        entry.delete(0, tk.END)  # Clear input
    else:
        entry.insert(tk.END, button_text)  # Add button text to entry

# Create main window
root = tk.Tk()
root.title("Calculator")
root.geometry("300x400")

# Entry widget for display
entry = tk.Entry(root, font=("Arial", 20), borderwidth=5, relief="ridge", justify="right")
entry.pack(fill=tk.BOTH, ipadx=8, ipady=8, pady=10)

# Buttons layout
buttons = [
    ["7", "8", "9", "/"],
    ["4", "5", "6", "*"],
    ["1", "2", "3", "-"],
    ["0", ".", "=", "+"],
    ["C"]
]

# Create buttons in a grid
frame = tk.Frame(root)
frame.pack()

for row in buttons:
    row_frame = tk.Frame(frame)
    row_frame.pack(expand=True, fill="both")
    for btn_text in row:
        button = tk.Button(row_frame, text=btn_text, font=("Arial", 18), height=2, width=5,
                           command=lambda b=btn_text: click(b))
        button.pack(side="left", expand=True, fill="both")

root.mainloop()
