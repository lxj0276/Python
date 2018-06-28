import tkinter as tk

window = tk.Tk()
window.title("my window")
window.geometry("500x500")

frame = tk.Frame(window)
frame.pack()

frame_load = tk.Frame(frame)
frame_load.pack()

frame_entry_begin = tk.Frame(frame_load)
frame_entry_begin.grid(row=1, column=0, padx=10, pady=10)
l1 = tk.Label(frame_entry_begin, font=("Arial", 10), height=5, text='开始日期')
l1.pack(side='left')
e1 = tk.Entry(frame_entry_begin, font=("Arial", 12))
e1.pack(side='left', padx=10)

frame_entry_end = tk.Frame(frame_load)
frame_entry_end.grid(row=2, column=0, padx=10, pady=10)
l2 = tk.Label(frame_entry_end, font=("Arial", 10), text='结束日期')
l2.pack(side='left')
e2 = tk.Entry(frame_entry_end, font=("Arial", 12))
e2.pack(side='left', padx=10)


window.mainloop()