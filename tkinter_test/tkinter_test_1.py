import tkinter as tk
top = tk.Tk()
li = ['C','python','php','html','SQL','java']
movie = ['CSS','jQuery','Bootstrap']
listb = tk.Listbox(top)
listb2 = tk.Listbox(top)

for item in li:
    listb.insert(0, item)

for item in movie:
    listb2.insert(0, item)

listb.pack()
listb2.pack()
top.mainloop()
