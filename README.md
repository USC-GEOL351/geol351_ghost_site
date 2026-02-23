# GEOL351 Student Workspace (Spring 2026)

This repository is your **personal workspace for GEOL351**.  
You will keep it for the entire semester.

You will:

- view lab instructions (HTML) and lab notebooks (jupyter or quarto)
- keep notes or notebooks from labs
- develop your final project

You will **not** submit work through GitHub unless explicitly asked.

---

## What this repository is for

This repo will have three main folders:

```
lab_site/            # lab instructions (pulled from the instructor)
work/            # your lab notes, notebooks, figures, scratch work
final_project/   # your final project materials
```

### Important rule
**Do not edit anything inside `lab_site/`.**

Labs are distributed from Github Classroom and updated centrally.  

---


## Weekly workflow

### Pull materials
Each time a new lab is released or whenever there are updates to existing lab materials, you will need to open a command line prompt from your repo directory and type:

```
git pull upstream main
```

If you open your lab directory in Positron, the command line will likely be situated in the right place. If you aren't in the right place, this will throw an error.

### Copy lab folder (not lab_site) to `work/`
Then you will need to copy the lab# folder to `work/`. Delete the html files and the deliverable.md file. This should leave the jupyter notebook(s) (`ipynb` files), the `deliverable.qmd`, and any folders (e.g. `data` or `support_code`) inside the lab folder. 


### Work through the lab
Right clicking the lab_site/index.html file, selecting "open in browser" (or similar depending on your operating system), and navigate to the lab of interest. 

- Setup: You will want to have the lab_worksheet open in the viewer (Positron seems to have this functionality), and the jupyter notebook open in the adjacent window. 
- Execution: You will work through them together, taking notes as you like in the worksheet. 
- Saving notes: click the button at the top of the worksheet that says "download my notes" and add them to your `work/lab0t` folder
- Save (1): go to GitHub desktop, add a summary of what you did, click commit, then Push
- Deliverable: Open the `deliverable.qmd` file in Positron. Edit it (I reccommend using the "visual" editor, but your choice), then save it. 
- Render deliverable: in the terminal, type `quarto render lab#/lab#-deliverable.qmd` to make the deliverable.qmd file into a PDF
- Submit: submit your PDF in brightspace
- Save (2): go to GitHub desktop, add a summary of what you did, click commit, then Push

