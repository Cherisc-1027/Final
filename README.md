# Final assignment for Cognitive Modeling (COGS 107)

Using the code in `sdt_ddm.py1`, analyze the data in the file `data.csv`.

The code contains a function to read the data, a function to create a hierarchical SDT model, and a function to draw delta plots.  You will need to add code to run the analysis, and you may need to edit these functions to suit your purposes.

The data is choice response time data from a 2x2x2 experimental design (trial difficulty x stimulus type x signal presence).  Such data can be analyzed using Signal Detection Theory (SDT), or using a diffusion model.  Here we will do both (using delta plots for our diffusion model analysis) and compare the results.

The main questions pertain to the effect of the stimulus type (Simple vs Complex) and trial difficulty (Easy vs Hard) on different aspects of the participants' performance.  You will need to adapt the SDT model to quantify the effect of the stimulus type and trial difficulty on the participants' performance.  You will need to check convergence of the SDT model, and display (either in a figure or a table) the posterior distributions of the parameters.

During test time, you will be asked to interpret these results, including descriptive statistics of the data, including person-specific estimates, population-level estimates, convergence statistics, and about your general understanding of SDT and diffusion model parameters.

Be prepared to re-run your code while you are answering questions.