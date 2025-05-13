# Neuronal-response-to-high-frequency-electric-field-stimulation

Code used in "Neuronal response to high-frequency electric field stimulation" master's thesis.

## Code overview:

| Filename                           | Description                                                                                                   |
|------------------------------------|---------------------------------------------------------------------------------------------------------------|
| run_simulations.py                 | - Simulation code for all neocortical neuron models stimulated with external electric field.                 |
| run_simulations_simplified_neuron_models.py | - Simulation code for idealized neuron models stimulated with external electric field.                |
| Plots_neocortical.ipynb | - Loads simulation data for neocortical neuron models (From run_simulations.py)  <br> - Plot functions for Figures 5, 9, 14–19, C.4–14. <br> - Calculations for Tables 2 and C.1. |
| Plots_idealized_models.ipynb | - Loads simulation data for idealized neuron models (From run_simulations_simplified_neuron_models.py) <br> - Plot functions for Figures 8, 12, C.2, and C.3. |
| White_noise.ipynb                  | - White noise current and extracellular stimulation to three selected neurons. <br> - Plot functions for Figures 20 and 21. |
| Lambda_ac.ipynb                    | - Calculates frequency-dependent length constants. <br> - Plot function for Figure 13.                       |
| Hay_neuron.ipynb                   | - Simulations for active and passive Hay neuron. <br> - Plot functions for Figures 10, 11, and C.1.         |


## Figure overview
| Figure number  | Code avalible in       |
|------------------|----------------------------------------|
| 5                | Plots_neocortical.ipynb |
| 8                | Plots_idealized_models.ipynb   |
| 9                | Plots_neocortical.ipynb |
| 10, 11           | Hay_neuron.ipynb                      |
| 12               | Plots_idealized_models.ipynb   |
| 13               | Lambda_ac.ipynb                       |
| 14–19            | Plots_neocortical.ipynb |
| 20, 21           | White_noise.ipynb                     |
| C.1              | Hay_neuron.ipynb                      |
| C.2, C.3         | Plots_idealized_models.ipynb   |
| C.4–C.14         | Plots_neocortical.ipynb |
