
import os
import sys
from os.path import join, expanduser

from glob import glob
import numpy as np

import neuron
import LFPy

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colorbar as colorbar

from IPython.display import display, HTML

# Add the parent directory of brainsignals to sys.path
parent_path = '/Users/susannedahle/Python/brainsignals'
sys.path.append(parent_path)

import brainsignals.neural_simulations as ns
from brainsignals.plotting_convention import mark_subplots
import csv

# Real neuron model
def return_BBP_neuron(cell_name, tstop, dt):

    # load some required neuron-interface files
    neuron.h.load_file("stdrun.hoc")
    neuron.h.load_file("import3d.hoc")

    CWD = os.getcwd()
    cell_folder = join(join(bbp_folder, cell_name))
    if not os.path.isdir(cell_folder):
        ns.download_BBP_model(cell_name)

    neuron.load_mechanisms(bbp_mod_folder)
    os.chdir(cell_folder)
    add_synapses = False
    # get the template name
    f = open("template.hoc", 'r')
    templatename = ns.get_templatename(f)
    f.close()

    # get biophys template name
    f = open("biophysics.hoc", 'r')
    biophysics = ns.get_templatename(f)
    f.close()

    # get morphology template name
    f = open("morphology.hoc", 'r')
    morphology = ns.get_templatename(f)
    f.close()

    # get synapses template name
    f = open(ns.posixpth(os.path.join("synapses", "synapses.hoc")), 'r')
    synapses = ns.get_templatename(f)
    f.close()

    neuron.h.load_file('constants.hoc')

    if not hasattr(neuron.h, morphology):
        """Create the cell model"""
        # Load morphology
        neuron.h.load_file(1, "morphology.hoc")
    if not hasattr(neuron.h, biophysics):
        # Load biophysics
        neuron.h.load_file(1, "biophysics.hoc")
    if not hasattr(neuron.h, synapses):
        # load synapses
        neuron.h.load_file(1, ns.posixpth(os.path.join('synapses', 'synapses.hoc')
                                       ))
    if not hasattr(neuron.h, templatename):
        # Load main cell template
        neuron.h.load_file(1, "template.hoc")

    templatefile = ns.posixpth(os.path.join(cell_folder, 'template.hoc'))

    morphologyfile = glob(os.path.join('morphology', '*'))[0]


    # Instantiate the cell(s) using LFPy
    cell = LFPy.TemplateCell(morphology=morphologyfile,
                             templatefile=templatefile,
                             templatename=templatename,
                             templateargs=1 if add_synapses else 0,
                             tstop=tstop,
                             dt=dt,
                             lambda_f = 500,
                             nsegs_method='lambda_f',
                             v_init = -65)
    os.chdir(CWD)
    # set view as in most other examples
    cell.set_rotation(x=np.pi / 2)
    return cell



# Simplified neuron models ----
def return_ball_and_stick_cell(tstop, dt, apic_diam=2):

    h("forall delete_section()")
    h("""
    proc celldef() {
      topol()
      subsets()
      geom()
      biophys()
      geom_nseg()
    }

    create soma[1], dend[1]

    proc topol() { local i
      basic_shape()
      connect dend(0), soma(1)
    }
    proc basic_shape() {
      soma[0] {pt3dclear()
      pt3dadd(0, 0, -10., 20.)
      pt3dadd(0, 0, 10., 20.)}
      dend[0] {pt3dclear()
      pt3dadd(0, 0, 10., %s)
      pt3dadd(0, 0, 1000, %s)}
    }

    objref all
    proc subsets() { local i
      objref all
      all = new SectionList()
        soma[0] all.append()
        dend[0] all.append()

    }
    proc geom() {
    }
    proc geom_nseg() {
    soma[0] {nseg = 1}
    dend[0] {nseg = 1000}
    }
    proc biophys() {
    }
    celldef()

    Rm = 30000.

    forall {
        insert pas // 'pas' for passive, 'hh' for Hodgkin-Huxley
        g_pas = 1 / Rm
        Ra = 150.
        cm = 1.
        }
    """ % (apic_diam, apic_diam))
    cell_params = {
                'morphology': h.all,
                'delete_sections': False,
                'v_init': -70.,
                'passive': False,
                'nsegs_method': None,
                'dt': dt,
                'tstart': -100.,
                'tstop': tstop,
                'pt3d': True,
            }
    cell = LFPy.Cell(**cell_params)
    # cell.set_pos(x=-cell.xstart[0])
    return cell


def return_ball_and_two_sticks_cell(tstop, dt, apic_diam=2, apic_upper_len = 1000, apic_bottom_len = -200):

    h("forall delete_section()")
    h("""
    proc celldef() {
      topol()
      subsets()
      geom()
      biophys()
      geom_nseg()
    }

    create soma[1], dend[2]

    proc topol() { local i
      basic_shape()
      connect dend[0](0), soma(1)
      connect dend[1](0), soma(0)
    }
    proc basic_shape() {
      soma[0] {pt3dclear()
      pt3dadd(0, 0, -10., 20.)
      pt3dadd(0, 0, 10., 20.)}
      dend[0] {pt3dclear()
      pt3dadd(0, 0, 10., %s)
      pt3dadd(0, 0, %s, %s)}
      dend[1] {pt3dclear()
      pt3dadd(0, 0, -10., %s)
      pt3dadd(0, 0, %s, %s)}
    }

    objref all
    proc subsets() { local i
      objref all
      all = new SectionList()
        soma[0] all.append()
        dend[0] all.append()
        dend[1] all.append()

    }
    proc geom() {
    }
    proc geom_nseg() {
    soma[0] {nseg = 1}
    dend[0] {nseg = 1000}
    dend[1] {nseg = 1000}
    }
    proc biophys() {
    }
    celldef()

    Rm = 30000.

    forall {
        insert pas // 'pas' for passive, 'hh' for Hodgkin-Huxley
        g_pas = 1 / Rm
        Ra = 150.
        cm = 1.
        }
    """ % (apic_diam, apic_upper_len, apic_diam, apic_diam, apic_bottom_len, apic_diam))
    cell_params = {
                'morphology': h.all,
                'delete_sections': False,
                'v_init': -70.,
                'passive': False,
                'nsegs_method': None,
                'dt': dt,
                'tstart': -100.,
                'tstop': tstop,
                'pt3d': True,
            }
    cell = LFPy.Cell(**cell_params)
    # cell.set_pos(x=-cell.xstart[0])
    return cell


def return_sticks_cell(tstop, dt, apic_diam=2, apic_upper_len = 1000, apic_bottom_len = -200):

    h("forall delete_section()")
    h("""
    proc celldef() {
      topol()
      subsets()
      geom()
      biophys()
      geom_nseg()
    }

    create soma[1], dend[2]

    proc topol() { local i
      basic_shape()
      connect dend[0](0), soma(1)
      connect dend[1](0), soma(0)
    }
    proc basic_shape() {
      soma[0] {pt3dclear()
      pt3dadd(0, 0, -10., %s)
      pt3dadd(0, 0, 10., %s)}
      dend[0] {pt3dclear()
      pt3dadd(0, 0, 10., %s)
      pt3dadd(0, 0, %s, %s)}
      dend[1] {pt3dclear()
      pt3dadd(0, 0, -10., %s)
      pt3dadd(0, 0, %s, %s)}
    }

    objref all
    proc subsets() { local i
      objref all
      all = new SectionList()
        soma[0] all.append()
        dend[0] all.append()
        dend[1] all.append()

    }
    proc geom() {
    }
    proc geom_nseg() {
    soma[0] {nseg = 1}
    dend[0] {nseg = 200}
    dend[1] {nseg = 200}
    }
    proc biophys() {
    }
    celldef()

    Rm = 30000.

    forall {
        insert pas // 'pas' for passive, 'hh' for Hodgkin-Huxley
        g_pas = 1 / Rm
        Ra = 150.
        cm = 1.
        }
    """ % (apic_diam, apic_diam, apic_diam, apic_upper_len, apic_diam, apic_diam, apic_bottom_len, apic_diam))
    cell_params = {
                'morphology': h.all,
                'delete_sections': False,
                'v_init': -70.,
                'passive': False,
                'nsegs_method': None,
                'dt': dt,
                'tstart': -100.,
                'tstop': tstop,
                'pt3d': True,
            }
    cell = LFPy.Cell(**cell_params)
    # cell.set_pos(x=-cell.xstart[0])
    return cell


def return_two_comp_cell(tstop, dt):

    h("forall delete_section()")
    h("""
    proc celldef() {
      topol()
      subsets()
      geom()
      biophys()
      geom_nseg()
    }

    create soma[1], dend[1]

    proc topol() { local i
      basic_shape()
      connect dend(0), soma(1)
    }
    proc basic_shape() {
      soma[0] {pt3dclear()
      pt3dadd(0, 0, -10.0, 20)
      pt3dadd(0, 0, 10., 20)}
      dend[0] {pt3dclear()
      pt3dadd(0, 0, 10.0, 20)
      pt3dadd(0, 0, 30.0, 20)}
    }

    objref all
    proc subsets() { local i
      objref all
      all = new SectionList()
        soma[0] all.append()
        dend[0] all.append()

    }
    proc geom() {
    }
    proc geom_nseg() {
    soma[0] {nseg = 1}
    dend[0] {nseg = 1}
    }
    proc biophys() {
    }
    celldef()

    Rm = 30000.

    forall {
        insert pas // 'pas' for passive, 'hh' for Hodgkin-Huxley
        g_pas = 1 / Rm
        Ra = 150.
        cm = 1.
        }
    """)
    cell_params = {
                'morphology': h.all,
                'delete_sections': False,
                'v_init': -70.,
                'passive': False,
                'nsegs_method': None,
                'dt': dt,
                'tstart': -100.,
                'tstop': tstop,
                'pt3d': True,
            }
    cell = LFPy.Cell(**cell_params)
    # cell.set_pos(x=-cell.xstart[0])
    return cell


# Simulation function ---------------------------------------

def check_existing_data(file_path, cell_name, frequency):
    if not os.path.exists(file_path):
        return False
    
    data = np.load(file_path, allow_pickle=True).item()
    
    if cell_name in data:
        if frequency in data[cell_name]['freq']:
            return True
    
    return False


def run_simulation_neuron_models(freq, neurons, diam, upper_len, tstop, dt, cutoff,
                                 local_E_field=1, directory='/Users/susannedahle/Python/Simulation_data_neur_models'):
    
    amp_data_file_path = os.path.join(directory, 'amp_data_neuron_models.npy')
    plot_data_file_path = os.path.join(directory, 'plot_data_neuron_models.npy')
    
    if os.path.exists(amp_data_file_path):
        amp_data = np.load(amp_data_file_path, allow_pickle=True).item()
    else:
        amp_data = {}
    
    if os.path.exists(plot_data_file_path):
        plot_data = np.load(plot_data_file_path, allow_pickle=True).item()
    else:
        plot_data = {}

    local_ext_pot = np.vectorize(lambda x, y, z: local_E_field * z / 1000)
    n_tsteps_ = int((tstop + cutoff) / dt + 1)
    t_ = np.arange(n_tsteps_) * dt

        
    for neuron_idx in range(len(neurons)):
        cell_name = neurons[neuron_idx]

        if cell_name == "Stick":
            print("Stick")
            bottom_len = np.array([-750, -500, -250])
            
            for l in bottom_len:
                dend_len = l

                print(f"Running simulation with {cell_name} and dend_len {l} µm")

                for f in freq:
                    if check_existing_data(amp_data_file_path, f"{cell_name}_dend_len_1000_and_{dend_len}", f):
                        print(f"Skipping {cell_name}_dend_len_1000_and_{dend_len} at {f} Hz (already exists in data)")
                        continue
                    
                    cell = return_sticks_cell(tstop, dt, apic_bottom_len = l)
                    print("Returned Stick model")

                    cell.extracellular = True

                    for sec in cell.allseclist:
                        sec.insert("extracellular")

                    # Calculate and insert extracellular potential
                    base_pot = local_ext_pot(
                        cell.x.mean(axis=-1),
                        cell.y.mean(axis=-1),
                        cell.z.mean(axis=-1)
                    ).reshape(cell.totnsegs, 1)

                    pulse = np.sin(2 * np.pi * f * t_ / 1000)

                    v_cell_ext = np.zeros((cell.totnsegs, n_tsteps_))
                    v_cell_ext = base_pot * pulse.reshape(1, n_tsteps_)

                    cell.insert_v_ext(v_cell_ext, t_)
                    cell.simulate(rec_vmem=True)

                    # Calculate soma amp with fourier
                    cut_tvec = cell.tvec[cell.tvec > 2000]
                    cut_soma_vmem = cell.vmem[0, cell.tvec > 2000]
                    freqs, vmem_amps = ns.return_freq_and_amplitude(cut_tvec, cut_soma_vmem)
                    freq_idx = np.argmin(np.abs(freqs - f))
                    soma_amp = vmem_amps[0, freq_idx]

                    # Write data to .npy file
                    if f"{cell_name}_dend_len_1000_and_{dend_len}" not in amp_data:
                        amp_data[f"{cell_name}_dend_len_1000_and_{dend_len}"] = {
                            'freq': [],
                            'soma_vmem_amp': []
                        }
                    amp_data[f"{cell_name}_dend_len_1000_and_{dend_len}"]['freq'].append(f)
                    amp_data[f"{cell_name}_dend_len_1000_and_{dend_len}"]['soma_vmem_amp'].append(soma_amp)

                    # Save amp data to .npy file
                    np.save(amp_data_file_path, amp_data)
                    print(f"Amplitude data has been saved to {os.path.abspath(amp_data_file_path)}")
    

                    if f in [10, 100, 1000]:
                        print(f"Storing data for selected frequency: {f} Hz") # Debug statement to make sure the data for the selected frequencies is stored for the plot
                        # Calculate amplitude in all segments and removing the part before the cell stabilizes 
                        vmem_amplitudes = []
                        for idx in range(cell.totnsegs):
                            cut_tvec = cell.tvec[cell.tvec > 2000]
                            cut_vmem = cell.vmem[idx, cell.tvec > 2000]
                            freqs, vmem_amps = ns.return_freq_and_amplitude(cut_tvec, cut_vmem)
                            freq_idx = np.argmin(np.abs(freqs - f))
                            amplitude = vmem_amps[0, freq_idx]
                            vmem_amplitudes.append(amplitude)

                        # Save plot-data in a .npy file
                        if f"{cell_name}_dend_len_1000_and_{dend_len}" not in plot_data:
                            plot_data[f"{cell_name}_dend_len_1000_and_{dend_len}"] = {
                                'freq': [],
                                "x": cell.x.tolist(),
                                "z": cell.z.tolist(),
                                "totnsegs": cell.totnsegs,
                                "tvec": cell.tvec.tolist(),
                                "soma_vmem": [],
                                "vmem_amps": []
                            }
    
                        plot_data[f"{cell_name}_dend_len_1000_and_{dend_len}"]['freq'].append(f)
                        plot_data[f"{cell_name}_dend_len_1000_and_{dend_len}"]['soma_vmem'].append(cell.vmem[0].tolist())
                        plot_data[f"{cell_name}_dend_len_1000_and_{dend_len}"]['vmem_amps'].append(vmem_amplitudes)
                        
                        np.save(plot_data_file_path, plot_data)
                        print(f"Plot data for selected frequency: {f} Hz has been saved to {os.path.abspath(plot_data_file_path)}")
                    
                    del cell

                    print(f"{f} Hz complete for dendrite len {l} µm")
                
        
        elif cell_name == "BnS":
            print("Ball and stick")
            for d in diam:
                diameter = d

                print(f"Running simulation with {cell_name} and diameter {d} µm")

                for f in freq:
                    if check_existing_data(amp_data_file_path, f"{cell_name}_diam_{diameter}", f):
                        print(f"Skipping {cell_name}_diam_{diameter} at {f} Hz (already exists in data)")
                        continue
                    
                    cell = return_ball_and_stick_cell(tstop, dt, apic_diam=d)
                    print("Returned Ball-and-Stick model")

                    cell.extracellular = True

                    for sec in cell.allseclist:
                        sec.insert("extracellular")

                    # Calculate and insert extracellular potential
                    base_pot = local_ext_pot(
                        cell.x.mean(axis=-1),
                        cell.y.mean(axis=-1),
                        cell.z.mean(axis=-1)
                    ).reshape(cell.totnsegs, 1)

                    pulse = np.sin(2 * np.pi * f * t_ / 1000)

                    v_cell_ext = np.zeros((cell.totnsegs, n_tsteps_))
                    v_cell_ext = base_pot * pulse.reshape(1, n_tsteps_)

                    cell.insert_v_ext(v_cell_ext, t_)
                    cell.simulate(rec_vmem=True)

                    # Calculate soma amp with fourier
                    cut_tvec = cell.tvec[cell.tvec > 2000]
                    cut_soma_vmem = cell.vmem[0, cell.tvec > 2000]
                    freqs, vmem_amps = ns.return_freq_and_amplitude(cut_tvec, cut_soma_vmem)
                    freq_idx = np.argmin(np.abs(freqs - f))
                    soma_amp = vmem_amps[0, freq_idx]

                    # Write data to .npy file
                    if f"{cell_name}_diam_{diameter}" not in amp_data:
                        amp_data[f"{cell_name}_diam_{diameter}"] = {
                            'freq': [],
                            'soma_vmem_amp': []
                        }
                    amp_data[f"{cell_name}_diam_{diameter}"]['freq'].append(f)
                    amp_data[f"{cell_name}_diam_{diameter}"]['soma_vmem_amp'].append(soma_amp)

                    # Save amp data to .npy file
                    np.save(amp_data_file_path, amp_data)
                    print(f"Amplitude data has been saved to {os.path.abspath(amp_data_file_path)}")
    

                    if f in [10, 100, 1000]:
                        print(f"Storing data for selected frequency: {f} Hz") # Debug statement to make sure the data for the selected frequencies is stored for the plot
                        # Calculate amplitude in all segments and removing the part before the cell stabilizes 
                        vmem_amplitudes = []
                        for idx in range(cell.totnsegs):
                            cut_tvec = cell.tvec[cell.tvec > 2000]
                            cut_vmem = cell.vmem[idx, cell.tvec > 2000]
                            freqs, vmem_amps = ns.return_freq_and_amplitude(cut_tvec, cut_vmem)
                            freq_idx = np.argmin(np.abs(freqs - f))
                            amplitude = vmem_amps[0, freq_idx]
                            vmem_amplitudes.append(amplitude)

                        # Save plot-data in a .npy file
                        if f"{cell_name}_diam_{diameter}" not in plot_data:
                            plot_data[f"{cell_name}_diam_{diameter}"] = {
                                'freq': [],
                                "x": cell.x.tolist(),
                                "z": cell.z.tolist(),
                                "totnsegs": cell.totnsegs,
                                "tvec": cell.tvec.tolist(),
                                "soma_vmem": [],
                                "vmem_amps": []
                            }
    
                        plot_data[f"{cell_name}_diam_{diameter}"]['freq'].append(f)
                        plot_data[f"{cell_name}_diam_{diameter}"]['soma_vmem'].append(cell.vmem[0].tolist())
                        plot_data[f"{cell_name}_diam_{diameter}"]['vmem_amps'].append(vmem_amplitudes)
                        
                        np.save(plot_data_file_path, plot_data)
                        print(f"Plot data for selected frequency: {f} Hz has been saved to {os.path.abspath(plot_data_file_path)}")
                    
                    del cell

                    print(f"{f} Hz complete for diameter {d} µm")

                
        elif cell_name == "Bn2S":
            print("Ball and Two sticks")
            
            for l_up in upper_len:
                upper_dend = l_up
                # Calculate bottom_len as 75%, 50%, and 25% of upper length 
                bottom_len = np.array([-l_up, (-0.75) * l_up, (-0.5) * l_up, (-0.25) * l_up])
                
                for l in bottom_len: 
                    bottom_dend = l
    
                    print(f"Running simulation with dendrite lengths {l_up} and {l} µm") 
    
                    for f in freq:
                        if check_existing_data(amp_data_file_path, f"{cell_name}_dendrite_len_{upper_dend}_and_{bottom_dend}", f):
                            print(f"Skipping {cell_name}_dendrite_len_{upper_dend}_and_{bottom_dend} at {f} Hz (already exists in data)")
                            continue
    
                        cell = return_ball_and_two_sticks_cell(tstop, dt, apic_upper_len = l_up, apic_bottom_len = l)
    
                        cell.extracellular = True
    
                        for sec in cell.allseclist:
                            sec.insert("extracellular")
    
                        # Calculate and instert extracellular potential
                        base_pot = local_ext_pot(
                            cell.x.mean(axis=-1),
                            cell.y.mean(axis=-1),
                            cell.z.mean(axis=-1)
                        ).reshape(cell.totnsegs, 1)
    
                        pulse = np.sin(2 * np.pi * f * t_ / 1000)
    
                        v_cell_ext = np.zeros((cell.totnsegs, n_tsteps_))
                        v_cell_ext = base_pot * pulse.reshape(1, n_tsteps_)
    
                        cell.insert_v_ext(v_cell_ext, t_)
                        cell.simulate(rec_vmem=True)

                        # Calculate soma amp with fourier
                        cut_tvec = cell.tvec[cell.tvec > 2000]
                        cut_soma_vmem = cell.vmem[0, cell.tvec > 2000]
                        freqs, vmem_amps = ns.return_freq_and_amplitude(cut_tvec, cut_soma_vmem)
                        freq_idx = np.argmin(np.abs(freqs - f))
                        soma_amp = vmem_amps[0, freq_idx]

                        # Write data to .npy file
                        if f"{cell_name}_dendrite_len_{upper_dend}_and_{bottom_dend}" not in amp_data:
                            amp_data[f"{cell_name}_dendrite_len_{upper_dend}_and_{bottom_dend}"] = {
                                'freq': [],
                                'soma_vmem_amp': []
                            }
                        amp_data[f"{cell_name}_dendrite_len_{upper_dend}_and_{bottom_dend}"]['freq'].append(f)
                        amp_data[f"{cell_name}_dendrite_len_{upper_dend}_and_{bottom_dend}"]['soma_vmem_amp'].append(soma_amp)
    
                        # Save amp data to .npy file
                        np.save(amp_data_file_path, amp_data)
                        print(f"Amplitude data has been saved to {os.path.abspath(amp_data_file_path)}")


                        if f in [10, 100, 1000]:
                            print(f"Storing data for selected frequency: {f} Hz") # Debug statement to make sure the data for the selected frequencies is stored for the plot
                            # Calculate amplitude in all segments and removing the part before the cell stabilizes 
                            vmem_amplitudes = []
                            for idx in range(cell.totnsegs):
                                cut_tvec = cell.tvec[cell.tvec > 2000]
                                cut_vmem = cell.vmem[idx, cell.tvec > 2000]
                                freqs, vmem_amps = ns.return_freq_and_amplitude(cut_tvec, cut_vmem)
                                freq_idx = np.argmin(np.abs(freqs - f))
                                amplitude = vmem_amps[0, freq_idx]
                                vmem_amplitudes.append(amplitude)
    
                            # Save plot-data in a .npy file
                            if f"{cell_name}_dendrite_len_{upper_dend}_and_{bottom_dend}" not in plot_data:
                                plot_data[f"{cell_name}_dendrite_len_{upper_dend}_and_{bottom_dend}"] = {
                                    'freq': [],
                                    "x": cell.x.tolist(),
                                    "z": cell.z.tolist(),
                                    "totnsegs": cell.totnsegs,
                                    "tvec": cell.tvec.tolist(),
                                    "soma_vmem": [],
                                    "vmem_amps": []
                                }
        
                            plot_data[f"{cell_name}_dendrite_len_{upper_dend}_and_{bottom_dend}"]['freq'].append(f)
                            plot_data[f"{cell_name}_dendrite_len_{upper_dend}_and_{bottom_dend}"]['soma_vmem'].append(cell.vmem[0].tolist())
                            plot_data[f"{cell_name}_dendrite_len_{upper_dend}_and_{bottom_dend}"]['vmem_amps'].append(vmem_amplitudes)
                            
                            np.save(plot_data_file_path, plot_data)
                            print(f"Plot data for selected frequency: {f} Hz has been saved to {os.path.abspath(plot_data_file_path)}")

                        del cell 
                        
                        print(f"{f}Hz complete for {cell_name}_dendrite_len_{upper_dend}_and_{bottom_dend}")              


        elif cell_name == "TwoComp":
            print("Two Compartment")

            for f in freq:
                if check_existing_data(amp_data_file_path, cell_name, f):
                    print(f"Skipping {cell_name} at {f} Hz (already exists in data)")
                    continue
                
                cell = return_two_comp_cell(tstop, dt)
                print("Returned Two-Compartment model")

                cell.extracellular = True

                for sec in cell.allseclist:
                    sec.insert("extracellular")

                # Calculate and insert extracellular potential
                base_pot = local_ext_pot(
                    cell.x.mean(axis=-1),
                    cell.y.mean(axis=-1),
                    cell.z.mean(axis=-1)
                ).reshape(cell.totnsegs, 1)

                pulse = np.sin(2 * np.pi * f * t_ / 1000)

                v_cell_ext = np.zeros((cell.totnsegs, n_tsteps_))
                v_cell_ext = base_pot * pulse.reshape(1, n_tsteps_)

                cell.insert_v_ext(v_cell_ext, t_)
                cell.simulate(rec_vmem=True)

                # Calculate soma amp with fourier
                cut_tvec = cell.tvec[cell.tvec > 2000]
                cut_soma_vmem = cell.vmem[0, cell.tvec > 2000]
                freqs, vmem_amps = ns.return_freq_and_amplitude(cut_tvec, cut_soma_vmem)
                freq_idx = np.argmin(np.abs(freqs - f))
                soma_amp = vmem_amps[0, freq_idx]

                # Write data to .npy file
                if cell_name not in amp_data:
                    amp_data[cell_name] = {
                        'freq': [],
                        'soma_vmem_amp': []
                    }
                amp_data[cell_name]['freq'].append(f)
                amp_data[cell_name]['soma_vmem_amp'].append(soma_amp)

                # Save amp data to .npy file
                np.save(amp_data_file_path, amp_data)
                print(f"Amplitude data has been saved to {os.path.abspath(amp_data_file_path)}")


                if f in [10, 100, 1000]:
                    print(f"Storing data for selected frequency: {f} Hz") # Debug statement to make sure the data for the selected frequencies is stored for the plot
                    # Calculate amplitude in all segments and removing the part before the cell stabilizes 
                    vmem_amplitudes = []
                    for idx in range(cell.totnsegs):
                        cut_tvec = cell.tvec[cell.tvec > 2000]
                        cut_vmem = cell.vmem[idx, cell.tvec > 2000]
                        freqs, vmem_amps = ns.return_freq_and_amplitude(cut_tvec, cut_vmem)
                        freq_idx = np.argmin(np.abs(freqs - f))
                        amplitude = vmem_amps[0, freq_idx]
                        vmem_amplitudes.append(amplitude)

                    # Save plot-data in a .npy file
                    if cell_name not in plot_data:
                        plot_data[cell_name] = {
                            'freq': [],
                            "x": cell.x.tolist(),
                            "z": cell.z.tolist(),
                            "totnsegs": cell.totnsegs,
                            "tvec": cell.tvec.tolist(),
                            "soma_vmem": [],
                            "vmem_amps": []
                        }

                    plot_data[cell_name]['freq'].append(f)
                    plot_data[cell_name]['soma_vmem'].append(cell.vmem[0].tolist())
                    plot_data[cell_name]['vmem_amps'].append(vmem_amplitudes)
                    
                    np.save(plot_data_file_path, plot_data)
                    print(f"Plot data for selected frequency: {f} Hz has been saved to {os.path.abspath(plot_data_file_path)}")
                    
                del cell

                print(f"{f} Hz complete for Two Compartment model")


        else:
            ns.compile_bbp_mechanisms(cell_name)

            for f in freq:
                if check_existing_data(amp_data_file_path, cell_name, f):
                    print(f"Skipping {cell_name} at {f} Hz (already exists in data)")
                    continue
                
                cell = return_BBP_neuron(cell_name, tstop, dt)
                ns.remove_active_mechanisms(remove_list, cell)
                
                cell.extracellular = True

                for sec in cell.allseclist:
                    sec.insert("extracellular")

                # Calculate and insert extracellular potential
                base_pot = local_ext_pot(
                    cell.x.mean(axis=-1),
                    cell.y.mean(axis=-1),
                    cell.z.mean(axis=-1)
                ).reshape(cell.totnsegs, 1)

                pulse = np.sin(2 * np.pi * f * t_ / 1000)

                v_cell_ext = np.zeros((cell.totnsegs, n_tsteps_))
                v_cell_ext = base_pot * pulse.reshape(1, n_tsteps_)

                cell.insert_v_ext(v_cell_ext, t_)
                cell.simulate(rec_vmem=True)

                # Calculate soma amp with fourier
                cut_tvec = cell.tvec[cell.tvec > 2000]
                cut_soma_vmem = cell.vmem[0, cell.tvec > 2000]
                freqs, vmem_amps = ns.return_freq_and_amplitude(cut_tvec, cut_soma_vmem)
                freq_idx = np.argmin(np.abs(freqs - f))
                soma_amp = vmem_amps[0, freq_idx]

                # Write data to .npy file
                if cell_name not in amp_data:
                    amp_data[cell_name] = {
                        'freq': [],
                        'soma_vmem_amp': []
                    }
                amp_data[cell_name]['freq'].append(f)
                amp_data[cell_name]['soma_vmem_amp'].append(soma_amp)

                # Save amp data to .npy file
                np.save(amp_data_file_path, amp_data)
                print(f"Amplitude data has been saved to {os.path.abspath(amp_data_file_path)}")

                if f in [10, 100, 1000]:
                    # Calculate amplitude in all segments by removing the part before the cell stabilizes 
                    vmem_amplitudes = []
                    for idx in range(cell.totnsegs):
                        cut_tvec = cell.tvec[cell.tvec > 2000]
                        cut_vmem = cell.vmem[idx, cell.tvec > 2000]
                        freqs, vmem_amps = ns.return_freq_and_amplitude(cut_tvec, cut_vmem)
                        freq_idx = np.argmin(np.abs(freqs - f))
                        amplitude = vmem_amps[0, freq_idx]
                        vmem_amplitudes.append(amplitude)

                    # Save plot-data in a .npy file
                    if cell_name not in plot_data:
                        plot_data[cell_name] = {
                            'freq': [],
                            "x": cell.x.tolist(),
                            "z": cell.z.tolist(),
                            "totnsegs": cell.totnsegs,
                            "tvec": cell.tvec.tolist(),
                            "soma_vmem": [],
                            "vmem_amps": []
                        }

                    plot_data[cell_name]['freq'].append(f)
                    plot_data[cell_name]['soma_vmem'].append(cell.vmem[0].tolist())
                    plot_data[cell_name]['vmem_amps'].append(vmem_amplitudes)
                    
                    np.save(plot_data_file_path, plot_data)
                    print(f"Plot data for selected frequency: {f} Hz has been saved to {os.path.abspath(plot_data_file_path)}")
                    
                del cell

                print(f"{f} Hz complete for {cell_name}")


if __name__=='__main__':
    ns.load_mechs_from_folder(ns.cell_models_folder)

    h = neuron.h

    real_cells_folder = '/Users/susannedahle/Python/brainsignals/cell_models/bbp_models'
    bbp_folder = os.path.abspath(real_cells_folder)                              # Make this the bbp_folder

    cell_models_folder = '/Users/susannedahle/Python/brainsignals/cell_models'
    bbp_mod_folder = join(cell_models_folder, "bbp_mod")                        # Mappen med ulike parametere og mekanismer 

    cell_title_dict = {"Stick": "Stick neuron",
                    "BnS": "Ball-and-Stick",
                    "Bn2S": "Ball-and-Two-Sticks",
                    "TwoComp": "Two compartments model",
                    "L5_MC_bAC217_1": "L5 martinotti cell",
                    "L5_TTPC2_cADpyr232_2": "L5 pyramidal cell",
                    "L5_NGC_bNAC219_5": "L5 neurogliaform cell"}

    # One pyramidal cell and two interneurons from L5
    neurons = ["Stick",
            "BnS",
            "Bn2S",
            "TwoComp", 
            "L5_TTPC2_cADpyr232_2",
            "L5_MC_bAC217_1",
            "L5_NGC_bNAC219_5"
            ]
    
    # List of active mechanisms to remove to make passive
    remove_list = ["Ca_HVA", "Ca_LVAst", "Ca", "CaDynamics_E2", 
                "Ih", "Im", "K_Pst", "K_Tst", "KdShu2007", "Nap_Et2",
                "NaTa_t", "NaTs2_t", "SK_E2", "SKv3_1", "StochKv"]


    # Simulation time 
    tstop = 5000.
    dt = 2**-4

    cutoff = 20

    # El field frequency 
    freq1 = np.arange(1, 10, 1) # Shorter steplength in beginning
    freq2 = np.arange(10, 100, 10)
    freq3 = np.arange(100, 2200, 200) # Longer steplength to save calculation time
    freq = sorted(np.concatenate((freq1, freq2, freq3, np.array([1000]))))

    # El field strengths 
    local_E_field = 1  # V/m

    # Properties for the ball and stick(s) neurons
    diam = np.arange(2, 9, 2)
    upper_len = np.array([1000, 100])

    run_simulation_neuron_models(freq, neurons, diam, upper_len, tstop, dt, cutoff)
