# Simulation code all neocortical neuron models from Blue Brain Project

import os
import sys
from os.path import join, expanduser

from glob import glob
import numpy as np

import neuron
import LFPy

parent_path = '/Users/susannedahle/Python/brainsignals'
sys.path.append(parent_path)

import brainsignals.neural_simulations as ns
from brainsignals.plotting_convention import mark_subplots
not_working_cells = []
not_working_plot_cells = []


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

def check_existing_data(data_dict, cell_name, frequency):
    if cell_name in data_dict:
        if frequency in data_dict[cell_name]['freq']:
            return True
    return False

def run_passive_simulation_Ex(freq, 
                              neurons,
                              remove_list,
                              tstop, 
                              dt, 
                              cutoff,
                              local_E_field=1,  # V/m
                              amp_data_filename='amp_data_all_cells_Ex.npy',
                              directory='/Users/susannedahle/Python/Simulation_data',
                              plot_data_filename='plot_data_all_cells_Ex.npy'):

    amp_data_file_path = os.path.join(directory, amp_data_filename)
    plot_data_file_path = os.path.join(directory, plot_data_filename)
    
    # Initialize or load existing data
    if os.path.exists(amp_data_file_path):
        amp_data = np.load(amp_data_file_path, allow_pickle=True).item()
    else:
        amp_data = {}
        amp_file_exist = False

    if os.path.exists(plot_data_file_path):
        plot_data = np.load(plot_data_file_path, allow_pickle=True).item()
    else:
        plot_data = {}
        plot_file_exist = False

    local_ext_pot = np.vectorize(lambda x, y, z: local_E_field * x / 1000)
    n_tsteps_ = int((tstop + cutoff) / dt + 1)
    t_ = np.arange(n_tsteps_) * dt
    ns.compile_bbp_mechanisms(neurons[0])
    
    for neuron_idx, cell_name in enumerate(neurons):
        cell_failed_simulation = False
        cell_failed_plotting = False
        
        for f in freq:
            if check_existing_data(amp_data, cell_name, f):
                print(f"Skipping {cell_name} at {f} Hz (already exists in data)")
                continue
            
            try:
                cell = return_BBP_neuron(cell_name, tstop, dt)
                ns.remove_active_mechanisms(remove_list, cell)
                
                cell.extracellular = True   
                for sec in cell.allseclist:
                    sec.insert("extracellular")

                base_pot = local_ext_pot(
                    cell.x.mean(axis=-1),
                    cell.y.mean(axis=-1),
                    cell.z.mean(axis=-1)
                ).reshape(cell.totnsegs, 1)

                pulse = np.sin(2 * np.pi * f * t_ / 1000)
                v_cell_ext = base_pot * pulse.reshape(1, n_tsteps_)

                cell.insert_v_ext(v_cell_ext, t_)
                cell.simulate(rec_vmem=True)

                # Calculate soma amp with fourier
                cut_tvec = cell.tvec[cell.tvec > 2000]
                cut_soma_vmem = cell.vmem[0, cell.tvec > 2000]
                freqs, vmem_amps = ns.return_freq_and_amplitude(cut_tvec, cut_soma_vmem)
                freq_idx = np.argmin(np.abs(freqs - f))
                soma_amp = vmem_amps[0, freq_idx]

                # Distance from soma to closest endpoint                  
                upper_x_endpoint = cell.x.mean(axis=-1)[cell.get_closest_idx(x=10000)]
                bottom_x_endpoint = cell.x.mean(axis=-1)[cell.get_closest_idx(x=-10000)]
                closest_x_endpoint = min(upper_x_endpoint, abs(bottom_x_endpoint))
                distant_x_endpoint = max(upper_x_endpoint, abs(bottom_x_endpoint))
                total_len_x_direction = closest_x_endpoint + distant_x_endpoint
                symmetry_factor_x_direction = closest_x_endpoint/distant_x_endpoint

                # Dendrites connected to Soma:
                soma_name = None
                for sec in cell.allseclist:
                    if 'soma[0]' in sec.name():
                        soma_name = sec.name()
                child_list = cell.get_idx_children(parent=soma_name)

                number_of_soma_dendrites = len(child_list)

                dend_diam_to_soma = cell.d[child_list]
                total_diam_soma_dendrites = np.sum(dend_diam_to_soma)
                soma_diam = cell.d[0]

                # Store data in dictionary
                if cell_name not in amp_data:
                    amp_data[cell_name] = {
                        'freq': [],
                        'soma_amp': [],
                        'closest_z_endpoint': closest_x_endpoint,
                        'distant_z_endpoint': distant_x_endpoint,
                        'upper_z_endpoint': upper_x_endpoint,
                        'bottom_z_endpoint': bottom_x_endpoint,
                        'total_len': total_len_x_direction,
                        'symmetry_factor': symmetry_factor_x_direction,
                        'number_of_soma_dendrites': number_of_soma_dendrites,
                        'total_diam_soma_dendrites': total_diam_soma_dendrites,
                        'soma_diam': soma_diam
                    }
                
                amp_data[cell_name]['freq'].append(f)
                amp_data[cell_name]['soma_amp'].append(soma_amp)
                
                # Save amp data to .npy file
                np.save(amp_data_file_path, amp_data)
                print(f"Amplitude data has been saved to {os.path.abspath(amp_data_file_path)}")
                
            except:
                cell_failed_simulation = True
                break
            
            if f in [10, 100, 1000]:
                try:
                    print(f"Storing data for selected frequency: {f} Hz")
                    amplitudes = []
                    for idx in range(cell.totnsegs):
                        cut_vmem = cell.vmem[idx, cell.tvec > 2000]
                        freqs, vmem_amps = ns.return_freq_and_amplitude(cut_tvec, cut_vmem)
                        freq_idx = np.argmin(np.abs(freqs - f))
                        amplitude = vmem_amps[0, freq_idx]
                        amplitudes.append(amplitude)

                    if cell_name not in plot_data:
                        plot_data[cell_name] = {
                            'freq': [],
                            'x': [],
                            'z': [],
                            'amplitudes': [],
                            'totnsegs': [],
                            'tvec': [],
                            'vmem': []
                        }

                    plot_data[cell_name]['freq'].append(f)
                    plot_data[cell_name]['x'].append(cell.x.tolist())
                    plot_data[cell_name]['z'].append(cell.z.tolist())
                    plot_data[cell_name]['amplitudes'].append(amplitudes)
                    plot_data[cell_name]['totnsegs'].append(cell.totnsegs)
                    plot_data[cell_name]['tvec'].append(cell.tvec.tolist())
                    plot_data[cell_name]['vmem'].append(cell.vmem[0].tolist())

                    # Save plot data to .npy file
                    np.save(plot_data_file_path, plot_data)
                    print(f"Plot data has been saved to {os.path.abspath(plot_data_file_path)}")
                    
                except:
                    cell_failed_plotting = True
                    break
            
            del cell
            print(f"{f} Hz complete for {cell_name}")
        
        if cell_failed_simulation:
            not_working_cells.append(cell_name)
            print(f"Neuron {cell_name}, idx:{neuron_idx} does not work for simulation")
        elif cell_failed_plotting:
            not_working_plot_cells.append(cell_name)
            print(f"Neuron {cell_name}, idx:{neuron_idx} does not work for plotting")
        
        print(f"Simulation with E-field in x direction complete for Neuron nr.{neuron_idx+1} of {len(neurons)} neurons\n")


def run_passive_simulation_Ey(freq, 
                              neurons,
                              remove_list,
                              tstop, 
                              dt, 
                              cutoff,
                              local_E_field=1,  # V/m
                              amp_data_filename='amp_data_all_cells_Ey.npy',
                              directory='/Users/susannedahle/Python/Simulation_data',
                              plot_data_filename='plot_data_all_cells_Ey.npy'):
                              #directory='/Users/susannedahle/Documents/Master/Code/Simulation_data'):

    amp_data_file_path = os.path.join(directory, amp_data_filename)
    plot_data_file_path = os.path.join(directory, plot_data_filename)
    
    # Initialize or load existing data
    if os.path.exists(amp_data_file_path):
        amp_data = np.load(amp_data_file_path, allow_pickle=True).item()
    else:
        amp_data = {}
        amp_file_exist = False

    if os.path.exists(plot_data_file_path):
        plot_data = np.load(plot_data_file_path, allow_pickle=True).item()
    else:
        plot_data = {}
        plot_file_exist = False

    local_ext_pot = np.vectorize(lambda x, y, z: local_E_field * y / 1000)
    n_tsteps_ = int((tstop + cutoff) / dt + 1)
    t_ = np.arange(n_tsteps_) * dt
    ns.compile_bbp_mechanisms(neurons[0])
    
    for neuron_idx, cell_name in enumerate(neurons):
        cell_failed_simulation = False
        cell_failed_plotting = False
        
        for f in freq:
            if check_existing_data(amp_data, cell_name, f):
                print(f"Skipping {cell_name} at {f} Hz (already exists in data)")
                continue
            
            try:
                cell = return_BBP_neuron(cell_name, tstop, dt)
                ns.remove_active_mechanisms(remove_list, cell)
                
                cell.extracellular = True   
                for sec in cell.allseclist:
                    sec.insert("extracellular")

                base_pot = local_ext_pot(
                    cell.x.mean(axis=-1),
                    cell.y.mean(axis=-1),
                    cell.z.mean(axis=-1)
                ).reshape(cell.totnsegs, 1)

                pulse = np.sin(2 * np.pi * f * t_ / 1000)
                v_cell_ext = base_pot * pulse.reshape(1, n_tsteps_)

                cell.insert_v_ext(v_cell_ext, t_)
                cell.simulate(rec_vmem=True)

                # Calculate soma amp with fourier
                cut_tvec = cell.tvec[cell.tvec > 2000]
                cut_soma_vmem = cell.vmem[0, cell.tvec > 2000]
                freqs, vmem_amps = ns.return_freq_and_amplitude(cut_tvec, cut_soma_vmem)
                freq_idx = np.argmin(np.abs(freqs - f))
                soma_amp = vmem_amps[0, freq_idx]

                # Distance from soma to closest endpoint                  
                upper_y_endpoint = cell.y.mean(axis=-1)[cell.get_closest_idx(y=10000)]
                bottom_y_endpoint = cell.y.mean(axis=-1)[cell.get_closest_idx(y=-10000)]
                closest_y_endpoint = min(upper_y_endpoint, abs(bottom_y_endpoint))
                distant_y_endpoint = max(upper_y_endpoint, abs(bottom_y_endpoint))
                total_len_y_direction = closest_y_endpoint + distant_y_endpoint
                symmetry_factor_y_direction = closest_y_endpoint/distant_y_endpoint

                # Dendrites connected to Soma:
                soma_name = None
                for sec in cell.allseclist:
                    if 'soma[0]' in sec.name():
                        soma_name = sec.name()
                child_list = cell.get_idx_children(parent=soma_name)

                number_of_soma_dendrites = len(child_list)

                dend_diam_to_soma = cell.d[child_list]
                total_diam_soma_dendrites = np.sum(dend_diam_to_soma)

                soma_diam = cell.d[0]

                # Store data in dictionary
                if cell_name not in amp_data:
                    amp_data[cell_name] = {
                        'freq': [],
                        'soma_amp': [],
                        'closest_z_endpoint': closest_y_endpoint,
                        'distant_z_endpoint': distant_y_endpoint,
                        'upper_z_endpoint': upper_y_endpoint,
                        'bottom_z_endpoint': bottom_y_endpoint,
                        'total_len': total_len_y_direction,
                        'symmetry_factor': symmetry_factor_y_direction,
                        'number_of_soma_dendrites': number_of_soma_dendrites,
                        'total_diam_soma_dendrites': total_diam_soma_dendrites,
                        'soma_diam': soma_diam       
                    }
                
                amp_data[cell_name]['freq'].append(f)
                amp_data[cell_name]['soma_amp'].append(soma_amp)

                # Save amp data to .npy file
                np.save(amp_data_file_path, amp_data)
                print(f"Amplitude data has been saved to {os.path.abspath(amp_data_file_path)}")
                
            except:
                cell_failed_simulation = True
                break
            
            if f in [10, 100, 1000]:
                try:
                    print(f"Storing data for selected frequency: {f} Hz")
                    amplitudes = []
                    for idx in range(cell.totnsegs):
                        cut_vmem = cell.vmem[idx, cell.tvec > 2000]
                        freqs, vmem_amps = ns.return_freq_and_amplitude(cut_tvec, cut_vmem)
                        freq_idx = np.argmin(np.abs(freqs - f))
                        amplitude = vmem_amps[0, freq_idx]
                        amplitudes.append(amplitude)

                    if cell_name not in plot_data:
                        plot_data[cell_name] = {
                            'freq': [],
                            'x': [],
                            'z': [],
                            'amplitudes': [],
                            'totnsegs': [],
                            'tvec': [],
                            'vmem': []
                        }

                    plot_data[cell_name]['freq'].append(f)
                    plot_data[cell_name]['x'].append(cell.x.tolist())
                    plot_data[cell_name]['z'].append(cell.z.tolist())
                    plot_data[cell_name]['amplitudes'].append(amplitudes)
                    plot_data[cell_name]['totnsegs'].append(cell.totnsegs)
                    plot_data[cell_name]['tvec'].append(cell.tvec.tolist())
                    plot_data[cell_name]['vmem'].append(cell.vmem[0].tolist())

                    # Save plot data to .npy file
                    np.save(plot_data_file_path, plot_data)
                    print(f"Plot data has been saved to {os.path.abspath(plot_data_file_path)}")
                    
                except:
                    cell_failed_plotting = True
                    break
            
            del cell
            print(f"{f} Hz complete for {cell_name}")
        
        if cell_failed_simulation:
            not_working_cells.append(cell_name)
            print(f"Neuron {cell_name}, idx:{neuron_idx} does not work for simulation")
        elif cell_failed_plotting:
            not_working_plot_cells.append(cell_name)
            print(f"Neuron {cell_name}, idx:{neuron_idx} does not work for plotting")
        
        print(f"Simulation with E-field in y direction complete for Neuron nr.{neuron_idx+1} of {len(neurons)} neurons\n")


def run_passive_simulation_Ez(freq, 
                              neurons,
                              remove_list,
                              tstop, 
                              dt, 
                              cutoff,
                              local_E_field=1,  # V/m
                              amp_data_filename='amp_data_all_cells.npy',
                              plot_data_filename='plot_data_all.npy',
                              directory='/Users/susannedahle/Python/Simulation_data'):
                              #directory='/Users/susannedahle/Documents/Master/Code/Simulation_data'):

    amp_data_file_path = os.path.join(directory, amp_data_filename)
    plot_data_file_path = os.path.join(directory, plot_data_filename)
    
    # Initialize or load existing data
    if os.path.exists(amp_data_file_path):
        amp_data = np.load(amp_data_file_path, allow_pickle=True).item()
    else:
        amp_data = {}
        amp_file_exist = False

    if os.path.exists(plot_data_file_path):
        plot_data = np.load(plot_data_file_path, allow_pickle=True).item()
    else:
        plot_data = {}
        plot_file_exist = False

    local_ext_pot = np.vectorize(lambda x, y, z: local_E_field * z / 1000)
    n_tsteps_ = int((tstop + cutoff) / dt + 1)
    t_ = np.arange(n_tsteps_) * dt
    ns.compile_bbp_mechanisms(neurons[0])
    
    for neuron_idx, cell_name in enumerate(neurons):
        cell_failed_simulation = False
        cell_failed_plotting = False
        
        for f in freq:
            if check_existing_data(amp_data, cell_name, f):
                print(f"Skipping {cell_name} at {f} Hz (already exists in data)")
                continue
            
            try:
                cell = return_BBP_neuron(cell_name, tstop, dt)
                ns.remove_active_mechanisms(remove_list, cell)
                
                cell.extracellular = True   
                for sec in cell.allseclist:
                    sec.insert("extracellular")

                base_pot = local_ext_pot(
                    cell.x.mean(axis=-1),
                    cell.y.mean(axis=-1),
                    cell.z.mean(axis=-1)
                ).reshape(cell.totnsegs, 1)

                pulse = np.sin(2 * np.pi * f * t_ / 1000)
                v_cell_ext = base_pot * pulse.reshape(1, n_tsteps_)

                cell.insert_v_ext(v_cell_ext, t_)
                cell.simulate(rec_vmem=True)

                # Calculate soma amp with fourier
                cut_tvec = cell.tvec[cell.tvec > 2000]
                cut_soma_vmem = cell.vmem[0, cell.tvec > 2000]
                freqs, vmem_amps = ns.return_freq_and_amplitude(cut_tvec, cut_soma_vmem)
                freq_idx = np.argmin(np.abs(freqs - f))
                soma_amp = vmem_amps[0, freq_idx]

                # Distance from soma to closest endpoint                  
                upper_z_endpoint = cell.z.mean(axis=-1)[cell.get_closest_idx(z=10000)]
                bottom_z_endpoint = cell.z.mean(axis=-1)[cell.get_closest_idx(z=-10000)]
                closest_z_endpoint = min(upper_z_endpoint, abs(bottom_z_endpoint))
                distant_z_endpoint = max(upper_z_endpoint, abs(bottom_z_endpoint))
                total_len = closest_z_endpoint + distant_z_endpoint
                symmetry_factor = closest_z_endpoint/distant_z_endpoint

                # Dendrites connected to Soma:
                soma_name = None
                for sec in cell.allseclist:
                    if 'soma[0]' in sec.name():
                        soma_name = sec.name()
                child_list = cell.get_idx_children(parent=soma_name)

                number_of_soma_dendrites = len(child_list)

                dend_diam_to_soma = cell.d[child_list]
                total_diam_soma_dendrites = np.sum(dend_diam_to_soma)

                soma_diam = cell.d[0]

                # Store data in dictionary
                if cell_name not in amp_data:
                    amp_data[cell_name] = {
                        'freq': [],
                        'soma_amp': [],
                        'closest_z_endpoint': closest_z_endpoint,
                        'distant_z_endpoint': distant_z_endpoint,
                        'upper_z_endpoint': upper_z_endpoint,
                        'bottom_z_endpoint': bottom_z_endpoint,
                        'total_len': total_len,
                        'symmetry_factor': symmetry_factor,
                        'number_of_soma_dendrites': number_of_soma_dendrites,
                        'total_diam_soma_dendrites': total_diam_soma_dendrites,
                        'soma_diam': soma_diam
                    }

                amp_data[cell_name]['freq'].append(f)
                amp_data[cell_name]['soma_amp'].append(soma_amp)

                # Save amp data to .npy file
                np.save(amp_data_file_path, amp_data)
                print(f"Amplitude data has been saved to {os.path.abspath(amp_data_file_path)}")
                
            except:
                cell_failed_simulation = True
                break
            
            if f in [10, 100, 1000]:
                try:
                    print(f"Storing data for selected frequency: {f} Hz")
                    amplitudes = []
                    for idx in range(cell.totnsegs):
                        cut_vmem = cell.vmem[idx, cell.tvec > 2000]
                        freqs, vmem_amps = ns.return_freq_and_amplitude(cut_tvec, cut_vmem)
                        freq_idx = np.argmin(np.abs(freqs - f))
                        amplitude = vmem_amps[0, freq_idx]
                        amplitudes.append(amplitude)

                    if cell_name not in plot_data:
                        plot_data[cell_name] = {
                            'freq': [],
                            'x': [],
                            'z': [],
                            'amplitudes': [],
                            'totnsegs': [],
                            'tvec': [],
                            'vmem': []
                        }

                    plot_data[cell_name]['freq'].append(f)
                    plot_data[cell_name]['x'].append(cell.x.tolist())
                    plot_data[cell_name]['z'].append(cell.z.tolist())
                    plot_data[cell_name]['amplitudes'].append(amplitudes)
                    plot_data[cell_name]['totnsegs'].append(cell.totnsegs)
                    plot_data[cell_name]['tvec'].append(cell.tvec.tolist())
                    plot_data[cell_name]['vmem'].append(cell.vmem[0].tolist())

                    # Save plot data to .npy file
                    np.save(plot_data_file_path, plot_data)
                    print(f"Plot data has been saved to {os.path.abspath(plot_data_file_path)}")
                    
                except:
                    cell_failed_plotting = True
                    break
            
            del cell
            print(f"{f} Hz complete for {cell_name}")
        
        if cell_failed_simulation:
            not_working_cells.append(cell_name)
            print(f"Neuron {cell_name}, idx:{neuron_idx} does not work for simulation")
        elif cell_failed_plotting:
            not_working_plot_cells.append(cell_name)
            print(f"Neuron {cell_name}, idx:{neuron_idx} does not work for plotting")
        
        print(f"Neuron nr.{neuron_idx+1} of {len(neurons)} neurons complete \n")

if __name__=='__main__':
    ns.load_mechs_from_folder(ns.cell_models_folder)

    h = neuron.h

    all_cells_folder = '/Users/susannedahle/Python/all_cells_folder'
    bbp_folder = os.path.abspath(all_cells_folder)                       

    cell_models_folder = '/Users/susannedahle/Python/brainsignals/cell_models'
    bbp_mod_folder = join(cell_models_folder, "bbp_mod")    

    # List to store the neuron names
    neurons = []

    # Check if the directory exists
    if os.path.exists(all_cells_folder):
        # Iterate over the directories in the all_cells_folder
        for folder_name in os.listdir(all_cells_folder):
            folder_path = os.path.join(all_cells_folder, folder_name)
            if os.path.isdir(folder_path):
                neurons.append(folder_name)
    else:
        print(f"The directory {all_cells_folder} does not exist.")
    

    remove_list = ["Ca_HVA", "Ca_LVAst", "Ca", "CaDynamics_E2", 
                   "Ih", "Im", "K_Pst", "K_Tst", "KdShu2007", "Nap_Et2",
                   "NaTa_t", "NaTs2_t", "SK_E2", "SKv3_1", "StochKv"]
    # Simulation time 
    tstop = 5000.
    dt = 2**-4

    cutoff = 1

    # El field frequency 
    freq1 = np.arange(1, 10, 5) # Shorter steplength in beginning
    freq2 = np.arange(10, 100, 50)
    freq3 = np.arange(100, 2000, 300) # Longer steplength to save calculation time
    freq4 = np.arange(2000, 2001, 1)
    
    freq = np.concatenate((freq1, freq2, freq3, freq4))

    # El field strengths 
    local_E_field = 1  # V/m

    # Plot data stored in different files to save dynamic memory
    run_passive_simulation_Ez(freq, neurons[0:150], remove_list, tstop, dt, cutoff, local_E_field=1, plot_data_filename = 'plot_data_all_cells_1to150.npy')
    run_passive_simulation_Ez(freq, neurons[150:300], remove_list, tstop, dt, cutoff, local_E_field=1, plot_data_filename = 'plot_data_all_cells_150to300.npy')
    run_passive_simulation_Ez(freq, neurons[300:450], remove_list, tstop, dt, cutoff, local_E_field=1, plot_data_filename = 'plot_data_all_cells_300to450.npy')
    run_passive_simulation_Ez(freq, neurons[450:600], remove_list, tstop, dt, cutoff, local_E_field=1, plot_data_filename = 'plot_data_all_cells_450to600.npy')
    run_passive_simulation_Ez(freq, neurons[600:750], remove_list, tstop, dt, cutoff, local_E_field=1, plot_data_filename = 'plot_data_all_cells_600to750.npy')
    run_passive_simulation_Ez(freq, neurons[750:900], remove_list, tstop, dt, cutoff, local_E_field=1, plot_data_filename = 'plot_data_all_cells_750to900.npy')
    run_passive_simulation_Ez(freq, neurons[900:], remove_list, tstop, dt, cutoff, local_E_field=1, plot_data_filename ='plot_data_all_cells_900to1034.npy')

    run_passive_simulation_Ex(freq, neurons[0:150], remove_list, tstop, dt, cutoff, local_E_field=1, plot_data_filename = 'plot_data_all_cells_Ex_1to150.npy')
    run_passive_simulation_Ex(freq, neurons[150:300], remove_list, tstop, dt, cutoff, local_E_field=1, plot_data_filename = 'plot_data_all_cells_Ex_150to300.npy')
    run_passive_simulation_Ex(freq, neurons[300:450], remove_list, tstop, dt, cutoff, local_E_field=1, plot_data_filename = 'plot_data_all_cells_Ex_300to450.npy')
    run_passive_simulation_Ex(freq, neurons[450:600], remove_list, tstop, dt, cutoff, local_E_field=1, plot_data_filename = 'plot_data_all_cells_Ex_450to600.npy')
    run_passive_simulation_Ex(freq, neurons[600:750], remove_list, tstop, dt, cutoff, local_E_field=1, plot_data_filename = 'plot_data_all_cells_Ex_600to750.npy')
    run_passive_simulation_Ex(freq, neurons[750:900], remove_list, tstop, dt, cutoff, local_E_field=1, plot_data_filename = 'plot_data_all_cells_Ex_750to900.npy')
    run_passive_simulation_Ex(freq, neurons[900:], remove_list, tstop, dt, cutoff, local_E_field=1, plot_data_filename = 'plot_data_all_cells_Ex_900to1034.npy')

    run_passive_simulation_Ey(freq, neurons[0:150], remove_list, tstop, dt, cutoff, local_E_field=1, plot_data_filename = 'plot_data_all_cells_Ey_1to150.npy')
    run_passive_simulation_Ey(freq, neurons[150:300], remove_list, tstop, dt, cutoff, local_E_field=1, plot_data_filename = 'plot_data_all_cells_Ey_150to300.npy')
    run_passive_simulation_Ey(freq, neurons[300:450], remove_list, tstop, dt, cutoff, local_E_field=1, plot_data_filename = 'plot_data_all_cells_Ey_300to450.npy')
    run_passive_simulation_Ey(freq, neurons[450:600], remove_list, tstop, dt, cutoff, local_E_field=1, plot_data_filename = 'plot_data_all_cells_Ey_450to600.npy')
    run_passive_simulation_Ey(freq, neurons[600:750], remove_list, tstop, dt, cutoff, local_E_field=1, plot_data_filename = 'plot_data_all_cells_Ey_600to750.npy')
    run_passive_simulation_Ey(freq, neurons[750:900], remove_list, tstop, dt, cutoff, local_E_field=1, plot_data_filename = 'plot_data_all_cells_Ey_750to900.npy')
    run_passive_simulation_Ey(freq, neurons[900:], remove_list, tstop, dt, cutoff, local_E_field=1, plot_data_filename = 'plot_data_all_cells_Ey_900to1034.npy')
